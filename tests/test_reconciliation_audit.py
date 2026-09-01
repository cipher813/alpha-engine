"""Tests for executor/reconciliation_audit.py (config#859).

Locks the ledger-vs-IB integrity contract: position-parity match_rate from
ledger-reconstructed net shares vs IB snapshot, the daily-delta check, and
the explicit avoidance of the IB-vs-IB NAV tautology.
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from executor.reconciliation_audit import (
    build_reconciliation_audit,
    fetch_same_day_split_ratios,
    reconstruct_ledger_positions,
    same_day_split_ratios,
    write_reconciliation_audit,
)
from polygon_client import PolygonClient, PolygonRateLimitError


def _conn(rows: list[dict]) -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute(
        "CREATE TABLE trades (trade_id TEXT, date TEXT, ticker TEXT, action TEXT, "
        "shares INTEGER, filled_shares INTEGER, status TEXT, fill_time TEXT)"
    )
    for i, r in enumerate(rows):
        c.execute(
            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)",
            (str(i), r.get("date", "2026-06-10"), r["ticker"], r["action"],
             r.get("shares"), r.get("filled_shares"), r.get("status"),
             r.get("fill_time")),
        )
    c.commit()
    return c


def _pos(**kw) -> dict:
    return {t: {"shares": s} for t, s in kw.items()}


class TestReconstruct:
    def test_enter_adds_exit_reduce_subtract(self):
        conn = _conn([
            {"ticker": "AAPL", "action": "ENTER", "filled_shares": 100, "status": "Filled"},
            {"ticker": "AAPL", "action": "REDUCE", "filled_shares": 30, "status": "Filled"},
            {"ticker": "MSFT", "action": "ENTER", "filled_shares": 50, "status": "Filled"},
            {"ticker": "MSFT", "action": "EXIT", "filled_shares": 50, "status": "Filled"},
        ])
        assert reconstruct_ledger_positions(conn, as_of_date="2026-06-30") == {"AAPL": 70}

    def test_rejected_and_unknown_excluded(self):
        conn = _conn([
            {"ticker": "NVDA", "action": "ENTER", "filled_shares": 10, "status": "Rejected"},
            {"ticker": "NVDA", "action": "ENTER", "filled_shares": 5, "status": "Filled"},
            {"ticker": "TSLA", "action": "FROBNICATE", "filled_shares": 9, "status": "Filled"},
        ])
        # Rejected ENTER ignored; unknown action contributes 0 -> TSLA drops.
        assert reconstruct_ledger_positions(conn, as_of_date="2026-06-30") == {"NVDA": 5}

    def test_filled_shares_preferred_over_intended(self):
        conn = _conn([
            {"ticker": "AMD", "action": "ENTER", "shares": 100, "filled_shares": 80, "status": "Filled"},
        ])
        assert reconstruct_ledger_positions(conn, as_of_date="2026-06-30") == {"AMD": 80}

    def test_legacy_null_filled_falls_back_to_shares(self):
        conn = _conn([
            {"ticker": "INTC", "action": "ENTER", "shares": 40, "filled_shares": None, "status": None},
        ])
        assert reconstruct_ledger_positions(conn, as_of_date="2026-06-30") == {"INTC": 40}

    def test_on_date_restricts_to_single_day(self):
        conn = _conn([
            {"ticker": "AAPL", "action": "ENTER", "filled_shares": 100, "status": "Filled", "date": "2026-06-09"},
            {"ticker": "AAPL", "action": "REDUCE", "filled_shares": 25, "status": "Filled", "date": "2026-06-10"},
        ])
        assert reconstruct_ledger_positions(conn, on_date="2026-06-10") == {"AAPL": -25}

    def test_mutually_exclusive_date_args(self):
        conn = _conn([])
        with pytest.raises(ValueError):
            reconstruct_ledger_positions(conn, as_of_date="x", on_date="y")

    # ── config#1454: trading_day-vs-calendar (real fill date wins) ────────

    def test_on_date_uses_fill_time_not_trading_day_tag(self):
        # Tagged date=D-1 (trading_day) but the ACTUAL fill lands D — the
        # daily-delta comparison must attribute it to D (where IB's book
        # would actually reflect it), not D-1.
        conn = _conn([
            {"ticker": "AAPL", "action": "ENTER", "filled_shares": 50,
             "status": "Filled", "date": "2026-06-09", "fill_time": "2026-06-10T09:31:00"},
        ])
        assert reconstruct_ledger_positions(conn, on_date="2026-06-10") == {"AAPL": 50}
        assert reconstruct_ledger_positions(conn, on_date="2026-06-09") == {}

    def test_on_date_excludes_trade_tagged_today_but_filled_tomorrow(self):
        # Tagged date=D but the fill's real timestamp is D+1 — must NOT be
        # attributed to D (IB's book on D would not yet show it).
        conn = _conn([
            {"ticker": "MSFT", "action": "ENTER", "filled_shares": 20,
             "status": "Filled", "date": "2026-06-10", "fill_time": "2026-06-11T08:15:00"},
        ])
        assert reconstruct_ledger_positions(conn, on_date="2026-06-10") == {}
        assert reconstruct_ledger_positions(conn, on_date="2026-06-11") == {"MSFT": 20}

    def test_legacy_row_without_fill_time_falls_back_to_date_tag(self):
        # Pre-fill_time-column rows (NULL) keep the old date-tag behavior —
        # no regression for historical data.
        conn = _conn([
            {"ticker": "NVDA", "action": "ENTER", "filled_shares": 5,
             "status": "Filled", "date": "2026-06-10", "fill_time": None},
        ])
        assert reconstruct_ledger_positions(conn, on_date="2026-06-10") == {"NVDA": 5}

    def test_as_of_date_also_uses_effective_fill_date(self):
        # Cumulative-through-date replay: a trade tagged D but filled D+1
        # should NOT be included in a cumulative view as-of D (it hadn't
        # really happened yet); it SHOULD be included as-of D+1.
        conn = _conn([
            {"ticker": "TSLA", "action": "ENTER", "filled_shares": 15,
             "status": "Filled", "date": "2026-06-10", "fill_time": "2026-06-11T09:31:00"},
        ])
        assert reconstruct_ledger_positions(conn, as_of_date="2026-06-10") == {}
        assert reconstruct_ledger_positions(conn, as_of_date="2026-06-11") == {"TSLA": 15}

    def test_daily_delta_no_longer_double_counts_across_the_boundary(self):
        # The config#1454 failure mode: a trade tagged trading_day=D but
        # filled D+1 used to make BOTH days disagree with IB (D: ledger has
        # a fill IB doesn't show yet; D+1: IB shows a fill the ledger
        # attributed to D). Anchored on fill date, only D+1 sees the delta.
        conn = _conn([
            {"ticker": "AAPL", "action": "ENTER", "filled_shares": 30,
             "status": "Filled", "date": "2026-06-17", "fill_time": "2026-06-18T09:31:05"},
        ])
        audit_d = build_reconciliation_audit(
            conn, today_positions=_pos(AAPL=100), prior_positions=_pos(AAPL=100),
            run_date="2026-06-17",
        )
        # D: IB unchanged from prior (100); ledger's real fill isn't D's yet.
        assert audit_d["reconciliation_match_rate"] == 1.0
        assert audit_d["daily_delta"]["mismatches"] == []

        audit_d1 = build_reconciliation_audit(
            conn, today_positions=_pos(AAPL=130), prior_positions=_pos(AAPL=100),
            run_date="2026-06-18",
        )
        # D+1: IB now shows the +30 fill, and the ledger (keyed on fill date)
        # agrees — clean match, no manufactured mismatch on either day.
        assert audit_d1["reconciliation_match_rate"] == 1.0
        assert audit_d1["daily_delta"]["mismatches"] == []


class TestBuildAudit:
    def test_perfect_match_is_green(self):
        # Anchored headline: prior broker book + today's recorded fills == IB.
        conn = _conn([
            {"ticker": "AAPL", "action": "ENTER", "filled_shares": 30, "status": "Filled", "date": "2026-06-18"},
        ])
        audit = build_reconciliation_audit(
            conn, today_positions=_pos(AAPL=130, MSFT=50),
            prior_positions=_pos(AAPL=100, MSFT=50), run_date="2026-06-18",
            ib_nav=1_000_000.0,
        )
        assert audit["reconciliation_match_rate"] == 1.0
        assert audit["anchored"] is True
        assert audit["status"] == "OK"
        assert audit["n_mismatched"] == 0
        assert audit["position_parity"]["basis"] == "anchored"

    def test_ib_only_position_is_mismatch(self):
        # IB holds GOOG that neither the prior snapshot nor today's fills
        # explain — a real same-day integrity hit (expected 0, actual 20).
        conn = _conn([])
        audit = build_reconciliation_audit(
            conn, today_positions=_pos(AAPL=100, GOOG=20),
            prior_positions=_pos(AAPL=100), run_date="2026-06-18",
        )
        assert audit["reconciliation_match_rate"] == 0.5
        assert audit["status"] == "DRIFT"
        kinds = {m["ticker"]: m["kind"] for m in audit["position_parity"]["mismatches"]}
        assert kinds == {"GOOG": "ib_only"}

    def test_share_mismatch_detected(self):
        # Prior 100, ledger records a +30 ENTER today → expected 130, but IB
        # shows only 120: a 10-share unexplained shortfall today.
        conn = _conn([
            {"ticker": "AAPL", "action": "ENTER", "filled_shares": 30, "status": "Filled", "date": "2026-06-18"},
        ])
        audit = build_reconciliation_audit(
            conn, today_positions=_pos(AAPL=120),
            prior_positions=_pos(AAPL=100), run_date="2026-06-18",
        )
        m = audit["position_parity"]["mismatches"][0]
        assert m == {
            "ticker": "AAPL", "prior_ib_shares": 100, "ledger_today_shares": 30,
            "expected_shares": 130, "ib_shares": 120, "delta": -10,
            "kind": "share_mismatch",
        }

    def test_anchoring_clears_pre_ledger_baseline_gap(self):
        # THE config#1301 FIX: IB holds 100 AAPL from before the ledger began;
        # no trades today. Cumulative replay from inception sees an unexplained
        # 100-share IB position → false 0.0 DRIFT. Anchored on the prior broker
        # snapshot (also 100) it reconciles exactly → 1.0 headline.
        conn = _conn([])  # empty ledger (positions predate it)
        audit = build_reconciliation_audit(
            conn, today_positions=_pos(AAPL=100),
            prior_positions=_pos(AAPL=100), run_date="2026-06-18",
        )
        assert audit["reconciliation_match_rate"] == 1.0   # anchored headline
        assert audit["anchored"] is True
        assert audit["status"] == "OK"
        # The cumulative diagnostic still honestly shows the baseline-gap drift.
        cum = audit["cumulative_ledger_parity"]
        assert cum["match_rate"] == 0.0
        assert cum["n_mismatched"] == 1
        assert cum["mismatches"][0]["kind"] == "ib_only"

    def test_cold_start_no_prior_falls_back_to_cumulative(self):
        # No prior snapshot to anchor on → headline falls back to cumulative
        # replay and is flagged anchored=false (never a phantom empty baseline).
        conn = _conn([
            {"ticker": "AAPL", "action": "ENTER", "filled_shares": 100, "status": "Filled"},
        ])
        audit = build_reconciliation_audit(
            conn, today_positions=_pos(AAPL=100, GOOG=20),
            prior_positions=None, run_date="2026-06-18",
        )
        assert audit["anchored"] is False
        assert audit["position_parity"]["basis"] == "cumulative_ledger"
        assert audit["reconciliation_match_rate"] == 0.5  # GOOG unexplained
        assert audit["daily_delta"]["computed"] is False
        # Diagnostic mirrors the headline in the fallback case.
        assert audit["cumulative_ledger_parity"]["match_rate"] == 0.5

    def test_empty_universe_is_vacuous_green(self):
        audit = build_reconciliation_audit(
            _conn([]), today_positions={}, prior_positions={}, run_date="2026-06-18",
        )
        assert audit["reconciliation_match_rate"] == 1.0
        assert audit["anchored"] is True

    def test_daily_delta_matches_recorded_fills(self):
        # Prior held AAPL 100; today IB shows 130; ledger recorded a +30 ENTER today.
        conn = _conn([
            {"ticker": "AAPL", "action": "ENTER", "filled_shares": 100, "status": "Filled", "date": "2026-06-17"},
            {"ticker": "AAPL", "action": "ENTER", "filled_shares": 30, "status": "Filled", "date": "2026-06-18"},
        ])
        audit = build_reconciliation_audit(
            conn, today_positions=_pos(AAPL=130), prior_positions=_pos(AAPL=100),
            run_date="2026-06-18",
        )
        assert audit["daily_delta"]["computed"] is True
        assert audit["daily_delta"]["match_rate"] == 1.0
        assert audit["daily_delta"]["mismatches"] == []

    def test_daily_delta_flags_unrecorded_ib_change(self):
        # IB jumped +30 today but the ledger has no trade for it.
        conn = _conn([
            {"ticker": "AAPL", "action": "ENTER", "filled_shares": 100, "status": "Filled", "date": "2026-06-17"},
        ])
        audit = build_reconciliation_audit(
            conn, today_positions=_pos(AAPL=130), prior_positions=_pos(AAPL=100),
            run_date="2026-06-18",
        )
        dd = audit["daily_delta"]
        assert dd["match_rate"] == 0.0
        assert dd["mismatches"] == [{"ticker": "AAPL", "ib_delta": 30, "ledger_delta": 0}]

    def test_nav_is_informational_not_the_metric(self):
        audit = build_reconciliation_audit(
            _conn([]), today_positions={}, prior_positions=None,
            run_date="2026-06-18", ib_nav=500_000.0,
        )
        assert audit["ib_nav"] == 500_000.0
        # match_rate is built from the ledger, not NAV.
        assert "reconciliation_match_rate" in audit
        assert audit["daily_delta"]["computed"] is False


class TestSameDaySplitRatios:
    """Pure ratio extraction for same-day corporate actions (config#1682)."""

    def test_forward_split_ratio_on_run_date(self):
        splits = {"AAPL": [{"execution_date": "2026-06-18", "split_from": 1, "split_to": 2}]}
        assert same_day_split_ratios(splits, "2026-06-18") == {"AAPL": 2.0}

    def test_reverse_split_ratio(self):
        splits = {"CMPS": [{"execution_date": "2026-06-18", "split_from": 10, "split_to": 1}]}
        assert same_day_split_ratios(splits, "2026-06-18") == {"CMPS": 0.1}

    def test_ignores_splits_dated_other_days(self):
        splits = {"AAPL": [{"execution_date": "2026-06-17", "split_from": 1, "split_to": 2}]}
        assert same_day_split_ratios(splits, "2026-06-18") == {}

    def test_multiple_same_day_splits_compound(self):
        splits = {"X": [
            {"execution_date": "2026-06-18", "split_from": 1, "split_to": 2},
            {"execution_date": "2026-06-18", "split_from": 1, "split_to": 3},
        ]}
        assert same_day_split_ratios(splits, "2026-06-18") == {"X": 6.0}

    def test_malformed_and_nonpositive_skipped_not_defaulted(self):
        splits = {
            "BAD": [{"execution_date": "2026-06-18", "split_to": 2}],          # missing split_from
            "ZERO": [{"execution_date": "2026-06-18", "split_from": 0, "split_to": 2}],
        }
        # Neither silently becomes 1.0 — they are simply absent.
        assert same_day_split_ratios(splits, "2026-06-18") == {}


class TestCorporateActionAdjustment:
    """A same-day split no longer produces a false reconciliation mismatch."""

    def _split_day_audit(self, corporate_actions):
        # AAPL 2-for-1 split on 2026-06-18: IB doubles 100 -> 200 with NO ledger
        # trade. Prior snapshot holds the pre-split 100.
        conn = _conn([
            {"ticker": "AAPL", "action": "ENTER", "filled_shares": 100,
             "status": "Filled", "date": "2026-06-01"},
        ])
        return build_reconciliation_audit(
            conn,
            today_positions=_pos(AAPL=200),
            prior_positions=_pos(AAPL=100),
            run_date="2026-06-18",
            corporate_actions=corporate_actions,
        )

    def test_unadjusted_split_day_false_mismatches(self):
        # Baseline: without the corporate-action ratio the anchored parity reads
        # a false mismatch (expected 100 != IB 200) — the bug this issue closes.
        audit = self._split_day_audit(corporate_actions=None)
        assert audit["reconciliation_match_rate"] < 1.0
        assert audit["status"] == "DRIFT"
        assert audit["corporate_actions_applied"] == {}

    def test_split_adjusted_reconciles_clean(self):
        # With the 2.0 ratio the pre-split baseline is rebased 100 -> 200 and the
        # day reconciles exactly; the cumulative diagnostic is rebased too.
        audit = self._split_day_audit(corporate_actions={"AAPL": 2.0})
        assert audit["reconciliation_match_rate"] == 1.0
        assert audit["status"] == "OK"
        assert audit["corporate_actions_applied"] == {"AAPL": 2.0}
        assert audit["cumulative_ledger_parity"]["match_rate"] == 1.0

    def test_real_fill_on_split_day_still_reconciles(self):
        # Pre-split 100 (snapshot), a real same-day BUY of 20 recorded post-split,
        # then a 2:1 split of the 100 carry -> IB = 100*2 + 20 = 220.
        conn = _conn([
            {"ticker": "AAPL", "action": "ENTER", "filled_shares": 100,
             "status": "Filled", "date": "2026-06-01"},
            {"ticker": "AAPL", "action": "ENTER", "filled_shares": 20,
             "status": "Filled", "date": "2026-06-18"},
        ])
        audit = build_reconciliation_audit(
            conn,
            today_positions=_pos(AAPL=220),
            prior_positions=_pos(AAPL=100),
            run_date="2026-06-18",
            corporate_actions={"AAPL": 2.0},
        )
        assert audit["reconciliation_match_rate"] == 1.0


class TestFetchSameDaySplitRatios:
    """alpha-engine-config-I9646 — ONE date-filtered query, not N per-ticker ones."""

    def test_one_query_resolves_the_whole_book(self):
        client = MagicMock()
        client.get_splits_for_date.return_value = [
            {"ticker": "AAPL", "execution_date": "2026-06-18",
             "split_from": 1, "split_to": 2},
            # A split on a name we do not hold — must not leak into the result.
            {"ticker": "WZRD", "execution_date": "2026-06-18",
             "split_from": 15, "split_to": 1},
        ]
        out, unresolved = fetch_same_day_split_ratios(
            ["AAPL", "MSFT"], "2026-06-18", client=client,
        )
        assert out == {"AAPL": 2.0}
        # MSFT's absence from the day's split set is a POSITIVE finding of no
        # split, not an unmeasured one — that is the whole point of I9646.
        assert unresolved == []
        client.get_splits_for_date.assert_called_once_with("2026-06-18")

    def test_call_count_does_not_scale_with_the_book(self):
        """The defect was N calls against a 5/min budget. One query, always."""
        client = MagicMock()
        client.get_splits_for_date.return_value = []
        _, unresolved = fetch_same_day_split_ratios(
            [f"T{i}" for i in range(50)], "2026-06-18", client=client,
        )
        assert client.get_splits_for_date.call_count == 1
        assert unresolved == []

    def test_a_split_dated_another_day_is_not_applied(self):
        client = MagicMock()
        client.get_splits_for_date.return_value = [
            {"ticker": "AAPL", "execution_date": "2026-06-17",
             "split_from": 1, "split_to": 2},
        ]
        out, unresolved = fetch_same_day_split_ratios(
            ["AAPL"], "2026-06-18", client=client,
        )
        assert out == {}
        assert unresolved == []

    def test_query_failure_leaves_every_ticker_unresolved(self):
        """One query covers the whole book, so its failure leaves the whole
        book unmeasured — all-or-nothing, never a silent partial clean."""
        client = MagicMock()
        client.get_splits_for_date.side_effect = RuntimeError("polygon down")
        ratios, unresolved = fetch_same_day_split_ratios(
            ["AAPL", "MSFT"], "2026-06-18", client=client,
        )
        assert ratios == {}
        assert unresolved == ["AAPL", "MSFT"]

    def test_rate_limit_is_reported_unresolved_not_assumed_absent(self):
        """alpha-engine-config-I9630/I9641. The pre-I9630 contract returned {}
        and logged 'treating as no same-day action', which made an unmeasured
        split indistinguishable from a checked one."""
        client = MagicMock()
        client.get_splits_for_date.side_effect = PolygonRateLimitError(
            "Rate limited after 3 retries"
        )
        ratios, unresolved = fetch_same_day_split_ratios(
            ["AAPL"], "2026-06-18", client=client,
        )
        assert ratios == {}
        assert unresolved == ["AAPL"]

    def test_a_client_that_cannot_be_built_leaves_every_ticker_unresolved(self):
        # Not zero of them: this branch pre-I9630 returned {} and the whole
        # book silently read as split-checked.
        ratios, unresolved = fetch_same_day_split_ratios(
            ["AAPL", "MSFT"], "2026-06-18", client=None,
        )
        assert ratios == {}
        assert sorted(unresolved) == ["AAPL", "MSFT"]

    def test_empty_tickers_short_circuits(self):
        client = MagicMock()
        assert fetch_same_day_split_ratios(
            [], "2026-06-18", client=client,
        ) == ({}, [])
        client.get_splits_for_date.assert_not_called()


class TestGetSplitsForDate:
    """The client method behind the single query (alpha-engine-config-I9646)."""

    def _client(self):
        c = PolygonClient.__new__(PolygonClient)  # bypass __init__ (needs a key)
        return c

    def test_exact_date_filter_not_a_gte_cursor(self):
        """The gotcha in I9646: `start`/`execution_date.gte` is a cursor BOUND,
        `execution_date` is an exact-match filter. A wrong parameter here fails
        OPEN into 'no splits found', which is the silent-clean failure I9630
        closed."""
        c = self._client()
        c._get = MagicMock(return_value={"results": []})
        c.get_splits_for_date("2026-08-31")
        path, kwargs = c._get.call_args[0][0], c._get.call_args[1]
        assert path == "/v3/reference/splits"
        assert kwargs["params"]["execution_date"] == "2026-08-31"
        assert "execution_date.gte" not in kwargs["params"]
        assert "ticker" not in kwargs["params"]

    def test_paginates_on_next_url(self):
        c = self._client()
        c._get = MagicMock(return_value={
            "results": [{"ticker": "A"}], "next_url": "https://x/page2",
        })
        c._get_raw_url = MagicMock(side_effect=[
            {"results": [{"ticker": "B"}], "next_url": "https://x/page3"},
            {"results": [{"ticker": "C"}]},
        ])
        rows = c.get_splits_for_date("2026-08-31")
        assert [r["ticker"] for r in rows] == ["A", "B", "C"]
        assert c._get_raw_url.call_count == 2

    def test_raises_rather_than_reporting_an_unmeasured_day_as_empty(self):
        c = self._client()
        c._get = MagicMock(side_effect=PolygonRateLimitError("rate limited"))
        with pytest.raises(PolygonRateLimitError):
            c.get_splits_for_date("2026-08-31")


class TestWrite:
    def test_write_key_and_payload(self):
        stub = MagicMock()
        key = write_reconciliation_audit(
            {"reconciliation_match_rate": 1.0}, bucket="alpha-engine-research",
            run_date="2026-06-18", s3_client=stub,
        )
        assert key == "trades/2026-06-18/reconciliation_audit.json"
        _, kwargs = stub.put_object.call_args
        assert kwargs["Key"] == key
        assert json.loads(kwargs["Body"].decode())["reconciliation_match_rate"] == 1.0
