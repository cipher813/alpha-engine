"""alpha-engine-config-I8188, second pass — the HISTORICAL measurement series.

The forward path's gates were closed by PR490/491/509. These tests cover what
those left: the benchmark leg (which no gate touched at all), and the backfill
that puts a cost line, a dividend line and a total-return benchmark onto the
114 sessions written before any of it existed.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from executor import pnl_measurement_backfill as B
from executor.pnl_integrity import (
    BENCHMARK_ANCHOR_TOLERANCE_BPS,
    check_benchmark_vendor_anchor,
    gross_net_returns,
    session_costs,
    verify_benchmark_chain_closes,
)

# ─────────────────────────────────────────────────────────────────────────────
# Defect 2 — an ABSENT commission must never render as a measured $0.00
# ─────────────────────────────────────────────────────────────────────────────

def test_absent_commission_is_none_not_zero():
    """Fills executed, IB attached no commissionReport → None, and the flag agrees."""
    costs = session_costs([
        {"action": "BUY", "filled_shares": 10, "fill_price": 100.0,
         "price_at_order": 99.0},
    ])
    assert costs["commission_usd"] is None
    assert costs["commission_available"] is False
    assert costs["slippage_usd"] == pytest.approx(10.0)


def test_no_fills_commission_is_a_measured_zero():
    """No trades → $0.00 is a real measurement, not an absence."""
    costs = session_costs([])
    assert costs["commission_usd"] == 0.0
    assert costs["commission_available"] is True


def test_present_commission_is_summed():
    costs = session_costs([
        {"action": "BUY", "filled_shares": 10, "fill_price": 100.0,
         "price_at_order": 100.0, "commission_usd": 1.25},
        {"action": "SELL", "filled_shares": 5, "fill_price": 100.0,
         "price_at_order": 100.0},
    ])
    assert costs["commission_usd"] == pytest.approx(1.25)
    assert costs["commission_available"] is True


def test_gross_return_is_suppressed_when_commission_is_absent():
    """A gross return with a $0.00 commission leg is a net return mislabelled."""
    split = gross_net_returns(
        nav_change_usd=1_000.0, prior_nav=1_000_000.0,
        commission_usd=None, slippage_usd=500.0,
    )
    assert split["daily_return_net_pct"] == pytest.approx(0.1)
    assert split["daily_return_gross_pct"] is None
    assert split["total_cost_usd"] is None
    assert split["gross_available"] is False
    assert "commission absent" in split["gross_unavailable_reason"]


def test_gross_return_published_when_commission_is_known():
    split = gross_net_returns(
        nav_change_usd=1_000.0, prior_nav=1_000_000.0,
        commission_usd=100.0, slippage_usd=500.0,
    )
    assert split["daily_return_gross_pct"] == pytest.approx(0.16)
    assert split["gross_available"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Defect 1/3 — the benchmark leg
# ─────────────────────────────────────────────────────────────────────────────

def _row(date, close, ret, nav=1_000_000.0, port=0.0, div=None):
    return {"date": date, "spy_close": close, "spy_return_pct": ret,
            "portfolio_nav": nav, "daily_return_pct": port,
            "spy_dividend_per_share": div}


def test_benchmark_chain_closes_on_a_consistent_series():
    rows = [_row("2026-03-09", 100.0, None), _row("2026-03-10", 101.0, 1.0),
            _row("2026-03-11", 102.01, 1.0)]
    out = verify_benchmark_chain_closes(rows)
    assert out["closes"] is True
    assert out["offenders"] == []


def test_benchmark_chain_flags_a_disagreeing_row():
    rows = [_row("2026-03-09", 100.0, None), _row("2026-03-10", 101.0, 1.0),
            _row("2026-03-11", 102.01, 0.2)]  # stored says 0.2, closes say 1.0
    out = verify_benchmark_chain_closes(rows)
    assert out["closes"] is False
    assert [o["date"] for o in out["offenders"]] == ["2026-03-11"]
    assert "daily_alpha_pct" in out["message"]


def test_a_non_contiguous_pair_is_excluded_from_both_chains():
    """The live 2026-03-13 case: a missing session makes the close leg span two.

    Without the calendar the close-implied return is a two-session move against
    a one-session stored return, and the gate reports a 151bp defect that does
    not exist. This is the check that would have stopped the first draft of
    this module from rewriting a CORRECT stored value.
    """
    rows = [_row("2026-03-11", 100.0, None), _row("2026-03-13", 98.0, -1.0)]
    naive = verify_benchmark_chain_closes(rows)
    assert naive["closes"] is False  # -2.0% implied vs -1.0% stored

    def prior_session_of(date):
        return {"2026-03-13": "2026-03-12"}.get(date, "")

    calendar_aware = verify_benchmark_chain_closes(
        rows, prior_session_of=prior_session_of,
    )
    assert calendar_aware["status"] == "n/a"
    assert calendar_aware["excluded_pairs"] == [
        {"date": "2026-03-13", "reason": "non-contiguous session pair"}
    ]


def test_dividends_enter_the_implied_leg():
    """A price-return close with the distribution added is the total-return leg."""
    rows = [_row("2026-06-17", 100.0, None),
            _row("2026-06-18", 100.0, 2.0, div=2.0)]
    out = verify_benchmark_chain_closes(rows)
    assert out["closes"] is True


# ─────────────────────────────────────────────────────────────────────────────
# The vendor anchor — the one measurement this system does not write
# ─────────────────────────────────────────────────────────────────────────────

def test_vendor_anchor_catches_a_close_basis_divergence():
    """The live case: 8 rows carrying dividend-BACK-ADJUSTED closes at -27.2bp."""
    rows = [_row("2026-03-09", 676.4227, None), _row("2026-03-10", 675.3356, -0.1607)]
    breaches = check_benchmark_vendor_anchor(
        rows, vendor_closes={"2026-03-09": 678.27, "2026-03-10": 677.18},
    )
    kinds = [b["kind"] for b in breaches]
    assert kinds.count("benchmark_close_divergence") == 2
    assert all(
        b["divergence_bps"] == pytest.approx(-27.2, abs=0.5)
        for b in breaches if b["kind"] == "benchmark_close_divergence"
    )


def test_vendor_anchor_catches_whole_window_drift():
    rows = [_row("2026-03-09", 100.0, None), _row("2026-03-10", 100.0, 5.0)]
    breaches = check_benchmark_vendor_anchor(
        rows, vendor_closes={"2026-03-09": 100.0, "2026-03-10": 100.0},
    )
    drift = [b for b in breaches if b["kind"] == "benchmark_anchor_drift"]
    assert len(drift) == 1
    assert drift[0]["drift_bps"] == pytest.approx(500.0)
    assert drift[0]["tolerance_bps"] == BENCHMARK_ANCHOR_TOLERANCE_BPS


def test_vendor_anchor_reports_its_own_coverage_holes():
    """The live 2026-04-03 case: an eod_pnl row on a market holiday.

    An anchor that is silent on a session has not agreed with it.
    """
    rows = [_row("2026-04-02", 100.0, None), _row("2026-04-03", 100.0, 0.0)]
    breaches = check_benchmark_vendor_anchor(
        rows, vendor_closes={"2026-04-02": 100.0},
    )
    coverage = [b for b in breaches if b["kind"] == "benchmark_vendor_coverage"]
    assert coverage and coverage[0]["missing_sessions"] == ["2026-04-03"]


def test_vendor_anchor_not_evaluated_is_not_a_pass():
    audit = B.audit_history(
        [_row("2026-03-09", 100.0, None), _row("2026-03-10", 101.0, 1.0)],
        vendor_closes=None,
    )
    assert audit["benchmark_vendor_anchor_status"].startswith("NOT EVALUATED")
    assert audit["benchmark_vendor_anchor_breaches"] == []


# ─────────────────────────────────────────────────────────────────────────────
# The restatement is planned, never taken
# ─────────────────────────────────────────────────────────────────────────────

def _prior(date):
    return {"2026-03-10": "2026-03-09", "2026-03-11": "2026-03-10"}.get(date, "")


def test_restatement_is_vendor_anchored_and_restates_alpha_alongside():
    rows = [{"date": "2026-03-10", "spy_return_pct": 0.0, "daily_return_pct": 1.0,
             "daily_alpha_pct": 1.0}]
    plan = B.plan_benchmark_restatement(
        rows, vendor_closes={"2026-03-09": 100.0, "2026-03-10": 101.0},
        vendor_dividends={}, prior_session_of=_prior,
    )
    assert len(plan["corrections"]) == 1
    c = plan["corrections"][0]
    assert c["to_pct"] == pytest.approx(1.0)
    assert c["to_alpha_pct"] == pytest.approx(0.0)


def test_restatement_refuses_a_row_the_vendor_cannot_cover():
    rows = [{"date": "2026-03-10", "spy_return_pct": 0.0, "daily_return_pct": 1.0}]
    plan = B.plan_benchmark_restatement(
        rows, vendor_closes={"2026-03-10": 101.0}, vendor_dividends={},
        prior_session_of=_prior,
    )
    assert plan["corrections"] == []
    assert "vendor has no close" in plan["refused"][0]["reason"]


def test_restatement_refuses_when_alpha_cannot_be_restated_alongside():
    rows = [{"date": "2026-03-10", "spy_return_pct": 0.0, "daily_return_pct": None}]
    plan = B.plan_benchmark_restatement(
        rows, vendor_closes={"2026-03-09": 100.0, "2026-03-10": 101.0},
        vendor_dividends={}, prior_session_of=_prior,
    )
    assert plan["corrections"] == []
    assert "internally inconsistent" in plan["refused"][0]["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# Ex-dividend mapping and the position dividend leg
# ─────────────────────────────────────────────────────────────────────────────

def test_ex_dividend_maps_to_the_session_that_spans_it():
    rows = [{"date": "2026-06-17"}, {"date": "2026-06-19"}]
    out = B.map_ex_dividends_to_sessions(
        rows, [{"ex_dividend_date": "2026-06-18", "cash_amount": 1.9}],
    )
    assert out == {"2026-06-19": 1.9}  # the interval (06-17, 06-19] contains it


def test_ex_dividend_after_the_last_session_is_dropped_not_backdated():
    rows = [{"date": "2026-06-17"}]
    assert B.map_ex_dividends_to_sessions(
        rows, [{"ex_dividend_date": "2026-06-18", "cash_amount": 1.9}],
    ) == {}


def test_dividend_backfill_uses_prior_day_share_count_and_skips_unknowns():
    rows = [
        {"date": "2026-06-17", "positions_snapshot":
         '{"LMT": {"shares": 100, "prior_price": 400.0}}'},
        {"date": "2026-06-18", "positions_snapshot":
         '{"LMT": {"shares": 150, "prior_price": 400.0}}'},
        {"date": "2026-06-19", "positions_snapshot": None},
    ]
    plans = B.plan_dividend_backfill(rows, {"2026-06-18": {"LMT": 3.0}})
    assert plans[0]["skipped"] == "no prior snapshot for entitlement"
    # entitlement settles before the ex-date: 100 prior shares, not today's 150
    assert plans[1]["dividend_usd"] == pytest.approx(300.0)
    assert plans[2]["skipped"] == "no parseable positions_snapshot"


def test_a_skipped_dividend_session_is_left_null_never_written_zero():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE eod_pnl (date TEXT PRIMARY KEY, dividend_usd REAL, "
                 "dividend_accrual_available INTEGER)")
    conn.execute("INSERT INTO eod_pnl VALUES ('2026-06-17', NULL, NULL)")
    conn.commit()
    B.apply_dividend_backfill(conn, [{"date": "2026-06-17", "skipped": "no snapshot"}])
    row = conn.execute("SELECT dividend_usd, dividend_accrual_available "
                       "FROM eod_pnl").fetchone()
    assert row == (None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Cost backfill: idempotent, and NULL where the source cannot support a value
# ─────────────────────────────────────────────────────────────────────────────

def _cost_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE eod_pnl (date TEXT PRIMARY KEY, portfolio_nav REAL, "
        "nav_change_usd REAL, commission_usd REAL, commission_available INTEGER, "
        "slippage_usd REAL, traded_notional_usd REAL, daily_return_net_pct REAL, "
        "daily_return_gross_pct REAL)"
    )
    conn.executemany(
        "INSERT INTO eod_pnl (date, portfolio_nav, nav_change_usd) VALUES (?,?,?)",
        [("2026-03-09", 1_000_000.0, None), ("2026-03-10", 1_001_000.0, 1_000.0)],
    )
    conn.commit()
    return conn


def test_cost_backfill_writes_null_commission_and_suppresses_gross():
    conn = _cost_conn()
    rows = B._rows(conn)
    trades = {"2026-03-10": [
        {"action": "BUY", "filled_shares": 10, "fill_price": 100.0,
         "price_at_order": 99.0},
    ]}
    plans = B.plan_cost_backfill(rows, trades)
    assert B.apply_cost_backfill(conn, plans) == 2
    row = conn.execute(
        "SELECT commission_usd, commission_available, slippage_usd, "
        "daily_return_net_pct, daily_return_gross_pct FROM eod_pnl "
        "WHERE date='2026-03-10'"
    ).fetchone()
    assert row[0] is None            # commission ABSENT, not $0.00
    assert row[1] == 0
    assert row[2] == pytest.approx(10.0)
    assert row[3] == pytest.approx(0.1)
    assert row[4] is None            # gross suppressed with the commission leg


def test_cost_backfill_is_idempotent():
    conn = _cost_conn()
    trades = {"2026-03-10": [], "2026-03-09": []}
    B.apply_cost_backfill(conn, B.plan_cost_backfill(B._rows(conn), trades))
    assert B.plan_cost_backfill(B._rows(conn), trades) == []


# ─────────────────────────────────────────────────────────────────────────────
# The retroactive custodian-mark reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def test_mark_divergence_reconstructed_from_persisted_market_values():
    """ib_mark_outside_range does not exist on historical rows; MV pairs do."""
    row = {
        "date": "2026-08-04", "portfolio_nav": 1_036_000.0,
        "positions_snapshot":
            '{"AMD": {"market_value": 116700.0, "ib_market_value": 107794.0},'
            ' "OK": {"market_value": 50000.0, "ib_market_value": 50100.0}}',
    }
    out = B.reconstruct_mark_divergences(row)
    assert [b["ticker"] for b in out] == ["AMD"]
    assert out[0]["divergence_usd"] == pytest.approx(-8906.0)
    assert out[0]["kind"] == "custodian_mark_divergence"


def test_mark_divergence_needs_both_marks():
    row = {"date": "2026-08-04", "portfolio_nav": 1_000_000.0,
           "positions_snapshot": '{"X": {"market_value": 100000.0}}'}
    assert B.reconstruct_mark_divergences(row) == []


# ─────────────────────────────────────────────────────────────────────────────
# Session-axis coverage in the retroactive audit — alpha-engine-config-I9615
# ─────────────────────────────────────────────────────────────────────────────

import datetime as _dt  # noqa: E402

_FAKE_HOLIDAYS = {"2026-04-03"}


def _fake_is_trading_day(date_str: str) -> bool:
    if date_str in _FAKE_HOLIDAYS:
        return False
    return _dt.date.fromisoformat(date_str).weekday() < 5


def _fake_next_trading_day(date_str: str) -> str:
    d = _dt.date.fromisoformat(date_str) + _dt.timedelta(days=1)
    while not _fake_is_trading_day(d.isoformat()):
        d += _dt.timedelta(days=1)
    return d.isoformat()


def test_audit_history_reports_coverage_not_evaluated_without_a_calendar():
    """A gate that did not run has not agreed with anything — same convention
    as the vendor anchor's own NOT EVALUATED status."""
    audit = B.audit_history(
        [_row("2026-04-02", 100.0, None), _row("2026-04-03", 100.0, 0.0)],
    )
    assert audit["session_axis_coverage_status"].startswith("NOT EVALUATED")
    assert audit["session_axis_coverage"]["breaches"] == []


def test_audit_history_flags_the_live_good_friday_row():
    audit = B.audit_history(
        [_row("2026-04-02", 100.0, None), _row("2026-04-03", 100.0, 0.0)],
        is_trading_day=_fake_is_trading_day, next_trading_day=_fake_next_trading_day,
    )
    assert audit["session_axis_coverage_status"] == "evaluated"
    kinds = {(b["kind"], b["date"]) for b in audit["session_axis_coverage"]["breaches"]}
    assert ("non_trading_day_row", "2026-04-03") in kinds
    assert audit["breach_count"] >= 1


def test_plan_non_trading_day_flags_only_takes_that_breach_kind():
    coverage = {
        "breaches": [
            {"kind": "non_trading_day_row", "date": "2026-04-03", "message": "x"},
            {"kind": "missing_session", "date": "2026-03-12", "message": "y"},
        ],
    }
    plans = B.plan_non_trading_day_flags([], coverage)
    assert [p["date"] for p in plans] == ["2026-04-03"]
    assert plans[0]["breach"]["kind"] == "non_trading_day_row"


def _axis_flag_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE eod_pnl (date TEXT PRIMARY KEY, integrity_breach_json TEXT)"
    )
    conn.executemany(
        "INSERT INTO eod_pnl (date, integrity_breach_json) VALUES (?,?)",
        [("2026-04-03", None), ("2026-08-04", '[{"kind": "residual_breach"}]')],
    )
    conn.commit()
    return conn


def test_apply_non_trading_day_flags_writes_a_fresh_breach_list():
    conn = _axis_flag_conn()
    plans = [{"date": "2026-04-03",
              "breach": {"kind": "non_trading_day_row", "date": "2026-04-03"}}]
    assert B.apply_non_trading_day_flags(conn, plans) == 1
    import json as _json
    stored = _json.loads(
        conn.execute(
            "SELECT integrity_breach_json FROM eod_pnl WHERE date='2026-04-03'"
        ).fetchone()[0]
    )
    assert stored == [{"kind": "non_trading_day_row", "date": "2026-04-03"}]


def test_apply_non_trading_day_flags_merges_never_overwrites():
    """A session already carrying a residual breach must keep it — this is an
    ADDITIVE flag, not a replace."""
    conn = _axis_flag_conn()
    plans = [{"date": "2026-08-04",
              "breach": {"kind": "non_trading_day_row", "date": "2026-08-04"}}]
    assert B.apply_non_trading_day_flags(conn, plans) == 1
    import json as _json
    stored = _json.loads(
        conn.execute(
            "SELECT integrity_breach_json FROM eod_pnl WHERE date='2026-08-04'"
        ).fetchone()[0]
    )
    kinds = {b["kind"] for b in stored}
    assert kinds == {"residual_breach", "non_trading_day_row"}


def test_apply_non_trading_day_flags_is_idempotent():
    conn = _axis_flag_conn()
    plans = [{"date": "2026-04-03",
              "breach": {"kind": "non_trading_day_row", "date": "2026-04-03"}}]
    assert B.apply_non_trading_day_flags(conn, plans) == 1
    assert B.apply_non_trading_day_flags(conn, plans) == 0  # already flagged


# ─────────────────────────────────────────────────────────────────────────────
# Restating the historical NAV series — alpha-engine-config-I9629
# ─────────────────────────────────────────────────────────────────────────────
#
# The fixture is the measured AMD 2026-08-04 instance: 700 shares, broker MV
# $116,700 against a settled MV of $107,794, on a $1,036,000 NAV. Materiality is
# max($500, 15bp) = $1,554 and the correction bound is max($10,000, 100bp) =
# $10,360, so the -$8,906 correction is material AND inside the bound — the same
# two verdicts the live path reaches.

_AMD_SNAP = ('{"AMD": {"shares": 700, "market_value": 107794.0, '
             '"ib_market_value": 116700.0}, '
             '"OK": {"shares": 100, "market_value": 50000.0, '
             '"ib_market_value": 50050.0}}')
_CLEAN_SNAP = ('{"OK": {"shares": 100, "market_value": 50000.0, '
               '"ib_market_value": 50050.0}}')

_NAV_0, _NAV_1, _NAV_2 = 1_000_000.0, 1_036_000.0, 1_040_000.0
_CORRECTION = 107_794.0 - 116_700.0          # -8,906.00
_NAV_1_NEW = _NAV_1 + _CORRECTION            # 1,027,094.00


def _mark_conn(path=":memory:"):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE eod_pnl (date TEXT PRIMARY KEY, portfolio_nav REAL, "
        "daily_return_pct REAL, spy_return_pct REAL, daily_alpha_pct REAL, "
        "positions_snapshot TEXT, nav_ib_raw_usd REAL, "
        "nav_mark_correction_usd REAL, nav_mark_correction_json TEXT)"
    )
    r1 = (_NAV_1 - _NAV_0) / _NAV_0 * 100.0
    r2 = (_NAV_2 - _NAV_1) / _NAV_1 * 100.0
    conn.executemany(
        "INSERT INTO eod_pnl (date, portfolio_nav, daily_return_pct, "
        "spy_return_pct, daily_alpha_pct, positions_snapshot) VALUES (?,?,?,?,?,?)",
        [
            ("2026-08-03", _NAV_0, None, 0.1, None, _CLEAN_SNAP),
            ("2026-08-04", _NAV_1, r1, 0.1, r1 - 0.1, _AMD_SNAP),
            ("2026-08-05", _NAV_2, r2, 0.1, r2 - 0.1, _CLEAN_SNAP),
        ],
    )
    conn.commit()
    return conn


def test_reconstructed_inputs_carry_the_degenerate_range_and_the_live_materiality():
    """The only bound the persisted data justifies is [close, close] — see §5."""
    row = {"date": "2026-08-04", "portfolio_nav": _NAV_1,
           "positions_snapshot": _AMD_SNAP}
    got = B.reconstruct_mark_correction_inputs(row)
    assert [f["ticker"] for f in got["flags"]] == ["AMD"]      # OK is immaterial
    assert got["flags"][0]["ib_mark"] == pytest.approx(116_700.0 / 700)
    assert got["settled_closes"]["AMD"] == pytest.approx(107_794.0 / 700)
    assert got["day_low"]["AMD"] == got["day_high"]["AMD"] == pytest.approx(
        got["settled_closes"]["AMD"]
    )
    assert got["flags"][0]["materiality_usd"] == pytest.approx(1_554.0)


def test_a_name_without_ib_market_value_is_named_and_skipped_never_zeroed():
    """Most of the history is pre-schema-2.1. Silence there would read as clean."""
    row = {"date": "2026-06-02", "portfolio_nav": 1_000_000.0,
           "positions_snapshot": '{"MU": {"shares": 100, "market_value": 50000.0}}'}
    got = B.reconstruct_mark_correction_inputs(row)
    assert got["flags"] == []
    assert got["refused"][0]["ticker"] == "MU"
    assert "pre-schema-2.1" in got["refused"][0]["reason"]


def test_an_unparseable_snapshot_is_named_and_skipped():
    row = {"date": "2026-06-02", "portfolio_nav": 1_000_000.0,
           "positions_snapshot": "{not json"}
    got = B.reconstruct_mark_correction_inputs(row)
    assert got["flags"] == []
    assert got["refused"] == [{
        "date": "2026-06-02", "ticker": None,
        "reason": "positions_snapshot absent or unparseable — the session "
                  "cannot be examined and is NOT reported as clean",
    }]


def test_a_row_without_nav_is_named_and_skipped():
    row = {"date": "2026-06-02", "portfolio_nav": None,
           "positions_snapshot": _AMD_SNAP}
    got = B.reconstruct_mark_correction_inputs(row)
    assert got["flags"] == []
    assert "no portfolio_nav" in got["refused"][0]["reason"]


def test_plan_restates_the_nav_and_labels_the_basis_reconstructed():
    plan = B.plan_historical_mark_restatement(B._rows(_mark_conn()))
    assert len(plan["restatements"]) == 1
    r = plan["restatements"][0]
    assert r["date"] == "2026-08-04"
    assert r["nav_ib_raw_usd"] == pytest.approx(_NAV_1)
    assert r["portfolio_nav"] == pytest.approx(_NAV_1_NEW)
    assert r["nav_mark_correction_usd"] == pytest.approx(_CORRECTION)
    assert r["tickers"] == ["AMD"]
    payload = r["nav_mark_correction_json"]
    assert payload["basis"] == B.MARK_BASIS_RECONSTRUCTED
    assert payload["discriminator_evaluated"] is False
    assert "UPPER BOUND" in payload["instrument"]


def test_the_basis_label_distinguishes_a_restated_row_from_a_live_corrected_one():
    """A consumer tells the two apart by reading one field, never by guessing."""
    plan = B.plan_historical_mark_restatement(B._rows(_mark_conn()))
    assert plan["restatements"][0]["nav_mark_correction_json"]["basis"] == (
        B.MARK_BASIS_RECONSTRUCTED
    )
    assert B.MARK_BASIS_LIVE == "live_gate"
    assert B.MARK_BASIS_RECONSTRUCTED != B.MARK_BASIS_LIVE


def test_the_chain_moves_both_the_corrected_session_and_the_one_after_it():
    plan = B.plan_historical_mark_restatement(B._rows(_mark_conn()))
    moved = {c["date"]: c for c in plan["chain"]}
    assert sorted(moved) == ["2026-08-04", "2026-08-05"]
    c1, c2 = moved["2026-08-04"], moved["2026-08-05"]
    assert c1["daily_return_pct_to"] == pytest.approx(
        (_NAV_1_NEW - _NAV_0) / _NAV_0 * 100.0
    )
    assert c1["daily_alpha_pct_to"] == pytest.approx(c1["daily_return_pct_to"] - 0.1)
    # session t+1's NAV never moved; its return moved because its BASE did
    assert c2["nav_from"] == c2["nav_to"] == pytest.approx(_NAV_2)
    assert c2["daily_return_pct_to"] == pytest.approx(
        (_NAV_2 - _NAV_1_NEW) / _NAV_1_NEW * 100.0
    )
    assert c2["daily_alpha_pct_to"] == pytest.approx(c2["daily_return_pct_to"] - 0.1)


def test_the_chain_refuses_a_row_whose_alpha_cannot_move_alongside():
    conn = _mark_conn()
    conn.execute("UPDATE eod_pnl SET spy_return_pct=NULL WHERE date='2026-08-05'")
    conn.commit()
    plan = B.plan_historical_mark_restatement(B._rows(conn))
    assert [c["date"] for c in plan["chain"]] == ["2026-08-04"]
    assert plan["chain_refused"][0]["date"] == "2026-08-05"
    assert "internally inconsistent" in plan["chain_refused"][0]["reason"]


def test_a_stored_return_that_already_disagreed_with_its_nav_chain_is_named():
    conn = _mark_conn()
    conn.execute("UPDATE eod_pnl SET daily_return_pct=99.0 WHERE date='2026-08-05'")
    conn.commit()
    plan = B.plan_historical_mark_restatement(B._rows(conn))
    assert [m["date"] for m in plan["chain_basis_mismatch"]] == ["2026-08-05"]
    assert plan["chain_basis_mismatch"][0]["stored_daily_return_pct"] == 99.0


def test_a_correction_over_the_bound_is_refused_not_applied():
    """max($10,000, 100bp) — a disagreement that size is a different book."""
    conn = _mark_conn()
    conn.execute(
        "UPDATE eod_pnl SET positions_snapshot=? WHERE date='2026-08-04'",
        ('{"AMD": {"shares": 700, "market_value": 100000.0, '
         '"ib_market_value": 200000.0}}',),
    )
    conn.commit()
    plan = B.plan_historical_mark_restatement(B._rows(conn))
    assert plan["restatements"] == []
    assert plan["chain"] == []
    refused = [r for r in plan["refused"] if r.get("refused_by_bound")]
    assert refused and "REFUSED" in refused[0]["reason"]


def test_a_row_already_restated_by_the_live_path_is_left_alone():
    """The live 2026-08-31 DUOL row: non-null nav_mark_correction_json → no-op."""
    conn = _mark_conn()
    conn.execute(
        "UPDATE eod_pnl SET nav_ib_raw_usd=?, nav_mark_correction_usd=?, "
        "nav_mark_correction_json=? WHERE date='2026-08-04'",
        (_NAV_1, _CORRECTION, '{"basis": "live_gate"}'),
    )
    conn.commit()
    plan = B.plan_historical_mark_restatement(B._rows(conn))
    assert plan["restatements"] == []
    assert plan["chain"] == []
    assert [a["date"] for a in plan["already_restated"]] == ["2026-08-04"]
    assert plan["already_restated"][0]["basis"] == B.MARK_BASIS_LIVE


def test_apply_writes_the_nav_the_original_and_the_chain():
    conn = _mark_conn()
    plan = B.plan_historical_mark_restatement(B._rows(conn))
    assert B.apply_historical_mark_restatement(conn, plan) == {
        "nav_rows_written": 1, "chain_rows_written": 2,
    }
    nav, raw, corr, payload, ret, alpha = conn.execute(
        "SELECT portfolio_nav, nav_ib_raw_usd, nav_mark_correction_usd, "
        "nav_mark_correction_json, daily_return_pct, daily_alpha_pct "
        "FROM eod_pnl WHERE date='2026-08-04'"
    ).fetchone()
    assert nav == pytest.approx(_NAV_1_NEW)
    assert raw == pytest.approx(_NAV_1)          # reversible from the ledger alone
    assert corr == pytest.approx(_CORRECTION)
    import json as _json
    assert _json.loads(payload)["basis"] == B.MARK_BASIS_RECONSTRUCTED
    assert ret == pytest.approx((_NAV_1_NEW - _NAV_0) / _NAV_0 * 100.0)
    assert alpha == pytest.approx(ret - 0.1)
    nxt = conn.execute(
        "SELECT daily_return_pct FROM eod_pnl WHERE date='2026-08-05'"
    ).fetchone()[0]
    assert nxt == pytest.approx((_NAV_2 - _NAV_1_NEW) / _NAV_1_NEW * 100.0)


def test_apply_is_idempotent():
    conn = _mark_conn()
    B.apply_historical_mark_restatement(
        conn, B.plan_historical_mark_restatement(B._rows(conn))
    )
    second = B.plan_historical_mark_restatement(B._rows(conn))
    assert second["restatements"] == []
    assert second["chain"] == []
    assert B.apply_historical_mark_restatement(conn, second) == {
        "nav_rows_written": 0, "chain_rows_written": 0,
    }


def _seed_file_db(tmp_path):
    path = str(tmp_path / "trades.db")
    _mark_conn(path).close()
    return path


def test_cli_dry_run_is_the_default_and_writes_nothing(tmp_path, capsys):
    path = _seed_file_db(tmp_path)
    assert B.main(["--db", path, "--restate-marks"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["applied"] is False
    assert out["mark_restatement"]["status"] == "planned"
    assert out["mark_restatement"]["sessions_restated"] == 1
    assert out["mark_restatement"]["downstream_rows_moved"] == 2
    conn = sqlite3.connect(path)
    assert conn.execute(
        "SELECT portfolio_nav, nav_ib_raw_usd FROM eod_pnl WHERE date='2026-08-04'"
    ).fetchone() == (pytest.approx(_NAV_1), None)


def test_cli_apply_restates_and_a_second_run_is_a_no_op(tmp_path, capsys):
    path = _seed_file_db(tmp_path)
    assert B.main(["--db", path, "--restate-marks", "--apply"]) == 0
    first = json.loads(capsys.readouterr().out)["mark_restatement"]
    assert first["nav_rows_written"] == 1
    assert first["chain_rows_written"] == 2
    assert B.main(["--db", path, "--restate-marks", "--apply"]) == 0
    second = json.loads(capsys.readouterr().out)["mark_restatement"]
    assert second["nav_rows_written"] == 0
    assert second["chain_rows_written"] == 0
    assert second["sessions_restated"] == 0
    assert [a["date"] for a in second["already_restated"]] == ["2026-08-04"]


def test_cli_no_flags_does_not_select_the_restatement_leg():
    """The one leg that rewrites a published NAV is never on by default."""
    assert B._build_parser().parse_args([]).restate_marks is False
    assert B._build_parser().parse_args(["--restate-marks"]).restate_marks is True
    assert B._build_parser().parse_args([]).apply is False
