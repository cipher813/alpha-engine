"""alpha-engine-config-I8188, second pass — the HISTORICAL measurement series.

The forward path's gates were closed by PR490/491/509. These tests cover what
those left: the benchmark leg (which no gate touched at all), and the backfill
that puts a cost line, a dividend line and a total-return benchmark onto the
114 sessions written before any of it existed.
"""

from __future__ import annotations

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
