"""Residual-window basis purity and the sleeve backfill that gives it depth.

This file protects the defect that failed `eod-2026-08-24-1787601606`, the
FIRST postclose run after the integrity gates shipped. The three sleeve
columns were added as bare ``ALTER TABLE ... ADD COLUMN``, so the trailing
window held 62 rows of the RAW plug and one true residual. The raw plug
carries realized rotation P&L and sums to -$20,293 over that window by
construction; the bound is derived for the true residual, measured at +$522.
The gate therefore breached at -$19,750 on a book whose same-day true residual
was -$637, and would have breached every run after it.

The invariant: the cumulative window holds ONE quantity, or the check does not
run and says so.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from executor.pnl_backfill import (
    Unreconstructible,
    backfill_residual_sleeves,
    reconstruct_sleeves,
)
from executor.pnl_integrity import (
    RESIDUAL_CUMULATIVE_WINDOW_SESSIONS,
    check_residual_bounds,
)
from executor.trade_logger import init_db, log_eod, log_trade

NAV = 1_026_629.14


@pytest.fixture()
def conn(tmp_path):
    c = init_db(str(tmp_path / "trades.db"))
    yield c
    c.close()


def _positions(**tickers):
    return {
        t: {"shares": sh, "closing_price": px, "market_value": sh * px}
        for t, (sh, px) in tickers.items()
    }


# ── The production scenario ──────────────────────────────────────────────────


def test_raw_plug_history_would_breach_a_true_residual_bound():
    """The 2026-08-24 failure, reproduced: 62 raw plugs + one true residual."""
    raw_plug_history = [-330.0] * (RESIDUAL_CUMULATIVE_WINDOW_SESSIONS - 1)  # ≈ -$20k
    breaches = check_residual_bounds(
        unattributed_true_usd=-637.39,
        nav=NAV,
        trailing_residuals_usd=raw_plug_history,
        run_date="2026-08-24",
    )
    assert [b["kind"] for b in breaches] == ["cumulative"]

    # Same day, same bound, a window on the SAME basis: no breach. The gate was
    # measuring the mixture, not the book.
    true_history = [-4.10] * (RESIDUAL_CUMULATIVE_WINDOW_SESSIONS - 1)
    assert check_residual_bounds(
        unattributed_true_usd=-637.39,
        nav=NAV,
        trailing_residuals_usd=true_history,
        run_date="2026-08-24",
    ) == []


def test_cumulative_check_is_skipped_when_today_is_the_raw_plug():
    """A plug summed into a true-residual window is the same defect, one row."""
    true_history = [-4.10] * (RESIDUAL_CUMULATIVE_WINDOW_SESSIONS - 1)
    breaches = check_residual_bounds(
        unattributed_true_usd=-30_000.0,   # a raw plug: rotation still inside it
        nav=NAV,
        trailing_residuals_usd=true_history,
        run_date="2026-08-24",
        basis_is_true_residual=False,
    )
    # The per-session bound still applies — loudly. The cumulative one does not
    # run at all rather than running on two different quantities.
    assert [b["kind"] for b in breaches] == ["per_session"]


def test_a_genuine_systematic_gap_still_breaches_on_one_basis():
    """Basis purity must not be a way to switch the gate off."""
    drifting = [-400.0] * (RESIDUAL_CUMULATIVE_WINDOW_SESSIONS - 1)
    breaches = check_residual_bounds(
        unattributed_true_usd=-400.0,
        nav=NAV,
        trailing_residuals_usd=drifting,
        run_date="2026-08-24",
    )
    assert [b["kind"] for b in breaches] == ["cumulative"]


def test_a_short_window_still_fires_on_an_absolute_breach():
    """The bound is absolute dollars, so two sessions can breach it."""
    breaches = check_residual_bounds(
        unattributed_true_usd=-12_000.0,
        nav=NAV,
        trailing_residuals_usd=[-4.10],
        run_date="2026-08-24",
    )
    assert {b["kind"] for b in breaches} == {"per_session", "cumulative"}
    cumulative = next(b for b in breaches if b["kind"] == "cumulative")
    assert cumulative["n_sessions"] == 2, "window depth must be reported honestly"


# ── Reconstruction ───────────────────────────────────────────────────────────


def test_reconstruct_sleeves_matches_the_live_decomposition():
    prior = {
        "date": "2026-08-20",
        "portfolio_nav": 1_000_000.0,
        "total_cash": 100_000.0,
        "accrued_interest": 0.0,
        "positions_snapshot": json.dumps(_positions(AAA=(100, 50.0))),
    }
    row = {
        "date": "2026-08-21",
        "portfolio_nav": 1_002_000.0,
        "total_cash": 102_000.0,
        "accrued_interest": 0.0,
        "unattributed_usd": 900.0,
        "positions_snapshot": json.dumps(_positions(AAA=(60, 52.0))),
    }
    trades = [{"ticker": "AAA", "action": "REDUCE", "shares": 40, "price": 53.0}]

    sleeves = reconstruct_sleeves(row=row, prior_row=prior, trades_today=trades)

    # 40 shares sold at 53.00 against a 50.00 prior close.
    assert sleeves["rotation_realized_usd"] == pytest.approx(120.0)
    # mark_basis today  = 1_002_000 - (102_000 + 60*52) = 896_880
    # mark_basis prior  = 1_000_000 - (100_000 + 100*50) = 895_000
    assert sleeves["pricing_timing_usd"] == pytest.approx(1_880.0)
    assert sleeves["unattributed_true_usd"] == pytest.approx(900.0 - 120.0 - 1_880.0)


def test_a_rotation_with_no_priced_fill_is_refused_not_zeroed():
    """A $0 rotation is raw-plug contamination wearing the true residual's name."""
    prior = {
        "date": "2026-08-20",
        "portfolio_nav": 1_000_000.0,
        "total_cash": 100_000.0,
        "positions_snapshot": json.dumps(_positions(AAA=(100, 50.0))),
    }
    row = {
        "date": "2026-08-21",
        "portfolio_nav": 1_002_000.0,
        "total_cash": 102_000.0,
        "unattributed_usd": 900.0,
        "positions_snapshot": json.dumps(_positions(AAA=(60, 52.0))),
    }
    with pytest.raises(Unreconstructible, match="no priced sell fill"):
        reconstruct_sleeves(row=row, prior_row=prior, trades_today=[])


def test_a_prior_position_without_a_settled_close_is_refused():
    prior = {
        "date": "2026-08-20",
        "portfolio_nav": 1_000_000.0,
        "total_cash": 100_000.0,
        "positions_snapshot": json.dumps({"AAA": {"shares": 100, "market_value": 5_000}}),
    }
    row = {
        "date": "2026-08-21",
        "portfolio_nav": 1_002_000.0,
        "total_cash": 102_000.0,
        "unattributed_usd": 900.0,
        "positions_snapshot": json.dumps(_positions(AAA=(100, 52.0))),
    }
    with pytest.raises(Unreconstructible, match="closing_price"):
        reconstruct_sleeves(row=row, prior_row=prior, trades_today=[])


# ── The backfill against the real schema ─────────────────────────────────────


def _log(conn, date, nav, cash, unattributed, positions, **extra):
    log_eod(conn, {
        "date": date,
        "portfolio_nav": nav,
        "total_cash": cash,
        "accrued_interest": 0.0,
        "unattributed_usd": unattributed,
        "positions_snapshot": positions,
        **extra,
    })


def test_backfill_fills_null_sleeves_and_leaves_the_rest_alone(conn):
    _log(conn, "2026-08-20", 1_000_000.0, 100_000.0, 0.0, _positions(AAA=(100, 50.0)))
    _log(conn, "2026-08-21", 1_002_000.0, 102_000.0, 900.0, _positions(AAA=(60, 52.0)))
    _log(conn, "2026-08-24", 1_003_000.0, 103_000.0, 10.0, _positions(AAA=(60, 53.0)),
         rotation_realized_usd=0.0, pricing_timing_usd=0.0, unattributed_true_usd=10.0)
    log_trade(conn, {
        "date": "2026-08-21", "ticker": "AAA", "action": "REDUCE",
        "shares": 40, "price_at_order": 53.0,
    })

    result = backfill_residual_sleeves(conn)

    assert result["filled"] == 1
    assert result["skipped"] == []
    conn.row_factory = sqlite3.Row
    rows = {r["date"]: dict(r) for r in conn.execute("SELECT * FROM eod_pnl")}
    conn.row_factory = None
    assert rows["2026-08-21"]["rotation_realized_usd"] == pytest.approx(120.0)
    assert rows["2026-08-21"]["unattributed_true_usd"] == pytest.approx(-1_100.0)
    # The already-populated row is untouched, and the first row has no prior.
    assert rows["2026-08-24"]["unattributed_true_usd"] == pytest.approx(10.0)
    assert rows["2026-08-20"]["unattributed_true_usd"] is None


def test_backfill_is_idempotent(conn):
    _log(conn, "2026-08-20", 1_000_000.0, 100_000.0, 0.0, _positions(AAA=(100, 50.0)))
    _log(conn, "2026-08-21", 1_002_000.0, 102_000.0, 900.0, _positions(AAA=(100, 52.0)))

    first = backfill_residual_sleeves(conn)
    second = backfill_residual_sleeves(conn)

    assert first["filled"] == 1
    assert second["filled"] == 0


def test_an_unreconstructible_row_is_named_not_substituted(conn):
    _log(conn, "2026-08-20", 1_000_000.0, 100_000.0, 0.0, _positions(AAA=(100, 50.0)))
    # Rotated out with no priced fill: must stay NULL, and be reported.
    _log(conn, "2026-08-21", 1_002_000.0, 102_000.0, 900.0, _positions(AAA=(60, 52.0)))

    result = backfill_residual_sleeves(conn)

    assert result["filled"] == 0
    assert [d for d, _ in result["skipped"]] == ["2026-08-21"]
    row = conn.execute(
        "SELECT unattributed_true_usd FROM eod_pnl WHERE date='2026-08-21'"
    ).fetchone()
    assert row[0] is None


# ── The cost vocabulary the slippage leg is classified by ────────────────────


def test_a_forced_exit_is_priced_into_slippage_not_dropped():
    """LIQUIDATION_SELL/EMERGENCY_SELL are real actions; both were unclassified,
    so their shortfall vanished while their notional still diluted the bps."""
    from executor.pnl_integrity import session_costs

    fills = [
        {"action": "EMERGENCY_SELL", "shares": 100, "fill_price": 49.0,
         "price_at_order": 50.0},
        {"action": "LIQUIDATION_SELL", "shares": 100, "fill_price": 48.0,
         "price_at_order": 50.0},
    ]
    costs = session_costs(fills)

    # Sold below arrival on both: side −1 × 100 × (49 − 50) = +$100, +$200.
    assert costs["slippage_usd"] == pytest.approx(300.0)
    assert costs["n_fills_unclassified_action"] == 0


def test_an_unclassified_action_is_counted_so_the_dilution_is_visible():
    from executor.pnl_integrity import session_costs

    costs = session_costs([
        {"action": "SOMETHING_NEW", "shares": 100, "fill_price": 49.0,
         "price_at_order": 50.0},
    ])

    assert costs["slippage_usd"] == 0.0
    assert costs["traded_notional_usd"] == pytest.approx(4_900.0)
    assert costs["n_fills_unclassified_action"] == 1
