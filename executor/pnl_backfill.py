"""Reconstruct the P&L attribution sleeves for eod_pnl rows written before
they were persisted (alpha-engine-config-I8188 residual, found by the
2026-08-24 postclose failure).

Why this exists. `pricing_timing_usd`, `rotation_realized_usd` and
`unattributed_true_usd` were added to `eod_pnl` as bare
``ALTER TABLE ... ADD COLUMN`` with no backfill, so every row written before
that migration carries NULL for all three. The cumulative residual gate then
had exactly TWO true-basis sessions of history against a 63-session window.

That is not a cosmetic gap. The cumulative gate is bounded on
``unattributed_true_usd`` — the residual AFTER the rotation and pricing&timing
sleeves are lifted out, measured at +$522 over 74 sessions. The RAW plug it
falls back to sums to −$20,293 over the same window *by construction*, because
realized rotation P&L lives inside it. Summing a window of raw plugs against a
bound derived for the true residual breaches on the first run and every run
after it, which is what failed `eod-2026-08-24-1787601606`.

The window must therefore be one basis, and the honest way to get depth is to
compute the missing sleeves rather than to substitute a different quantity for
them. Every input is already persisted: the day's NAV, cash and accrued
interest are columns; the settled marks are in ``positions_snapshot``; the
sell fills are in ``trades``. A row whose inputs are NOT all present is left
NULL and excluded from the window — named, never substituted.

The reconstruction mirrors ``eod_reconcile.run``'s live computation term for
term. Any divergence between the two is a defect in this module, not a
licence to approximate: a backfilled sleeve is bounded by the same gate as a
live one.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

# How far back to reconstruct. Two quarters — the cumulative gate's own
# 63-session window plus enough history that the window is full on the first
# run after this ships rather than filling in over a quarter.
BACKFILL_LOOKBACK_SESSIONS = 126


class Unreconstructible(Exception):
    """A row's sleeves cannot be reconstructed from what was persisted."""


def _settled_mv_from_marks(positions: dict[str, Any]) -> float:
    """Σ settled market value, priced from the per-name settled close.

    Mirrors ``eod_reconcile``'s PRIOR-day leg, which prices from
    ``closing_price`` when present. A position lacking ``closing_price``
    makes the row unreconstructible rather than silently falling back to the
    IB mark: mixing an IB mark into the settled leg puts the pricing&timing
    difference back into the residual this module exists to remove.
    """
    total = 0.0
    for ticker, pos in (positions or {}).items():
        cp = (pos or {}).get("closing_price")
        if cp is None:
            raise Unreconstructible(f"{ticker} carries no closing_price")
        total += float(cp) * float((pos or {}).get("shares", 0) or 0)
    return total


def reconstruct_sleeves(
    *,
    row: dict[str, Any],
    prior_row: dict[str, Any],
    trades_today: list[dict[str, Any]] | None,
) -> dict[str, float]:
    """Recompute one session's sleeves from persisted state. PURE.

    ``row`` and ``prior_row`` are ``eod_pnl`` rows as dicts, consecutive in
    date order. Raises ``Unreconstructible`` when any input the live path
    required is absent — the caller leaves the row NULL and counts it.
    """
    from executor.eod_report import compute_rotation_realized

    unattributed = row.get("unattributed_usd")
    if unattributed is None:
        raise Unreconstructible("no unattributed_usd persisted")

    nav = row.get("portfolio_nav")
    cash = row.get("total_cash")
    prior_nav = prior_row.get("portfolio_nav")
    prior_cash = prior_row.get("total_cash")
    if nav is None or cash is None or prior_nav is None or prior_cash is None:
        raise Unreconstructible("NAV/cash missing on this row or its prior")

    positions = _positions(row)
    prior_positions = _positions(prior_row)

    # mark_basis = nav_ib − (cash + accrued + Σ settled_mv); the day-over-day
    # difference is the pricing&timing sleeve. The row's OWN leg prices from
    # the stored market_value (already the settled-close override at write
    # time, exactly as the live path's today-leg does); the prior leg prices
    # from closing_price, as the live path's prior-leg does.
    settled_mv_today = sum(
        float((p or {}).get("market_value", 0) or 0) for p in positions.values()
    )
    settled_mv_prior = _settled_mv_from_marks(prior_positions)
    mark_basis_today = float(nav) - (
        float(cash) + float(row.get("accrued_interest") or 0.0) + settled_mv_today
    )
    mark_basis_prior = float(prior_nav) - (
        float(prior_cash)
        + float(prior_row.get("accrued_interest") or 0.0)
        + settled_mv_prior
    )
    pricing_timing_usd = mark_basis_today - mark_basis_prior

    # compute_rotation_realized falls back to the PRIOR CLOSE when a rotated-out
    # name has no sell fill, which yields $0 realized and leaves the real
    # realized P&L inside the residual. Live that is a visible same-day
    # degradation; backfilled it would be a silent one — and a rotation of $0
    # written into the window is precisely the raw-plug contamination this
    # module exists to remove, only harder to see. Refuse the row instead.
    _require_priced_rotation(positions, prior_positions, trades_today)
    rotation_realized_usd = compute_rotation_realized(
        positions, prior_positions, list(trades_today or []),
    )

    return {
        "pricing_timing_usd": pricing_timing_usd,
        "rotation_realized_usd": rotation_realized_usd,
        "unattributed_true_usd": (
            float(unattributed) - rotation_realized_usd - pricing_timing_usd
        ),
    }


def _require_priced_rotation(
    positions: dict[str, Any],
    prior_positions: dict[str, Any],
    trades_today: list[dict[str, Any]] | None,
) -> None:
    """Raise unless every rotated-out name carries a real sell fill price."""
    from executor.eod_report import _sell_exit_prices

    exit_px = _sell_exit_prices(list(trades_today or []))
    for ticker, prior_pos in (prior_positions or {}).items():
        try:
            prior_shares = float((prior_pos or {}).get("shares", 0) or 0)
            today_shares = float((positions.get(ticker) or {}).get("shares", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise Unreconstructible(f"{ticker} share count is not numeric") from exc
        if prior_shares - today_shares > 0 and ticker not in exit_px:
            raise Unreconstructible(
                f"{ticker} rotated out with no priced sell fill"
            )


def _positions(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("positions_snapshot")
    if not raw:
        raise Unreconstructible("no positions_snapshot persisted")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise Unreconstructible(f"positions_snapshot is not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise Unreconstructible("positions_snapshot is not an object")
    return parsed


def backfill_residual_sleeves(
    conn: sqlite3.Connection,
    *,
    lookback_sessions: int = BACKFILL_LOOKBACK_SESSIONS,
) -> dict[str, Any]:
    """Fill the missing sleeves on historical ``eod_pnl`` rows, idempotently.

    Runs on every EOD reconciliation so the gate's window heals itself with no
    operator step (principles §2.3 — detect, act, verify, close). Rows that
    already carry ``unattributed_true_usd`` are untouched; rows that cannot be
    reconstructed are left NULL and NAMED in the returned counts, so a short
    window is visible as a short window rather than as a full one.
    """
    from executor.trade_logger import get_todays_trades

    conn.row_factory = sqlite3.Row
    try:
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM eod_pnl ORDER BY date DESC LIMIT ?",
                (lookback_sessions + 1,),
            ).fetchall()
        ][::-1]
    finally:
        conn.row_factory = None

    filled = 0
    skipped: list[tuple[str, str]] = []
    for prior_row, row in zip(rows, rows[1:], strict=False):
        if row.get("unattributed_true_usd") is not None:
            continue
        run_date = row.get("date")
        try:
            sleeves = reconstruct_sleeves(
                row=row,
                prior_row=prior_row,
                trades_today=get_todays_trades(conn, run_date),
            )
        except Unreconstructible as exc:
            skipped.append((str(run_date), str(exc)))
            continue
        conn.execute(
            "UPDATE eod_pnl SET pricing_timing_usd=?, rotation_realized_usd=?, "
            "unattributed_true_usd=? WHERE date=?",
            (
                sleeves["pricing_timing_usd"],
                sleeves["rotation_realized_usd"],
                sleeves["unattributed_true_usd"],
                run_date,
            ),
        )
        filled += 1
    conn.commit()

    if filled or skipped:
        logger.info(
            "Residual-sleeve backfill: %d row(s) reconstructed, %d unreconstructible%s",
            filled,
            len(skipped),
            (" — " + ", ".join(f"{d} ({why})" for d, why in skipped[:5])) if skipped else "",
        )
    return {"filled": filled, "skipped": skipped}
