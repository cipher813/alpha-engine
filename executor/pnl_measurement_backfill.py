"""Backfill the performance-measurement columns over the HISTORICAL eod_pnl series.

alpha-engine-config-I8188, second pass. The first pass (PR490/PR491/PR509)
closed all four defects on the FORWARD path and left the history untouched.
Measured against ``s3://alpha-engine-research/trades/eod_pnl.csv`` on
2026-08-31, 120 sessions, 2026-03-09 → 2026-08-28:

    commission_usd              populated on   6 / 120 sessions
    slippage_usd                populated on   6 / 120
    traded_notional_usd         populated on   6 / 120
    daily_return_gross_pct      populated on   6 / 120
    dividend_usd                populated on  79 / 120, and 0.00 on all 79
    spy_dividend_per_share      populated on   6 / 120

Six sessions is every session since the 2026-08-21 deploy. So the cost line,
the dividend line and the total-return benchmark exist prospectively and are
absent from the series any threshold would actually be set against — including
`alpha-engine-config-I9005`, the Crucible viability predicate this issue blocks.
A viability bar set on a 120-session record whose cost line covers 5% of it is
set against a number that is optimistic by an unmeasured amount, and BOTH
omissions run the same direction:

* no cost line  → implementation cost reads as $0 on 114 sessions;
* price-return SPY → the benchmark is understated by its distribution yield.

WHAT IS AND IS NOT RECONSTRUCTIBLE — measured against the live ledger
(``trades_latest.db``, 513 rows / 504 filled, 2026-03-13 → 2026-08-31), not
assumed:

    price_at_order  present on 504 / 504 filled rows  → slippage IS backfillable
    commission_usd  present on  35 / 504 filled rows, ALL of them 2026-08

So implementation shortfall is recoverable over the whole history and
commission is not. The commission leg is therefore written **NULL with
``commission_available = 0``**, never 0.0, and ``daily_return_gross_pct`` is
left NULL on those sessions rather than published with a $0.00 commission leg —
a gross return computed that way is a net return wearing a gross label. Whether
IB can be made to reissue historical commissionReports is a separate question;
until it is answered the honest state of that column is ABSENT.

RUN LOCATION. This is a manual one-off write to a production data repo. It runs
IN-REGION on EC2 (the trading box off-market-hours, or a data-spot instance),
never from a laptop — a full-history pass is dominated by S3 round-trip latency
and, for the dividend leg, by Polygon's 5-calls-per-minute limiter.

    python -m executor.pnl_measurement_backfill --dry-run      # default
    python -m executor.pnl_measurement_backfill --apply --costs --benchmark
    python -m executor.pnl_measurement_backfill --apply --dividends   # Polygon

NOT WIRED INTO THE EOD PATH. ``pnl_backfill.backfill_residual_sleeves`` runs on
every reconciliation and self-heals its own window, which is the pattern this
module should eventually follow (principles §2.3 — no operator step). It is
deliberately NOT wired here: the preopen/postclose/weekly Step Functions are
change-quiet until Crucible's clause-1 four-week reliability clock starts on
2026-09-05 (`alpha-engine-config-I9041`), and adding a call to
``eod_reconcile.run`` alters what the postclose SF executes. The wiring is one
line beside the existing ``backfill_residual_sleeves(conn)`` call and is filed
rather than taken.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from typing import Any

from executor.dividends import SPY_TICKER, accrue_position_dividends
from executor.pnl_integrity import (
    check_benchmark_vendor_anchor,
    check_custodian_marks,
    check_residual_bounds,
    gross_net_returns,
    session_costs,
    verify_benchmark_chain_closes,
    verify_twr_closes,
)

logger = logging.getLogger(__name__)

# The whole persisted history. Unlike the residual-sleeve backfill — which only
# has to keep a 63-session gate window full — the audience for these columns is
# the full track record a viability threshold is set against, so a lookback
# shorter than the history would reintroduce the gap it exists to close.
FULL_HISTORY = 10_000


class NothingToBackfill(Exception):
    """The requested leg has no reconstructible input on this row."""


def _rows(conn: sqlite3.Connection, *, limit: int = FULL_HISTORY) -> list[dict[str, Any]]:
    """The eod_pnl series as dicts, OLDEST-first."""
    conn.row_factory = sqlite3.Row
    try:
        out = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM eod_pnl ORDER BY date DESC LIMIT ?", (limit,)
            ).fetchall()
        ]
    finally:
        conn.row_factory = None
    return out[::-1]


def _positions(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = row.get("positions_snapshot")
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _f(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cost lines
# ─────────────────────────────────────────────────────────────────────────────

def plan_cost_backfill(
    rows: list[dict[str, Any]],
    trades_by_date: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Plan the ``slippage_usd`` / ``traded_notional_usd`` / commission writes. PURE.

    One entry per session that currently carries no ``slippage_usd``. Each
    reuses :func:`executor.pnl_integrity.session_costs` — the SAME function the
    live path calls — so a backfilled cost line is computed by the code that
    computes a live one. Any divergence between the two would be a defect here,
    not a licence to approximate.

    ``commission_usd`` is carried through exactly as ``session_costs`` reports
    it, which is ``None`` when fills executed and none carried a commission.
    ``daily_return_gross_pct`` follows :func:`gross_net_returns` and is
    therefore None on the same rows.
    """
    plans: list[dict[str, Any]] = []
    prior_nav: float | None = None
    for row in rows:
        nav = _f(row.get("portfolio_nav"))
        date = str(row.get("date"))
        if row.get("slippage_usd") is None:
            costs = session_costs(trades_by_date.get(date) or [])
            split = gross_net_returns(
                nav_change_usd=_f(row.get("nav_change_usd")),
                prior_nav=prior_nav,
                commission_usd=costs["commission_usd"],
                slippage_usd=costs["slippage_usd"],
            )
            plans.append({
                "date": date,
                "commission_usd": costs["commission_usd"],
                "commission_available": 1 if costs["commission_available"] else 0,
                "slippage_usd": costs["slippage_usd"],
                "traded_notional_usd": costs["traded_notional_usd"],
                "daily_return_net_pct": split["daily_return_net_pct"],
                "daily_return_gross_pct": split["daily_return_gross_pct"],
                "n_fills": costs["n_fills"],
                "slippage_bps": costs["slippage_bps"],
            })
        if nav is not None:
            prior_nav = nav
    return plans


def apply_cost_backfill(
    conn: sqlite3.Connection, plans: list[dict[str, Any]],
) -> int:
    """Write the planned cost lines. Idempotent — planning skips populated rows."""
    for p in plans:
        conn.execute(
            "UPDATE eod_pnl SET commission_usd=?, commission_available=?, "
            "slippage_usd=?, traded_notional_usd=?, daily_return_net_pct=?, "
            "daily_return_gross_pct=? WHERE date=?",
            (
                p["commission_usd"], p["commission_available"], p["slippage_usd"],
                p["traded_notional_usd"], p["daily_return_net_pct"],
                p["daily_return_gross_pct"], p["date"],
            ),
        )
    conn.commit()
    return len(plans)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Total-return benchmark
# ─────────────────────────────────────────────────────────────────────────────

def plan_benchmark_restatement(
    rows: list[dict[str, Any]],
    *,
    vendor_closes: dict[str, float],
    vendor_dividends: dict[str, float],
    prior_session_of: Any,
) -> dict[str, Any]:
    """Plan a VENDOR-ANCHORED restatement of ``spy_return_pct``/``daily_alpha_pct``.

    Neither persisted column can repair the other — ``spy_close`` changes basis
    at 2026-03-20 and ``spy_return_pct`` is what it is being checked against
    (see ``pnl_integrity`` §6). The only well-founded restatement is the
    vendor's own series:

        spy_return_pct[t] = (close[t] + div[t]) / close[prior_session(t)] − 1

    with ``prior_session`` from the TRADING CALENDAR, not from adjacency in the
    table. That is what makes the three coverage gaps (2026-03-12, 2026-07-27
    missing; 2026-04-03 a market holiday with a row) span correctly instead of
    silently comparing a two-session move against a one-session one.

    A row the vendor cannot cover — no close for the session, or none for the
    prior session — is REFUSED and named, never filled from the persisted
    column it is supposed to be verifying.

    THIS IS A RESTATEMENT OF A PUBLISHED TRACK RECORD, not a self-heal, and the
    caller gates it behind an explicit flag. Every ``daily_alpha_pct`` ever
    reported moves. That is a ruling for Brian, not a repair an agent takes.
    """
    corrections: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for row in rows:
        date = str(row.get("date"))
        stored = _f(row.get("spy_return_pct"))
        port = _f(row.get("daily_return_pct"))
        close = _f(vendor_closes.get(date))
        try:
            prior = str(prior_session_of(date))
        except Exception as exc:  # noqa: BLE001
            refused.append({"date": date, "reason": f"calendar lookup failed: {exc}"})
            continue
        prior_close = _f(vendor_closes.get(prior))
        if close is None or not prior_close:
            refused.append({
                "date": date,
                "reason": (
                    f"vendor has no close for {date if close is None else prior} "
                    "— the benchmark leg cannot be rebuilt for this session, and "
                    "keeping the persisted value is the honest state"
                ),
            })
            continue
        to_pct = ((close + float(vendor_dividends.get(date, 0.0) or 0.0))
                  / prior_close - 1.0) * 100.0
        if stored is not None and abs(stored - to_pct) * 100.0 <= 0.5:
            continue  # already correct to half a basis point
        if port is None:
            refused.append({
                "date": date,
                "from_pct": stored,
                "to_pct": to_pct,
                "reason": (
                    "row carries no daily_return_pct, so daily_alpha_pct cannot "
                    "be restated alongside spy_return_pct — correcting one leg "
                    "alone leaves the row internally inconsistent"
                ),
            })
            continue
        corrections.append({
            "date": date,
            "prior_session": prior,
            "from_pct": stored,
            "to_pct": to_pct,
            "delta_pct": (stored - to_pct) if stored is not None else None,
            "from_alpha_pct": _f(row.get("daily_alpha_pct")),
            "to_alpha_pct": port - to_pct,
            "spy_dividend_per_share": float(vendor_dividends.get(date, 0.0) or 0.0),
        })
    return {"corrections": corrections, "refused": refused}


def plan_dividend_per_share_writes(
    rows: list[dict[str, Any]], spy_ex_dividends: dict[str, float],
) -> list[dict[str, Any]]:
    """Plan the additive ``spy_dividend_per_share`` writes. PURE.

    Written on every row that lacks it, INCLUDING the zeros: "nothing went ex
    in this interval" is a real measurement, and it is what lets a later reader
    tell an unmeasured session from a measured one. This write restates
    nothing — it names a quantity the column never carried.
    """
    return [
        {"date": str(r.get("date")),
         "spy_dividend_per_share": float(spy_ex_dividends.get(str(r.get("date")), 0.0))}
        for r in rows
        if r.get("spy_dividend_per_share") is None
    ]


def apply_dividend_per_share_writes(
    conn: sqlite3.Connection, writes: list[dict[str, Any]],
) -> int:
    for w in writes:
        conn.execute(
            "UPDATE eod_pnl SET spy_dividend_per_share=? WHERE date=?",
            (w["spy_dividend_per_share"], w["date"]),
        )
    conn.commit()
    return len(writes)


def apply_benchmark_restatement(
    conn: sqlite3.Connection, plan: dict[str, Any],
) -> int:
    """Write the vendor-anchored benchmark restatement. GATED — see the planner."""
    for c in plan.get("corrections", []):
        conn.execute(
            "UPDATE eod_pnl SET spy_return_pct=?, daily_alpha_pct=? WHERE date=?",
            (c["to_pct"], c["to_alpha_pct"], c["date"]),
        )
    conn.commit()
    return len(plan.get("corrections", []))


def map_ex_dividends_to_sessions(
    rows: list[dict[str, Any]], events: list[dict[str, Any]],
) -> dict[str, float]:
    """Map raw Polygon dividend events onto the persisted session grid. PURE.

    ``events`` is what ``PolygonClient.get_dividends`` returns. Each event is
    credited to the FIRST persisted session on or after its ex-date, so the
    distribution lands in the interval that spans it. An ex-date after the last
    persisted session is dropped and counted by the caller — crediting it to
    the last row would put a future distribution inside a closed interval.

    Mirrors ``executor.dividends._in_interval``'s half-open ``(prior, through]``
    convention: the return for session *t* spans ``(t-1, t]``, so a dividend
    going ex ON session *t* belongs to session *t*.
    """
    dates = sorted(str(r.get("date")) for r in rows if r.get("date"))
    out: dict[str, float] = {}
    for ev in events or []:
        ex = ev.get("ex_dividend_date")
        amount = ev.get("cash_amount")
        if not ex or amount in (None, ""):
            continue
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            continue
        if amount <= 0:
            continue
        target = next((d for d in dates if d >= str(ex)), None)
        if target is None:
            continue
        out[target] = out.get(target, 0.0) + amount
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Position dividends
# ─────────────────────────────────────────────────────────────────────────────

def plan_dividend_backfill(
    rows: list[dict[str, Any]],
    ex_dividends_by_session: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Plan ``dividend_usd`` per session from the persisted position snapshots. PURE.

    ``ex_dividends_by_session[date][ticker]`` is the per-share cash going ex in
    the interval ending on ``date``. Entitlement is taken from the PRIOR
    session's share count, which is why this runs on consecutive persisted rows
    and reuses :func:`executor.dividends.accrue_position_dividends` rather than
    reimplementing the settle-before-ex rule.

    A session whose snapshot is absent or unparseable is SKIPPED with a reason
    and left NULL — it is not written as 0.00. ``dividend_usd`` reading 0.00 on
    all 79 populated historical sessions is the exact defect being repaired;
    replacing an unknown with a zero here would recreate it one layer down.
    """
    plans: list[dict[str, Any]] = []
    prior_positions: dict[str, Any] | None = None
    for row in rows:
        date = str(row.get("date"))
        positions = _positions(row)
        per_share = ex_dividends_by_session.get(date) or {}
        if positions is None:
            plans.append({"date": date, "skipped": "no parseable positions_snapshot"})
            prior_positions = None
            continue
        if prior_positions is None:
            plans.append({"date": date, "skipped": "no prior snapshot for entitlement"})
            prior_positions = positions
            continue
        # accrue_position_dividends mutates; operate on a copy so planning is pure
        # with respect to the caller's rows.
        working = {t: dict(p) for t, p in positions.items()}
        accruals = accrue_position_dividends(working, prior_positions, per_share)
        plans.append({
            "date": date,
            "dividend_usd": sum(a["amount_usd"] for a in accruals),
            "n_accruals": len(accruals),
            "tickers": [a["ticker"] for a in accruals],
        })
        prior_positions = positions
    return plans


def apply_dividend_backfill(
    conn: sqlite3.Connection, plans: list[dict[str, Any]],
) -> int:
    """Write the planned ``dividend_usd`` figures. Skipped rows stay NULL."""
    n = 0
    for p in plans:
        if "dividend_usd" not in p:
            continue
        conn.execute(
            "UPDATE eod_pnl SET dividend_usd=?, dividend_accrual_available=1 "
            "WHERE date=?",
            (p["dividend_usd"], p["date"]),
        )
        n += 1
    conn.commit()
    return n


# ─────────────────────────────────────────────────────────────────────────────
# 4. Retroactive audit — run every gate over the history, loudly
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_mark_divergences(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rebuild the custodian-mark check for a session from what was PERSISTED.

    ``ib_mark_outside_range`` / ``ib_mark_range_error_usd`` — the fields
    :func:`executor.pnl_integrity.check_custodian_marks` consumes — were only
    added to ``positions_snapshot`` in 2026-08. Measured over the persisted
    history: 5 flagged position-days, ALL of them 2026-08-17 or later, against
    989 position-days carrying ``market_value`` and ``ib_market_value``. So the
    live gate reads a field the history does not have, and running it over the
    history reports zero breaches — a silence that would be indistinguishable
    from a clean record.

    THE INSTRUMENT IS DIFFERENT AND WEAKER, and is labelled as such rather than
    merged into the same count. The live gate measures the distance past that
    day's ArcticDB traded RANGE — a mark outside the range is provably wrong
    regardless of volatility. The traded range is not persisted, so the only
    reference available retroactively is the settled close (``market_value``).
    A settled close lies INSIDE the traded range, so this divergence is an
    upper bound on the range breach: it can flag a mark the live gate would
    pass (a genuine intraday move away from the close), and it can never miss
    one the live gate would raise. That direction is the safe one for an audit
    of a record nobody has ever checked, and it is why this leg is reported
    under its own key with its own instrument named.

    Materiality is the live gate's own ``mark_materiality_usd`` — max($500,
    15bp of NAV) — so the audit and the forward path agree on what "material"
    means even where they disagree on what is being measured.
    """
    from executor.pnl_integrity import mark_materiality_usd

    nav = _f(row.get("portfolio_nav"))
    positions = _positions(row) or {}
    if nav is None or not positions:
        return []
    materiality = mark_materiality_usd(nav)
    out: list[dict[str, Any]] = []
    for ticker, pos in positions.items():
        settled_mv = _f((pos or {}).get("market_value"))
        ib_mv = _f((pos or {}).get("ib_market_value"))
        if settled_mv is None or ib_mv is None:
            continue
        divergence = ib_mv - settled_mv
        if abs(divergence) <= materiality:
            continue
        out.append({
            "kind": "custodian_mark_divergence",
            "instrument": "broker MV vs settled-close MV (upper bound on the "
                          "traded-range breach the live gate measures)",
            "run_date": str(row.get("date")),
            "ticker": ticker,
            "settled_mv_usd": settled_mv,
            "ib_market_value_usd": ib_mv,
            "divergence_usd": divergence,
            "divergence_pct": (divergence / settled_mv * 100.0) if settled_mv else None,
            "materiality_usd": materiality,
            "nav": nav,
        })
    return out


def audit_history(
    rows: list[dict[str, Any]],
    *,
    spy_ex_dividends: dict[str, float] | None = None,
    vendor_closes: dict[str, float] | None = None,
    prior_session_of: Any | None = None,
) -> dict[str, Any]:
    """Run the integrity gates over the WHOLE persisted series and report breaches.

    The forward path evaluates each gate on the session it is reconciling. None
    of them has ever been evaluated against the history, so a breach that
    happened before the gate existed has never been seen. This runs all of them
    retroactively — TWR closure, benchmark-chain closure, the per-session and
    cumulative residual bounds, and the custodian-mark materiality gate — and
    returns every breach with its session.

    Returns a dict with ``breach_count``; the CLI exits NON-ZERO when it is
    nonzero, so a breach in the history is a failure rather than a paragraph in
    a report nobody reads.
    """
    twr = verify_twr_closes(rows)
    bench = verify_benchmark_chain_closes(
        rows,
        ex_dividends=spy_ex_dividends or {},
        prior_session_of=prior_session_of,
    )
    # The vendor anchor is the only leg here that is not built from columns this
    # system wrote. Its ABSENCE is reported as a named degradation rather than
    # as a pass — a gate that could not run has not agreed with anything.
    if vendor_closes:
        anchor_breaches = check_benchmark_vendor_anchor(
            rows,
            vendor_closes=vendor_closes,
            vendor_dividends=spy_ex_dividends or {},
        )
        anchor_status = "evaluated"
    else:
        anchor_breaches = []
        anchor_status = "NOT EVALUATED — no vendor close series supplied"

    residual_breaches: list[dict[str, Any]] = []
    mark_breaches: list[dict[str, Any]] = []
    divergence_breaches: list[dict[str, Any]] = []
    true_window: list[float] = []
    for row in rows:
        nav = _f(row.get("portfolio_nav"))
        date = str(row.get("date"))
        true_residual = _f(row.get("unattributed_true_usd"))
        if nav is None or true_residual is None:
            continue
        true_window.append(true_residual)
        residual_breaches.extend(
            check_residual_bounds(
                unattributed_true_usd=true_residual,
                nav=nav,
                trailing_residuals_usd=true_window[:-1],
                run_date=date,
            )
        )
        positions = _positions(row) or {}
        flags = [
            {
                "ticker": t,
                "mark_error_usd": p.get("ib_mark_range_error_usd"),
                "ib_market_value": p.get("ib_market_value"),
                "shares": p.get("shares"),
            }
            for t, p in positions.items()
            if p.get("ib_mark_outside_range")
            and p.get("ib_mark_range_error_usd") is not None
        ]
        mark_breaches.extend(check_custodian_marks(flags, nav=nav, run_date=date))
        divergence_breaches.extend(reconstruct_mark_divergences(row))

    breach_count = (
        (0 if twr.get("closes") is not False else 1)
        + (0 if bench.get("closes") is not False else 1)
        + len(residual_breaches)
        + len(mark_breaches)
        + len(divergence_breaches)
        + len(anchor_breaches)
    )
    return {
        "n_sessions": len(rows),
        "twr_closure": twr,
        "benchmark_closure": bench,
        "benchmark_vendor_anchor_status": anchor_status,
        "benchmark_vendor_anchor_breaches": anchor_breaches,
        "residual_breaches": residual_breaches,
        "mark_breaches": mark_breaches,
        "mark_divergence_breaches": divergence_breaches,
        "breach_count": breach_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_spy_events(start: str) -> list[dict[str, Any]]:
    from polygon_client import PolygonClient

    return PolygonClient().get_dividends(SPY_TICKER, start=start)


def fetch_vendor_closes(start: str, end: str) -> dict[str, float]:
    """SPY daily closes from the vendor, keyed by session date.

    The independent anchor for
    :func:`executor.pnl_integrity.check_benchmark_vendor_anchor`. Uses
    ``PolygonClient.get_daily_bars``, whose own docstring states the prices are
    split-adjusted but **NOT dividend-adjusted** — which is the series required
    here, because the distributions are added explicitly from
    ``/v3/reference/dividends`` rather than folded into the price level.

    That distinction is the whole finding: ``PolygonClient`` also carries
    ``get_daily_bars_dividend_adjusted``, and the persisted ``spy_close``
    values through 2026-03-19 sit exactly 27.2bp below the unadjusted series —
    $1.797/$659.80, SPY's 2026-03-20 distribution. Mixing the two in one column
    is what made the benchmark leg unverifiable.
    """
    from polygon_client import PolygonClient

    bars = PolygonClient().get_daily_bars(SPY_TICKER, start, end, adjusted=True)
    if bars is None or bars.empty:
        raise RuntimeError(
            f"vendor returned no SPY bars for {start} → {end}; the benchmark "
            "anchor cannot be evaluated and is reported NOT EVALUATED"
        )
    return {
        idx.date().isoformat(): float(close)
        for idx, close in bars["Close"].items()
    }


def _prior_session_of(date_str: str) -> str:
    import datetime as _dt

    from krepis.trading_calendar import previous_trading_day

    return previous_trading_day(_dt.date.fromisoformat(date_str)).isoformat()


def _fetch_holding_events(tickers: list[str], start: str) -> dict[str, list[dict]]:
    from polygon_client import PolygonClient

    client = PolygonClient()
    out: dict[str, list[dict]] = {}
    for t in sorted(set(tickers)):
        try:
            out[t] = client.get_dividends(t, start=start)
        except Exception:  # noqa: BLE001 — per-ticker isolation, mirrors fetch_ex_dividends
            # (a) swallowed: a third-party HTTP/credential failure on ONE ticker;
            # (b) the primary deliverable — every other ticker's accrual and the
            #     cost/benchmark legs — survives; (c) recorded: the ticker is
            #     absent from the returned map, the CLI prints the shortfall, and
            #     the row's dividend stays inside the bounded residual.
            logger.warning("[backfill] dividend fetch failed for %s", t, exc_info=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="trades.db", help="path to the trades sqlite db")
    ap.add_argument("--apply", action="store_true", help="write (default is dry-run)")
    ap.add_argument("--costs", action="store_true")
    ap.add_argument("--benchmark", action="store_true",
                    help="write spy_dividend_per_share (additive, restates nothing)")
    ap.add_argument(
        "--restate-benchmark", action="store_true",
        help="ALSO rewrite spy_return_pct/daily_alpha_pct from the vendor series. "
             "This restates a published track record and needs a ruling; without "
             "it the restatement is planned and printed but never written.",
    )
    ap.add_argument("--dividends", action="store_true")
    ap.add_argument("--audit", action="store_true", help="run the retroactive gates")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not any([args.costs, args.benchmark, args.dividends, args.audit]):
        args.costs = args.benchmark = args.audit = True

    conn = sqlite3.connect(args.db)
    rows = _rows(conn)
    if not rows:
        logger.error("eod_pnl is empty in %s — nothing to backfill", args.db)
        return 2
    start, end = str(rows[0]["date"]), str(rows[-1]["date"])
    report: dict[str, Any] = {
        "db": args.db, "n_sessions": len(rows), "window": [start, end],
        "applied": bool(args.apply),
        "benchmark_restated": bool(args.apply and args.restate_benchmark),
    }

    if args.costs:
        from executor.trade_logger import get_todays_trades

        trades_by_date = {str(r["date"]): get_todays_trades(conn, str(r["date"]))
                          for r in rows}
        plans = plan_cost_backfill(rows, trades_by_date)
        n_absent = sum(1 for p in plans if p["commission_usd"] is None)
        report["costs"] = {
            "sessions_planned": len(plans),
            "sessions_with_commission_ABSENT": n_absent,
            "slippage_usd_total": sum(p["slippage_usd"] for p in plans),
            "traded_notional_usd_total": sum(p["traded_notional_usd"] for p in plans),
        }
        if args.apply:
            report["costs"]["written"] = apply_cost_backfill(conn, plans)
        logger.info("costs: %s", json.dumps(report["costs"]))

    spy_map: dict[str, float] = {}
    vendor_closes: dict[str, float] = {}
    if args.benchmark or args.audit:
        try:
            spy_map = map_ex_dividends_to_sessions(rows, _fetch_spy_events(start))
            vendor_closes = fetch_vendor_closes(start, end)
        except Exception as exc:  # noqa: BLE001
            # NOT swallowed into a zero. The benchmark leg is REFUSED outright
            # when its vendor source is unavailable: writing a "total-return"
            # benchmark with an unverified zero distribution, or passing an
            # anchor check that never ran, is the defect this module exists to
            # remove wearing a green badge.
            logger.error("Vendor benchmark fetch failed (%s) — benchmark leg REFUSED", exc)
            report["benchmark"] = {"status": "refused", "reason": str(exc)}
            args.benchmark = False

    if args.benchmark:
        writes = plan_dividend_per_share_writes(rows, spy_map)
        restatement = plan_benchmark_restatement(
            rows, vendor_closes=vendor_closes, vendor_dividends=spy_map,
            prior_session_of=_prior_session_of,
        )
        report["benchmark"] = {
            "status": "planned",
            "spy_ex_dividends": spy_map,
            "spy_dividend_per_share_writes": len(writes),
            "restatement_corrections": restatement["corrections"],
            "restatement_refused": restatement["refused"],
        }
        if args.apply:
            report["benchmark"]["spy_dividend_per_share_written"] = (
                apply_dividend_per_share_writes(conn, writes)
            )
            if args.restate_benchmark:
                report["benchmark"]["restatement_written"] = (
                    apply_benchmark_restatement(conn, restatement)
                )
            else:
                logger.warning(
                    "%d benchmark restatement(s) planned and NOT written — "
                    "rerun with --restate-benchmark once the restatement is "
                    "ruled on. Every daily_alpha_pct on those sessions moves.",
                    len(restatement["corrections"]),
                )

    if args.dividends:
        tickers = [t for r in rows for t in (_positions(r) or {})]
        events = _fetch_holding_events(tickers, start)
        by_session: dict[str, dict[str, float]] = {}
        for tkr, evs in events.items():
            for date, amount in map_ex_dividends_to_sessions(rows, evs).items():
                by_session.setdefault(date, {})[tkr] = amount
        div_plans = plan_dividend_backfill(rows, by_session)
        report["dividends"] = {
            "tickers_fetched": len(events),
            "tickers_requested": len(set(tickers)),
            "sessions_planned": sum(1 for p in div_plans if "dividend_usd" in p),
            "sessions_skipped": [p for p in div_plans if "skipped" in p],
            "dividend_usd_total": sum(p.get("dividend_usd", 0.0) for p in div_plans),
        }
        if args.apply:
            report["dividends"]["written"] = apply_dividend_backfill(conn, div_plans)

    exit_code = 0
    if args.audit:
        rows = _rows(conn)  # re-read so the audit sees anything just written
        audit = audit_history(
            rows, spy_ex_dividends=spy_map, vendor_closes=vendor_closes,
            prior_session_of=_prior_session_of,
        )
        report["audit"] = audit
        if audit["breach_count"]:
            logger.error(
                "HISTORICAL INTEGRITY: %d breach(es) over %d sessions — "
                "TWR closes=%s, benchmark columns agree=%s, vendor anchor=%s "
                "(%d breach), residual=%d, mark flags=%d, mark divergences=%d",
                audit["breach_count"], audit["n_sessions"],
                audit["twr_closure"].get("closes"),
                audit["benchmark_closure"].get("closes"),
                audit["benchmark_vendor_anchor_status"],
                len(audit["benchmark_vendor_anchor_breaches"]),
                len(audit["residual_breaches"]), len(audit["mark_breaches"]),
                len(audit["mark_divergence_breaches"]),
            )
            exit_code = 1

    payload = json.dumps(report, indent=2, default=str)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(payload)
    else:
        print(payload)
    conn.close()
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
