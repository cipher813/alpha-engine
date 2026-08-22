"""Performance-measurement integrity gates for the EOD P&L path.

Every function here is PURE (no IO, no IB, no S3, no SQLite) so each gate is
unit-testable without mocking ``eod_reconcile.run``'s IO. ``eod_reconcile``
owns dispatch — logging, flow-doctor paging, and raising.

Why this module exists (alpha-engine-config-I8188). Four measured defects, all
against ``s3://alpha-engine-research/trades/eod_pnl.csv`` (115 sessions,
2026-03-09 → 2026-08-21):

1. ``nav_change_usd = position_pnl_usd + interest_usd + dividend_usd +
   unattributed_usd`` held on every session **because ``unattributed_usd`` is
   defined as the remainder**. The identity was tautological and could never
   fail, so −$20,293 cumulative accumulated with nothing firing.
2. Chain-linking ``daily_return_pct`` gave +3.8261% against a NAV ratio of
   +3.6526% — 17.4bp of drift that is identically zero by construction absent
   external flows.
3. There was no transaction-cost line anywhere in the schema, so gross and net
   performance were the same number and neither was labelled.
4. ``ib_mark_outside_range`` existed but only flagged: 24 of 229 position-days
   (10.5%) differed from the broker mark by more than 0.5% with nothing
   failing.

The residual-bound derivation below rests on one measurement that the issue
body did NOT have, and which changes what the bound is FOR. Reconstructing
``rotation_realized_usd`` (realized P&L on shares rotated out, which
``eod_report.compute_rotation_realized`` already computes but which was never
persisted) over the 74 sessions carrying attribution columns:

    Σ unattributed_usd (raw)              = −$20,293
    Σ rotation_realized_usd               = −$20,815
    Σ residual after lifting rotation out = **+$522**  (mean +$7/session)

    turnover days (57):  raw mean −$406  →  ex-rotation mean −$41
    no-turnover days (17): raw mean +$168 →  ex-rotation mean +$168

So the residual "tracking turnover" was realized rotation P&L — mechanically
turnover-linked — not unmodelled trading costs. The bound therefore applies to
``unattributed_true_usd`` (after the rotation and pricing&timing sleeves are
lifted out), which is the number that should actually be ~0, and the sleeves
are persisted alongside it so the CSV carries the decomposition rather than one
undifferentiated plug.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Residual bounds — per-session and cumulative
# ─────────────────────────────────────────────────────────────────────────────
#
# DERIVATION (measured, 74 sessions with attribution columns, NAV ≈ $1.03M).
#
# Per-session. |unattributed_usd − rotation_realized_usd| distribution:
#   p50 = $754 (7bp of NAV) · p90 = $2,789 (27bp) · p95 = $3,560 (35bp) ·
#   p99 = $8,573 (83bp). The p99 day is 2026-08-04, whose residual (−$9,713)
#   is dominated by a single stale IB mark on AMD (+$8,906, 8.26% off the
#   settled close) — i.e. the pricing&timing sleeve, which this gate lifts out
#   separately. A HARD raise must therefore sit ABOVE the ordinary
#   settlement/mark noise band and only fire on a residual no named sleeve can
#   absorb. max($5,000, 50bp of NAV) ≈ $5,150 at today's NAV clears p95 by 1.4x
#   and fires on 1 of 74 historical sessions (1.4%) — the one session that was
#   genuinely broken.
#
#   The pre-existing SOFT warning (max($100, 5bp of NAV), data_warnings →
#   EOD email only) is UNCHANGED and still fires first. Same two-band shape
#   as NAV_HARD_GATE_TOLERANCE_* in eod_reconcile: a hard gate set at the soft
#   band's rate pages exactly as often as the email already flags and trains
#   the operator to ignore it.
#
# Cumulative. The whole point of defect 1 is that per-session-invisible,
# same-sign residuals compounded to −$20,293 (−1.97% of NAV) with no detector.
# A per-session bound structurally cannot catch that. Measured true drift over
# 74 sessions after the sleeves are lifted: +$522 = 5bp of NAV. A trailing
# 63-session (one quarter) window bounded at max($10,000, 100bp of NAV) is
# ~20x the measured drift — it cannot fire on the observed behaviour of a
# correctly-attributed book, and it would have fired on the RAW plug within
# ~30 sessions.
RESIDUAL_HARD_PER_SESSION_USD_FLOOR = 5_000.0
RESIDUAL_HARD_PER_SESSION_NAV_BPS = 50.0
RESIDUAL_CUMULATIVE_WINDOW_SESSIONS = 63
RESIDUAL_HARD_CUMULATIVE_USD_FLOOR = 10_000.0
RESIDUAL_HARD_CUMULATIVE_NAV_BPS = 100.0


def residual_per_session_tolerance_usd(nav: float) -> float:
    """Dollar bound on one session's TRUE residual at this NAV level."""
    return max(
        RESIDUAL_HARD_PER_SESSION_USD_FLOOR,
        RESIDUAL_HARD_PER_SESSION_NAV_BPS / 10_000.0 * nav,
    )


def residual_cumulative_tolerance_usd(nav: float) -> float:
    """Dollar bound on the trailing-window cumulative residual at this NAV."""
    return max(
        RESIDUAL_HARD_CUMULATIVE_USD_FLOOR,
        RESIDUAL_HARD_CUMULATIVE_NAV_BPS / 10_000.0 * nav,
    )


def check_residual_bounds(
    *,
    unattributed_true_usd: float | None,
    nav: float | None,
    trailing_residuals_usd: Sequence[float] | None = None,
    run_date: str = "",
) -> list[dict[str, Any]]:
    """Bound the P&L residual per-session AND cumulatively.

    ``unattributed_true_usd`` is the residual AFTER the rotation-realized and
    pricing&timing sleeves are lifted out — the number that should be ~0. Pass
    the raw plug only if the sleeves are unavailable (the caller records that
    as a degradation).

    ``trailing_residuals_usd`` is the persisted history of the same quantity,
    OLDEST-first and EXCLUDING today; today's value is appended here so the
    cumulative check always includes the session being written.

    Returns a list of breach dicts (empty when both bounds hold). Each breach
    carries ``kind`` (``"per_session"`` / ``"cumulative"``), the measured
    value, the tolerance, and a ``message`` the caller raises with. Returning
    rather than raising keeps the gate testable and lets the caller persist the
    EOD row BEFORE failing the run — a lost artifact is a worse outcome than a
    red pipeline, and the breach is itself persisted (``residual_breach``).
    """
    if unattributed_true_usd is None or not nav:
        return []

    breaches: list[dict[str, Any]] = []

    per_tol = residual_per_session_tolerance_usd(nav)
    if abs(unattributed_true_usd) > per_tol:
        breaches.append({
            "kind": "per_session",
            "run_date": run_date,
            "value_usd": unattributed_true_usd,
            "tolerance_usd": per_tol,
            "nav": nav,
            "pct_of_nav": unattributed_true_usd / nav * 100,
            "message": (
                f"P&L residual bound breached for {run_date}: unattributed "
                f"${unattributed_true_usd:+,.0f} "
                f"({unattributed_true_usd / nav * 100:+.3f}% of NAV) exceeds the "
                f"per-session bound ${per_tol:,.0f} "
                f"({RESIDUAL_HARD_PER_SESSION_NAV_BPS:.0f}bps of NAV). This is the "
                "residual AFTER rotation-realized and pricing&timing are lifted "
                "out — no named sleeve explains it."
            ),
        })

    window = list(trailing_residuals_usd or [])[-(RESIDUAL_CUMULATIVE_WINDOW_SESSIONS - 1):]
    window.append(float(unattributed_true_usd))
    cumulative = sum(window)
    cum_tol = residual_cumulative_tolerance_usd(nav)
    if abs(cumulative) > cum_tol:
        breaches.append({
            "kind": "cumulative",
            "run_date": run_date,
            "value_usd": cumulative,
            "tolerance_usd": cum_tol,
            "nav": nav,
            "n_sessions": len(window),
            "pct_of_nav": cumulative / nav * 100,
            "message": (
                f"Cumulative P&L residual bound breached as of {run_date}: "
                f"${cumulative:+,.0f} ({cumulative / nav * 100:+.3f}% of NAV) over "
                f"the trailing {len(window)} session(s) exceeds "
                f"${cum_tol:,.0f} ({RESIDUAL_HARD_CUMULATIVE_NAV_BPS:.0f}bps of "
                "NAV). Same-sign residuals are accumulating — a systematic "
                "attribution gap, not day-to-day noise."
            ),
        })

    return breaches


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transaction costs — commission and implementation shortfall
# ─────────────────────────────────────────────────────────────────────────────

_BUY_ACTIONS = {"BUY", "ENTER", "ADD", "COVER"}
_SELL_ACTIONS = {"SELL", "EXIT", "REDUCE", "TRIM"}


def session_costs(trades_today: Iterable[Mapping[str, Any]] | None) -> dict[str, Any]:
    """Explicit transaction costs for one session, sourced from FILLS.

    Two named lines, both POSITIVE = a cost to the book:

    ``commission_usd``
        Σ of the per-execution commission IB reports on each fill
        (``Fill.commissionReport.commission``, captured onto ``trades`` at
        order time). This is a real cash debit and is already inside
        ``nav_change_usd``; naming it is what lets net be distinguished from
        gross.

    ``slippage_usd``
        Implementation shortfall against the ARRIVAL price
        (``price_at_order``, the mark at order submission):
        ``Σ side · shares · (fill_price − price_at_order)``, side +1 for buys
        and −1 for sells. Measured over the whole live window (468 fills):
        **+6.4bp of traded notional** — a real and entirely unrecorded cost. Unlike commission this NEVER debited NAV (the fill
        price IS the book cost); it is the difference between the return the
        book earned and the return the decision would have earned at arrival.
        Gross-of-cost return is therefore explicitly hypothetical and labelled
        as such.

    ``commission_available`` is False when no filled row carried a commission
    figure — fail loud: a $0 commission line and an ABSENT commission line are
    different facts and must not render identically. The pre-existing code
    treated the absence as "paper-account commissions are trivial" and dropped
    it silently, which is exactly how the cost line came not to exist.
    """
    n_fills = 0
    n_with_commission = 0
    commission = 0.0
    slippage = 0.0
    notional = 0.0
    for t in trades_today or []:
        shares = t.get("filled_shares")
        if shares in (None, ""):
            shares = t.get("shares")
        fill_price = t.get("fill_price")
        if shares in (None, "") or fill_price in (None, ""):
            continue
        try:
            shares = float(shares)
            fill_price = float(fill_price)
        except (TypeError, ValueError):
            continue
        if shares <= 0 or fill_price <= 0:
            continue
        n_fills += 1
        notional += shares * fill_price

        comm = t.get("commission_usd")
        if comm not in (None, ""):
            try:
                commission += abs(float(comm))
                n_with_commission += 1
            except (TypeError, ValueError):
                pass

        arrival = t.get("price_at_order")
        action = str(t.get("action") or "").upper()
        if arrival in (None, "") or action not in (_BUY_ACTIONS | _SELL_ACTIONS):
            continue
        try:
            arrival = float(arrival)
        except (TypeError, ValueError):
            continue
        if arrival <= 0:
            continue
        side = 1.0 if action in _BUY_ACTIONS else -1.0
        slippage += side * shares * (fill_price - arrival)

    return {
        "commission_usd": commission,
        "slippage_usd": slippage,
        "traded_notional_usd": notional,
        "n_fills": n_fills,
        "n_fills_with_commission": n_with_commission,
        "commission_available": n_fills == 0 or n_with_commission > 0,
        "slippage_bps": (slippage / notional * 10_000.0) if notional else None,
    }


def gross_net_returns(
    *,
    nav_change_usd: float | None,
    prior_nav: float | None,
    commission_usd: float,
    slippage_usd: float,
) -> dict[str, float | None]:
    """Split the day's return into net-of-cost and gross-of-cost.

    ``daily_return_net_pct`` is the return the book actually earned — the
    existing ``daily_return_pct``, restated under a name that says which side
    of costs it is on. Commissions have already debited NAV and fills already
    happened at the fill price, so the NAV-based number IS net.

    ``daily_return_gross_pct`` adds both cost lines back: the return the same
    decisions would have produced with zero commission and zero implementation
    shortfall. It is a counterfactual, not a second measurement of the book —
    the shortfall leg never touched NAV.

    Both are None when there is no prior NAV (first session).
    """
    if nav_change_usd is None or not prior_nav:
        return {
            "daily_return_net_pct": None,
            "daily_return_gross_pct": None,
            "total_cost_usd": commission_usd + slippage_usd,
        }
    net = nav_change_usd / prior_nav * 100.0
    gross = (nav_change_usd + commission_usd + slippage_usd) / prior_nav * 100.0
    return {
        "daily_return_net_pct": net,
        "daily_return_gross_pct": gross,
        "total_cost_usd": commission_usd + slippage_usd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. TWR closure — chain-linked daily returns must equal the NAV ratio
# ─────────────────────────────────────────────────────────────────────────────
#
# Absent external cash flows these are identically equal by construction, since
# every stored daily_return_pct is (nav − prior_nav) / prior_nav against the
# immediately preceding persisted row. Any drift is therefore a STORED value
# that no longer agrees with the NAV series it was derived from.
#
# MEASURED CAUSE of the live 17.4bp drift — it is not external flows, and it is
# not distributed: it is ONE row. On 2026-04-07 the stored daily_return_pct is
# +0.026827% while the persisted NAVs imply −0.140311%. Its implied prior NAV
# is $1,007,786.32; the persisted 2026-04-06 row says $1,009,473.08. The
# 2026-04-06 NAV was corrected AFTER the 2026-04-07 row was written (that row's
# created_at is 21:42Z against the usual 20:20Z — a re-run) and 04-07's stored
# return was never recomputed. Recomputing that single row moves the chain-link
# from +3.8261% to +3.6526%, matching the NAV ratio to 3e-5pp.
TWR_CLOSURE_TOLERANCE_BPS = 1.0
# A self-heal must not be able to launder an arbitrary rewrite of the track
# record. A correction larger than this on a single session is not a stale
# stored value, it is a different NAV series (an external flow, a restated
# snapshot, a mis-keyed row) — that raises instead of being silently rewritten.
TWR_SELF_HEAL_MAX_CORRECTION_PCT = 1.0


def nav_implied_returns(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """For each row, the daily return its own NAV series implies.

    ``rows`` is the persisted eod_pnl series OLDEST-first, each carrying
    ``date``, ``portfolio_nav`` and ``daily_return_pct``. The first row has no
    predecessor so its implied return is None.
    """
    out: list[dict[str, Any]] = []
    prior_nav: float | None = None
    for r in rows:
        nav = r.get("portfolio_nav")
        try:
            nav = float(nav) if nav not in (None, "") else None
        except (TypeError, ValueError):
            nav = None
        stored = r.get("daily_return_pct")
        try:
            stored = float(stored) if stored not in (None, "") else None
        except (TypeError, ValueError):
            stored = None
        implied = None
        if nav is not None and prior_nav:
            implied = (nav / prior_nav - 1.0) * 100.0
        out.append({
            "date": r.get("date"),
            "portfolio_nav": nav,
            "stored_pct": stored,
            "implied_pct": implied,
            "delta_pct": (
                stored - implied
                if (stored is not None and implied is not None)
                else None
            ),
        })
        if nav is not None:
            prior_nav = nav
    return out


def verify_twr_closes(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance_bps: float = TWR_CLOSURE_TOLERANCE_BPS,
) -> dict[str, Any]:
    """Assert chain-linked TWR equals the NAV ratio to ``tolerance_bps``.

    Returns a dict with ``closes`` (bool), both figures, the drift in bps, and
    ``offenders`` — the rows whose stored return disagrees with the return
    their own NAV series implies, which is where the drift comes from.

    ``status`` is ``"n/a"`` when there are fewer than two usable rows.
    """
    detail = [d for d in nav_implied_returns(rows) if d["portfolio_nav"] is not None]
    if len(detail) < 2:
        return {"status": "n/a", "closes": None, "offenders": []}

    chain = 1.0
    for d in detail[1:]:
        if d["stored_pct"] is None:
            return {
                "status": "n/a",
                "closes": None,
                "offenders": [],
                "reason": f"row {d['date']} has no stored daily_return_pct",
            }
        chain *= 1.0 + d["stored_pct"] / 100.0
    chain_pct = (chain - 1.0) * 100.0

    first, last = detail[0]["portfolio_nav"], detail[-1]["portfolio_nav"]
    if not first:
        return {"status": "n/a", "closes": None, "offenders": []}
    nav_ratio_pct = (last / first - 1.0) * 100.0

    drift_bps = (chain_pct - nav_ratio_pct) * 100.0
    offenders = [
        d for d in detail[1:]
        if d["delta_pct"] is not None and abs(d["delta_pct"]) * 100.0 > tolerance_bps
    ]
    return {
        "status": "ok",
        "closes": abs(drift_bps) <= tolerance_bps,
        "chain_linked_pct": chain_pct,
        "nav_ratio_pct": nav_ratio_pct,
        "drift_bps": drift_bps,
        "tolerance_bps": tolerance_bps,
        "n_sessions": len(detail),
        "offenders": offenders,
        "message": (
            f"TWR does not close: chain-linked daily_return_pct = "
            f"{chain_pct:+.4f}% vs NAV ratio {nav_ratio_pct:+.4f}% over "
            f"{len(detail)} sessions — {drift_bps:+.1f}bp of drift against a "
            f"{tolerance_bps:.0f}bp tolerance. Absent external flows these are "
            f"identically equal by construction. Disagreeing row(s): "
            + ", ".join(
                f"{o['date']} stored {o['stored_pct']:+.6f}% vs NAV-implied "
                f"{o['implied_pct']:+.6f}%"
                for o in offenders[:5]
            )
            + ("" if len(offenders) <= 5 else f" (+{len(offenders) - 5} more)")
        ),
    }


def plan_twr_self_heal(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance_bps: float = TWR_CLOSURE_TOLERANCE_BPS,
    max_correction_pct: float = TWR_SELF_HEAL_MAX_CORRECTION_PCT,
) -> dict[str, Any]:
    """Plan the corrections that would make TWR close, without applying them.

    A stored ``daily_return_pct`` that disagrees with the persisted NAV series
    is stale — the NAV series is ground truth (it is what the broker reported)
    and the stored percentage is a derived value that was never recomputed
    after an upstream NAV correction. Rewriting it is a repair, and it CLOSES
    THE LOOP without an operator step; the alternative — a detector that goes
    red until a human runs a backfill — is a page, not a fix.

    Two guards keep the repair from becoming a licence to restate history:

    * any single correction larger than ``max_correction_pct`` is REFUSED and
      surfaces in ``refused`` — at that size the disagreement is a different
      NAV series (an external flow, a restated snapshot), not a stale
      derived value, and it must be ruled on rather than rewritten;
    * ``spy_return_pct``/``daily_alpha_pct`` are NOT touched — only the leg
      whose ground truth is the persisted NAV.

    Returns ``{"corrections": [...], "refused": [...]}`` where each correction
    is ``{date, from_pct, to_pct, delta_pct}``.
    """
    corrections: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for d in nav_implied_returns(rows):
        if d["delta_pct"] is None:
            continue
        if abs(d["delta_pct"]) * 100.0 <= tolerance_bps:
            continue
        entry = {
            "date": d["date"],
            "from_pct": d["stored_pct"],
            "to_pct": d["implied_pct"],
            "delta_pct": d["delta_pct"],
        }
        if abs(d["delta_pct"]) > max_correction_pct:
            entry["reason"] = (
                f"correction {d['delta_pct']:+.4f}pp exceeds the "
                f"{max_correction_pct:.2f}pp self-heal ceiling — this is a NAV-series "
                "disagreement (external flow / restated snapshot), not a stale "
                "derived value"
            )
            refused.append(entry)
        else:
            corrections.append(entry)
    return {"corrections": corrections, "refused": refused}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Custodian mark divergence — promoted from flag to failure
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT IS BEING COMPARED matters, and it decides the shape of the test.
# ``market_value`` is the CANONICAL valuation (settled ArcticDB close x shares).
# ``ib_market_value`` is IB Gateway's paper-account mark — a <=15-minute DELAYED
# quote captured at ~4:05pm ET. Measured over the 229 position-days carrying
# both, |mv - ib_mv| / ib_mv runs p50 = 0.085%, p90 = 0.52%, p95 = 0.79%. That
# body of the distribution is just one name's last-quarter-hour price move
# (sigma_15min ~ sigma_daily x sqrt(15/390) ~ 0.4% for the names actually held),
# which is why the pre-existing 0.5% band flags 24 of 229 (10.5%).
#
# So a flat percentage is the WRONG INSTRUMENT to promote to a failure: it fails
# high-volatility names for being volatile and passes stale marks on quiet ones,
# and any threshold picked to make today's data pass would be worthless.
#
# The repo already has the right test. ``eod_reconcile._detect_ib_mark_outside_range``
# (config#6349/#6818) flags an IB mark that fell outside that day's own ArcticDB
# [Low, High]. That is volatility-normalised with no arbitrary tolerance at all:
# a mark outside the range the security actually traded in is PROVABLY wrong
# however volatile the name is. This gate promotes THAT check rather than
# inventing a parallel percentage band beside it.
#
# The only remaining question is materiality, and the answer reuses the number
# the NAV three-way hard gate already declares (15bps of NAV) with the soft
# data_warnings floor beneath it ($500, ``NAV_BREACH_RESIDUAL_FLOOR_USD``)
# rather than minting a fourth threshold.
#
# Against the 8 out-of-range marks in the live window the separation is clean —
# the distribution is bimodal, and the gap between the two modes is 4x wide:
#   2026-08-04 AMD   mark $479.00 vs [$502.20, $530.13]  -$5,220  50.4bp  RAISE
#   2026-07-30 COIN  mark $154.45 vs [$159.31, $164.78]  -$2,999  29.7bp  RAISE
#   2026-06-26 LNTH  mark $105.12 vs [$108.51, $111.46]  -$2,532  25.5bp  RAISE
#   2026-07-29 SPY   ...                                   -$584   5.9bp  pass
#   2026-07-29 COIN  ...                                   -$420   4.2bp  pass
#   2026-08-21 DECK / 2026-07-16 TWLO / 2026-08-17 SPY   <=$102  <=1.0bp  pass
# The three that raise are genuinely wrong marks — AMD 2026-08-04 is also the
# single worst residual session in the entire window (-$9,713). The five that
# pass are edge-of-range rounding.
MARK_HARD_MATERIALITY_USD_FLOOR = 500.0
MARK_HARD_MATERIALITY_NAV_BPS = 15.0


def mark_materiality_usd(nav: float) -> float:
    """Dollar materiality floor for an out-of-range broker mark at this NAV."""
    return max(
        MARK_HARD_MATERIALITY_USD_FLOOR,
        MARK_HARD_MATERIALITY_NAV_BPS / 10_000.0 * nav,
    )


def check_custodian_marks(
    mark_range_flags: Sequence[Mapping[str, Any]] | None,
    *,
    nav: float | None,
    run_date: str = "",
) -> list[dict[str, Any]]:
    """Promote an out-of-range broker mark from a flag to a FAILURE.

    ``mark_range_flags`` is what ``eod_reconcile._detect_ib_mark_outside_range``
    returns: one entry per held ticker whose ``ib_market_value / shares`` fell
    outside that day's ArcticDB ``[Low, High]``, carrying ``mark_error_usd``
    (signed, the distance past the breached bound x shares).

    Every flag stays a flag — the observational surface that measured the
    distribution above is not deleted by promoting the check. A breach is
    returned only when the mark error is also MATERIAL against NAV, so a
    provably-wrong mark on a trivially small position does not halt the
    pipeline.
    """
    if not mark_range_flags or not nav:
        return []
    materiality_usd = mark_materiality_usd(nav)
    breaches: list[dict[str, Any]] = []
    for flag in mark_range_flags:
        error_usd = flag.get("mark_error_usd")
        try:
            error_usd = float(error_usd)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(error_usd) or abs(error_usd) <= materiality_usd:
            continue
        breaches.append({
            "kind": "custodian_mark",
            "run_date": run_date,
            "ticker": flag.get("ticker"),
            "ib_mark": flag.get("ib_mark"),
            "day_low": flag.get("day_low"),
            "day_high": flag.get("day_high"),
            "shares": flag.get("shares"),
            "error_usd": error_usd,
            "materiality_usd": materiality_usd,
            "pct_of_nav": error_usd / nav * 100.0,
            "message": (
                f"Custodian mark failure for {flag.get('ticker')} on {run_date}: "
                f"broker mark ${flag.get('ib_mark'):,.2f} fell outside the day's "
                f"traded range [${flag.get('day_low'):,.2f}, "
                f"${flag.get('day_high'):,.2f}] — ${error_usd:+,.0f} "
                f"({error_usd / nav * 100:+.3f}% of NAV) past the breached bound, "
                f"above the ${materiality_usd:,.0f} materiality floor "
                f"({MARK_HARD_MATERIALITY_NAV_BPS:.0f}bps of NAV). A mark outside "
                "the range the security actually traded in is wrong regardless of "
                "volatility, and every return, risk metric and attribution number "
                "is computed on top of these marks."
            ),
        })
    return breaches
