"""
Constrained mean-variance portfolio optimizer — PR 1 of the portfolio-optimizer
arc (plan: `alpha-engine-docs/private/portfolio-optimizer-260511.md`).

The institutional benchmark-as-null pattern: SPY (the benchmark) is the
no-conviction fill, cash is a pinned operational sleeve, conviction picks
express deviation from SPY within sector + position + vol-target constraints.

Math:
    maximize   wᵀα̂  −  λ · wᵀΣ_H w  −  γ · wᵀΩw  −  C(w − w_prev)/NAV
    s.t.       Σwᵢ = 1                                    (budget)
               w[CASH] = cash_sleeve                       (sleeve pin)
               0 ≤ wᵢ ≤ stance_capᵢ                       (per-name cap)
               Σ_{i∈sector S} wᵢ ≤ max_sector_pct          (sector cap)
               |wᵢ − w_prevᵢ| · NAV ≤ max_pct_adv · ADVᵢ   (participation cap)
               wᵢ = 0 for i with eligibility=False         (gate mask)
               wᵀΣ_H w ≤ σ²_target_H                       (vol-target SOC)

Transaction-cost term (tradeability arc, §43 — config#1401): the objective's
cost term is the participation-aware **square-root market-impact** cost from
the fleet's ONE shared engine (``nousergon_lib.quant.transaction_cost``,
lib#144), NOT a flat L1 turnover penalty. Per-name one-side dollar cost is

    C_i(Δwᵢ) = |Δwᵢ|·NAV · (half_spread + commission)/1e4              (linear)
             + impact_coef/1e4 · NAV^{1.5} · |Δwᵢ|^{1.5} / √ADVᵢ        (impact)

i.e. cost ∝ half_spread + c·√(participation) per Almgren-Chriss/Kissell, so the
DOLLAR cost scales as |Δwᵢ|^{1.5} (participation^{1.5}) — CONVEX in the trade
size, so ``−Σ C_i / NAV`` is concave and the objective stays DCP. Keying the
impact term on per-name ADV$ (from the scanner tradeability artifact) makes
turnover cost participation-aware: rebalancing an illiquid name is penalized far
more than a liquid one, where the flat 5bps L1 penalty was liquidity-blind. When
a name has no ADV coverage the impact term drops to the half-spread+commission
floor (the lib's conservative fallback) — never an error, never a silent zero.
The impact coefficient is a literature default (impact_coef_bps≈10 at 100%
participation) and configurable via the ``transaction_cost`` config block; a
TCA calibration loop against realized slippage_vs_signal (daemon.py:1570) is a
documented FOLLOW-ON, not built here. When ``adv_usd`` is None (or all-NaN, or
``portfolio_notional`` is None — pre-tradeability-artifact rollout) the term
degrades to the legacy flat L1 turnover penalty (``tcost_bps``), preserving
bit-identical fail-soft behavior. See §43 + optimizer-sota-upgrades-260526.md.

Horizon convention: Σ_H is the H-day covariance, where H is set via
``cfg["sigma_horizon_days"]`` (default 1 = daily, preserves legacy behavior).
Under i.i.d. log-return assumption, Σ_H = H · Σ_daily — see
`alpha-engine-docs/private/optimizer-sota-upgrades-260526.md` §A.1 for the
rationale (align Σ horizon with the canonical 21d log-domain α̂).

α̂-uncertainty term (workstream B.3): Ω = diag(σ_α̂²) penalizes positions
in proportion to per-name predictor variance — Garlappi-Uppal-Wang 2007
diagonal-Ω form. γ = cfg["alpha_uncertainty_penalty"] (default 0.0 = OFF,
preserves bit-identical legacy MVO). σ_α̂ comes from the predictor's
BayesianRidge posterior (`predicted_alpha_std` in predictions JSON, shipped
in alpha-engine-predictor B.1 #199). When `alpha_uncertainty=None` or all
entries are NaN, the term is skipped regardless of γ — covers the 1-week
soak window before the next training cycle promotes a BayesianRidge model.

This module is a pure function over numpy inputs. It does no I/O, no logging
config side effects, no S3 calls — easy to unit-test (PR 1) and easy to wire
into shadow mode (PR 2).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_CLARABEL = "CLARABEL"
_FALLBACK_SOLVERS = ("SCS", "OSQP")


@dataclass(frozen=True)
class OptimizerResult:
    weights: np.ndarray
    diagnostics: dict


def solve_target_weights(
    tickers: list[str],
    alpha_hat: np.ndarray,
    returns_panel: np.ndarray | None,
    w_prev: np.ndarray,
    sectors: list[str],
    stance_caps: np.ndarray,
    eligibility: np.ndarray,
    spy_idx: int,
    cash_idx: int,
    cfg: dict,
    *,
    alpha_uncertainty: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
    adv_usd: np.ndarray | None = None,
    portfolio_notional: float | None = None,
    name_sigma: np.ndarray | None = None,
) -> OptimizerResult:
    """
    Solve the constrained MVO and return target weights + diagnostics.

    Args:
        tickers: length-N universe. Must contain SPY (benchmark fill) and a
            CASH sentinel ticker.
        alpha_hat: shape (N,) predicted alpha vector. Convention: SPY entry is
            0.0 (benchmark = null hypothesis), CASH entry is a small negative
            number so the optimizer prefers SPY over cash when ε-indifferent.
        returns_panel: shape (T, N) daily returns history for covariance
            estimation. Rows with NaN are dropped pre-shrinkage. Caller is
            responsible for ensuring CASH column is ~0 (no return) and SPY
            column has real history.
        w_prev: shape (N,) current portfolio weights (positions / NAV). Used
            for the L1 turnover penalty.
        sectors: length-N sector labels. Use a stable string like "tech",
            "healthcare". SPY and CASH should have unique sentinel sectors
            (e.g., "__benchmark__", "__cash__") so they're not summed into
            real sector caps.
        stance_caps: shape (N,) per-name upper bound on weight. The caller
            composes this from base max_pos × stance multiplier × drawdown
            tier × earnings × coverage. For SPY use a high cap (e.g., 1.0);
            for CASH the cap is overridden by the equality pin.
        eligibility: shape (N,) bool. Names with eligibility=False are pinned
            to w_i = 0. SPY and CASH must be eligibility=True.
        spy_idx, cash_idx: positions in tickers list.
        cfg: dict with optimizer parameters. See OPTIMIZER_CONFIG_DEFAULTS.
        alpha_uncertainty: optional shape (N,) array of σ_α̂ per ticker —
            the BayesianRidge posterior std emitted by the predictor
            (predicted_alpha_std field in predictions JSON, B.1). When
            provided AND cfg["alpha_uncertainty_penalty"] > 0, adds the
            Garlappi-Uppal-Wang 2007 diagonal-Ω penalty term to the MVO
            objective so noisy picks size down proportionally. NaN entries
            are treated as zero uncertainty (no penalty for that name);
            covers the partial-rollout case during the 1-week soak between
            B.1 landing and the first BayesianRidge model being promoted
            in production. None ↔ no penalty regardless of γ.
        covariance: optional shape (N,N) DAILY covariance matrix Σ_daily
            (pre-horizon-scaling). When provided, the returns-panel estimator
            step is skipped and this matrix is used directly (horizon scaling
            still applied: Σ_H = H · Σ_daily). This is the intraday-re-solve
            path: the daemon reuses the morning Σ (cached in the optimizer
            shadow log) so an event-driven re-solve after a hard-risk exit is
            mechanism-identical to the morning solve and adds zero alpha
            look-ahead (Σ is daily-stable). ``returns_panel`` may be None when
            ``covariance`` is provided. None ↔ estimate Σ from returns_panel
            as before.
        adv_usd: optional shape (N,) per-name average daily DOLLAR volume
            (price × shares), read from the scanner tradeability artifact's
            ``tradeability.adv_usd`` block (crucible-research#343). Drives BOTH
            the participation-aware √-impact cost term AND the max-%-ADV
            participation constraint. Entries that are NaN / ≤0 (coverage gap)
            fall back to the half-spread+commission cost floor and are exempt
            from the participation constraint (no ADV → no participation bound
            can be formed — conservative degrade, never a crash). SPY and CASH
            entries are ignored (benchmark fill / sleeve carry no market
            impact). None ↔ no ADV info → the cost term degrades to the legacy
            flat ``tcost_bps`` L1 penalty and the participation constraint is
            skipped (bit-identical pre-tradeability behavior).
        portfolio_notional: optional book size in dollars (NAV) — required to
            convert weight deltas to trade notionals for the √-impact term and
            the max-%-ADV constraint. When None (or ≤0) the participation-aware
            path is disabled and the optimizer falls back to the legacy flat
            ``tcost_bps`` penalty, preserving fail-soft behavior.
        name_sigma: optional shape (N,) per-name daily return volatility used
            for the Almgren-Chriss σ-scaling of the impact term (σᵢ/refσ, where
            refσ = the cross-sectional median). None ↔ σ-agnostic √-impact (the
            lib default), which is the safe institutional baseline.

    Returns:
        OptimizerResult with weights (length N, sums to 1, sleeve pinned) and
        diagnostics dict including solver status, portfolio vol, active share
        vs SPY, and n_active.

    On infeasibility, returns the fallback weights (current weights with cash
    absorbing the residual) and diagnostics["status"] = "infeasible_fallback".
    """
    cfg = {**OPTIMIZER_CONFIG_DEFAULTS, **cfg}
    N = len(tickers)
    _validate_inputs(
        tickers, alpha_hat, returns_panel, w_prev,
        sectors, stance_caps, eligibility, spy_idx, cash_idx,
        covariance_provided=covariance is not None,
    )

    if covariance is not None:
        # Intraday re-solve: reuse the morning DAILY Σ instead of re-estimating.
        # Apply the SAME horizon scaling the estimator path applies (Σ_H =
        # H · Σ_daily); persisting/injecting an already-horizon-scaled Σ here
        # would double-scale and silently corrupt every vol diagnostic + the
        # vol-target SOC constraint — guarded by _validate_covariance + the
        # caller's vol-parity assertion.
        sigma_daily = _validate_covariance(covariance, N)
        horizon = int(cfg.get("sigma_horizon_days", 1))
        if horizon < 1:
            raise ValueError(f"sigma_horizon_days must be ≥ 1; got {horizon}")
        sigma = horizon * sigma_daily
    else:
        sigma = _estimate_covariance(returns_panel, cfg)
    omega_diag, alpha_unc_used = _resolve_alpha_uncertainty(alpha_uncertainty, N, cfg)

    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "cvxpy is required for portfolio_optimizer. Install via "
            "`pip install 'cvxpy>=1.9.2,<1.10'`. See requirements.txt."
        ) from e

    sigma_psd = cp.psd_wrap(sigma)
    w = cp.Variable(N)

    tcost = _build_tcost_term(
        cp, w, w_prev, adv_usd, portfolio_notional, name_sigma,
        spy_idx, cash_idx, cfg,
    )
    objective_terms = [
        alpha_hat @ w,
        - cfg["risk_aversion"] * cp.quad_form(w, sigma_psd),
        tcost.objective_term,
    ]
    if alpha_unc_used:
        # γ · sum_i (σ_α̂_i² · w_i²) — diagonal-Ω Garlappi-Uppal-Wang penalty.
        # cp.square(w) on a Variable is convex; sum with non-negative weights
        # remains convex; negated in a Maximize is concave (well-formed).
        gamma = float(cfg["alpha_uncertainty_penalty"])
        objective_terms.append(- gamma * (omega_diag @ cp.square(w)))
    objective = cp.Maximize(sum(objective_terms))

    eligibility_idx = np.where(~eligibility)[0]
    effective_caps = np.where(eligibility, stance_caps, 0.0)

    constraints = [
        cp.sum(w) == 1.0,
        w >= 0,
        w <= effective_caps,
        w[cash_idx] == cfg["cash_sleeve_pct"],
    ]
    if cfg.get("vol_target_annual") is not None:
        # Σ is at horizon H. Under i.i.d. log-returns, Var_ann = Var_H · (252/H),
        # so the H-day variance budget that corresponds to annual vol_target is
        # vol_target² · H/252. At default H=1 this reduces to (vol_target/√252)².
        horizon = int(cfg.get("sigma_horizon_days", 1))
        sigma_target_squared = (cfg["vol_target_annual"] ** 2) * horizon / 252
        constraints.append(cp.quad_form(w, sigma_psd) <= sigma_target_squared)
    if eligibility_idx.size > 0:
        constraints.append(w[eligibility_idx] == 0)

    for sector_label in _real_sectors(sectors):
        idx = [i for i, s in enumerate(sectors) if s == sector_label]
        constraints.append(cp.sum(w[idx]) <= cfg["max_sector_pct"])

    # ── max-%-ADV participation constraint (tradeability arc, config#1401) ──
    # Bound the single-solve trade in each name to a fraction of its average
    # daily dollar volume: |Δwᵢ|·NAV ≤ max_pct_adv · ADVᵢ. This is the HARD
    # capacity guardrail the √-impact objective term complements — the cost
    # term prices participation, this constraint refuses to trade a name so
    # thin that even a small book move would move the market. Only applied to
    # names with usable ADV coverage (NaN/≤0 → no bound can be formed → exempt,
    # conservative degrade) and only when a book notional is known; SPY/CASH
    # (benchmark fill / sleeve) carry no market impact and are always exempt.
    adv_cap_meta = _apply_max_pct_adv_constraint(
        cp, w, w_prev, constraints, adv_usd, portfolio_notional,
        spy_idx, cash_idx, cfg,
    )

    # ── daily turnover budget, as a CONSTRAINT (alpha-engine-config-I7346) ──
    # ‖w − w_prev‖₁ / 2 ≤ max_daily_turnover. cvxpy expresses the L1 norm
    # directly and the problem stays DCP (a norm is convex, so a ≤ bound is a
    # convex set). The optimizer therefore CHOOSES which trades fit the budget
    # and each surviving name lands at a size it actually wants.
    #
    # This replaces a post-solve uniform shrink
    # (w_exec = w_prev + (w_target − w_prev)·cap/requested), which answered
    # "how much may I trade" by trading a bit less of EVERYTHING — including
    # names whose entire delta IS an intended new position. Downstream,
    # optimizer_shadow's rebalance band drops any |Δw| < rebalance_band_pct,
    # so a hard enough shrink could push an entire entry cohort under the band
    # and delete it while the solve still reported `optimal`. A budget spent
    # by the objective cannot do that: nothing is scaled after the fact.
    turnover_meta = _apply_turnover_constraint(
        cp, w, w_prev, constraints, effective_caps, cash_idx, cfg,
        eligibility=eligibility,
    )

    problem = cp.Problem(objective, constraints)
    weights, status = _solve_with_fallback(problem, w, cfg)

    if weights is None:
        weights = _fallback_weights(w_prev, cash_idx, cfg["cash_sleeve_pct"])
        diagnostics = _build_diagnostics(
            weights, w_prev, sigma, alpha_hat, spy_idx, "infeasible_fallback", cfg,
            omega_diag=omega_diag, alpha_unc_used=alpha_unc_used,
        )
        diagnostics.update(tcost.diagnostics)
        diagnostics.update(adv_cap_meta)
        diagnostics.update(_turnover_diagnostics(weights, w_prev, turnover_meta))
        return OptimizerResult(weights=weights, diagnostics=diagnostics)

    weights, clip_mass_zeroed = _clip_and_renormalize(
        weights, effective_caps, cash_idx, cfg
    )
    weights, governor = _apply_turnover_governor(
        weights, w_prev, cfg,
        turnover_meta=turnover_meta,
        clip_mass_zeroed=clip_mass_zeroed,
    )
    diagnostics = _build_diagnostics(
        weights, w_prev, sigma, alpha_hat, spy_idx, status, cfg,
        omega_diag=omega_diag, alpha_unc_used=alpha_unc_used,
    )
    diagnostics.update(governor)
    diagnostics.update(tcost.diagnostics)
    diagnostics.update(adv_cap_meta)
    return OptimizerResult(weights=weights, diagnostics=diagnostics)


_VOL_TARGET_COMMENT = """
vol_target_annual default is None (no SOC constraint). For a long-only
benchmark-aware portfolio that uses SPY as the no-conviction fill, the
portfolio's natural volatility is bounded below by SPY's vol (≈16% annual),
since SPY absorbs ~89% of the book on conviction-light days. Setting
vol_target_annual below SPY vol is structurally infeasible without bonds.
Set explicitly (e.g., 0.25) to enable a stress-regime cap that only binds
during high-vol periods. Reserved for v2 multi-asset / risk-parity layer.
""".strip()


OPTIMIZER_CONFIG_DEFAULTS: dict = {
    "vol_target_annual": None,
    "risk_aversion": 5.0,
    "tcost_bps": 5.0,
    "cash_sleeve_pct": 0.03,
    "max_sector_pct": 0.25,
    "covariance_shrinkage": "ledoit_wolf",
    "min_position_pct": 0.005,
    # ── Participation-aware transaction cost (tradeability arc, config#1401) ─
    # ``tcost_mode`` selects the objective's turnover-cost term:
    #   "sqrt_impact" (default) — the canonical participation-aware √-impact
    #     dollar cost from nousergon_lib.quant.transaction_cost (lib#144),
    #     keyed on per-name ADV$. Requires adv_usd + portfolio_notional; when
    #     either is absent it AUTOMATICALLY degrades to the flat L1 penalty
    #     (fail-soft). This is the institutional-correct construction cost.
    #   "flat_l1" — the legacy flat ``tcost_bps`` L1 penalty (liquidity-blind).
    #     Kept for A/B and as the explicit fallback the auto-degrade lands on.
    # The impact COEFFICIENT (impact_coef_bps, half_spread_bps, commission_bps)
    # lives in the ``transaction_cost`` config block consumed by
    # TransactionCostModel.from_config — literature defaults today; a TCA
    # calibration loop against realized slippage_vs_signal (daemon.py:1570) is
    # the documented FOLLOW-ON that will tune impact_coef_bps. See §43.
    "tcost_mode": "sqrt_impact",
    # ── max-%-ADV participation constraint (config#1401) ─────────────────────
    # HARD capacity guardrail: |Δwᵢ|·NAV ≤ max_pct_adv · ADVᵢ per name per
    # solve. None → constraint OFF (bit-identical legacy behavior). 0.05 = a
    # single-solve trade may consume at most 5% of a name's average daily
    # dollar volume — a conservative institutional participation ceiling that
    # keeps the √-impact objective term honest (the cost prices participation;
    # this refuses to trade a name so thin the model can't be trusted). Only
    # binds on names with ADV coverage; requires portfolio_notional to convert
    # weight deltas to notionals — skipped (with a diagnostic) when absent.
    "max_pct_adv": 0.05,
    # Horizon (trading days) at which Σ is expressed. 1 = legacy daily Σ
    # (bit-identical to pre-260526 behavior); set to 21 to align Σ with the
    # canonical 21d log-domain α̂. See optimizer-sota-upgrades-260526.md §A.1.
    "sigma_horizon_days": 1,
    # EWMA decay for ``covariance_shrinkage="ewma"``. RiskMetrics 1996
    # canonical value 0.94 ↔ ~11d half-life; 0.97 ↔ ~23d half-life (closer
    # to canonical 21d α̂ horizon). See optimizer-sota-upgrades-260526.md §A.2.
    "ewma_lambda_decay": 0.94,
    # γ for the Garlappi-Uppal-Wang 2007 α̂-uncertainty penalty term
    # γ · sum_i(σ_α̂_i² · w_i²). 0.0 (default) disables the term and
    # preserves bit-identical legacy MVO behavior. Backtester-tunable.
    # See optimizer-sota-upgrades-260526.md §B.3.
    "alpha_uncertainty_penalty": 0.0,
    # ── Turnover governor (gradual-rebalance guardrail) ──────────────────
    # SAFETY guardrail — NOT an alpha knob, NOT backtester-tuned. Caps the
    # one-way turnover the book may execute in a single day by scaling the
    # step from w_prev toward the optimizer's target, so the portfolio WALKS
    # to the target over several daily re-solves instead of jumping in one
    # session. Institutional books rebalance gradually; a large single-day
    # reallocation is the rare exception that should be operator-reviewed,
    # not the default. Defaults ON (unlike the optional α̂-uncertainty term)
    # because it's a fail-safe — a too-tight cap only slows rebalancing, it
    # can never produce a worse trade.
    #   max_daily_turnover: one-way turnover cap/day (None → governor OFF,
    #     bit-identical legacy behavior).
    #   large_move_turnover_flag: when REQUESTED (uncapped) one-way turnover
    #     exceeds this, the solve sets large_move_flagged so the planner
    #     alerts for approval. The move is STILL executed gradually under the
    #     cap — flagging never bypasses the cap, and the cap never waits on
    #     the flag.
    "max_daily_turnover": 0.20,
    "large_move_turnover_flag": 0.35,
    # ── Turnover tripwire (L4515) ─────────────────────────────────────────
    # SAFETY alarm — NOT an alpha knob, NOT backtester-tuned. Band-checks the
    # EXECUTED one-way turnover daily in the planner and pages on breach
    # (executor/turnover_tripwire.py): daily = cap × multiple at ERROR (the
    # governor should make a breach impossible, so one means the cap was
    # bypassed/disabled); rolling = sum over the last N sessions at WARN
    # (churn-by-a-thousand-cuts — each day under the cap, week abnormal; the
    # signature of the 5/29, 6/01, 6/04 incidents this generalizes).
    "turnover_tripwire_enabled": True,
    "turnover_tripwire_daily_multiple": 1.25,
    "turnover_tripwire_rolling_days": 5,
    "turnover_tripwire_rolling_sum_band": 0.60,
}


def _resolve_alpha_uncertainty(
    alpha_uncertainty: np.ndarray | None,
    N: int,
    cfg: dict,
) -> tuple[np.ndarray, bool]:
    """Build omega_diag = σ_α̂² and decide whether the penalty term is
    active for this solve.

    Returns (omega_diag, used) where ``used`` is True iff γ > 0 AND at
    least one σ_α̂ entry is finite AND non-zero. On used=False the caller
    skips the penalty term, preserving bit-identical legacy behavior.

    Negative or non-finite σ_α̂ entries are coerced to 0 (no penalty for
    that name) so partial-rollout (legacy Ridge std=None → NaN) does not
    raise. Caller's alpha_uncertainty contract is "predictor posterior
    std or NaN per ticker"; we enforce the σ ≥ 0 invariant defensively
    here too — a negative entry IS an upstream bug (BR posterior is
    always positive), but the optimizer is the wrong place to crash the
    morning planner over it. Log loud, treat as missing.
    """
    gamma = float(cfg.get("alpha_uncertainty_penalty", 0.0))
    if alpha_uncertainty is None or gamma <= 0.0:
        return np.zeros(N), False
    arr = np.asarray(alpha_uncertainty, dtype=np.float64).ravel()
    if arr.shape != (N,):
        raise ValueError(
            f"alpha_uncertainty shape {arr.shape} != ({N},) — must be one entry per ticker"
        )
    # Any negative entry is an upstream contract violation. Don't crash the
    # morning planner — log loud, coerce to 0 (per partial-rollout policy).
    if np.any(arr[np.isfinite(arr)] < 0.0):
        n_bad = int(np.sum((arr < 0.0) & np.isfinite(arr)))
        logger.warning(
            "alpha_uncertainty has %d negative entries — coercing to 0. "
            "Predictor BayesianRidge posterior is always positive; investigate "
            "upstream (B.1 #199 wiring).", n_bad,
        )
    # NaN / inf / negative → 0 → no penalty contribution
    arr = np.where(np.isfinite(arr) & (arr >= 0.0), arr, 0.0)
    omega = arr ** 2
    used = bool(np.any(omega > 0.0))
    return omega, used


@dataclass(frozen=True)
class _TCostTerm:
    """The optimizer objective's turnover-cost term + its observability."""
    objective_term: object          # a cvxpy expression (concave, DCP-safe)
    diagnostics: dict


def _clean_adv(
    adv_usd: np.ndarray | None, N: int, spy_idx: int, cash_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize an ADV$ vector → (adv, usable_mask).

    ``adv`` has NaN/≤0/non-finite entries coerced to 0.0 (no coverage).
    ``usable_mask`` is True only for real names (not SPY/CASH) with ADV>0.
    SPY (benchmark fill) and CASH (sleeve) carry no market impact and are
    always excluded from both the cost term and the participation constraint.
    ``adv_usd=None`` → all-zero adv, all-False mask (no ADV info at all).
    """
    if adv_usd is None:
        return np.zeros(N), np.zeros(N, dtype=bool)
    adv = np.asarray(adv_usd, dtype=np.float64).ravel()
    if adv.shape != (N,):
        raise ValueError(
            f"adv_usd shape {adv.shape} != ({N},) — one ADV$ entry per ticker"
        )
    adv = np.where(np.isfinite(adv) & (adv > 0.0), adv, 0.0)
    usable = adv > 0.0
    usable[spy_idx] = False
    usable[cash_idx] = False
    return adv, usable


def _resolve_ref_sigma(
    name_sigma: np.ndarray | None, usable_mask: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    """Return (sigma_used, ref_sigma) for the Almgren-Chriss σ-scaling.

    ref_sigma is the cross-sectional MEDIAN σ over usable names (the lib's
    self-calibrating reference: the median-vol name reproduces the σ-agnostic
    cost). Returns (None, None) — σ-agnostic — when no per-name σ is supplied
    or no usable finite-positive σ exists (safe institutional default).
    """
    if name_sigma is None:
        return None, None
    sig = np.asarray(name_sigma, dtype=np.float64).ravel()
    finite_pos = np.isfinite(sig) & (sig > 0.0) & usable_mask
    if not np.any(finite_pos):
        return None, None
    ref = float(np.median(sig[finite_pos]))
    if not (ref > 0.0):
        return None, None
    return sig, ref


def _build_tcost_term(
    cp, w, w_prev: np.ndarray,
    adv_usd: np.ndarray | None,
    portfolio_notional: float | None,
    name_sigma: np.ndarray | None,
    spy_idx: int, cash_idx: int,
    cfg: dict,
) -> _TCostTerm:
    """Build the objective's turnover-cost term.

    Consumes the fleet's canonical √-impact ``TransactionCostModel`` (lib#144)
    for the coefficients — the executor never re-derives the impact math. The
    per-name one-side DOLLAR cost decomposes into a cvxpy-DCP-safe convex sum:

        C_i = (half_spread+commission)/1e4 · NAV · |Δwᵢ|          (linear)
            + impact_coef·(σᵢ/refσ)/1e4 · NAV^{1.5}/√ADVᵢ · |Δwᵢ|^{1.5}

    |Δwᵢ| is convex; |Δwᵢ|^{1.5} = power(|Δwᵢ|,1.5) is convex; both enter the
    objective as ``−ΣC_i/NAV`` (concave → valid in cp.Maximize). Dividing by
    NAV keeps the term commensurate with the (weight-space) α̂ and risk terms.

    Fail-soft: when tcost_mode="flat_l1", OR portfolio_notional is missing/≤0,
    OR no name has usable ADV coverage, the term degrades to the legacy flat
    ``tcost_bps`` L1 penalty ``−(tcost_bps/1e4)·‖w−w_prev‖₁`` — bit-identical
    to pre-1401 behavior. The chosen mode is surfaced in diagnostics.
    """
    flat_l1 = - (cfg["tcost_bps"] / 1e4) * cp.norm(w - w_prev, 1)
    mode = str(cfg.get("tcost_mode", "sqrt_impact"))
    N = w.shape[0]
    adv, usable = _clean_adv(adv_usd, N, spy_idx, cash_idx)
    n_usable = int(np.sum(usable))

    def _flat(reason: str) -> _TCostTerm:
        return _TCostTerm(
            objective_term=flat_l1,
            diagnostics={
                "tcost_term_mode": "flat_l1",
                "tcost_fallback_reason": reason,
                "tcost_n_names_with_adv": n_usable,
            },
        )

    if mode == "flat_l1":
        return _flat("configured_flat_l1")
    if mode != "sqrt_impact":
        raise ValueError(f"Unknown tcost_mode: {mode!r} (expected sqrt_impact|flat_l1)")
    if portfolio_notional is None or not (float(portfolio_notional) > 0.0):
        return _flat("no_portfolio_notional")
    if n_usable == 0:
        # No ADV coverage anywhere (pre-tradeability-artifact rollout, or an
        # all-gap universe) → cannot form the participation-aware term.
        return _flat("no_adv_coverage")

    try:
        from nousergon_lib.quant.transaction_cost import TransactionCostModel
    except ImportError as e:  # pragma: no cover - pin guarantees availability
        logger.warning(
            "nousergon_lib.quant.transaction_cost unavailable (%s) — falling "
            "back to flat L1 turnover penalty. Bump the nousergon-lib pin to "
            ">=v0.75.0 (lib#144).", e,
        )
        return _flat("lib_unavailable")

    model = TransactionCostModel.from_config(cfg)
    nav = float(portfolio_notional)
    sig_used, ref_sigma = _resolve_ref_sigma(name_sigma, usable)

    # Linear (half-spread + commission) part — applies to EVERY name that
    # trades, ADV-covered or not: it's the spread/commission floor, not impact.
    linear_bps = model.half_spread_bps + model.commission_bps
    dw = w - w_prev
    linear_cost = (linear_bps / 1e4) * nav * cp.abs(dw)  # per-name $, vector

    # Impact part — only names with ADV coverage. k_i · |Δwᵢ|^{1.5}, where
    # k_i = impact_coef·(σᵢ/refσ)/1e4 · NAV^{1.5} / √ADVᵢ.
    impact_terms = []
    for i in np.where(usable)[0]:
        sigma_scale = 1.0
        if sig_used is not None and ref_sigma:
            s = sig_used[i]
            if np.isfinite(s) and s > 0.0:
                sigma_scale = float(s) / ref_sigma
        k_i = (
            model.impact_coef_bps * sigma_scale / 1e4
            * (nav ** 1.5) / math.sqrt(adv[i])
        )
        if k_i > 0.0:
            impact_terms.append(k_i * cp.power(cp.abs(dw[i]), 1.5))

    total_cost = cp.sum(linear_cost)
    if impact_terms:
        total_cost = total_cost + cp.sum(impact_terms)
    # Normalize dollar cost back to weight units (÷NAV) so the cost term is
    # commensurate with the weight-space α̂ / risk / uncertainty terms.
    objective_term = - total_cost / nav
    return _TCostTerm(
        objective_term=objective_term,
        diagnostics={
            "tcost_term_mode": "sqrt_impact",
            "tcost_n_names_with_adv": n_usable,
            "tcost_impact_coef_bps": float(model.impact_coef_bps),
            "tcost_sigma_scaled": bool(sig_used is not None and ref_sigma),
            "tcost_portfolio_notional": nav,
        },
    )


def _apply_max_pct_adv_constraint(
    cp, w, w_prev: np.ndarray, constraints: list,
    adv_usd: np.ndarray | None,
    portfolio_notional: float | None,
    spy_idx: int, cash_idx: int,
    cfg: dict,
) -> dict:
    """Append the per-name max-%-ADV participation constraint, in place.

    ``|wᵢ − w_prevᵢ| · NAV ≤ max_pct_adv · ADVᵢ`` for every real name with
    usable ADV coverage. cvxpy encodes ``|Δwᵢ| ≤ bound_i`` as the linear pair
    ``Δwᵢ ≤ bound_i``, ``−Δwᵢ ≤ bound_i`` (kept affine so the LP/SOCP stays
    convex). Skipped (returns a diagnostic) when the cap is disabled, no book
    notional is known, or no name has ADV coverage — fail-soft, never a crash.

    Returns a diagnostics dict recording whether the cap was applied + to how
    many names, so an operator can see the constraint is (or isn't) live.
    """
    cap = cfg.get("max_pct_adv")
    N = w.shape[0]
    adv, usable = _clean_adv(adv_usd, N, spy_idx, cash_idx)
    n_usable = int(np.sum(usable))
    if cap is None or not (float(cap) > 0.0):
        return {"max_pct_adv_applied": False, "max_pct_adv_reason": "disabled"}
    if portfolio_notional is None or not (float(portfolio_notional) > 0.0):
        return {"max_pct_adv_applied": False, "max_pct_adv_reason": "no_portfolio_notional"}
    if n_usable == 0:
        return {"max_pct_adv_applied": False, "max_pct_adv_reason": "no_adv_coverage"}

    nav = float(portfolio_notional)
    cap = float(cap)
    idx = np.where(usable)[0]
    # Per-name weight-space bound: max_pct_adv·ADVᵢ / NAV.
    bounds = cap * adv[idx] / nav
    dw = w[idx] - w_prev[idx]
    constraints.append(dw <= bounds)
    constraints.append(-dw <= bounds)
    return {
        "max_pct_adv_applied": True,
        "max_pct_adv": cap,
        "max_pct_adv_n_names_constrained": int(len(idx)),
        "max_pct_adv_min_bound_weight": float(np.min(bounds)) if len(bounds) else None,
    }


def _validate_inputs(
    tickers: list[str],
    alpha_hat: np.ndarray,
    returns_panel: np.ndarray,
    w_prev: np.ndarray,
    sectors: list[str],
    stance_caps: np.ndarray,
    eligibility: np.ndarray,
    spy_idx: int,
    cash_idx: int,
    covariance_provided: bool = False,
) -> None:
    N = len(tickers)
    if N == 0:
        raise ValueError("Empty universe — cannot optimize")
    for name, arr in (
        ("alpha_hat", alpha_hat),
        ("w_prev", w_prev),
        ("stance_caps", stance_caps),
        ("eligibility", eligibility),
    ):
        if arr.shape != (N,):
            raise ValueError(f"{name} shape {arr.shape} != ({N},)")
    # ``returns_panel`` is only required when estimating Σ from history. In the
    # intraday re-solve path the caller supplies a precomputed covariance and
    # may pass returns_panel=None.
    if not covariance_provided:
        if returns_panel is None:
            raise ValueError("returns_panel is required when covariance is not provided")
        if returns_panel.ndim != 2 or returns_panel.shape[1] != N:
            raise ValueError(
                f"returns_panel shape {returns_panel.shape} incompatible with N={N}"
            )
    if len(sectors) != N:
        raise ValueError(f"sectors length {len(sectors)} != N={N}")
    if not (0 <= spy_idx < N) or not (0 <= cash_idx < N):
        raise ValueError(f"spy_idx={spy_idx} cash_idx={cash_idx} out of range [0,{N})")
    if not eligibility[spy_idx]:
        raise ValueError("SPY must be eligible (benchmark fill)")
    if not eligibility[cash_idx]:
        raise ValueError("CASH must be eligible (sleeve pin)")


def _ewma_covariance(returns: np.ndarray, lambda_decay: float) -> np.ndarray:
    """RiskMetrics 1996 EWMA covariance with zero-mean assumption.

    Σ_EWMA = (1−λ) · Σ_{k=0}^{T-1} λ^k · r_{t-k} r_{t-k}ᵀ, normalized so weights
    sum to 1 over the finite window. The zero-mean simplification is standard
    for daily equity returns (E[r] ≪ σ); RiskMetrics 1996 §5.3.2.

    With λ=0.94 the effective half-life is log(0.5)/log(0.94) ≈ 11.2 trading days
    (RiskMetrics canonical); 0.97 → ~22.8 days (closer to 21d α̂ horizon).

    Degenerate at λ=1.0: weights become uniform 1/T → reduces to (unbiased
    only up to the 1/T vs 1/(T-1) factor) sample covariance. Tested.
    """
    if not 0.5 <= lambda_decay <= 1.0:
        raise ValueError(
            f"ewma_lambda_decay must be in [0.5, 1.0]; got {lambda_decay}. "
            f"RiskMetrics 1996 canonical is 0.94 (daily) or 0.97 (monthly)."
        )
    T = returns.shape[0]
    if lambda_decay >= 1.0 - 1e-12:
        # Uniform weights (degenerate). Treat λ=1 as plain sample-cov-equivalent.
        return (returns.T @ returns) / T
    # Newest observation first; row 0 carries the largest weight.
    R = returns[::-1]
    weights = (1.0 - lambda_decay) * lambda_decay ** np.arange(T)
    weights /= weights.sum()  # normalize for finite-window truncation
    return (R.T * weights) @ R


def _validate_covariance(cov: np.ndarray, N: int) -> np.ndarray:
    """Validate + symmetrize a precomputed DAILY covariance for the re-solve.

    JSON round-tripping can introduce tiny asymmetry / non-PSD perturbations.
    Symmetrize (0.5·(Σ+Σᵀ)) and fail LOUD on shape mismatch, non-finite
    entries, or a materially negative eigenvalue — a silently mis-shaped Σ
    would corrupt the entire vol math (vol-target SOC + every vol diagnostic).
    """
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (N, N):
        raise ValueError(f"covariance shape {cov.shape} != ({N}, {N})")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance contains non-finite entries")
    cov = 0.5 * (cov + cov.T)
    min_eig = float(np.linalg.eigvalsh(cov).min())
    tol = -1e-8 * max(1.0, float(np.trace(cov)) / N)
    if min_eig < tol:
        raise ValueError(
            f"covariance is not PSD (min eigenvalue {min_eig:.3e} < tol {tol:.3e})"
        )
    return cov


def _estimate_covariance_daily(returns_panel: np.ndarray, cfg: dict) -> np.ndarray:
    """Estimate the DAILY covariance Σ_daily (pre-horizon-scaling).

    Split out of ``_estimate_covariance`` so the optimizer shadow log can
    persist Σ_daily for an intraday re-solve (see ``solve_target_weights``'s
    ``covariance`` argument). Horizon scaling lives in ``_estimate_covariance``.

    Estimators (cfg["covariance_shrinkage"]):
      * "ledoit_wolf" (default): Ledoit-Wolf 2004 constant-correlation shrinkage
        on equal-weighted samples. Institutional default.
      * "oas": Chen et al. 2010 Oracle Approximating Shrinkage. Lower-MSE than
        LW when T/N is small (our universe ~27 × T~252 → T/N≈9 is modestly
        small-sample). Drop-in alternative; same shrinkage-target family
        (multiple of identity). See optimizer-sota-upgrades-260526.md §A.3.
      * "sample": raw sample covariance, no shrinkage. Test-only.
      * "ewma": RiskMetrics 1996 EWMA with cfg["ewma_lambda_decay"] (default
        0.94). Captures vol-clustering; weights recent observations more.
        See optimizer-sota-upgrades-260526.md §A.2.
    """
    clean = returns_panel[~np.isnan(returns_panel).any(axis=1)]
    if clean.shape[0] < 20:
        raise ValueError(
            f"Need ≥20 clean return rows for covariance; got {clean.shape[0]}"
        )
    estimator = cfg["covariance_shrinkage"]
    if estimator == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for Ledoit-Wolf shrinkage. Install "
                "via `pip install 'scikit-learn>=1.3,<1.6'`."
            ) from e
        sigma_daily = LedoitWolf().fit(clean).covariance_
    elif estimator == "oas":
        try:
            from sklearn.covariance import OAS
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for OAS shrinkage. Install via "
                "`pip install 'scikit-learn>=1.3,<1.6'`."
            ) from e
        sigma_daily = OAS().fit(clean).covariance_
    elif estimator == "sample":
        sigma_daily = np.cov(clean, rowvar=False)
    elif estimator == "ewma":
        sigma_daily = _ewma_covariance(clean, float(cfg.get("ewma_lambda_decay", 0.94)))
    else:
        raise ValueError(f"Unknown covariance_shrinkage: {estimator}")

    return sigma_daily


def _estimate_covariance(returns_panel: np.ndarray, cfg: dict) -> np.ndarray:
    """Return covariance at horizon ``cfg["sigma_horizon_days"]``.

    Estimates Σ_daily via ``_estimate_covariance_daily``, then scales by
    horizon-days under i.i.d. log-return assumption: Σ_H = H · Σ_daily.
    Default H=1 preserves legacy daily Σ bit-identical (1 × Σ = Σ).
    """
    sigma_daily = _estimate_covariance_daily(returns_panel, cfg)
    horizon = int(cfg.get("sigma_horizon_days", 1))
    if horizon < 1:
        raise ValueError(f"sigma_horizon_days must be ≥ 1; got {horizon}")
    return horizon * sigma_daily


def _real_sectors(sectors: list[str]) -> set[str]:
    return {s for s in sectors if not (s.startswith("__") and s.endswith("__"))}


def _solve_with_fallback(problem, w, cfg: dict):
    import cvxpy as cp
    for solver in (_CLARABEL, *_FALLBACK_SOLVERS):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver)
        except (cp.error.SolverError, ValueError) as e:
            logger.warning(f"Solver {solver} raised {e!r}, trying next")
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            return np.asarray(w.value, dtype=float), problem.status
        logger.warning(
            f"Solver {solver} returned status={problem.status}, trying next"
        )
    return None, problem.status if problem.status else "no_solver_available"


def _fallback_weights(
    w_prev: np.ndarray, cash_idx: int, cash_sleeve_pct: float,
) -> np.ndarray:
    weights = np.maximum(w_prev.copy(), 0.0)
    weights[cash_idx] = 0.0
    equity_sum = weights.sum()
    target_equity = 1.0 - cash_sleeve_pct
    if equity_sum > 0:
        weights *= target_equity / equity_sum
    weights[cash_idx] = cash_sleeve_pct
    return weights


def _clip_and_renormalize(
    weights: np.ndarray,
    effective_caps: np.ndarray,
    cash_idx: int,
    cfg: dict,
) -> tuple[np.ndarray, float]:
    """Clip to the box, drop sub-``min_position_pct`` dust, renormalize.

    Returns ``(weights, mass_zeroed)``. ``mass_zeroed`` is the total weight
    removed by the dust rule and is NOT bookkeeping trivia: it is the exact
    budget by which this post-solve step may legitimately push the vector past
    the solver's turnover constraint (zeroing mass m adds ≤ m of one-way
    turnover from the drop itself, and ≤ m more from renormalizing the
    survivors up by 1/(1−m)). ``_apply_turnover_governor`` uses it as the
    assertion tolerance, so the tolerance is DERIVED from what actually
    happened rather than being a hand-picked epsilon that silently absorbs a
    real breach.
    """
    weights = np.maximum(weights, 0.0)
    weights = np.minimum(weights, effective_caps + 1e-8)
    small = (weights < cfg["min_position_pct"]) & (np.arange(len(weights)) != cash_idx)
    mass_zeroed = float(np.sum(weights[small]))
    weights = np.where(small, 0.0, weights)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights, mass_zeroed


def _mandatory_turnover_floor(
    w_prev: np.ndarray,
    effective_caps: np.ndarray,
    cash_idx: int,
    cfg: dict,
) -> float:
    """One-way turnover that the OTHER constraints force, regardless of alpha.

    Adding ``‖w − w_prev‖₁/2 ≤ cap`` can only be safe if the feasible set is
    still non-empty. ``w_prev`` has turnover 0 and so always satisfies the
    turnover constraint itself — but it does not necessarily satisfy the rest
    of the program: a held name that went ineligible is pinned to 0, and the
    cash sleeve is pinned to ``cash_sleeve_pct``. Those pins MANDATE movement.
    If that mandated movement exceeds the budget, the program is infeasible
    and the whole book falls to the hold path — a new failure mode, introduced
    by the fix, on a day when a forced exit is exactly what must happen.

    So the budget governs DISCRETIONARY trading only. This returns a bound on
    the forced movement; the caller raises the constraint's right-hand side to
    it when it is larger than the configured cap, which makes the feasible set
    non-empty BY CONSTRUCTION rather than by assumption:

      * ``d_i`` = distance from ``w_prev_i`` to its box ``[l_i, u_i]``. The
        projection ``p`` of ``w_prev`` into the box costs ``Σ d_i`` of L1.
      * ``p`` need not sum to 1; the residual ``r = 1 − Σp`` must be absorbed
        by names with slack (SPY, the unconstrained benchmark fill, always
        has some), costing a further ``|r|`` of L1.

    ``(Σd + |r|) / 2`` is therefore an attainable one-way turnover for a point
    satisfying the box, the sleeve pin and the budget identity. Sector and
    participation caps can only be satisfied at that point or need more
    movement; they are inequality caps on sums that the projection does not
    tighten, so this is the operative bound in practice, and any residual
    infeasibility still lands on the pre-existing ``_fallback_weights`` path
    rather than anywhere new.
    """
    lower = np.zeros_like(w_prev)
    upper = np.array(effective_caps, dtype=float)
    sleeve = float(cfg["cash_sleeve_pct"])
    lower[cash_idx] = sleeve
    upper[cash_idx] = sleeve
    projected = np.clip(w_prev, lower, upper)
    forced_l1 = float(np.sum(np.abs(projected - w_prev)))
    residual = abs(1.0 - float(projected.sum()))
    return (forced_l1 + residual) / 2.0


def _decompose_mandatory_turnover(
    w_prev: np.ndarray,
    effective_caps: np.ndarray,
    eligibility: np.ndarray,
    cash_idx: int,
    cfg: dict,
) -> dict:
    """Which constraint forced each unit of the mandatory turnover floor.

    alpha-engine-config-I8753. ``turnover_mandatory_floor`` has been emitted
    since the turnover constraint shipped, and it answers "how much of today's
    trading was forced" — but not BY WHAT, and the three causes have entirely
    different fixes:

    * ``cash_sleeve_pin``   — the sleeve is an equality pin; drift into or out
      of it is mandated every session and is not a defect.
    * ``ineligibility_pin`` — a held name went ineligible (research EXIT, GBM
      veto, score gate) and is pinned to zero. A forced exit is the system
      working.
    * ``position_cap``      — a held name sits above its per-name cap
      ``max_position_pct × stance_multiplier``. The stance is re-derived every
      morning with no persistence, so a name whose stance flips
      momentum→quality must be cut from 10% of NAV to 4% THAT MORNING,
      regardless of its alpha, and rebuilt when the stance flips back.

    Measured 2026-08-27 across the nine sessions since the v2 cutover
    (``predictor/optimizer_shadow/{date}.json``, read-only): cap cuts on held
    names mandated **0.278 of NAV in one-way selling**, ~3.1% per session,
    entirely alpha-independent. On 2026-08-27 the floor was 0.105 against an
    executed one-way turnover of 0.170 — 62% of the day's trading forced before
    the objective was consulted.

    That total was legible only by reading nine artifacts and diffing
    ``stance_caps`` by hand, which is what this function removes. Whether the
    right fix is hold-side stance hysteresis, a different cap schedule, or
    nothing at all is NOT decided here — it is decided by watching this
    decomposition accrue. Nine sessions gave only four cap flips with a full
    three-session look-ahead (two reverted), which is too thin to set a
    hysteresis default on.

    Emitted on every solve, healthy included: a component emitting nothing is
    not healthy, it is unobserved.

    Returns per-cause one-way turnover plus ``total``, which reconciles to
    ``_mandatory_turnover_floor`` to within floating-point tolerance. The
    ``renormalization`` term is the residual the projection leaves behind —
    the mass that must be absorbed by names with slack once every pin is
    satisfied — and it belongs to no single name.
    """
    sleeve = float(cfg["cash_sleeve_pct"])
    lower = np.zeros_like(w_prev)
    upper = np.array(effective_caps, dtype=float)
    lower[cash_idx] = sleeve
    upper[cash_idx] = sleeve
    projected = np.clip(w_prev, lower, upper)
    per_name = np.abs(projected - w_prev)

    idx = np.arange(len(w_prev))
    is_cash = idx == cash_idx
    # An ineligible name carries an effective cap of 0, so its whole holding is
    # forced out. Attributing that to "position_cap" would read as a sizing
    # artifact when it is a deliberate exit — the two have opposite responses.
    is_ineligible = (~np.asarray(eligibility, dtype=bool)) & (~is_cash)

    sleeve_l1 = float(per_name[is_cash].sum())
    ineligible_l1 = float(per_name[is_ineligible].sum())
    cap_l1 = float(per_name[~is_cash & ~is_ineligible].sum())
    residual = abs(1.0 - float(projected.sum()))

    names_at_cap = [
        int(i) for i in idx
        if not is_cash[i] and not is_ineligible[i] and per_name[i] > 1e-9
    ]
    names_pinned_out = [int(i) for i in idx if is_ineligible[i] and per_name[i] > 1e-9]

    return {
        "cash_sleeve_pin": sleeve_l1 / 2.0,
        "ineligibility_pin": ineligible_l1 / 2.0,
        "position_cap": cap_l1 / 2.0,
        "renormalization": residual / 2.0,
        "total": (sleeve_l1 + ineligible_l1 + cap_l1 + residual) / 2.0,
        "n_names_over_cap": len(names_at_cap),
        "n_names_pinned_out": len(names_pinned_out),
    }


def _apply_turnover_constraint(
    cp,
    w,
    w_prev: np.ndarray,
    constraints: list,
    effective_caps: np.ndarray,
    cash_idx: int,
    cfg: dict,
    eligibility: np.ndarray | None = None,
) -> dict:
    """Append the L1 daily-turnover budget to ``constraints``.

    Returns metadata the diagnostics and the post-solve assertion both read:
    the configured cap, the mandatory floor, and the EFFECTIVE right-hand side
    actually imposed. ``max_daily_turnover=None`` disables the budget entirely
    (legacy behaviour, bit-identical).
    """
    cap = cfg.get("max_daily_turnover")
    meta: dict = {
        "turnover_constraint_applied": False,
        "turnover_constraint_cap": None,
        "turnover_mandatory_floor": None,
        "turnover_mandatory_floor_by_cause": None,
        "turnover_constraint": None,
    }
    if cap is None or cap <= 0:
        return meta
    floor = _mandatory_turnover_floor(w_prev, effective_caps, cash_idx, cfg)
    # alpha-engine-config-I8753 — WHICH constraint forced each unit of it.
    # `eligibility` is optional so every existing caller keeps working; without
    # it the ineligibility pin cannot be separated from the position cap, and
    # the decomposition says so rather than guessing.
    by_cause = (
        _decompose_mandatory_turnover(
            w_prev, effective_caps, eligibility, cash_idx, cfg,
        )
        if eligibility is not None
        else None
    )
    # A small slack above the floor: the floor is a bound on an attainable
    # point, and pinning the RHS exactly to it would leave a feasible set of
    # measure ~zero that an interior-point solver reports as infeasible.
    effective_cap = max(float(cap), floor * (1.0 + 1e-6) + 1e-9)
    constraint = cp.norm(w - w_prev, 1) / 2 <= effective_cap
    constraints.append(constraint)
    meta.update({
        "turnover_constraint_applied": True,
        "turnover_constraint_cap": float(effective_cap),
        "turnover_mandatory_floor": float(floor),
        "turnover_mandatory_floor_by_cause": by_cause,
        "turnover_constraint": constraint,
    })
    if effective_cap > float(cap) + 1e-9:
        logger.warning(
            "turnover budget RAISED from the configured %.4f to %.4f: the "
            "eligibility mask and cash-sleeve pin mandate %.4f one-way "
            "turnover on their own. The budget governs discretionary trading; "
            "a forced exit is not discretionary and must not be starved of "
            "budget.",
            float(cap), effective_cap, floor,
        )
    return meta


def _turnover_diagnostics(
    weights: np.ndarray, w_prev: np.ndarray, turnover_meta: dict
) -> dict:
    """Turnover observability, emitted on EVERY solve including the fallback
    and the budget-disabled path — a field that appears only on the
    interesting path is indistinguishable from a dead emitter.

    ``requested_turnover_one_way`` deliberately equals the EXECUTED one-way
    turnover now. Under the old post-solve shrink the two differed, and the
    gap was the only published evidence that the optimizer wanted to move
    more than it was allowed to. That evidence has not been dropped, it has
    moved to a better instrument: ``turnover_constraint_binding`` says the
    budget bound the solve, and ``turnover_constraint_shadow_price`` is the
    constraint's dual — the marginal objective value of one more unit of
    turnover budget, i.e. exactly what the restraint cost.
    """
    executed = float(np.sum(np.abs(weights - w_prev)) / 2)
    cap = turnover_meta.get("turnover_constraint_cap")
    out: dict = {
        "requested_turnover_one_way": executed,
        "turnover_constraint_applied": bool(
            turnover_meta.get("turnover_constraint_applied")
        ),
        "turnover_constraint_cap": cap,
        "turnover_mandatory_floor": turnover_meta.get("turnover_mandatory_floor"),
        # alpha-engine-config-I8753 — the floor split by the constraint that
        # forced it. Emitted on every solve, healthy included: "how much of
        # today's trading was forced, and by what" was previously answerable
        # only by reading nine artifacts and diffing stance_caps by hand.
        "turnover_mandatory_floor_by_cause": turnover_meta.get(
            "turnover_mandatory_floor_by_cause"
        ),
        "turnover_constraint_binding": False,
        "turnover_constraint_shadow_price": None,
    }
    if cap is not None:
        constraint = turnover_meta.get("turnover_constraint")
        dual = getattr(constraint, "dual_value", None) if constraint is not None else None
        if dual is not None:
            try:
                out["turnover_constraint_shadow_price"] = float(np.ravel(dual)[0])
            except (TypeError, ValueError, IndexError):
                out["turnover_constraint_shadow_price"] = None
        # The DUAL is the primary binding test, not a comparison of the final
        # turnover against the cap. `weights` here is post-clip-and-renormalize,
        # so its turnover is a few bp off the solver's own — measured 2026-08-14
        # on the live book: solver at the 0.2000 cap, post-clip 0.19978, which a
        # numeric ">= cap − ε" test reads as NOT binding while the dual is
        # 0.00926. A strictly positive dual is the definition of an active
        # constraint; the numeric test is kept only as the fallback for solvers
        # or statuses that return no dual.
        shadow = out["turnover_constraint_shadow_price"]
        if shadow is not None:
            out["turnover_constraint_binding"] = bool(shadow > 1e-9)
        else:
            out["turnover_constraint_binding"] = bool(
                executed >= float(cap) * (1.0 - 1e-3) - 1e-6
            )
    # ``turnover_capped`` keeps its consumer-facing meaning — "the daily
    # turnover budget bound this solve" — which is now the binding test
    # rather than "a post-hoc shrink was applied".
    out["turnover_capped"] = out["turnover_constraint_binding"]
    return out


class TurnoverBudgetError(RuntimeError):
    """The solved weight vector exceeds the daily turnover budget.

    Raised, not silently corrected. Under the constraint construction
    (alpha-engine-config-I7346) the budget is enforced INSIDE the convex
    program, so a vector that violates it means the solver returned a point
    that does not satisfy a constraint it was given, or the post-solve
    clip/renormalize moved it further than that step can account for. Both
    are bugs in this module, not market conditions.

    RAISE rather than degrade, deliberately. ``run_shadow_optimizer`` catches
    it, writes the ``shadow_status: "failed"`` sentinel, and
    ``optimizer_cutover.is_log_usable`` then returns False, so the planner
    falls back to an EMPTY order book: the book is HELD for the session. The
    alternative — shrinking the vector back under the budget — is precisely
    the mechanism this change removes, and it would restore the failure it
    fixed while now also hiding a solver bug behind a plausible-looking
    order book. Holding a book for one session is recoverable and loud;
    trading a book the optimizer did not sanction is neither.
    """


def _apply_turnover_governor(
    weights: np.ndarray,
    w_prev: np.ndarray,
    cfg: dict,
    *,
    turnover_meta: dict | None = None,
    clip_mass_zeroed: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Post-solve ASSERTION that the daily turnover budget held.

    Historically this function ENFORCED the budget, by scaling the whole step
    ``w_prev → weights`` uniformly by ``cap / requested``. That is no longer
    its job: the budget is a constraint inside the convex program
    (``_apply_turnover_constraint``), so the solver returns a vector that
    already satisfies it. This function's role is inverted — it measures and
    raises, and NEVER modifies ``weights``.

    The tolerance is derived, not chosen. ``_clip_and_renormalize`` runs
    between the solve and this check and may legitimately add turnover by
    dropping sub-``min_position_pct`` dust and renormalizing the survivors;
    that step's whole footprint is bounded by ``2 · clip_mass_zeroed`` of L1,
    i.e. ``clip_mass_zeroed`` of one-way turnover. Anything beyond that plus
    solver numerical slack is unexplained, and unexplained is what this
    raises on.

    ``max_daily_turnover=None`` disables the budget; the function then only
    reports, exactly as before.
    """
    meta = turnover_meta or {}
    requested = float(np.sum(np.abs(weights - w_prev)) / 2)
    flag = cfg.get("large_move_turnover_flag")
    gov = _turnover_diagnostics(weights, w_prev, meta)
    binding = bool(gov["turnover_constraint_binding"])
    above_flag = bool(flag is not None and requested > flag)
    gov["large_move_flagged"] = bool(above_flag or binding)
    # Why it was flagged, so the operator alert can say the true thing. Under
    # the constraint construction the executed turnover can no longer exceed
    # the flag on a capped day, so a flag driven ONLY by the raw comparison
    # would go permanently silent — a detector killed by a fix is a worse
    # outcome than the fix is good. `binding` is the honest successor signal:
    # the optimizer wanted to move more than the budget allowed.
    if above_flag:
        gov["large_move_reason"] = "executed_turnover_above_flag"
    elif binding:
        gov["large_move_reason"] = "turnover_budget_binding"
    else:
        gov["large_move_reason"] = None

    cap = meta.get("turnover_constraint_cap")
    if cap is not None:
        tolerance = float(clip_mass_zeroed) + 1e-6
        if requested > float(cap) + tolerance:
            raise TurnoverBudgetError(
                f"solved one-way turnover {requested:.6f} exceeds the daily "
                f"budget {float(cap):.6f} by more than the post-solve clip can "
                f"account for (clip zeroed {float(clip_mass_zeroed):.6f} of "
                f"weight, tolerance {tolerance:.6f}). The budget is a "
                f"constraint inside the convex program, so this is a solver or "
                f"config defect, not a market condition. Holding the book for "
                f"this session rather than trading an unsanctioned vector "
                f"(alpha-engine-config-I7346)."
            )
    return weights, gov


def _build_diagnostics(
    weights: np.ndarray,
    w_prev: np.ndarray,
    sigma: np.ndarray,
    alpha_hat: np.ndarray,
    spy_idx: int,
    status: str,
    cfg: dict,
    *,
    omega_diag: np.ndarray | None = None,
    alpha_unc_used: bool = False,
) -> dict:
    # sigma is at horizon H per _estimate_covariance. Annualize:
    # Var_ann = Var_H · (252/H) → vol_ann = √(252/H · Var_H). At default
    # H=1 this is the legacy √(252 · daily_var).
    horizon = int(cfg.get("sigma_horizon_days", 1))
    horizon_var = float(weights @ sigma @ weights)
    horizon_var = max(horizon_var, 0.0)
    vol_ann = float(np.sqrt((252 / horizon) * horizon_var))
    spy_only = np.zeros_like(weights)
    spy_only[spy_idx] = 1.0 - cfg["cash_sleeve_pct"]
    active_share = float(np.sum(np.abs(weights - spy_only)) / 2)
    n_active = int(np.sum(weights > cfg["min_position_pct"]))
    turnover = float(np.sum(np.abs(weights - w_prev)) / 2)
    out = {
        "status": status,
        "portfolio_vol_ann": vol_ann,
        "active_share_vs_spy": active_share,
        "n_active_positions": n_active,
        "turnover_one_way": turnover,
        "expected_alpha": float(weights @ alpha_hat),
        "weight_sum": float(weights.sum()),
        "alpha_uncertainty_penalty_used": alpha_unc_used,
    }
    # α̂-uncertainty observability (workstream B.3). Mean σ_α̂ across the
    # active book (omega_diag = σ²) — operator-readable signal for how
    # confident the predictor is on the names being sized today.
    if omega_diag is not None and np.any(omega_diag > 0.0):
        active_mask = weights > cfg["min_position_pct"]
        active_omega = omega_diag[active_mask]
        if active_omega.size > 0:
            out["mean_alpha_std_active"] = float(np.sqrt(active_omega.mean()))
            out["alpha_uncertainty_penalty_contribution"] = float(
                cfg.get("alpha_uncertainty_penalty", 0.0) * (omega_diag @ (weights ** 2))
            )
    return out


def make_cash_sentinel_returns(n_rows: int) -> np.ndarray:
    """Helper for callers: cash has zero return (treated as risk-free at sleeve)."""
    return np.zeros(n_rows)
