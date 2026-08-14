"""One alpha scale per solve — the optimizer's alpha input contract
(alpha-engine-config-I7337, layer 3).

## What went wrong

``executor/optimizer_shadow.py::_build_alpha_hat`` assembles ONE
``alpha_hat`` vector from ``predictions_by_ticker``, and the mean-variance
solve compares every entry of that vector against a **SPY = 0.0 anchor**.
That anchor is only meaningful if every entry is *market-relative* — i.e.
the cross-sectional mean of the name alphas is ≈ 0.

Two producers were writing a field called ``predicted_alpha`` into that one
dict, on two different anchors, and nothing checked:

* **The predictor** (``crucible-predictor/inference/stages/run_inference.py``
  → ``inference/level_neutralization.py::apply_cross_sectional_neutralization``)
  subtracts the batch's cross-sectional mean, so its ``predicted_alpha`` IS
  market-relative. Measured 2026-08-14 on ``predictor/predictions/latest.json``:
  n=23, mean **4.3e-07**, range **-0.0400 .. +0.0948**, 10/23 positive.
* **The champion arm** ``scanner_predictor_direct``
  (``executor/champion.py``) injects the raw
  ``MetaModel.predict_single`` output carried in
  ``predictor/research_free_backfill/predictor_outcomes_research_free.parquet``
  (written by ``crucible-backtester/analysis/
  scanner_predictor_research_free_backfill.py``, ``alpha = float(
  mm.predict_single(feats))``). That output is **never level-neutralized**,
  so it carries the meta-L2's common-mode macro level. Measured 2026-08-14
  on the 2026-08-13 cohort: n=72, mean **-0.2882**, range
  **-0.3184 .. -0.2094**, **0 of 72 positive**, cross-sectional std 0.0177.

A champion-injected name therefore sat ~29 points of 21-day log alpha below
SPY's 0.0 anchor purely as an artifact of where the level was measured from.
It could never win a solve — every one landed at weight 0.000, rendered in
the order book as ``optimizer_target_zero`` / ``optimizer_scale_down``.

This exact failure was predicted, in writing, by the producer-side module
that fixed the predictor's half of it — see the "Downstream" list at the top
of ``crucible-predictor/inference/level_neutralization.py``:

    the portfolio optimizer compares each name's alpha against the SPY=0
    anchor (executor/optimizer_shadow.py::_build_alpha_hat), so a
    common-mode -X% makes every name lose to SPY -> mass
    optimizer_target_zero flush

The prediction was correct and the guard was a docstring. This module is
that guard, executable.

## The contract

Every numeric ``predicted_alpha`` reaching ``_build_alpha_hat`` MUST declare
its anchor in the ``alpha_anchor`` field, and the ONLY anchor the optimizer
can consume is :data:`OPTIMIZER_ALPHA_ANCHOR`. A batch carrying a second
anchor, or carrying a numeric alpha with no anchor at all, raises
:class:`AlphaAnchorError` naming the offending tickers and their anchors.

Three deliberate properties:

1. **The optimizer never converts.** A common-mode level can only be
   measured over a complete cross-section, and by ``_build_alpha_hat`` time
   the dict is a MIXTURE of two batches (the predictor's and the champion's)
   — the one place in the system where the correction is not computable.
   Normalising happens at each producing adapter, which holds its own whole
   batch; the optimizer only ASSERTS. An input contract that accepts two
   incompatible scales and quietly resolves them is how this shipped.
2. **Declared, not inferred.** The anchor is stamped from a fact the
   producing artifact states about itself — for the predictor, its own
   ``level_neutralization.applied`` block — never guessed from the field
   name or from the sign of the numbers.
3. **Centering only, never rescaling.** The correction is a subtraction of
   the cross-sectional mean. The research-free arm's dispersion (std 0.0177)
   is genuinely narrower than the predictor's because it zeroes four
   research meta-features; scaling its spread up to match would fabricate
   conviction the arm does not have. Mean-variance handles a
   lower-dispersion signal correctly on its own — smaller tilts.

Centering is rank-preserving (a constant shift), so an arm's SELECTED
candidate set is byte-identical before and after. This is a units fix, not
an arm swap: no ``policy-champion-challenger`` promotion, demotion or
retirement is implied by it, and the arm's identity is unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# The one anchor the portfolio optimizer can consume: 21-day log-domain
# alpha measured relative to the market, so that the solve's SPY = 0.0
# sentinel is the true neutral. Named rather than described — the fleet's
# units-in-the-name convention (see ~/Development/AGENTS.md, feature-store
# column rule) applied to a JSON payload field, because a units contract
# enforced only by a docstring has already cost this fleet a silent
# months-long failure twice.
OPTIMIZER_ALPHA_ANCHOR = "market_relative_21d_log"

# The same quantity BEFORE cross-sectional level-neutralization: it still
# carries the meta-L2's common-mode macro level, so it is NOT comparable to
# a SPY = 0.0 anchor. Legal to transport and to record; illegal to solve on.
RAW_ALPHA_ANCHOR = "raw_21d_log"

# Field names on a prediction record.
ANCHOR_FIELD = "alpha_anchor"
ANCHOR_SOURCE_FIELD = "alpha_anchor_source"


class AlphaAnchorError(RuntimeError):
    """Raised when the optimizer's alpha vector is not on one declared,
    market-relative anchor.

    Deliberately fatal rather than degrading: the degraded outcome is a
    silent mass flush of the book to zero weights, which reads in the order
    book as a normal optimizer decision. A halt is visible; that is not.
    """


def center_to_market_relative(values: Sequence[float]) -> tuple[list[float], float]:
    """Subtract the cross-sectional mean, returning ``(centered, mean_removed)``.

    This is the executor-side mirror of ``crucible-predictor``'s
    ``inference/level_neutralization.py::apply_cross_sectional_neutralization``
    — same transform, same justification, applied to a batch that repo never
    sees. Kept as a small local function rather than a cross-repo import:
    ``crucible-executor`` has no package dependency on ``crucible-predictor``
    and a one-line demean does not justify creating one.

    Raises on a cross-section too small to define a market level: centering a
    single name against its own mean zeroes its alpha, manufacturing a
    "neutral" opinion out of nothing. Two names is the minimum at which a
    common mode is a measurable quantity rather than an assumption.
    """
    vals = [float(v) for v in values]
    if len(vals) < 2:
        raise AlphaAnchorError(
            f"Cannot center a cross-section of {len(vals)} name(s) to a market-"
            "relative anchor: the cross-sectional mean of a single name is that "
            "name, so centering would zero its alpha and assert a neutral "
            "opinion the model never expressed. Need >= 2 names."
        )
    if not all(math.isfinite(v) for v in vals):
        bad = [i for i, v in enumerate(vals) if not math.isfinite(v)]
        raise AlphaAnchorError(
            f"Cannot center: {len(bad)} non-finite alpha value(s) at index(es) "
            f"{bad[:10]}. A NaN/inf poisons the cross-sectional mean and would "
            "silently anchor every other name to NaN."
        )
    mean_removed = sum(vals) / len(vals)
    return [v - mean_removed for v in vals], mean_removed


def _numeric_alpha(pred: dict) -> float | None:
    """The record's numeric ``predicted_alpha``, or None when it carries no
    numeric alpha opinion at all.

    ``None`` is a legitimate, honest value here — the ``thinktank_coverage``
    arm injects it deliberately, because a subjective 0-100 analyst rating is
    not a log-alpha estimate and fabricating one would misrepresent it. Such a
    record contributes 0.0 to ``alpha_hat`` (identical to SPY) and carries no
    anchor obligation, because it asserts no level.

    An exact 0.0 IS a numeric opinion and is returned as such — it is not
    "missing". (Treating it as missing is the falsy-``or`` bug this function
    replaces at the ``_build_alpha_hat`` call site.)
    """
    raw = pred.get("predicted_alpha")
    if raw is None:
        raw = pred.get("canonical_predicted_alpha")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def assert_optimizer_anchor(
    tickers: Sequence[str],
    predictions_by_ticker: dict[str, dict],
    *,
    spy_idx: int,
    cash_idx: int,
) -> dict:
    """Assert every numeric alpha in the solve's universe is on
    :data:`OPTIMIZER_ALPHA_ANCHOR`; raise :class:`AlphaAnchorError` otherwise.

    Returns an observability block recorded on the optimizer shadow log, so
    the check's result is a published number rather than an inference from
    the absence of an exception. It is emitted on EVERY run, healthy
    included: a field that appears only on the bad path is indistinguishable
    from a dead emitter.

    Skipped, with reason:

    * ``spy_idx`` / ``cash_idx`` — solver sentinels, not predictions. SPY is
      the anchor itself (0.0 by definition) and CASH carries a config hint.
    * a ticker with no prediction record, or a record whose
      ``predicted_alpha`` is ``None``/non-numeric — it asserts no level, and
      contributes exactly 0.0. See :func:`_numeric_alpha`.
    """
    anchors: dict[str, int] = {}
    undeclared: list[str] = []
    wrong_anchor: list[tuple[str, str]] = []
    n_checked = 0

    for i, ticker in enumerate(tickers):
        if i in (spy_idx, cash_idx):
            continue
        pred = predictions_by_ticker.get(ticker) or {}
        if _numeric_alpha(pred) is None:
            continue
        n_checked += 1
        anchor = pred.get(ANCHOR_FIELD)
        if anchor is None:
            undeclared.append(ticker)
            continue
        anchor = str(anchor)
        anchors[anchor] = anchors.get(anchor, 0) + 1
        if anchor != OPTIMIZER_ALPHA_ANCHOR:
            wrong_anchor.append((ticker, anchor))

    if undeclared:
        raise AlphaAnchorError(
            f"{len(undeclared)} name(s) carry a numeric predicted_alpha with no "
            f"{ANCHOR_FIELD!r} declaration: {sorted(undeclared)[:15]}"
            f"{' ...' if len(undeclared) > 15 else ''}. The optimizer compares "
            f"every alpha against a SPY=0.0 anchor, so an undeclared alpha is an "
            f"unknown-anchor alpha — it may be market-relative or it may carry a "
            f"common-mode level, and the two are indistinguishable by value. "
            f"Stamp {ANCHOR_FIELD}={OPTIMIZER_ALPHA_ANCHOR!r} at the producing "
            f"adapter (alpha-engine-config-I7337)."
        )

    if wrong_anchor:
        detail = ", ".join(f"{t}={a}" for t, a in sorted(wrong_anchor)[:15])
        raise AlphaAnchorError(
            f"{len(wrong_anchor)} of {n_checked} name(s) declare an alpha anchor "
            f"the optimizer cannot solve on: {detail}"
            f"{' ...' if len(wrong_anchor) > 15 else ''}. Distinct anchors present "
            f"in this batch: {sorted(anchors)}. Only "
            f"{OPTIMIZER_ALPHA_ANCHOR!r} is comparable to the solve's SPY=0.0 "
            f"anchor; a {RAW_ALPHA_ANCHOR!r} alpha still carries the meta-L2's "
            f"common-mode macro level, which makes every name lose to SPY and "
            f"flushes the book to zero weights. Refusing to solve a mixed-anchor "
            f"alpha vector (alpha-engine-config-I7337)."
        )

    return {
        "expected_anchor": OPTIMIZER_ALPHA_ANCHOR,
        "n_checked": n_checked,
        "anchors": anchors,
        "single_anchor": len(anchors) <= 1,
    }
