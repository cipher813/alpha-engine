"""Per-run visibility into feature-flagged risk-config safety gates.

Motivation (alpha-engine-config-I9021): ``_apply_batch_confidence_tightening``
(``executor.deciders``) was written 2026-05-07 as a backstop for degenerate
predictor batches, shipped feature-flagged off, and never turned on — the
live ``config/risk.yaml`` had no ``batch_confidence_tightening_enabled`` key
at all, so the flag silently defaulted ``False`` for the entire time the
condition it guards against was live (batch-mean ``prediction_confidence``
0.11/0.097/0.098/0.092/0.094 across 2026-08-24..28, roughly 3x below the
0.30 trigger every single day). A feature flag that has never once fired is
indistinguishable from a flag that doesn't exist; the yaml is not read by
anything that would have surfaced this.

``SAFETY_FLAGS_DEFAULT_OFF`` is the single source of truth for every
``*_enabled`` gate in ``config/risk.yaml.example`` that ships with a default
of ``false`` (swept 2026-08-28 — see the I9021 comment for the full audit).
Each is wired to a real consumer (``config.get(<name>, False)``), so
"absent from the live yaml" and "explicitly false" are behaviorally
identical — both leave the gate OFF. This module makes that state visible
per run rather than requiring someone to diff two yaml files.

Deliberately excludes ``momentum_gate_enabled`` — it appears twice in
``risk.yaml.example`` (line 51 ``false``, then redeclared ``true`` at line
60 under the DE-STANCED 2026-06-07 section); YAML's last-key-wins makes the
effective documented default ``true``, so the ``false`` line is a stale
duplicate-key artifact, not a real off-by-default gate. Flagged separately
in the I9021 sweep comment as a documentation defect, not folded into this
list.
"""

from __future__ import annotations

# Name, and the config/risk.yaml.example line it was swept from (2026-08-28).
# Keep in sync with risk.yaml.example — a new "<x>_enabled: false" default
# added there belongs in this list too.
SAFETY_FLAGS_DEFAULT_OFF: tuple[str, ...] = (
    "batch_confidence_tightening_enabled",   # risk.yaml.example:45
    "urgency_weighted_entry_ranking_enabled",  # risk.yaml.example:83
    "regime_sizing_enabled",                 # risk.yaml.example:141
    "barrier_win_prob_sizing_enabled",       # risk.yaml.example:156
    "regime_drawdown_enabled",               # risk.yaml.example:170
    "regime_forced_bear_enabled",            # risk.yaml.example:190
    "drawdown_regime_enabled",               # risk.yaml.example:205
    "regime_min_score_enabled",              # risk.yaml.example:216
    "derisk_on_expectancy_enabled",          # risk.yaml.example:239
)


def list_off_safety_flags(config: dict | None) -> list[str]:
    """Names, from ``SAFETY_FLAGS_DEFAULT_OFF``, that are OFF in ``config``.

    A flag counts as OFF whether it is explicitly ``false`` or simply
    absent — both reach the consumer's ``config.get(name, False)`` the same
    way. Returns a sorted list so run-to-run diffs and JSON output are
    stable. ``config=None`` (a defensive caller) returns every flag as OFF,
    matching the consumer-side default.
    """
    cfg = config or {}
    return sorted(name for name in SAFETY_FLAGS_DEFAULT_OFF if not cfg.get(name, False))
