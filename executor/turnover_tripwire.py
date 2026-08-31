"""Turnover tripwire — ROADMAP L4515 (fast standalone live-system ops fix).

``turnover_one_way`` has been computed in ``portfolio_optimizer`` since the
governor shipped, but never ALARMED. Three extreme-decision incidents (idle
cash after a hard-risk exit 2026-05-29 #229; a ~90%-DOWN-skew prediction day
2026-06-01 #230; an optimizer target silently dropped on a missing price
2026-06-04 #234/#235) were each caught by a point fix after the fact. This is
the GENERAL surface: a band check on the existing executed-turnover metric
that pages when the book churns abnormally, whatever the upstream cause.

Two independent bands, checked daily in the morning planner (via
``run_shadow_optimizer``) and persisted into the shadow artifact:

- **daily** — executed ``turnover_one_way`` above the governor cap × a
  multiple. The governor is supposed to make this impossible; a breach means
  the cap was bypassed/disabled and pages at ERROR.
- **rolling** — the sum of executed turnover over the last N sessions (read
  from the prior ``predictor/optimizer_shadow/{date}.json`` artifacts) above a
  band. Catches churn-by-a-thousand-cuts: every day under the cap but the
  week's cumulative rebalance abnormal — the actual signature of the three
  incidents above. Pages at WARN.

The rolling alert CARRIES ITS DRIVER (alpha-engine-config-I9315). It used to
end "review the optimizer shadow logs for the driver", which asks a human to
perform the diagnosis the detector is already standing on the data to perform —
partial coverage under ``principles.md`` §2.3 and an unactionable surface under
§2.7. ``_attribute`` now splits the window into forced versus discretionary
turnover and names the dominant driver out of a closed set (forced exits,
conviction-gate throttling, predictor conviction collapse, budget saturation,
or an explicit ``unattributed`` when the combination is one the attribution
does not recognise — a new failure mode is reported as new, never rounded to
the nearest known one). The block is written into the shadow artifact on every
run, breach or not, so a condition can be watched building.

Posture (per [[feedback_no_silent_fails]]): the tripwire itself RAISES on a
breach via ``alerts.publish`` (SNS + Telegram, deduped per run_date). It is
secondary observability hung off the planner's primary path, so an internal
failure must never block order planning — but the failure is RECORDED, not
swallowed: WARN log + a status/sentinel block written into the daily shadow
artifact (``turnover_tripwire`` key), so a dead tripwire is itself visible.
"""
from __future__ import annotations

import json
import logging
import math
import re

import boto3

logger = logging.getLogger(__name__)

_SHADOW_PREFIX = "predictor/optimizer_shadow/"
_DATED_KEY_RE = re.compile(r"optimizer_shadow/(\d{4}-\d{2}-\d{2})\.json$")

# Absolute daily band when the governor is OFF (max_daily_turnover: None) —
# with no cap to multiply, a full-book one-way move above this is the same
# "should have been operator-reviewed" event the governor would have capped.
_DAILY_BAND_GOVERNOR_OFF = 0.25

# The `status` vocabulary this block reports (alpha-engine-config-I8752).
# Named rather than inline so a console adapter can enumerate them instead of
# string-matching, and so "which values are NOT ok" is answerable from the
# module rather than by reading the branch that assigns them.
STATUS_OK = "ok"
STATUS_BREACH_DAILY = "breach_daily"
STATUS_BREACH_ROLLING = "breach_rolling"
STATUS_DISABLED = "disabled"
STATUS_NO_METRIC = "no_turnover_metric"
STATUS_ERROR = "error"

TRIPWIRE_STATUSES = (
    STATUS_OK,
    STATUS_BREACH_DAILY,
    STATUS_BREACH_ROLLING,
    STATUS_DISABLED,
    STATUS_NO_METRIC,
    STATUS_ERROR,
)

#: Statuses that mean "the band was breached". A reader asking "is this
#: component healthy?" tests membership here rather than `!= "ok"` — `disabled`
#: and `no_turnover_metric` are not breaches, and lumping them in would page on
#: a deliberately-off tripwire.
TRIPWIRE_BREACH_STATUSES = frozenset({STATUS_BREACH_DAILY, STATUS_BREACH_ROLLING})


def check_turnover_tripwire(
    diagnostics: dict,
    optimizer_cfg: dict,
    signals_bucket: str,
    run_date: str,
    s3_client=None,
) -> dict:
    """Run both bands and alert on breach. Returns the block persisted into
    the shadow artifact. Never raises (see module docstring posture)."""
    try:
        if not optimizer_cfg.get("turnover_tripwire_enabled", True):
            return {"status": STATUS_DISABLED}
        today = (diagnostics or {}).get("turnover_one_way")
        if today is None or not math.isfinite(float(today)):
            # Upstream contract violation — the optimizer always writes this
            # field on a solved run. Surface it, don't quietly skip (the
            # silent-skip is how the 6/04 dropped-target class hid).
            logger.warning(
                "turnover tripwire: diagnostics carry no finite "
                "turnover_one_way (run_date=%s) — tripwire DID NOT RUN",
                run_date,
            )
            return {"status": STATUS_NO_METRIC}
        today = float(today)

        cap = optimizer_cfg.get("max_daily_turnover")
        multiple = float(optimizer_cfg.get("turnover_tripwire_daily_multiple", 1.25))
        daily_band = float(cap) * multiple if cap else _DAILY_BAND_GOVERNOR_OFF
        rolling_days = int(optimizer_cfg.get("turnover_tripwire_rolling_days", 5))
        rolling_band = float(
            optimizer_cfg.get("turnover_tripwire_rolling_sum_band", 0.60)
        )

        prior = _read_prior_turnovers(
            signals_bucket, run_date, rolling_days - 1, s3_client
        )
        window = [_driver_row(run_date, diagnostics or {})] + prior
        rolling_sum = float(sum(r["turnover_one_way"] for r in window))

        daily_breach = today > daily_band
        rolling_breach = rolling_sum > rolling_band
        # alpha-engine-config-I8752: DERIVED, never a literal.
        #
        # This field was a hardcoded "ok" that nothing revised. The breach
        # booleans were recorded beside it and drove `alerts.publish`, but
        # `status` — the one field a console pane or a sweep reads as this
        # component's verdict — stayed "ok" through a breach. Measured
        # 2026-08-27 on the live shadow artifacts: eight CONSECUTIVE sessions
        # (2026-08-18 through 2026-08-27) carried rolling_breach=true with
        # rolling_sum 0.69..0.95 against a 0.60 band, and every one of them
        # said status "ok". Brian discovered the churn himself, which is the
        # signal that the WARN publish alone is not a sufficient surface.
        #
        # The module docstring's posture — "a dead tripwire is itself visible"
        # via this block — was inverted here: a BREACHING tripwire was invisible
        # on the same field.
        #
        # Daily outranks rolling: a daily breach means the governor was bypassed
        # and pages at ERROR, which is strictly the more urgent finding.
        if daily_breach:
            status = STATUS_BREACH_DAILY
        elif rolling_breach:
            status = STATUS_BREACH_ROLLING
        else:
            status = STATUS_OK
        # Attribution runs on EVERY invocation, breach or not. A driver block
        # that exists only on the alerting path cannot be used to watch a
        # condition BUILD, and a slow accumulation each single day looks fine
        # against is exactly what this tripwire generalizes
        # (alpha-engine-config-I9315).
        attribution = _attribute(window)
        out = {
            "status": status,
            "attribution": attribution,
            "turnover_one_way": round(today, 6),
            "daily_band": round(daily_band, 6),
            "daily_breach": daily_breach,
            "rolling_days": rolling_days,
            "n_days_used": len(window),
            "rolling_sum": round(rolling_sum, 6),
            "rolling_band": round(rolling_band, 6),
            "rolling_breach": rolling_breach,
        }
        if daily_breach:
            _publish(
                out,
                severity="ERROR",
                dedup_key=f"turnover_tripwire_daily_{run_date}",
                message=(
                    f"[executor] TURNOVER TRIPWIRE (daily): executed one-way "
                    f"turnover {today:.1%} exceeds the {daily_band:.1%} band "
                    f"(governor cap {cap if cap is None else format(cap, '.0%')}, "
                    f"run_date={run_date}). The governor should make this "
                    f"impossible — investigate before the next session."
                ),
            )
            logger.warning(
                "TURNOVER TRIPWIRE daily breach: %.1f%% > %.1f%% (run_date=%s)",
                today * 100, daily_band * 100, run_date,
            )
        if rolling_breach:
            _publish(
                out,
                severity="WARN",
                dedup_key=f"turnover_tripwire_rolling_{run_date}",
                message=(
                    f"[executor] TURNOVER TRIPWIRE (rolling): one-way turnover "
                    f"summed {rolling_sum:.1%} over the last {len(window)} "
                    f"session(s) — above the {rolling_band:.0%}/{rolling_days}d "
                    f"band (run_date={run_date}). The book is churning "
                    f"abnormally even though each day is under the cap. "
                    f"DRIVER ({attribution['driver']}): {attribution['detail']}"
                    f" Per-session one-way turnover: "
                    + ", ".join(
                        f"{r['date']} {r['turnover']:.1%}"
                        for r in attribution["per_day"]
                    )
                    + "."
                ),
            )
            logger.warning(
                "TURNOVER TRIPWIRE rolling breach: sum %.1f%% over %d sessions "
                "> %.1f%% (run_date=%s) — driver=%s: %s",
                rolling_sum * 100, len(window), rolling_band * 100, run_date,
                attribution["driver"], attribution["detail"],
            )
        if not (daily_breach or rolling_breach):
            logger.info(
                "turnover tripwire OK: today=%.1f%% (band %.1f%%), "
                "rolling %.1f%%/%dd (band %.1f%%)",
                today * 100, daily_band * 100, rolling_sum * 100,
                len(window), rolling_band * 100,
            )
        return out
    except Exception as e:  # noqa: BLE001 — secondary observability: must not
        # block the planner; failure recorded in the shadow artifact + WARN.
        logger.warning("turnover tripwire failed (non-blocking): %s", e, exc_info=True)
        return {"status": STATUS_ERROR, "error": repr(e)}


def _read_prior_turnovers(
    bucket: str, run_date: str, n: int, s3_client=None,
) -> list[dict]:
    """Executed ``turnover_one_way`` from the most recent ``n`` dated shadow
    artifacts strictly before ``run_date``. Artifacts that are missing the
    metric (failed/sentinel days) are skipped with a log line — a short window
    still alerts when its partial sum already breaches (sum is monotonic)."""
    if n <= 0:
        return []
    s3 = s3_client or boto3.client("s3")
    dates: list[str] = []
    token = None
    # Hard page cap: the prefix accrues ~1 dated key per session (~250/yr) at
    # 1000 keys/page, so >10 pages is structurally impossible — the cap is a
    # guard against a pathological/non-conforming client looping forever on a
    # truthy IsTruncated (exactly how a MagicMock behaves in tests).
    for _page in range(10):
        kwargs = {"Bucket": bucket, "Prefix": _SHADOW_PREFIX}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            m = _DATED_KEY_RE.search(obj.get("Key", ""))
            if m and m.group(1) < run_date:
                dates.append(m.group(1))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    else:
        logger.warning(
            "turnover tripwire: shadow-prefix listing exceeded the 10-page "
            "cap — rolling window computed from the first %d keys only",
            len(dates),
        )
    out: list[dict] = []
    for d in sorted(dates, reverse=True)[:n]:
        try:
            body = s3.get_object(Bucket=bucket, Key=f"{_SHADOW_PREFIX}{d}.json")
            log_d = json.loads(body["Body"].read())
            dg = log_d.get("diagnostics") or {}
            v = dg.get("turnover_one_way")
            if v is not None and math.isfinite(float(v)):
                out.append(_driver_row(d, dg))
            else:
                logger.info(
                    "turnover tripwire: %s shadow log has no turnover "
                    "(sentinel/failed day) — excluded from rolling window", d,
                )
        except Exception as e:  # noqa: BLE001 — one unreadable day must not
            # kill the window; the exclusion is logged and n_days_used shows it.
            logger.warning(
                "turnover tripwire: could not read shadow log for %s: %s", d, e,
            )
    return out



def _driver_row(date: str, diagnostics: dict) -> dict:
    """One session's turnover reduced to the facts that EXPLAIN it.

    The rolling band answers "is the book churning". Until
    alpha-engine-config-I9315 the alert stopped there and told the operator to
    "review the optimizer shadow logs for the driver" — which is the diagnosis
    the detector is standing on top of the data to perform. These fields are
    already in every shadow artifact; reading them costs nothing extra (the
    artifact is fetched either way) and turns the alert from a question into an
    answer.
    """
    def _f(key):
        v = diagnostics.get(key)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        return fv if math.isfinite(fv) else None

    return {
        "date": date,
        "turnover_one_way": _f("turnover_one_way") or 0.0,
        "mandatory_floor": _f("turnover_mandatory_floor"),
        "budget_configured": _f("turnover_budget_configured"),
        "budget_discretionary": _f("turnover_budget_discretionary"),
        "conviction_ir_xs": _f("conviction_ir_xs"),
        "conviction_multiplier": _f("conviction_budget_multiplier"),
        "conviction_gate_applied": bool(diagnostics.get("conviction_gate_applied")),
        "conviction_gate_reason": diagnostics.get("conviction_gate_reason"),
        "binding": bool(diagnostics.get("turnover_constraint_binding")),
    }


def _attribute(window: list[dict]) -> dict:
    """Split the rolling window's turnover into FORCED and DISCRETIONARY, and
    name the dominant driver. Returns the block persisted in the artifact."""
    total = sum(r["turnover_one_way"] for r in window)
    forced = sum(min(r["mandatory_floor"] or 0.0, r["turnover_one_way"]) for r in window)
    discretionary = max(total - forced, 0.0)
    n_binding = sum(1 for r in window if r["binding"])
    n_gated = sum(1 for r in window if r["conviction_gate_applied"])
    irs = [r["conviction_ir_xs"] for r in window if r["conviction_ir_xs"] is not None]
    reasons = {r["conviction_gate_reason"] for r in window if r["conviction_gate_reason"]}
    median_ir = sorted(irs)[len(irs) // 2] if irs else None

    # Ordered most-specific first: the first predicate that holds IS the driver.
    if forced > 0.6 * total and total > 0:
        driver = "forced_exits"
        detail = (
            f"{forced:.1%} of the {total:.1%} was MANDATORY turnover — forced "
            f"exits, ineligibility pins or the cash sleeve. The optimizer is "
            f"not choosing to churn; hard risk rules are ejecting positions. "
            f"Look at turnover_mandatory_floor_by_cause in the shadow "
            f"artifacts, not at the alpha signal."
        )
    elif n_gated >= max(1, len(window) // 2):
        driver = "conviction_gate_throttling"
        detail = (
            f"the conviction gate throttled the discretionary budget on "
            f"{n_gated} of {len(window)} sessions (median cross-sectional "
            f"IR {median_ir:.3f}), so this turnover is what SURVIVED the "
            f"throttle. If the band is still breached with the gate engaged, "
            f"the residue is mandatory turnover or the gate floor is too high."
        )
    elif median_ir is not None and median_ir < 0.5:
        driver = "predictor_conviction_collapse"
        detail = (
            f"the predictor's cross-sectional alpha spread is only "
            f"{median_ir:.3f}x its own published per-name sigma_alpha (median "
            f"over {len(irs)} sessions). The optimizer is ranking names that "
            f"are not statistically distinguishable from one another, so the "
            f"target reshuffles daily on noise. This is an UPSTREAM condition: "
            f"the executor is behaving correctly on a signal that carries no "
            f"cross-sectional information. Check the predictor's "
            f"high-confidence count and the champion model's recent promotion."
        )
    elif n_binding >= max(1, len(window) // 2):
        driver = "budget_saturation"
        detail = (
            f"the daily budget bound the solve on {n_binding} of "
            f"{len(window)} sessions with signal quality PASSING the "
            f"conviction gate — the optimizer genuinely wants a book this far "
            f"from the current one and is walking there a capped step at a "
            f"time. Sustained saturation means the target is moving at least "
            f"as fast as the budget allows the book to travel, so the walk "
            f"never converges. Either the reallocation is real and the budget "
            f"should be spent deliberately, or the target is unstable."
        )
    else:
        driver = "unattributed"
        detail = (
            f"no single driver dominates: {forced:.1%} forced of {total:.1%} "
            f"total, budget binding on {n_binding}/{len(window)} sessions, "
            f"conviction gate engaged on {n_gated}. This combination is not "
            f"one the attribution knows; treat it as a new failure mode rather "
            f"than a known one."
        )
    return {
        "driver": driver,
        "detail": detail,
        "forced_sum": round(forced, 6),
        "discretionary_sum": round(discretionary, 6),
        "n_binding": n_binding,
        "n_conviction_gated": n_gated,
        "median_conviction_ir": median_ir,
        "gate_reasons": sorted(reasons),
        "per_day": [
            {
                "date": r["date"],
                "turnover": round(r["turnover_one_way"], 4),
                "forced": None if r["mandatory_floor"] is None
                else round(r["mandatory_floor"], 4),
                "ir": None if r["conviction_ir_xs"] is None
                else round(r["conviction_ir_xs"], 4),
            }
            for r in window
        ],
    }


def _publish(out: dict, *, severity: str, dedup_key: str, message: str) -> None:
    """Best-effort alert publish — mirrors the large-move flag posture: the
    band verdict is already recorded (shadow artifact + WARN log), so a
    publish failure must never block the planner; it is recorded in the
    artifact's ``publish_error`` field."""
    try:
        from executor.notifier import publish_ops_alert

        publish_ops_alert(
            message=message,
            severity=severity,
            source="alpha-engine/executor/turnover_tripwire.py",
            dedup_key=dedup_key,
        )
    except Exception as e:  # noqa: BLE001 — secondary observability
        logger.warning("turnover tripwire alert publish failed (non-fatal): %s", e)
        out["publish_error"] = repr(e)
