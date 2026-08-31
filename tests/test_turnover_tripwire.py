"""L4515 — turnover tripwire (daily + rolling band on executed turnover).

Pins: no-breach quiet path; daily breach pages at ERROR with the run-date
dedup key; rolling breach sums the prior shadow artifacts and pages at WARN;
sentinel/unreadable prior days are excluded (not fabricated); a partial
window still alerts when its sum already breaches; the disabled flag and the
missing-metric upstream-contract case; publish failure is recorded in the
artifact block, never raised into the planner.
"""
from __future__ import annotations

import io
import json

import pytest

from executor import turnover_tripwire as tw


class _FakeS3:
    """list_objects_v2 + get_object over a dict of {key: dict-payload}."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    def list_objects_v2(self, Bucket, Prefix, **kwargs):  # noqa: N803
        return {
            "Contents": [{"Key": k} for k in self.objects if k.startswith(Prefix)],
            "IsTruncated": False,
        }

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": io.BytesIO(json.dumps(self.objects[Key]).encode())}


def _shadow(date, turnover):
    diag = {} if turnover is None else {"turnover_one_way": turnover}
    return {"run_date": date, "diagnostics": diag}


_CFG = {
    "max_daily_turnover": 0.20,
    "turnover_tripwire_enabled": True,
    "turnover_tripwire_daily_multiple": 1.25,
    "turnover_tripwire_rolling_days": 5,
    "turnover_tripwire_rolling_sum_band": 0.60,
}


@pytest.fixture
def published(monkeypatch):
    calls = []
    from nousergon_lib import alerts

    monkeypatch.setattr(alerts, "publish", lambda **kw: calls.append(kw))
    return calls


def test_quiet_day_no_alert(published):
    s3 = _FakeS3({
        f"{tw._SHADOW_PREFIX}2026-06-09.json": _shadow("2026-06-09", 0.05),
    })
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.05}, _CFG, "bkt", "2026-06-10", s3)
    assert out["status"] == "ok"
    assert out["daily_breach"] is False and out["rolling_breach"] is False
    assert out["rolling_sum"] == pytest.approx(0.10)
    assert out["n_days_used"] == 2
    assert published == []


def test_daily_breach_pages_error(published):
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.30}, _CFG, "bkt", "2026-06-10", _FakeS3())
    assert out["daily_breach"] is True            # 0.30 > 0.20 × 1.25
    severities = [c["severity"] for c in published]
    assert "ERROR" in severities
    daily = next(c for c in published if c["severity"] == "ERROR")
    assert daily["dedup_key"] == "turnover_tripwire_daily_2026-06-10"
    assert daily["sns"] is True and daily["telegram"] is False


def test_rolling_breach_sums_prior_days_and_pages_warn(published):
    s3 = _FakeS3({
        f"{tw._SHADOW_PREFIX}2026-06-{d:02d}.json": _shadow(f"2026-06-{d:02d}", t)
        for d, t in [(4, 0.15), (5, 0.15), (8, 0.15), (9, 0.15)]
    })
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.10}, _CFG, "bkt", "2026-06-10", s3)
    assert out["daily_breach"] is False           # every day under the cap…
    assert out["rolling_breach"] is True          # …but the week churned 70%
    assert out["rolling_sum"] == pytest.approx(0.70)
    assert [c["severity"] for c in published] == ["WARN"]
    assert published[0]["dedup_key"] == "turnover_tripwire_rolling_2026-06-10"


def test_window_takes_newest_n_and_ignores_future_and_latest(published):
    objs = {
        f"{tw._SHADOW_PREFIX}latest.json": _shadow("x", 9.9),       # not dated
        f"{tw._SHADOW_PREFIX}2026-06-11.json": _shadow("f", 9.9),   # future
    }
    for d in range(2, 10):  # 06-02..06-09 all small
        objs[f"{tw._SHADOW_PREFIX}2026-06-{d:02d}.json"] = _shadow(f"2026-06-{d:02d}", 0.01)
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.01}, _CFG, "bkt", "2026-06-10", _FakeS3(objs))
    assert out["n_days_used"] == 5                # today + newest 4 dated
    assert out["rolling_breach"] is False
    assert published == []


def test_sentinel_prior_day_excluded_partial_window_still_alerts(published):
    s3 = _FakeS3({
        f"{tw._SHADOW_PREFIX}2026-06-08.json": _shadow("2026-06-08", None),  # sentinel
        f"{tw._SHADOW_PREFIX}2026-06-09.json": _shadow("2026-06-09", 0.45),
    })
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.20}, _CFG, "bkt", "2026-06-10", s3)
    assert out["n_days_used"] == 2                # sentinel day excluded
    assert out["rolling_breach"] is True          # 0.65 > 0.60 on a partial window
    assert [c["severity"] for c in published] == ["WARN"]


def test_disabled_and_missing_metric(published):
    off = tw.check_turnover_tripwire(
        {"turnover_one_way": 9.9}, {**_CFG, "turnover_tripwire_enabled": False},
        "bkt", "2026-06-10", _FakeS3())
    assert off == {"status": "disabled"}
    missing = tw.check_turnover_tripwire({}, _CFG, "bkt", "2026-06-10", _FakeS3())
    assert missing == {"status": "no_turnover_metric"}
    assert published == []


def test_governor_off_uses_absolute_band(published):
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.26}, {**_CFG, "max_daily_turnover": None},
        "bkt", "2026-06-10", _FakeS3())
    assert out["daily_band"] == pytest.approx(tw._DAILY_BAND_GOVERNOR_OFF)
    assert out["daily_breach"] is True


def test_publish_failure_recorded_not_raised(monkeypatch):
    from nousergon_lib import alerts

    def _boom(**kw):
        raise RuntimeError("sns down")

    monkeypatch.setattr(alerts, "publish", _boom)
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.30}, _CFG, "bkt", "2026-06-10", _FakeS3())
    assert out["daily_breach"] is True            # verdict still recorded
    assert "publish_error" in out                 # failure recorded in artifact


def test_internal_error_returns_sentinel(monkeypatch):
    class _ExplodingS3:
        def list_objects_v2(self, **kw):
            raise RuntimeError("s3 down")

    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.05}, _CFG, "bkt", "2026-06-10", _ExplodingS3())
    assert out["status"] == "error"               # recorded in the artifact
    assert "s3 down" in out["error"]


# ── status is DERIVED, not a literal (alpha-engine-config-I8752) ─────────
#
# `status` was a hardcoded "ok" that nothing revised. The breach booleans sat
# beside it and drove `alerts.publish`, but `status` — the one field a console
# pane or a sweep reads as this component's verdict — stayed "ok" through a
# breach. Measured 2026-08-27 on the live shadow artifacts: eight CONSECUTIVE
# sessions (2026-08-18 .. 2026-08-27) carried rolling_breach=true with
# rolling_sum 0.69..0.95 against a 0.60 band, every one reporting status "ok".
#
# The module docstring's posture — "a dead tripwire is itself visible" via this
# block — was inverted: a BREACHING tripwire was invisible on the same field.
#
# Every test below is RED against the pre-change code, which returns the
# literal "ok" on every path that gets as far as building the block.


def test_rolling_breach_sets_a_non_ok_status(published):
    """The live shape: every day under the daily cap, the week's sum over the
    band."""
    s3 = _FakeS3({
        f"{tw._SHADOW_PREFIX}2026-06-{d:02d}.json": _shadow(f"2026-06-{d:02d}", t)
        for d, t in [(4, 0.15), (5, 0.15), (8, 0.15), (9, 0.15)]
    })

    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.10}, _CFG, "bkt", "2026-06-10", s3)

    assert out["rolling_breach"] is True
    assert out["daily_breach"] is False
    assert out["status"] == tw.STATUS_BREACH_ROLLING
    assert out["status"] != tw.STATUS_OK


def test_daily_breach_outranks_rolling_in_the_status(published):
    """A daily breach means the governor was bypassed and pages at ERROR — it
    is strictly the more urgent finding, so it must be what `status` names even
    when the rolling band is also over."""
    s3 = _FakeS3({
        f"{tw._SHADOW_PREFIX}2026-06-{d:02d}.json": _shadow(f"2026-06-{d:02d}", t)
        for d, t in [(4, 0.20), (5, 0.20), (8, 0.20), (9, 0.20)]
    })

    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.30}, _CFG, "bkt", "2026-06-10", s3)

    assert out["daily_breach"] is True and out["rolling_breach"] is True
    assert out["status"] == tw.STATUS_BREACH_DAILY


def test_a_quiet_day_is_still_ok(published):
    s3 = _FakeS3({
        f"{tw._SHADOW_PREFIX}2026-06-09.json": _shadow("2026-06-09", 0.05),
    })

    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.05}, _CFG, "bkt", "2026-06-10", s3)

    assert out["status"] == tw.STATUS_OK
    assert out["status"] not in tw.TRIPWIRE_BREACH_STATUSES


def test_breach_statuses_exclude_disabled_and_missing_metric():
    """A reader asking 'is this healthy?' tests TRIPWIRE_BREACH_STATUSES, not
    `!= "ok"` — a deliberately-disabled tripwire is not a breach, and lumping
    it in would page on an off switch."""
    assert tw.TRIPWIRE_BREACH_STATUSES == {
        tw.STATUS_BREACH_DAILY, tw.STATUS_BREACH_ROLLING,
    }
    assert tw.STATUS_DISABLED not in tw.TRIPWIRE_BREACH_STATUSES
    assert tw.STATUS_NO_METRIC not in tw.TRIPWIRE_BREACH_STATUSES
    assert tw.STATUS_ERROR not in tw.TRIPWIRE_BREACH_STATUSES


def test_every_reachable_status_is_in_the_declared_vocabulary(published):
    """A console adapter enumerates TRIPWIRE_STATUSES rather than
    string-matching; a value it cannot render is a value it renders as
    nothing."""
    disabled = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.05},
        dict(_CFG, turnover_tripwire_enabled=False), "bkt", "2026-06-10", _FakeS3())
    no_metric = tw.check_turnover_tripwire({}, _CFG, "bkt", "2026-06-10", _FakeS3())
    breach = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.30}, _CFG, "bkt", "2026-06-10", _FakeS3())

    for out in (disabled, no_metric, breach):
        assert out["status"] in tw.TRIPWIRE_STATUSES, out


def test_the_live_2026_08_27_artifact_shape_would_not_report_ok(published):
    """Replay of the real rolling sums that reported 'ok' for eight sessions.

    0.170 today plus 0.200 + 0.200 + 0.201 + 0.150 prior = 0.921 against a 0.60
    band, with every single day under the 0.25 daily band.
    """
    s3 = _FakeS3({
        f"{tw._SHADOW_PREFIX}2026-08-{d}.json": _shadow(f"2026-08-{d}", t)
        for d, t in [("21", 0.150), ("24", 0.200), ("25", 0.201), ("26", 0.200)]
    })

    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.170}, _CFG, "bkt", "2026-08-27", s3)

    assert out["rolling_sum"] == pytest.approx(0.921)
    assert out["daily_breach"] is False
    assert out["status"] == tw.STATUS_BREACH_ROLLING


# ── Driver attribution (alpha-engine-config-I9315) ────────────────────────
# The rolling alert used to end "review the optimizer shadow logs for the
# driver" — asking a human to perform the diagnosis the detector is already
# standing on the data to perform. These pin that it now performs it.


def _rich_shadow(date, turnover, *, floor=0.0, ir=None, gated=False, binding=False):
    return {
        "run_date": date,
        "diagnostics": {
            "turnover_one_way": turnover,
            "turnover_mandatory_floor": floor,
            "turnover_budget_configured": 0.20,
            "turnover_budget_discretionary": 0.20 * (0.05 if gated else 1.0),
            "conviction_ir_xs": ir,
            "conviction_budget_multiplier": 0.05 if gated else 1.0,
            "conviction_gate_applied": gated,
            "conviction_gate_reason": (
                "alpha_spread_below_own_noise" if gated else "signal_quality_ok"
            ),
            "turnover_constraint_binding": binding,
        },
    }


def _window(rows):
    return [tw._driver_row(r["run_date"], r["diagnostics"]) for r in rows]


def test_attribution_names_predictor_conviction_collapse():
    # The measured live condition, 2026-08-24..2026-08-31: budget binding
    # daily, gate NOT yet deployed, cross-sectional IR ~0.03.
    a = tw._attribute(_window([
        _rich_shadow(f"2026-08-2{d}", 0.20, ir=0.03, binding=True) for d in range(4, 9)
    ]))
    assert a["driver"] == "predictor_conviction_collapse"
    assert "not statistically distinguishable" in a["detail"]
    assert "UPSTREAM" in a["detail"]
    assert a["median_conviction_ir"] == pytest.approx(0.03)


def test_attribution_names_forced_exits():
    a = tw._attribute(_window([
        _rich_shadow(f"2026-08-2{d}", 0.18, floor=0.17, ir=2.0) for d in range(4, 9)
    ]))
    assert a["driver"] == "forced_exits"
    assert "turnover_mandatory_floor_by_cause" in a["detail"]
    assert a["forced_sum"] == pytest.approx(0.85)


def test_attribution_names_the_gate_when_it_is_throttling():
    a = tw._attribute(_window([
        _rich_shadow(f"2026-08-2{d}", 0.14, ir=0.03, gated=True, binding=True)
        for d in range(4, 9)
    ]))
    assert a["driver"] == "conviction_gate_throttling"
    assert "SURVIVED the throttle" in a["detail"]
    assert a["n_conviction_gated"] == 5


def test_attribution_names_budget_saturation_on_a_signal_that_passes():
    a = tw._attribute(_window([
        _rich_shadow(f"2026-08-2{d}", 0.20, ir=3.0, binding=True) for d in range(4, 9)
    ]))
    assert a["driver"] == "budget_saturation"
    assert "never converges" in a["detail"]


def test_unknown_combination_is_reported_as_unattributed_not_guessed():
    # A new failure mode must be reported as new, never rounded to the
    # nearest known one.
    a = tw._attribute(_window([
        _rich_shadow(f"2026-08-2{d}", 0.20, ir=3.0, binding=False) for d in range(4, 9)
    ]))
    assert a["driver"] == "unattributed"
    assert "not one the attribution knows" in a["detail"]


def test_attribution_is_emitted_on_a_quiet_day_too(published):
    # A driver block that exists only on the alerting path cannot be used to
    # watch a condition build.
    s3 = _FakeS3({"predictor/optimizer_shadow/2026-08-28.json":
                  _rich_shadow("2026-08-28", 0.01, ir=3.0)})
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.01, "conviction_ir_xs": 3.0},
        _CFG, "b", "2026-08-31", s3_client=s3,
    )
    assert out["status"] == tw.STATUS_OK
    assert published == []
    assert "attribution" in out and out["attribution"]["driver"]


def test_rolling_alert_message_carries_the_driver_and_the_series(published):
    objs = {
        f"predictor/optimizer_shadow/2026-08-2{d}.json":
        _rich_shadow(f"2026-08-2{d}", 0.20, ir=0.03, binding=True)
        for d in range(4, 9)
    }
    out = tw.check_turnover_tripwire(
        {"turnover_one_way": 0.20, "conviction_ir_xs": 0.03,
         "turnover_constraint_binding": True, "turnover_mandatory_floor": 0.0},
        _CFG, "b", "2026-08-31", s3_client=_FakeS3(objs),
    )
    assert out["rolling_breach"] is True
    assert len(published) == 1
    msg = published[0]["message"]
    assert "DRIVER (predictor_conviction_collapse)" in msg
    # The old text asked the operator to go and do the diagnosis.
    assert "review the optimizer shadow logs" not in msg
    # The per-session series is in the message, not only in the artifact.
    assert "2026-08-31 20.0%" in msg
    assert out["attribution"]["per_day"][0]["date"] == "2026-08-31"
