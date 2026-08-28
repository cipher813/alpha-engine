"""Tests for the risk-config safety-flag visibility surface
(alpha-engine-config-I9021).

Two things are covered here:

1. ``executor.risk_flag_audit.list_off_safety_flags`` — the pure function
   that decides which feature-flagged safety gates are OFF for a given
   config.
2. Its wiring into ``executor.main._write_order_book_summary`` /
   ``_write_stops_and_finalize`` — the console-facing surface
   (``order_books/{date}/summary.json``) a component emitting nothing would
   otherwise leave unobserved (principle 7).

A third test (``TestBatchConfidenceTighteningAtMeasuredMean``) locks in the
exact condition this arc found live: batch-mean prediction_confidence
~0.094 against the (now armed) 0.30 threshold.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executor.risk_flag_audit import SAFETY_FLAGS_DEFAULT_OFF, list_off_safety_flags

# ── list_off_safety_flags ──────────────────────────────────────────────────


class TestListOffSafetyFlags:

    def test_all_flags_off_when_config_empty(self):
        assert list_off_safety_flags({}) == sorted(SAFETY_FLAGS_DEFAULT_OFF)

    def test_all_flags_off_when_config_none(self):
        assert list_off_safety_flags(None) == sorted(SAFETY_FLAGS_DEFAULT_OFF)

    def test_empty_when_every_flag_explicitly_true(self):
        config = dict.fromkeys(SAFETY_FLAGS_DEFAULT_OFF, True)
        assert list_off_safety_flags(config) == []

    def test_absent_and_explicit_false_are_both_reported(self):
        config = {
            "batch_confidence_tightening_enabled": False,  # explicit
            # urgency_weighted_entry_ranking_enabled: absent entirely
        }
        result = list_off_safety_flags(config)
        assert "batch_confidence_tightening_enabled" in result
        assert "urgency_weighted_entry_ranking_enabled" in result

    def test_flags_true_are_excluded(self):
        config = dict.fromkeys(SAFETY_FLAGS_DEFAULT_OFF, True)
        config["derisk_on_expectancy_enabled"] = False
        result = list_off_safety_flags(config)
        assert result == ["derisk_on_expectancy_enabled"]

    def test_result_is_sorted(self):
        config = {}
        result = list_off_safety_flags(config)
        assert result == sorted(result)

    def test_unrelated_keys_do_not_affect_result(self):
        config = {
            **dict.fromkeys(SAFETY_FLAGS_DEFAULT_OFF, True),
            "min_score_to_enter": 57,
            "allow_shorts": False,
        }
        assert list_off_safety_flags(config) == []

    def test_batch_confidence_tightening_now_armed_is_not_reported_off(self):
        # Regression guard for I9021 deliverable 1: once armed in risk.yaml,
        # it must disappear from the off-flags surface.
        config = {"batch_confidence_tightening_enabled": True}
        assert "batch_confidence_tightening_enabled" not in list_off_safety_flags(config)


# ── Wiring into the order-book summary (the console-read surface) ─────────


class TestOrderBookSummaryEmitsRiskFlagsOff:

    def _order_book(self, run_date="2026-08-28"):
        from executor.order_book import OrderBook, _default_book

        return OrderBook(_default_book(run_date))

    def test_risk_flags_off_present_and_populated(self, monkeypatch):
        import executor.main as main_mod

        put_calls = []

        class _FakeS3Put:
            def put_object(self, **kwargs):
                put_calls.append(kwargs)

        monkeypatch.setattr("boto3.client", lambda *a, **kw: _FakeS3Put())

        ob = self._order_book()
        main_mod._write_order_book_summary(
            ob, [], "test-bucket", "2026-08-28",
            risk_flags_off=["urgency_weighted_entry_ranking_enabled"],
        )

        body = json.loads(put_calls[0]["Body"])
        assert body["risk_flags_off"] == ["urgency_weighted_entry_ranking_enabled"]

    def test_risk_flags_off_defaults_to_empty_list_not_omitted(self, monkeypatch):
        # A component emitting nothing is unobserved, not healthy (principle
        # 7) — the key must always be present, even when the caller passes
        # nothing and even when nothing is off.
        import executor.main as main_mod

        put_calls = []

        class _FakeS3Put:
            def put_object(self, **kwargs):
                put_calls.append(kwargs)

        monkeypatch.setattr("boto3.client", lambda *a, **kw: _FakeS3Put())

        ob = self._order_book()
        main_mod._write_order_book_summary(ob, [], "test-bucket", "2026-08-28")

        body = json.loads(put_calls[0]["Body"])
        assert "risk_flags_off" in body
        assert body["risk_flags_off"] == []

    def test_write_stops_and_finalize_computes_and_forwards_off_flags(self, monkeypatch):
        # End-to-end: _write_stops_and_finalize must compute the off-flags
        # from ``config`` and thread them into the summary write without the
        # caller having to pass them explicitly.
        import executor.main as main_mod
        from executor.ibkr import SimulatedIBKRClient

        put_calls = []

        class _FakeS3Put:
            def put_object(self, **kwargs):
                put_calls.append(kwargs)

        monkeypatch.setattr("boto3.client", lambda *a, **kw: _FakeS3Put())
        monkeypatch.setattr("executor.order_book.OrderBook.save", lambda self: None)

        ibkr = SimulatedIBKRClient(prices={}, nav=1_000_000.0)
        ob = self._order_book()
        config = {
            "batch_confidence_tightening_enabled": True,  # armed — must NOT appear
        }

        main_mod._write_stops_and_finalize(
            ibkr, ob, {}, {}, {}, None, "2026-08-28",
            blocked_entries=[],
            signals_bucket="test-bucket",
            use_optimizer=False,
            signals_raw={},
            config=config,
        )

        summary_calls = [c for c in put_calls if c["Key"].endswith("summary.json")]
        assert len(summary_calls) == 1
        body = json.loads(summary_calls[0]["Body"])
        assert "batch_confidence_tightening_enabled" not in body["risk_flags_off"]
        assert "urgency_weighted_entry_ranking_enabled" in body["risk_flags_off"]

    def test_write_stops_and_finalize_reports_all_off_when_config_none(self, monkeypatch):
        import executor.main as main_mod
        from executor.ibkr import SimulatedIBKRClient

        put_calls = []

        class _FakeS3Put:
            def put_object(self, **kwargs):
                put_calls.append(kwargs)

        monkeypatch.setattr("boto3.client", lambda *a, **kw: _FakeS3Put())
        monkeypatch.setattr("executor.order_book.OrderBook.save", lambda self: None)

        ibkr = SimulatedIBKRClient(prices={}, nav=1_000_000.0)
        ob = self._order_book()

        main_mod._write_stops_and_finalize(
            ibkr, ob, {}, {}, {}, None, "2026-08-28",
            blocked_entries=[],
            signals_bucket="test-bucket",
            use_optimizer=False,
            signals_raw={},
            config=None,
        )

        summary_calls = [c for c in put_calls if c["Key"].endswith("summary.json")]
        body = json.loads(summary_calls[0]["Body"])
        assert sorted(body["risk_flags_off"]) == sorted(SAFETY_FLAGS_DEFAULT_OFF)


# ── The exact measured condition this arc found live ──────────────────────


class TestBatchConfidenceTighteningAtMeasuredMean:
    """2026-08-24..28 batch-mean prediction_confidence: 0.110, 0.097, 0.098,
    0.092, 0.094 — every session ~3x below the (now armed) 0.30 threshold.
    Locks in that the backstop actually fires at this measured mean, so a
    regression here would be caught rather than silently re-inerting the
    gate this arc just armed."""

    def test_fires_at_measured_mean_0_094(self):
        from executor.deciders import _apply_batch_confidence_tightening

        config = {
            "batch_confidence_tightening_enabled": True,
            "batch_confidence_threshold": 0.30,
            "batch_confidence_min_score_bump": 10,
            "min_score_to_enter": 57,
        }
        # Synthetic batch whose mean prediction_confidence is exactly the
        # measured 2026-08-28 value (0.094), well under the 0.30 trigger.
        preds = {
            "A": {"prediction_confidence": 0.094},
            "B": {"prediction_confidence": 0.094},
            "C": {"prediction_confidence": 0.094},
        }
        result = _apply_batch_confidence_tightening(config, preds, "2026-08-28")

        assert result is not config
        assert result["min_score_to_enter"] == 67  # 57 + 10 bump
        meta = result["_batch_confidence_tightening_applied"]
        assert meta["mean_confidence"] == pytest.approx(0.094)
        assert meta["threshold"] == 0.30
        assert meta["tightened_min_score"] == 67

    def test_fires_across_the_full_measured_five_session_window(self):
        from executor.deciders import _apply_batch_confidence_tightening

        # 2026-08-24 through 2026-08-28, in order.
        measured_means = [0.110, 0.097, 0.098, 0.092, 0.094]
        config = {
            "batch_confidence_tightening_enabled": True,
            "batch_confidence_threshold": 0.30,
            "batch_confidence_min_score_bump": 10,
            "min_score_to_enter": 57,
        }
        for i, mean_conf in enumerate(measured_means):
            preds = {"A": {"prediction_confidence": mean_conf}}
            result = _apply_batch_confidence_tightening(
                config, preds, f"2026-08-{24 + i}",
            )
            assert result is not config, f"failed to trigger at mean={mean_conf}"
            assert result["min_score_to_enter"] == 67

    def test_would_not_have_fired_while_the_flag_was_off(self):
        # The exact production state this arc found: flag absent (defaults
        # False) → no tightening, no protection, despite the same degenerate
        # batch confidence.
        from executor.deciders import _apply_batch_confidence_tightening

        config = {"min_score_to_enter": 57}  # no batch_confidence_* keys at all
        preds = {"A": {"prediction_confidence": 0.094}}
        result = _apply_batch_confidence_tightening(config, preds, "2026-08-28")

        assert result is config  # unchanged — this was the inert state
        assert result["min_score_to_enter"] == 57


# ── The threshold-miscalibration finding (alpha-engine-config-I9034) ──────
#
# Enabling batch_confidence_tightening_enabled at the default
# batch_confidence_threshold (0.30) was found NOT to be a safe arm: 35
# sessions of measured batch-mean prediction_confidence sit almost entirely
# below that threshold (range 0.052-0.305, median ~0.188), so 0.30 fires on
# nearly every session rather than selectively on genuinely degenerate ones.
# This is why the flag ships (and stays) OFF pending I9034's relative-
# threshold redesign. Locked in here as a regression guard: if someone
# quietly re-enables the flag at 0.30 without addressing I9034, this test
# documents exactly how often that would fire against the measured book.


class TestAbsoluteThresholdMiscalibration:

    # 35 trading sessions, predictor/predictions/{date}.json daily mean
    # prediction_confidence, measured 2026-08-28 (alpha-engine-config-I9034).
    MEASURED_35_SESSION_MEANS = [
        0.052, 0.052, 0.185, 0.177, 0.188, 0.184, 0.182, 0.194, 0.191, 0.194,
        0.178, 0.187, 0.299, 0.232, 0.243, 0.176, 0.166, 0.204, 0.257, 0.185,
        0.189, 0.188, 0.188, 0.138, 0.112, 0.191, 0.200, 0.220, 0.305, 0.281,
        0.110, 0.097, 0.098, 0.092, 0.094,
    ]

    def test_default_threshold_fires_on_nearly_every_measured_session(self):
        from executor.deciders import _apply_batch_confidence_tightening

        config_template = {
            "batch_confidence_tightening_enabled": True,
            "batch_confidence_threshold": 0.30,
            "batch_confidence_min_score_bump": 10,
            "min_score_to_enter": 57,
        }
        fired = 0
        for mean_conf in self.MEASURED_35_SESSION_MEANS:
            preds = {"A": {"prediction_confidence": mean_conf}}
            result = _apply_batch_confidence_tightening(
                dict(config_template), preds, "2026-08-28",
            )
            if result is not None and result.get("min_score_to_enter") != 57:
                fired += 1

        # 34 of 35 measured sessions are below 0.30 — only the single 0.305
        # session clears it. Asserting the exact count (not just ">majority")
        # so this test breaks loudly if the book's distribution shifts enough
        # to make 0.30 a reasonable absolute constant again.
        assert fired == 34
        fire_rate = fired / len(self.MEASURED_35_SESSION_MEANS)
        assert fire_rate > 0.9  # ~97% — the finding that kept this flag OFF

    def test_measured_distribution_shape(self):
        # Documents the numbers the I9021/I9034 rationale comments cite, so
        # a future change to config/risk.yaml.example's evidence comment has
        # a test tying it back to this exact dataset.
        means = self.MEASURED_35_SESSION_MEANS
        assert min(means) == pytest.approx(0.052)
        assert max(means) == pytest.approx(0.305)
        sorted_means = sorted(means)
        n = len(sorted_means)
        median = (
            sorted_means[n // 2]
            if n % 2
            else (sorted_means[n // 2 - 1] + sorted_means[n // 2]) / 2
        )
        assert median == pytest.approx(0.187, abs=0.001)
