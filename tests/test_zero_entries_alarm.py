"""Tests for executor.zero_entries_alarm (config#5713) — the consecutive
zero-entries floor alarm.

All hermetic — S3 is a tiny in-memory fake, no real boto3/network calls,
and the alert publish is monkeypatched.
"""

from __future__ import annotations

import io
import json
from datetime import date

from botocore.exceptions import ClientError

from executor.zero_entries_alarm import (
    DEFAULT_ZERO_ENTRIES_THRESHOLD_SESSIONS,
    _session_dates_ending_at,
    check_zero_entries_floor,
    compute_zero_entries_streak,
)


class _FakeS3:
    """Minimal get_object stand-in over a dict of {key: bytes}."""

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = dict(objects or {})

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 kwarg casing
        if Key not in self.objects:
            raise ClientError(
                error_response={"Error": {"Code": "NoSuchKey", "Message": "absent"}},
                operation_name="GetObject",
            )
        return {"Body": io.BytesIO(self.objects[Key])}


def _summary_key(d: date) -> str:
    return f"order_books/{d.isoformat()}/summary.json"


def _summary(entries: list[dict] | None = None) -> bytes:
    return json.dumps({"entries_approved": entries or []}).encode()


class TestSessionDates:
    def test_walks_back_over_trading_days_skipping_weekend(self):
        # 2026-07-27 is a Monday; the two prior sessions are Thu/Fri, not
        # Sat/Sun.
        assert _session_dates_ending_at(date(2026, 7, 27), 3) == [
            date(2026, 7, 23), date(2026, 7, 24), date(2026, 7, 27),
        ]

    def test_includes_run_date(self):
        dates = _session_dates_ending_at(date(2026, 7, 31), 3)
        assert dates[-1] == date(2026, 7, 31)  # run_date inclusive
        assert len(dates) == 3


class TestComputeStreak:
    def test_zero_entries_streak_of_one_ok(self):
        s3 = _FakeS3({_summary_key(date(2026, 7, 31)): _summary([])})
        assert compute_zero_entries_streak(
            "bucket", date(2026, 7, 31), threshold=3, s3_client=s3,
        ) == 1

    def test_full_streak_hits_threshold(self):
        objects = {
            _summary_key(d): _summary([])
            for d in [date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31)]
        }
        s3 = _FakeS3(objects)
        assert compute_zero_entries_streak(
            "bucket", date(2026, 7, 31), threshold=3, s3_client=s3,
        ) == 3

    def test_entry_breaks_the_streak(self):
        objects = {
            _summary_key(date(2026, 7, 29)): _summary([]),
            _summary_key(date(2026, 7, 30)): _summary([{"ticker": "SPY"}]),
            _summary_key(date(2026, 7, 31)): _summary([]),
        }
        s3 = _FakeS3(objects)
        assert compute_zero_entries_streak(
            "bucket", date(2026, 7, 31), threshold=3, s3_client=s3,
        ) == 1

    def test_missing_summary_breaks_the_streak(self):
        objects = {
            _summary_key(date(2026, 7, 30)): _summary([]),
            _summary_key(date(2026, 7, 31)): _summary([]),
        }
        s3 = _FakeS3(objects)
        assert compute_zero_entries_streak(
            "bucket", date(2026, 7, 31), threshold=3, s3_client=s3,
        ) == 2  # 07-29's summary absent → not counted, streak stops

    def test_unparseable_summary_breaks_the_streak(self):
        objects = {
            _summary_key(date(2026, 7, 30)): b"not json",
            _summary_key(date(2026, 7, 31)): _summary([]),
        }
        s3 = _FakeS3(objects)
        assert compute_zero_entries_streak(
            "bucket", date(2026, 7, 31), threshold=3, s3_client=s3,
        ) == 1

    def test_accepts_iso_string_run_date(self):
        s3 = _FakeS3({_summary_key(date(2026, 7, 31)): _summary([])})
        assert compute_zero_entries_streak(
            "bucket", "2026-07-31", threshold=3, s3_client=s3,
        ) == 1


class TestCheckZeroEntriesFloor:
    def test_pages_when_threshold_met(self, monkeypatch):
        calls: list[dict] = []
        monkeypatch.setattr(
            "executor.zero_entries_alarm._publish",
            lambda **kw: calls.append(kw),
        )
        objects = {
            _summary_key(d): _summary([])
            for d in [date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31)]
        }
        s3 = _FakeS3(objects)
        streak = check_zero_entries_floor(
            "bucket", date(2026, 7, 31), threshold=3, s3_client=s3,
        )
        assert streak == 3
        assert len(calls) == 1
        assert calls[0]["streak"] == 3
        assert calls[0]["threshold"] == 3
        assert calls[0]["run_date"] == date(2026, 7, 31)

    def test_no_page_below_threshold(self, monkeypatch):
        calls: list[dict] = []
        monkeypatch.setattr(
            "executor.zero_entries_alarm._publish",
            lambda **kw: calls.append(kw),
        )
        s3 = _FakeS3({_summary_key(date(2026, 7, 31)): _summary([])})
        streak = check_zero_entries_floor(
            "bucket", date(2026, 7, 31), threshold=3, s3_client=s3,
        )
        assert streak == 1
        assert calls == []

    def test_default_threshold_is_three(self):
        assert DEFAULT_ZERO_ENTRIES_THRESHOLD_SESSIONS == 3

    def test_publish_failure_is_non_fatal(self, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("sns down")

        monkeypatch.setattr("executor.zero_entries_alarm._publish", _boom)
        objects = {
            _summary_key(d): _summary([])
            for d in [date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31)]
        }
        s3 = _FakeS3(objects)
        streak = check_zero_entries_floor(
            "bucket", date(2026, 7, 31), threshold=3, s3_client=s3,
        )
        assert streak == 3  # alarm never raises

    def test_computation_failure_is_non_fatal(self, monkeypatch):
        def _boom(bucket, run_date, *, threshold, s3_client=None):
            raise RuntimeError("calendar broke")

        monkeypatch.setattr(
            "executor.zero_entries_alarm.compute_zero_entries_streak", _boom,
        )
        streak = check_zero_entries_floor("bucket", date(2026, 7, 31))
        assert streak == 0  # fail soft, no page
