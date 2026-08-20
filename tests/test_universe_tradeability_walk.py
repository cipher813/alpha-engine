"""``read_universe_tradeability`` resolves the NEWEST AVAILABLE scanner
universe artifact, not the exact run date (alpha-engine-config-I7811).

Why this exists. Brian ruled 2026-08-20 that the scanner forms its two cuts
WEEKLY and those feed research and the predictor for the week. This function
read ``scanner/universe/{run_date}/universe.json`` by exact date and is
fail-soft by contract: on a miss it returns ``{}``. Under weekly formation that
key is absent Tue-Fri, so both live consumers —
``main.py``'s ``adv_map`` (the position sizer's ADV size cap, config#1401) and
``optimizer_shadow._build_adv_usd`` (the participation-aware sqrt-impact term) —
would have silently fallen back to "no ADV coverage" on four days in five, with
nothing raising and nothing reporting it. The predictor was already safe: its
``load_universe`` walks back ``MEMBERSHIP_MAX_AGE_DAYS`` for the sibling
``universe_membership`` artifact. This closes the same gap on this side.

The window matches the predictor's deliberately: both artifacts are written by
the same weekly scanner run, so one number governs how stale either consumer
will tolerate.
"""

from __future__ import annotations

import json
import logging

import pytest

from executor import signal_reader
from executor.signal_reader import UNIVERSE_MAX_AGE_DAYS, read_universe_tradeability

BUCKET = "alpha-engine-research"

_BOARD = {
    "schema_version": 3,
    "stocks": [
        {"ticker": "AAPL", "tradeability": {"adv_usd": 1.0e9, "tradeability_score": 99}},
        {"ticker": "GD", "tradeability": {"adv_usd": 2.5e8, "tradeability_score": 80}},
    ],
}


class _S3:
    """Serves ``universe.json`` for exactly the dates in ``present``.

    Records every key requested so a test can assert the walk's shape and its
    bound, not merely its result.
    """

    def __init__(self, present: set[str], *, error_code: str = "NoSuchKey"):
        self.present = present
        self.error_code = error_code
        self.requested: list[str] = []

    def get_object(self, Bucket: str, Key: str):  # noqa: N803 — boto3 kwarg casing
        self.requested.append(Key)
        for d in self.present:
            if Key == f"scanner/universe/{d}/universe.json":
                body = json.dumps(_BOARD).encode()
                return {"Body": _Body(body)}
        raise _client_error(self.error_code)


class _Body:
    def __init__(self, raw: bytes):
        self._raw = raw

    def read(self) -> bytes:
        return self._raw


def _client_error(code: str):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code, "Message": code}}, "GetObject")


@pytest.fixture
def s3(monkeypatch):
    def _install(present, **kw):
        stub = _S3(present, **kw)
        monkeypatch.setattr(signal_reader.boto3, "client", lambda *a, **k: stub)
        return stub

    return _install


def test_exact_date_hit_does_not_walk(s3):
    stub = s3({"2026-08-20"})
    out = read_universe_tradeability(BUCKET, "2026-08-20")
    assert set(out) == {"AAPL", "GD"}
    assert stub.requested == ["scanner/universe/2026-08-20/universe.json"]


def test_weekly_cadence_resolves_a_prior_instance(s3, caplog):
    """The ruling's normal weekday: the cut was formed on Saturday and today is
    Wednesday. This is the case that returned ``{}`` before I7811."""
    stub = s3({"2026-08-15"})
    with caplog.at_level(logging.INFO):
        out = read_universe_tradeability(BUCKET, "2026-08-20")
    assert set(out) == {"AAPL", "GD"}, "a 5-day-old weekly cut must still supply ADV$"
    assert len(stub.requested) == 6, "must walk day by day, newest first"
    assert stub.requested[0].endswith("2026-08-20/universe.json")
    assert stub.requested[-1].endswith("2026-08-15/universe.json")
    assert "5 calendar day(s) back" in caplog.text


def test_nothing_in_window_is_empty_and_WARNS(s3, caplog):
    """Still fail-soft — the optimizer's flat-L1 fallback is preserved — but the
    absence is now a real condition, not an ordinary weekday, so it must not be
    logged at INFO. A silent fallback is the failure this issue was opened for."""
    stub = s3(set())
    with caplog.at_level(logging.INFO):
        out = read_universe_tradeability(BUCKET, "2026-08-20")
    assert out == {}
    assert len(stub.requested) == UNIVERSE_MAX_AGE_DAYS + 1, "walk must be bounded"
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "an empty window must WARN, not INFO"
    assert "No scanner universe artifact within" in caplog.text


def test_walk_is_bounded_by_the_declared_window(s3):
    """The instance just outside the window is NOT used — an unbounded walk
    would happily serve a months-old board and call it coverage."""
    from datetime import date, timedelta

    too_old = (date(2026, 8, 20) - timedelta(days=UNIVERSE_MAX_AGE_DAYS + 1)).isoformat()
    stub = s3({too_old})
    assert read_universe_tradeability(BUCKET, "2026-08-20") == {}
    assert len(stub.requested) == UNIVERSE_MAX_AGE_DAYS + 1


def test_non_missing_client_error_still_fails_soft(s3):
    """403/throttle must not propagate — this is a construction refinement,
    never a gate (the contract every caller relies on)."""
    stub = s3(set(), error_code="AccessDenied")
    assert read_universe_tradeability(BUCKET, "2026-08-20") == {}
    assert len(stub.requested) == 1, "a non-404 aborts the walk rather than retrying it 10x"


def test_unparseable_run_date_fails_soft(s3):
    s3({"2026-08-20"})
    assert read_universe_tradeability(BUCKET, "not-a-date") == {}
