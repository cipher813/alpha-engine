"""Consecutive zero-entries floor alarm (alpha-engine-config#5713).

Backstop for the "no new entries ever get proposed" failure class — the
general case the producer/champion coherence assertion
(``executor.champion.assert_producer_champion_coherence``) cannot
enumerate: ANY configuration or pipeline break that silently yields zero
approved entries for many sessions in a row. A single zero-entry day is
normal (observed 2026-07-25..28: 3, 0, 1, 2 approved entries — the ``0``
on 2026-07-28 was a legitimate quiet day indistinguishable from a broken
selection path on any single day's surface); a run of N consecutive
zero-entry sessions means the entry funnel is structurally broken and the
book trades down and never up.

Mechanism: after the day's order-book summary is written, walk back over
the last ``threshold`` trading sessions (fleet trading calendar — holidays
and weekends never count as sessions) and count consecutive sessions whose
``order_books/{date}/summary.json`` recorded ``entries_approved: []``. When
the streak reaches the threshold, publish an ops alert (SNS + Telegram,
deduped per session date).

Deliberately conservative edges:

* A session whose summary is MISSING is not counted — a missing summary
  means the executor did not run (uptime monitoring's business), not
  evidence of zero entries, and it breaks the streak.
* A session whose summary is unparseable is treated the same way.
* The check is best-effort observability: a computation or publish failure
  logs and returns, it never blocks the planner (mirrors
  ``turnover_tripwire``'s posture).

Threshold derivation (config#5713): the last four observed sessions
recorded 3/0/1/2 approved entries — the longest observed zero-run is 1
session, so a threshold of 3 consecutive zero-entry sessions has wide
margin while still catching a structurally-broken funnel inside a trading
week. Override via ``zero_entries_alarm_consecutive_sessions`` in the
executor's risk.yaml (private config repo).
"""

from __future__ import annotations

import json
import logging
from datetime import date

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

DEFAULT_ZERO_ENTRIES_THRESHOLD_SESSIONS = 3

_ORDER_BOOK_SUMMARY_KEY_TPL = "order_books/{session_date}/summary.json"


def _session_dates_ending_at(run_date: date, n: int) -> list[date]:
    """The ``n`` most recent trading sessions ending AT ``run_date``
    (inclusive), oldest first. Uses the fleet trading calendar so holidays
    and weekends never count as sessions."""
    from nousergon_lib.trading_calendar import previous_trading_day

    dates = [run_date]
    d = run_date
    for _ in range(n - 1):
        d = previous_trading_day(d)
        dates.append(d)
    return dates[::-1]


def compute_zero_entries_streak(
    bucket: str,
    run_date: str | date,
    *,
    threshold: int,
    s3_client=None,
) -> int:
    """Count consecutive trading sessions ending at ``run_date`` (inclusive)
    whose order-book summary recorded zero approved entries.

    A session is counted as zero-entry only when its summary EXISTS and its
    ``entries_approved`` is an empty list; a missing or unparseable summary
    breaks the streak (see module docstring). The count is capped at
    ``threshold`` — callers only need to know whether the floor was hit.
    """
    if isinstance(run_date, str):
        run_date = date.fromisoformat(run_date)
    s3 = s3_client or boto3.client("s3")
    streak = 0
    for d in reversed(_session_dates_ending_at(run_date, threshold)):
        key = _ORDER_BOOK_SUMMARY_KEY_TPL.format(session_date=d.isoformat())
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except ClientError:
            break  # no summary → session did not run → not zero-entry evidence
        try:
            summary = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            logger.warning(
                "zero-entries floor: unparseable summary at s3://%s/%s — "
                "breaking the streak", bucket, key,
            )
            break
        entries = summary.get("entries_approved")
        if not isinstance(entries, list) or entries:
            break
        streak += 1
    return streak


def check_zero_entries_floor(
    bucket: str,
    run_date: str | date,
    *,
    threshold: int = DEFAULT_ZERO_ENTRIES_THRESHOLD_SESSIONS,
    s3_client=None,
) -> int:
    """Compute the current zero-entries streak and page when it meets the
    threshold. Returns the streak. NEVER raises — best-effort observability
    (mirrors ``turnover_tripwire``: an alarm must not block the planner)."""
    try:
        streak = compute_zero_entries_streak(
            bucket, run_date, threshold=threshold, s3_client=s3_client,
        )
    except Exception as exc:  # noqa: BLE001 — documented best-effort posture
        logger.warning(
            "zero-entries floor: streak computation failed (non-fatal): %s", exc,
        )
        return 0
    if streak >= threshold:
        logger.error(
            "ZERO-ENTRIES FLOOR BREACH: %d consecutive trading sessions with "
            "zero approved entries (threshold %d) — the entry funnel is "
            "structurally broken (producer/champion incoherence, a vetoed "
            "selection path, or a silent break). Check the producer/champion "
            "coherence assertion and the order-book summaries; the book "
            "trades down and never up while this persists.",
            streak, threshold,
        )
        try:
            _publish(streak=streak, threshold=threshold, run_date=run_date)
        except Exception as exc:  # noqa: BLE001 — documented best-effort posture
            logger.warning(
                "zero-entries floor alert publish failed (non-fatal): %s", exc,
            )
    return streak


def _publish(*, streak: int, threshold: int, run_date: str | date) -> None:
    """Best-effort dual-channel ops alert (SNS + Telegram) — mirrors
    ``turnover_tripwire._publish``'s posture: a publish failure is logged,
    never raised."""
    from executor.notifier import publish_ops_alert

    msg = (
        f"ZERO-ENTRIES FLOOR BREACH: {streak} consecutive trading sessions "
        f"with zero approved entries (threshold {threshold}, through "
        f"{run_date}) — the entry funnel is structurally broken and the "
        "book trades down and never up. The producer/champion coherence "
        "assertion (config#5713) did not fire, so this is the general-case "
        "backstop: investigate the producer/champion pairing, the selection "
        "path, and today's order-book summary."
    )
    try:
        publish_ops_alert(
            message=msg,
            severity="error",
            source="alpha-engine/executor/zero_entries_alarm.py",
            dedup_key=f"executor_zero_entries_floor_{run_date}",
        )
    except Exception as exc:  # noqa: BLE001 — secondary observability
        logger.warning(
            "zero-entries floor alert publish failed (non-fatal): %s", exc,
        )
