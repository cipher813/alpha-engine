"""accepted_gaps — durable, queryable accepted-gap declarations for eod_pnl.

A permanently-unrecoverable eod_pnl gap (a date on which CaptureSnapshot
permanently failed and the snapshot was never written — see
alpha-engine-config#5569) needs a state other than "flag for manual backfill
forever" or "silently fabricate a NAV". This module provides that state: a
durable, versioned S3 record of accepted gaps that ``reconcile_audit`` reads
to distinguish "waiting for backfill" from "permanently unrecoverable."

Rules (alpha-engine-config#5570 gotcha):
  * Adding an accepted gap is an OPERATOR action tied to a ruling — never
    something an automated pass can do to clear its own alert (the same guard
    as the ``default-ok`` label, alpha-engine-config#1925). A gap suppressed
    without a ruling is a corrupted alpha series with the evidence deleted.
  * An accepted gap is reported at ``info`` by reconcile_audit, never
    re-flagged as a ``warning`` needing manual action. It still appears in the
    run's ``gaps`` output — suppressed from *paging*, never from *the record*.
  * Downstream consumers (alpha/Sharpe windows) MUST exclude the date rather
    than treating it as a zero-return day — this module provides a queryable
    list so consumers can skip the date without code changes per gap.

Schema of the list file (``trades/accepted_gaps.json``):
  {
    "schema_version": 1,
    "gaps": {
      "2026-07-27": {
        "reason": "CaptureSnapshot permanently failed before midnight — snapshot never written, unrecoverable (alpha-engine-config#5569). Accepted per Brian's 2026-07-29 ruling on alpha-engine-config#5325.",
        "ruling_ref": "alpha-engine-config#5325",
        "accepted_at": "2026-07-29T...",
        "accepted_by": "groom-bot"
      }
    }
  }
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3

logger = logging.getLogger(__name__)

_ACCEPTED_GAPS_KEY = "trades/accepted_gaps.json"
_SCHEMA_VERSION = 1


def _accepted_gaps_key() -> str:
    return _ACCEPTED_GAPS_KEY


def load_accepted_gaps(
    trades_bucket: str,
    region: str = "us-east-1",
    s3_client=None,
) -> dict[str, dict[str, Any]]:
    """Load the accepted-gaps registry from S3.

    Returns a dict mapping ``date`` -> ``{reason, ruling_ref, accepted_at,
    accepted_by}``. Returns an empty dict when the file does not yet exist
    (the common case — no permanently-unrecoverable gaps have been recorded).
    Any non-404 error RAISES (fail-loud): an unreadable registry must not
    silently degrade to "no accepted gaps" — that would re-flag a genuine
    accepted gap as needing manual backfill.
    """
    s3 = s3_client or boto3.client("s3", region_name=region)
    key = _accepted_gaps_key()
    try:
        obj = s3.get_object(Bucket=trades_bucket, Key=key)
    except Exception as exc:
        err_code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        if err_code in ("NoSuchKey", "404"):
            logger.info("[accepted_gaps] no accepted-gaps registry at s3://%s/%s — none recorded", trades_bucket, key)
            return {}
        raise

    try:
        doc = json.loads(obj["Body"].read())
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("[accepted_gaps] malformed accepted-gaps registry at s3://%s/%s: %s — treating as empty", trades_bucket, key, exc)
        return {}

    gaps = doc.get("gaps") if isinstance(doc, dict) else None
    if not isinstance(gaps, dict):
        logger.warning("[accepted_gaps] accepted-gaps registry at s3://%s/%s has no 'gaps' dict — treating as empty", trades_bucket, key)
        return {}
    logger.info("[accepted_gaps] loaded %d accepted gap(s) from s3://%s/%s", len(gaps), trades_bucket, key)
    return gaps


def is_accepted_gap(
    date: str,
    trades_bucket: str,
    region: str = "us-east-1",
    s3_client=None,
    registry: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Check whether ``date`` is a recorded accepted gap.

    Accepts an optional pre-loaded ``registry`` dict to avoid re-reading S3
    when the caller already loaded it (reconcile_audit loads once per run).
    """
    if registry is not None:
        return date in registry
    gaps = load_accepted_gaps(trades_bucket, region, s3_client)
    return date in gaps


def build_seed_gap(
    date: str,
    *,
    reason: str,
    ruling_ref: str,
    accepted_by: str = "groom-bot",
) -> dict[str, dict]:
    """Build the accepted-gaps document, seeded with one gap.

    Returns a dict ready to be serialised to JSON and written to
    ``trades/accepted_gaps.json``. Used by the seed script to bootstrap
    the first accepted gap(s) — after that all additions are operator
    actions via a PR that updates this file.
    """
    return {
        "schema_version": _SCHEMA_VERSION,
        "gaps": {
            date: {
                "reason": reason,
                "ruling_ref": ruling_ref,
                "accepted_by": accepted_by,
            },
        },
    }
