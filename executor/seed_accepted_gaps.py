#!/usr/bin/env python3
"""Seed the accepted-gaps S3 registry with the initial permanently-unrecoverable
eod_pnl gap for 2026-07-27 (alpha-engine-config#5570 / #5569).

Usage:
    python3 -m executor.seed_accepted_gaps [--dry-run]

The 2026-07-27 gap is the first accepted gap: CaptureSnapshot permanently failed
before midnight and the snapshot was never written, making 2026-07-27 unrecoverable
by any automatic path. Per Brian's 2026-07-29 ruling on alpha-engine-config#5325,
the gap is accepted rather than requiring a hand-built NAV.

This is a ONE-TIME bootstrap script. After seeding, all future accepted-gap
additions are operator actions via PRs that update the accepted_gaps module
and the S3 record — never an automated pass (same guard as the ``default-ok``
label, alpha-engine-config#1925).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

import boto3

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from executor.accepted_gaps import load_accepted_gaps
from executor.config_loader import load_config

logger = logging.getLogger(__name__)
ACCEPTED_GAPS_KEY = "trades/accepted_gaps.json"
SCHEMA_VERSION = 1


def seed(region: str = "us-east-1", trades_bucket: str | None = None, dry_run: bool = False) -> dict:
    """Seed the initial accepted gap for 2026-07-27.

    Reads existing registry first (if any), merges the new gap, and writes back.
    """
    config = load_config()
    bucket = trades_bucket or config["trades_bucket"]
    region = region or config.get("aws_region", "us-east-1")
    s3 = boto3.client("s3", region_name=region)

    # Load existing (may be empty dict if file doesn't exist yet)
    existing = load_accepted_gaps(bucket, region, s3_client=s3)

    # Build the 2026-07-27 gap
    gap_date = "2026-07-27"
    gap_record = {
        "reason": (
            "CaptureSnapshot permanently failed before midnight on 2026-07- — "
            "snapshot never written, unrecoverable by any automatic path "
            "(alpha-engine-config#5569). Accepted per Brian's 2026-07-29 "
            "ruling on alpha-engine-config#5325."
        ),
        "ruling_ref": "alpha-engine-config#5325",
        "accepted_at": datetime.now(UTC).isoformat(),
        "accepted_by": "seed-accepted-gaps",
    }

    # Merge: only add if not already present (idempotent)
    gaps = dict(existing)
    already_present = gap_date in gaps
    if not already_present:
        gaps[gap_date] = gap_record

    payload = {
        "schema_version": SCHEMA_VERSION,
        "gaps": gaps,
    }

    key = ACCEPTED_GAPS_KEY
    if dry_run:
        logger.info("[seed_accepted_gaps] DRY-RUN: would write s3://%s/%s", bucket, key)
        logger.info("[seed_accepted_gaps] DRY-RUN: payload=%s", json.dumps(payload, indent=2))
        return payload

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    action = "already present, unchanged" if already_present else "seeded"
    logger.info("[seed_accepted_gaps] %s: %s in s3://%s/%s", gap_date, action, bucket, key)
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dry_run = "--dry-run" in sys.argv
    seed(dry_run=dry_run)
    if not dry_run:
        print("Accepted-gaps registry seeded. Verify:")
        print(f"  aws s3 cp s3://$(python3 -c 'from executor.config_loader import load_config; c=load_config(); print(c[\"trades_bucket\"])')/{ACCEPTED_GAPS_KEY} -")
        print("  aws s3 ls s3://$(python3 -c 'from executor.config_loader import load_config; c=load_config(); print(c[\"trades_bucket\"])')/trades/accepted_gaps.json")


if __name__ == "__main__":
    main()
