"""Tests for executor/accepted_gaps.py — durable accepted-gap declarations for
eod_pnl (alpha-engine-config#5570)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from executor.accepted_gaps import (
    _SCHEMA_VERSION,
    build_seed_gap,
    is_accepted_gap,
    load_accepted_gaps,
)


class TestLoadAcceptedGaps:
    def test_no_file_returns_empty_dict(self):
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        # Simulate the error code path
        err = Exception("not found")
        err.response = {"Error": {"Code": "NoSuchKey"}}
        s3.get_object.side_effect = err
        gaps = load_accepted_gaps("my-bucket", s3_client=s3)
        assert gaps == {}

    def test_404_returns_empty_dict(self):
        s3 = MagicMock()
        err = Exception("not found")
        err.response = {"Error": {"Code": "404"}}
        s3.get_object.side_effect = err
        gaps = load_accepted_gaps("my-bucket", s3_client=s3)
        assert gaps == {}

    def test_unexpected_error_raises(self):
        s3 = MagicMock()
        err = Exception("AccessDenied")
        err.response = {"Error": {"Code": "AccessDenied"}}
        s3.get_object.side_effect = err
        import pytest
        with pytest.raises(Exception, match="AccessDenied"):
            load_accepted_gaps("my-bucket", s3_client=s3)

    def test_parses_valid_registry(self):
        payload = {
            "schema_version": 1,
            "gaps": {
                "2026-07-27": {
                    "reason": "CaptureSnapshot failed",
                    "ruling_ref": "alpha-engine-config#5325",
                    "accepted_by": "groom-bot",
                }
            },
        }
        s3 = MagicMock()
        s3.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps(payload).encode())}
        gaps = load_accepted_gaps("my-bucket", s3_client=s3)
        assert "2026-07-27" in gaps
        assert gaps["2026-07-27"]["ruling_ref"] == "alpha-engine-config#5325"

    def test_malformed_json_returns_empty(self):
        s3 = MagicMock()
        s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"not json{{}}")}
        gaps = load_accepted_gaps("my-bucket", s3_client=s3)
        assert gaps == {}

    def test_missing_gaps_key_returns_empty(self):
        payload = {"schema_version": 1}
        s3 = MagicMock()
        s3.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps(payload).encode())}
        gaps = load_accepted_gaps("my-bucket", s3_client=s3)
        assert gaps == {}


class TestIsAcceptedGap:
    def test_date_in_registry(self):
        registry = {"2026-07-27": {"reason": "test", "ruling_ref": "ref"}}
        assert is_accepted_gap("2026-07-27", "my-bucket", registry=registry)
        assert not is_accepted_gap("2026-07-28", "my-bucket", registry=registry)
        assert not is_accepted_gap("2026-07-26", "my-bucket", registry=registry)

    def test_empty_registry(self):
        assert not is_accepted_gap("2026-07-27", "my-bucket", registry={})


class TestBuildSeedGap:
    def test_builds_valid_seed(self):
        result = build_seed_gap(
            "2026-07-27",
            reason="CaptureSnapshot permanently failed",
            ruling_ref="alpha-engine-config#5325",
            accepted_by="groom-bot",
        )
        assert result["schema_version"] == _SCHEMA_VERSION
        assert "2026-07-27" in result["gaps"]
        assert result["gaps"]["2026-07-27"]["ruling_ref"] == "alpha-engine-config#5325"
