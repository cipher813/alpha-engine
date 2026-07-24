"""Unit tests for infrastructure/iam/check-drift.py.

Covers three paths required by the fail-loud gate sweep (§119 rule 1):

  1. AWS CLI missing (``shutil.which("aws")`` → ``None``)      → exit 1
  2. Success path (codified IAM matches live AWS)               → exit 0
  3. Drift detected (codified policy differs from live AWS)     → exit 1

Follows the module-load pattern from ``test_check_no_foreign_writers.py``
(``importlib.util.spec_from_file_location``) and patches ``_aws_iam`` /
``subprocess.run`` / ``shutil.which`` at the module level.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "infrastructure"
    / "iam"
    / "check-drift.py"
)
_spec = importlib.util.spec_from_file_location("check_drift", _SCRIPT_PATH)
cd = importlib.util.module_from_spec(_spec)
sys.modules["check_drift"] = cd
_spec.loader.exec_module(cd)

# Shared test policy document used by both success and drift tests.
_INLINE_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::example-bucket/*",
        }
    ],
}

_DRIFT_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::example-bucket/*",
        }
    ],
}


def _make_role_dir(tmp_path: Path, role_name: str, policies: dict[str, object]) -> Path:
    """Create a role directory under ``tmp_path`` with the given policy files.

    ``policies`` maps filenames (without ``.json``) to dict policy documents.
    """
    role_dir = tmp_path / role_name
    role_dir.mkdir(parents=True, exist_ok=True)
    for name, doc in policies.items():
        (role_dir / f"{name}.json").write_text(json.dumps(doc))
    return role_dir


# ---- AWS CLI missing guard (shutil.which -> None) --------------------------


def test_aws_cli_missing_exits_1():
    """When ``shutil.which("aws")`` returns ``None``, ``_aws_iam`` calls
    ``sys.exit(1)`` after writing the error message to stderr."""
    with patch.object(cd.shutil, "which", return_value=None):
        with pytest.raises(SystemExit) as exc_info:
            cd._aws_iam("list-role-policies", "--role-name", "test-role")
    assert exc_info.value.code == 1


# ---- Success path (codified == live) ---------------------------------------


def _patch_aws_for_clean_role():
    """Return a context manager that patches ``cd._aws_iam`` to simulate a
    role whose live IAM state exactly matches one inline policy file."""
    return patch.object(
        cd,
        "_aws_iam",
        side_effect=_aws_iam_side_effect({"my-policy": _INLINE_POLICY}),
    )


def _aws_iam_side_effect(
    policies: dict[str, object],
) -> object:
    """Build a side-effect callable for ``_aws_iam`` that returns plausible
    AWS responses given a map of policy-name → document.

    Handles the call sequences made by ``_check_role``:
      - ``list-role-policies --role-name <role>``
      - ``get-role-policy --role-name <role> --policy-name <name>``
      - ``get-role --role-name <role>``
      - ``list-attached-role-policies --role-name <role>``
    """

    def _side(*args: str) -> dict:
        if not args:
            return {}
        command = args[0]
        # Extract --role-name value
        role = None
        for i, a in enumerate(args):
            if a == "--role-name" and i + 1 < len(args):
                role = args[i + 1]
                break
        if command == "list-role-policies":
            return {"PolicyNames": list(policies.keys())}
        if command == "get-role-policy":
            pname = None
            for i, a in enumerate(args):
                if a == "--policy-name" and i + 1 < len(args):
                    pname = args[i + 1]
                    break
            doc = policies.get(pname, {})
            return {"PolicyDocument": doc, "RoleName": role, "PolicyName": pname}
        if command == "get-role":
            return {"Role": {"RoleName": role, "AssumeRolePolicyDocument": {}}}
        if command == "list-attached-role-policies":
            return {"AttachedPolicies": []}
        return {}

    return _side


def test_clean_role_returns_empty_findings(tmp_path):
    """When codified policy matches live AWS, ``_check_role`` returns an
    empty list (the no-drift signal)."""
    role_dir = _make_role_dir(tmp_path, "test-role", {"my-policy": _INLINE_POLICY})
    with _patch_aws_for_clean_role():
        findings = cd._check_role(role_dir)
    assert findings == []


def test_empty_role_dir_returns_finding(tmp_path):
    """When a role directory has no ``.json`` files, ``_check_role`` returns
    a finding reporting the empty directory."""
    role_dir = tmp_path / "empty-role"
    role_dir.mkdir()
    with patch.object(cd, "_aws_iam", return_value={"PolicyNames": []}):
        findings = cd._check_role(role_dir)
    assert len(findings) == 1
    assert "no .json files" in findings[0]


# ---- Drift paths -----------------------------------------------------------


def test_content_drift_returns_findings(tmp_path):
    """When a codified inline policy document differs from the live AWS
    document, ``_check_role`` returns a content-drift finding."""
    role_dir = _make_role_dir(tmp_path, "test-role", {"my-policy": _INLINE_POLICY})
    # Live returns a DIFFERENT policy document (drift).
    side = _aws_iam_side_effect({"my-policy": _DRIFT_POLICY})
    with patch.object(cd, "_aws_iam", side_effect=side):
        findings = cd._check_role(role_dir)
    assert len(findings) == 1
    assert "content drift" in findings[0]


def test_codified_policy_missing_in_aws_returns_finding(tmp_path):
    """When a policy file exists on disk but is absent from the live AWS
    role, ``_check_role`` returns a 'codified but not on AWS' finding."""
    role_dir = _make_role_dir(tmp_path, "test-role", {"extra-policy": _INLINE_POLICY})
    side = _aws_iam_side_effect({})  # No policies on AWS
    with patch.object(cd, "_aws_iam", side_effect=side):
        findings = cd._check_role(role_dir)
    assert len(findings) == 1
    assert "codified in source but not on AWS" in findings[0]


def test_aws_policy_missing_in_source_returns_finding(tmp_path):
    """When a policy exists on the live AWS role but has no matching file on
    disk, ``_check_role`` returns a 'present on AWS but not codified' finding.
    The role dir must have at least one non-reserved ``.json`` file to pass
    the empty-dir guard before reaching the AWS-set comparison."""
    role_dir = _make_role_dir(tmp_path, "test-role", {"existing-policy": _INLINE_POLICY})
    # AWS has existing-policy (matches) AND aws-only-policy (not in source)
    side = _aws_iam_side_effect({
        "existing-policy": _INLINE_POLICY,
        "aws-only-policy": _INLINE_POLICY,
    })
    with patch.object(cd, "_aws_iam", side_effect=side):
        findings = cd._check_role(role_dir)
    assert len(findings) == 1
    assert "present on AWS role but not codified" in findings[0]


def test_invalid_source_json_returns_finding(tmp_path):
    """When a policy file contains invalid JSON, ``_check_role`` returns
    a 'source JSON invalid' finding rather than crashing."""
    role_dir = tmp_path / "test-role"
    role_dir.mkdir()
    (role_dir / "bad-policy.json").write_text("not valid json")

    side = _aws_iam_side_effect({"bad-policy": _INLINE_POLICY})
    with patch.object(cd, "_aws_iam", side_effect=side):
        findings = cd._check_role(role_dir)
    assert len(findings) == 1
    assert "source JSON invalid" in findings[0]


# ---- Trust policy (opt-in) -------------------------------------------------


def test_trust_policy_drift_returns_finding(tmp_path):
    """When ``trust-policy.json`` exists and differs from the live
    ``AssumeRolePolicyDocument``, ``_check_role`` returns a drift finding."""
    trust_doc = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    role_dir = _make_role_dir(tmp_path, "test-role", {"trust-policy": trust_doc})

    # Live returns a DIFFERENT trust policy
    live_trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Deny", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}

    def _side(*args):
        if args[0] == "list-role-policies":
            return {"PolicyNames": []}
        if args[0] == "get-role":
            return {"Role": {"AssumeRolePolicyDocument": live_trust}}
        return {}

    with patch.object(cd, "_aws_iam", side_effect=_side):
        findings = cd._check_role(role_dir)
    assert len(findings) == 1
    assert "trust policy differs" in findings[0]


# ---- Managed policies (opt-in) ---------------------------------------------


def test_managed_policy_drift_returns_finding(tmp_path):
    """When ``managed-policies.json`` lists an ARN not attached to the live
    role, ``_check_role`` returns a 'codified but not attached' finding."""
    managed_arns = ["arn:aws:iam::123456789012:policy/MyManagedPolicy"]
    role_dir = _make_role_dir(tmp_path, "test-role", {"managed-policies": managed_arns})

    def _side(*args):
        if args[0] == "list-role-policies":
            return {"PolicyNames": []}
        if args[0] == "get-role":
            return {"Role": {"AssumeRolePolicyDocument": {}}}
        if args[0] == "list-attached-role-policies":
            return {"AttachedPolicies": []}  # Nothing attached live
        return {}

    with patch.object(cd, "_aws_iam", side_effect=_side):
        findings = cd._check_role(role_dir)
    assert len(findings) == 1
    assert "codified but not attached" in findings[0]
    assert "MyManagedPolicy" in findings[0]
