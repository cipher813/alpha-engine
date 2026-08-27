"""Tests for infrastructure/check_deps_match_manifest.sh.

alpha-engine-config-I8709: the assertion this script makes is deliberately
independent of every existing check — CodeFreshnessGate's import smoke test,
executor/preflight.py::check_deploy_drift, and boot-pull.sh's own post-pip-
install import gate are all inside the sequence they validate. This script
compares origin/main's requirements.txt (fetched via `git show`, never the
locally checked-out file) against the box's actual `pip freeze`.

These tests exercise the PURE comparison logic (compare_manifest_to_installed
/ parse_pins) by sourcing the script with AE_DEPS_CHECK_LIB_ONLY=1 against
fixture files — no git, no network, no real venv. Wiring this script into an
SF preflight stage (including the mandatory sf-pipeline-policy.md §7a
observe-mode staging) is explicitly out of scope for crucible-executor; see
the script's own header and the issue.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent / "infrastructure" / "check_deps_match_manifest.sh"


def _run_compare(manifest_text: str, installed_text: str, tmp_path: Path) -> subprocess.CompletedProcess:
    manifest_file = tmp_path / "requirements.txt"
    installed_file = tmp_path / "pip-freeze.txt"
    manifest_file.write_text(manifest_text)
    installed_file.write_text(installed_text)

    script = (
        f'source "{_SCRIPT}"; '
        f'compare_manifest_to_installed "{manifest_file}" "{installed_file}"'
    )
    import os

    env = dict(os.environ)
    env["AE_DEPS_CHECK_LIB_ONLY"] = "1"
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )


def test_script_exists_and_is_executable():
    assert _SCRIPT.is_file()
    import os

    assert os.access(_SCRIPT, os.X_OK)


def test_matching_manifest_and_installed_passes(tmp_path):
    manifest = "krepis==0.59.33\nboto3==1.43.53\n"
    installed = "krepis==0.59.33\nboto3==1.43.53\nsix==1.16.0\n"  # extra installed pkg is fine
    result = _run_compare(manifest, installed, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_mismatched_version_fails_and_names_both_versions(tmp_path):
    """The exact incident this issue names: krepis 0.54.0 installed while
    origin/main pins 0.59.33 must be caught and both versions named."""
    manifest = "krepis==0.59.33\n"
    installed = "krepis==0.54.0\n"
    result = _run_compare(manifest, installed, tmp_path)
    assert result.returncode == 1
    assert "krepis" in result.stdout
    assert "manifest=0.59.33" in result.stdout
    assert "installed=0.54.0" in result.stdout


def test_missing_installed_package_fails_and_says_not_installed(tmp_path):
    manifest = "krepis==0.59.33\n"
    installed = "boto3==1.43.53\n"
    result = _run_compare(manifest, installed, tmp_path)
    assert result.returncode == 1
    assert "krepis" in result.stdout
    assert "manifest=0.59.33" in result.stdout
    assert "installed=<not installed>" in result.stdout


def test_multiple_mismatches_are_all_named(tmp_path):
    manifest = "krepis==0.59.33\nboto3==1.43.53\nnousergon-lib==0.12.0\n"
    installed = "krepis==0.54.0\nboto3==1.43.53\nnousergon-lib==0.11.0\n"
    result = _run_compare(manifest, installed, tmp_path)
    assert result.returncode == 1
    assert "krepis" in result.stdout
    assert "nousergon-lib" in result.stdout
    assert "boto3" not in result.stdout
    assert "2 package(s) mismatched" in result.stdout


def test_pep503_name_normalization_treats_dash_and_underscore_as_equal(tmp_path):
    """pip freeze may render a package with underscores where
    requirements.txt uses dashes (or vice versa); PEP 503 treats them as the
    same distribution and this check must not manufacture a false mismatch
    from spelling alone."""
    manifest = "nousergon-lib==0.12.0\n"
    installed = "nousergon_lib==0.12.0\n"
    result = _run_compare(manifest, installed, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_pip_compile_via_comments_are_not_parsed_as_pins(tmp_path):
    manifest = (
        "krepis==0.59.33\n"
        "    # via\n"
        "    #   -r requirements.in\n"
        "boto3==1.43.53\n"
    )
    installed = "krepis==0.59.33\nboto3==1.43.53\n"
    result = _run_compare(manifest, installed, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_usage_error_without_repo_root_exits_2():
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2


def test_nonexistent_repo_root_exits_2(tmp_path):
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    result = subprocess.run(
        ["bash", str(_SCRIPT), str(not_a_repo)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "not a git checkout" in result.stdout + result.stderr
