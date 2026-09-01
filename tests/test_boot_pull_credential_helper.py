"""alpha-engine-config-I9739 deliverable 3: the trading box must stop writing
a long-lived GitHub PAT to disk and must instead depend on, and verify,
`git-credential-nousergon-app` (alpha-engine-config-I9628, nous-ergon-ops-PR962).

Root cause this closes, measured 2026-09-01: git sets `CURLOPT_NETRC` to
`CURL_NETRC_OPTIONAL` unconditionally, so libcurl answers GitHub's 401
challenge from the on-disk netrc file BEFORE git ever consults a configured
credential helper. The netrc file does not compete with the helper — it
outranks it. The dashboard box had the helper installed, configured, and
provably able to mint a token, and it was never once consulted, because its
own boot-pull.sh kept rewriting the netrc file on every boot. Every assertion
below is a regex over the script's source text (the same convention every
other guard in test_trading_box_boot_pull.py uses for this hardcoded-path
script) because there is no way to exercise the real
`/usr/local/bin/git-credential-nousergon-app` path or a real
`/home/ec2-user` HOME from a local test run without root.
"""
from __future__ import annotations

import re
from pathlib import Path

_BOOT_PULL = Path(__file__).parent.parent / "infrastructure" / "boot-pull.sh"
_NETRC_LITERAL = "/home/ec2-user/" + ".netrc"
_HELPER_PATH = "/usr/local/bin/git-credential-nousergon-app"


def _src() -> str:
    return _BOOT_PULL.read_text()


def test_ssm_netrc_hydration_is_gone():
    """The block that pulled /alpha-engine/GITHUB_TOKEN out of SSM and wrote
    it into the netrc file must be deleted entirely, not disabled behind a
    flag — a box that still WOULD write it on some code path is a box where
    the long-lived secret can still reappear. A historical comment naming the
    old parameter for context is fine; an actual SSM read of it is not."""
    src = _src()
    assert "aws ssm get-parameter --name /alpha-engine/GITHUB_TOKEN" not in src, (
        "boot-pull.sh must no longer read /alpha-engine/GITHUB_TOKEN — that "
        "PAT hydration is the mechanism this issue removes."
    )
    assert "GH_TOKEN" not in src
    assert "NEW_NETRC" not in src


def test_netrc_is_unconditionally_removed_every_run():
    """Every run must `rm -f` the netrc file and log that it did so — a box
    with a leftover file from an earlier boot is a box where the helper is
    installed and never consulted (measured on the dashboard box, I9739)."""
    src = _src()
    assert f'NETRC="{_NETRC_LITERAL}"' in src
    assert f'rm -f "$NETRC"' in src
    # Must be unconditional at the shell level: only gated on the file
    # existing (so `rm -f` doesn't spuriously log every boot), never gated on
    # whether a fresh token could be minted, an SSM read succeeded, or any
    # other condition that could leave a stale file in place.
    removal_idx = src.index('rm -f "$NETRC"')
    guard_window = src[max(0, removal_idx - 200) : removal_idx]
    assert re.search(r'if \[ -e "\$NETRC" \]; then', guard_window), (
        "the removal must be guarded only on file existence, not on any "
        "other condition (e.g. a successful SSM read)."
    )


def test_credential_helper_is_asserted_before_the_repo_loop():
    """The helper's presence and a live --check must be verified BEFORE the
    REPOS pull loop, so a box that cannot authenticate names that as the
    cause rather than surfacing as a generic 403 deep in the git-sync retry
    logic for alpha-engine-config specifically."""
    src = _src()
    assert f'HELPER="{_HELPER_PATH}"' in src

    helper_check_idx = src.index(f'HELPER="{_HELPER_PATH}"')
    # "\nREPOS=(" and not "REPOS=(" — the latter also matches inside
    # "FAILED_REPOS=()", which this script (deliberately) declares earlier
    # than either of these.
    repos_array_idx = src.index("\nREPOS=(")
    assert helper_check_idx < repos_array_idx, (
        "the credential helper assertion must run before the REPOS pull loop "
        "is even defined, so it is unconditionally the first thing boot-pull "
        "checks after removing the netrc file."
    )


def test_credential_helper_missing_is_fail_loud():
    """A box with no helper installed at all must not pass silently — this
    is the FAIL-LOUD condition the issue requires, reusing boot-pull's
    existing PULL_FAILURES/FAILED_REPOS accumulator rather than inventing a
    second failure-reporting mechanism."""
    src = _src()
    assert '[ ! -x "$HELPER" ]' in src
    missing_branch = src[src.index('[ ! -x "$HELPER" ]') :]
    missing_branch = missing_branch[: missing_branch.index("elif")]
    assert 'log "FAIL' in missing_branch
    assert "PULL_FAILURES=$((PULL_FAILURES + 1))" in missing_branch
    assert "FAILED_REPOS+=(" in missing_branch


def test_credential_helper_check_failure_is_fail_loud():
    """A box with the helper installed but unable to mint a token (e.g. the
    App installation lost read on this repo, or the instance role lost
    ssm:GetParameter) must also increment the same failure accumulator."""
    src = _src()
    assert f'"$HELPER" --check alpha-engine-config' in src
    check_branch = src[src.index(f'"$HELPER" --check alpha-engine-config') :]
    check_branch = check_branch[: check_branch.index("else")]
    assert 'log "FAIL' in check_branch
    assert "PULL_FAILURES=$((PULL_FAILURES + 1))" in check_branch
    assert "FAILED_REPOS+=(" in check_branch


def test_pull_failures_accumulator_is_declared_before_the_credential_check():
    """PULL_FAILURES/FAILED_REPOS moved up (from below the REPOS loop) so the
    credential-helper assertion — now the first possible failure in the
    script — can use the SAME counter every later failure uses instead of a
    second, parallel mechanism."""
    src = _src()
    decl_idx = src.index("PULL_FAILURES=0")
    helper_idx = src.index(f'HELPER="{_HELPER_PATH}"')
    assert decl_idx < helper_idx, (
        "PULL_FAILURES=0 / FAILED_REPOS=() must be declared before the "
        "credential helper assertion uses them."
    )
    # Declared exactly once — a stray second declaration further down would
    # silently reset the counter after the credential check already
    # incremented it, erasing that failure before the end-of-script report.
    assert src.count("PULL_FAILURES=0") == 1


def test_no_second_failure_reporting_mechanism_was_invented():
    """The credential assertion must route through log()/PULL_FAILURES/
    FAILED_REPOS — the script's one existing failure-accumulation idiom —
    not a bespoke exit or a separate alert call of its own."""
    src = _src()
    helper_block_start = src.index(f'HELPER="{_HELPER_PATH}"')
    repos_array_idx = src.index("REPOS=(")
    helper_block = src[helper_block_start:repos_array_idx]
    assert "exit 1" not in helper_block, (
        "the credential assertion must not exit the script directly — it "
        "must fall through to the existing end-of-script PULL_FAILURES check "
        "so the rest of boot-pull (systemd sync, trades.db restore) still "
        "runs on a box that can't authenticate."
    )
    assert "krepis.alerts" not in helper_block, (
        "alerting is the end-of-script block's job, driven off "
        "PULL_FAILURES/FAILED_REPOS — the credential check must not publish "
        "its own alert."
    )
