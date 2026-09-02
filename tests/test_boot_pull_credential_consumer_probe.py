"""alpha-engine-config-I9829: what boot-pull asserts about GitHub auth, and
what it alerts on when a failure repeats at the box's own boot cadence.

Two root causes, both measured 2026-09-02 on the trading box.

1. `--check` proved the helper could MINT and proved nothing about what git
   sent. The helper was installed, configured, and minting an installation
   token whose installation covers exactly `nousergon/alpha-engine-config`
   (that token returns 200 on the repo's API). `GIT_TRACE` on the failing
   fetch shows git never invoked the helper at all: git sets `CURLOPT_NETRC`
   unconditionally, libcurl answered from the credential dotfile first, the
   dotfile's token is VALID but unpermitted so GitHub replied 403 rather than
   401, and with no 401 challenge git never had a reason to consult a helper.
   A minting probe and the consumer path can disagree indefinitely, and the
   minting probe is the one that looks healthy.

2. The failure alert deduped itself into silence. `boot-pull.service` exited 1
   on every boot from 08-31 22:14, and the alert published once:

       08-31 22:14  dedup_skipped=True
       08-31 22:29  dedup_skipped=True
       09-01 12:16  dedup_skipped=True   <- the day preopen lost a session
       09-02 12:16  sns.ok=True          <- first and only publish

   A 1440-minute window against a once-a-day boot cadence resolves on clock
   drift, and it resolves toward "suppress" exactly when the failure is
   persistent rather than transient.

Assertions are regexes over the script source, matching the convention the
other guards for this hardcoded-path script use.
"""
from __future__ import annotations

import re
from pathlib import Path

_BOOT_PULL = Path(__file__).parent.parent / "infrastructure" / "boot-pull.sh"


def _src() -> str:
    return _BOOT_PULL.read_text()


def test_credential_assertion_probes_the_consumer_path_not_only_minting():
    """A probe of the path the repo loop actually uses, against the one
    private repo, before any repo is pulled."""
    src = _src()
    assert "ls-remote" in src, (
        "boot-pull asserts only that the helper can mint; on 2026-09-02 that "
        "passed on every boot while every real fetch 403'd"
    )
    assert re.search(
        r"ls-remote.*\n?.*alpha-engine-config", src
    ) or "alpha-engine-config.git HEAD" in src, (
        "the consumer probe must target the private repo — the public ones "
        "succeeded throughout the outage"
    )


def test_consumer_probe_runs_before_the_repo_loop():
    """Ordering is the point: an opaque 403 deep in the git-sync retry logic
    is what this replaces."""
    src = _src()
    probe = src.index("ls-remote")
    loop = src.index("for repo in")
    assert probe < loop, "the consumer probe must precede the repo loop"


def test_new_probes_are_observe_only_on_merge():
    """sf-pipeline-policy.md §7a: a newly added check whose verdict can halt a
    scheduled-pipeline stage observes first. The .gitconfig condition is
    present on the trading box today (I9835), so promoting on merge would fail
    boot-pull every boot for an already-tracked finding."""
    src = _src()
    for marker in ("ls-remote", "GITCONFIG"):
        assert marker in src

    # Both probe branches log OBSERVE and neither increments the accumulator.
    observe_branches = re.findall(
        r"log \"OBSERVE[^\"]*\"", src
    )
    assert len(observe_branches) >= 2, (
        f"expected both new probes to log OBSERVE, found {len(observe_branches)}"
    )

    # The ls-remote branch must not fail the run yet. Anchor on the elif that
    # opens it, not on the first mention of "ls-remote" in the file — the
    # commentary above the block names it too.
    ls_start = src.index("elif ! sudo -u ec2-user -H git ls-remote")
    ls_branch = src[ls_start:src.index("Promotion criterion")]
    assert "PULL_FAILURES=$((PULL_FAILURES + 1))" not in ls_branch, (
        "the ls-remote probe is enforcing on merge, violating §7a observe-first"
    )

    gc_branch = src[src.index("GITCONFIG="):]
    gc_branch = gc_branch[: gc_branch.index("\nfi\n") + 4]
    assert "PULL_FAILURES=$((PULL_FAILURES + 1))" not in gc_branch, (
        "the .gitconfig probe is enforcing on merge; the condition it detects "
        "is live on the box today (I9835), so this would be a self-inflicted red"
    )


def test_observe_mode_probes_carry_a_promotion_criterion():
    """§7a requires the promotion criterion to live in the guard's own module,
    not in a ticket, so the next reader can act on it."""
    src = _src()
    assert "Promotion criterion" in src
    assert "Re-exam:" in src, "an observe-mode guard with no re-exam date observes forever"
    assert "I9835" in src, "the .gitconfig promotion blocker must be named"


def test_gitconfig_probe_never_echoes_the_matched_secret():
    """The finding is the file and the line number. The value is a live
    credential and this log is read by agents."""
    src = _src()
    gc_branch = src[src.index("GITCONFIG="):]
    gc_branch = gc_branch[: gc_branch.index("\nfi\n") + 4]
    assert "cut -d: -f1" in gc_branch, (
        "the .gitconfig probe must reduce grep -n output to line numbers only"
    )
    # The log line interpolates line numbers, never the grep match itself.
    assert "_gc_lines" in gc_branch
    assert "grep -oE" not in gc_branch, "printing the matched token into the log"


def test_failure_alert_dedup_key_is_dated():
    """Without a date, a failure recurring at the once-a-day boot cadence
    collides with its own previous publish inside the 1440-minute window."""
    src = _src()
    dkey_line = next(
        line for line in src.splitlines() if line.strip().startswith("_dkey=")
    )
    assert "date -u" in dkey_line, (
        "dedup key carries no date; a daily-cadence failure is invisible after "
        "its first publish"
    )
    assert "%Y-%m-%d" in dkey_line


def test_dedup_window_still_suppresses_multiple_boots_in_one_day():
    """The box can boot more than once a day (08-31 booted at 22:14 and 22:29).
    Dating the key must not turn one failure into an alert per boot."""
    src = _src()
    assert "--dedup-window-min 1440" in src, (
        "the window is what collapses several boots on the same day into one "
        "alert; the date scopes it, it does not replace it"
    )
