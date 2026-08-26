"""ae-trading boot-pull scope invariants.

The executor box must only git-sync repos the weekday/EOD Step Functions
actually invoke via SSM. Dashboard + backtester run on ae-dashboard and
Saturday spots — pulling them on trading wastes disk and pip time.
"""
from __future__ import annotations

from pathlib import Path

_BOOT_PULL = Path(__file__).parent.parent / "infrastructure" / "boot-pull.sh"


def test_boot_pull_excludes_dashboard_and_backtester():
    src = _BOOT_PULL.read_text()
    assert "/home/ec2-user/alpha-engine-config" in src
    assert "/home/ec2-user/alpha-engine" in src
    assert "/home/ec2-user/alpha-engine-data" in src
    assert "/home/ec2-user/alpha-engine-dashboard" not in src
    assert "/home/ec2-user/alpha-engine-backtester" not in src


def test_trading_box_cleanup_script_exists():
    path = Path(__file__).parent.parent / "infrastructure" / "trading-box-cleanup.sh"
    assert path.is_file()
    text = path.read_text()
    assert "alpha-engine-dashboard" in text
    assert "predictor" in text


def test_boot_pull_reclaims_foreign_owned_files_before_git_reset():
    """Ownership reclaim must precede the git fetch/reset block.

    2026-07-06 incident (config#1811): a feature branch's sudo timer-install
    step left infrastructure/ops/ root-owned inside the ec2-user checkout;
    `git reset --hard origin/main` failed with "unable to unlink ...
    Permission denied" on every boot, the box silently ran 4-commits-stale
    code on a stray branch, and the day's pipeline burned ~40 min before the
    executor's deploy-drift preflight refused. The reclaim block makes that
    failure mode structurally impossible; this test pins its presence AND
    its ordering (reclaim before the sync — after would be useless).

    Ordering is measured against the per-repo loop's CALL to
    sync_repo_to_main, not against the first textual occurrence of `git reset
    --hard origin/main`. Since config-I4978 extracted the sync into a function
    defined above the loop, that string now appears near the top of the file
    and a raw text index would compare the reclaim against a function
    *definition* rather than against the point the reset actually runs —
    silently inverting this guard's meaning while still passing.
    """
    src = _BOOT_PULL.read_text()
    assert "-not -user ec2-user" in src, "foreign-ownership detection missing"
    assert 'chown -R ec2-user:ec2-user "$repo"' in src, "ownership reclaim missing"
    reclaim_pos = src.index("-not -user ec2-user")
    sync_call_pos = src.index('sync_repo_to_main "$repo"')
    assert reclaim_pos < sync_call_pos, (
        "ownership reclaim must run BEFORE the git sync"
    )


def test_boot_pull_git_sync_runs_under_shared_flock():
    """config#1944: the per-repo git fetch/checkout/reset must run under a
    shared advisory flock so boot-pull.service can't race the weekday
    CodeFreshnessGate / ChronicGapSelfHeal (nousergon-data
    step_function_daily.json) on .git/index.lock.

    2026-07-08 ne-preopen-trading FailExecution: boot-pull's `git reset --hard`
    held alpha-engine-data/.git/index.lock while the gate's checkout/reset ran
    -> "Another git process seems to be running" (exit 128) -> no orders placed.
    The flock is window-free (kernel mutex) and auto-releases on process death.
    This pins that a future edit can't silently drop back to bare, race-prone
    git calls.
    """
    import re

    src = _BOOT_PULL.read_text()
    # The lock must live in ec2-user's HOME, not /var/lock: /var/lock ->
    # /run/lock is root:root 0755, so an ec2-user boot-pull cannot create a
    # lock file there. The nousergon-data gate flocks this SAME path.
    assert "/home/ec2-user/.ae-git-sync.lock" in src, (
        "git-sync flock must use the shared /home/ec2-user/.ae-git-sync.lock "
        "path (the nousergon-data CodeFreshnessGate uses the identical inode)."
    )
    # A bounded flock must wrap the index-mutating reset (window-free), and the
    # bound must be > boot-pull's own 120s TimeoutStartSec so a genuinely stuck
    # writer fails loud rather than the flock timing out prematurely.
    assert re.search(r"flock -w \S+ \S+ bash -c '[^']*git reset --hard origin/main", src), (
        "the git fetch/checkout/reset group must run under `flock -w <wait> "
        "<lock> bash -c '...'` so the whole index mutation is serialized."
    )


def test_boot_pull_git_sync_lock_wait_exceeds_boot_pull_timeout():
    """The flock wait budget must exceed boot-pull.service's TimeoutStartSec
    (120s) so the gate can outwait a full boot-pull git-sync rather than the
    flock timing out and failing a healthy run."""
    src = _BOOT_PULL.read_text()
    assert 'GIT_SYNC_LOCK_WAIT="${AE_GIT_SYNC_LOCK_WAIT:-150}"' in src, (
        "flock wait must default to 150s (> boot-pull.service TimeoutStartSec=120)."
    )


def test_boot_pull_deploy_gate_uses_import_smoke_test():
    """config#2353: the deploy gate must perform a full import smoke test
    (not just ast.parse) to catch ImportErrors in transitive modules.

    An ImportError in any of the ~50 executor modules (not just the three
    entrypoints main/daemon/eod_reconcile) would previously pass the AST-only
    check and surface at planner runtime. The import test pulls the full
    dependency graph, catching broken deps pre-deployment.
    """
    src = _BOOT_PULL.read_text()
    # The gate must use import, not ast.parse
    assert "import executor.main, executor.daemon, executor.eod_reconcile" in src, (
        "deploy gate must use Python imports (not ast.parse) to catch transitive "
        "ImportErrors in executor modules."
    )
    # Verify the old ast.parse check is removed
    assert "ast.parse" not in src, (
        "obsolete ast.parse syntax-only check must be removed (config#2353)."
    )


def test_boot_pull_deploy_gate_runs_after_dependency_install():
    """alpha-engine-config-I8682: the import smoke test must run AFTER the pip
    install, never before it.

    Ordered the other way, the gate tests the NEW commit's code against the
    PREVIOUS commit's installed dependency set, so any commit that bumps a pin
    and uses the new API fails by construction — rolls back, reinstalls the
    rolled-back manifest, and repeats identically at the next boot. Measured on
    the trading box 2026-08-25/26: crucible-executor #493 (krepis
    0.54.0 -> 0.59.33 plus `from krepis.trading_calendar import
    is_market_hours`) was rolled back on every boot for three days, and the
    2026-08-26 preopen failed at CodeFreshnessGate's CODE-STALE-AFTER-HEAL.
    """
    src = _BOOT_PULL.read_text()

    gate = src.index("import executor.main, executor.daemon, executor.eod_reconcile")
    pip_install = src.index('.venv/bin/pip install --quiet -r "$REQUIREMENTS_FILE"')

    assert pip_install < gate, (
        "the deploy gate's import smoke test runs BEFORE the pip install — it "
        "would test new code against the previous commit's dependencies, which "
        "makes every dependency-bumping commit permanently un-deployable."
    )


def test_boot_pull_deploy_gate_rollback_is_counted_as_a_failure():
    """alpha-engine-config-I8682: a rollback must increment PULL_FAILURES.

    The rollback is the one path that knowingly leaves the trading box on stale
    code for a live session, and it was the only path in this script that
    emitted no signal — no counter, no alert, exit 0. On 2026-08-25 the preopen
    went green and the session traded #492 while origin/main was #493, with
    nothing said anywhere.
    """
    src = _BOOT_PULL.read_text()

    rollback = src.index('log "ROLLBACK $repo')
    tail = src[rollback : rollback + 400]

    assert "PULL_FAILURES=$((PULL_FAILURES + 1))" in tail, (
        "the deploy-gate rollback does not increment PULL_FAILURES, so "
        "boot-pull exits 0 and its krepis.alerts publish never fires."
    )
    assert "deploy-gate rollback" in tail, (
        "the rollback must name itself in FAILED_REPOS — the alert body is "
        "built from that list, and 'alpha-engine (git)' would misreport a "
        "successful sync that was deliberately reverted."
    )


def test_boot_pull_deploy_gate_rollback_runs_under_shared_flock():
    """alpha-engine-config-I8682: the rollback's `git reset --hard` must take
    the same advisory lock as every other git writer on the box.

    Unlocked, it can land between the weekday CodeFreshnessGate's self-heal and
    that gate's re-check — which is exactly what decided the 2026-08-26 preopen
    failure and the 2026-08-25 silent stale-code session. The two are opposed
    writers; serialising them makes the outcome deterministic instead of a race.
    """
    src = _BOOT_PULL.read_text()

    rollback_block = src[src.index('log "FAIL $repo — import smoke test failed') :]
    rollback_block = rollback_block[: rollback_block.index('log "ROLLBACK $repo')]

    assert '"$GIT_SYNC_LOCK"' in rollback_block, (
        "the rollback reset runs outside the shared git flock and can race the "
        "weekday CodeFreshnessGate's heal."
    )
