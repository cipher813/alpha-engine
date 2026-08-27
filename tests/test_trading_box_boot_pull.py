"""ae-trading boot-pull scope invariants.

The executor box must only git-sync repos the weekday/EOD Step Functions
actually invoke via SSM. Dashboard + backtester run on ae-dashboard and
Saturday spots — pulling them on trading wastes disk and pip time.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_BOOT_PULL = Path(__file__).parent.parent / "infrastructure" / "boot-pull.sh"
_LAUNCHER = Path(__file__).parent.parent / "infrastructure" / "boot-pull-launcher.sh"
_SERVICE_FILE = (
    Path(__file__).parent.parent / "infrastructure" / "systemd" / "boot-pull.service"
)
_INSTALL_SCRIPT = Path(__file__).parent.parent / "infrastructure" / "install-boot-pull.sh"
_SYNCED_REPO_PREFIX = "/home/ec2-user/alpha-engine"


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


def _exec_start_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("ExecStart="):
            return stripped
    raise AssertionError("no ExecStart= line found")


def test_boot_pull_unit_execstart_is_outside_the_synced_repo():
    """alpha-engine-config-I8734: boot-pull.service's ExecStart must not be a
    path under the alpha-engine checkout that boot-pull.sh itself hard-resets.

    boot-pull.sh's sync_repo_to_main() and its deploy-gate rollback both
    `git reset --hard` the SAME tree ExecStart used to point into
    (/home/ec2-user/alpha-engine/infrastructure/boot-pull.sh). Bash reads a
    running script incrementally and resumes at a byte offset after each
    command — a rewrite of the file underneath it silently resumes execution
    in the REPLACED content. ExecStart must instead point at a launcher
    installed outside the tree (infrastructure/install-boot-pull.sh installs
    infrastructure/boot-pull-launcher.sh to /usr/local/sbin).
    """
    exec_start = _exec_start_line(_SERVICE_FILE.read_text())
    path = exec_start.split("=", 1)[1].strip()
    assert not path.startswith(_SYNCED_REPO_PREFIX), (
        f"boot-pull.service ExecStart={path} is inside the synced alpha-engine "
        "checkout — bash resumes a running script at a byte offset, so a "
        "concurrent `git reset --hard` on this same file can silently swap "
        "the bytes bash is executing mid-run."
    )
    assert path == "/usr/local/sbin/boot-pull-launcher.sh"


def test_install_boot_pull_writes_the_same_execstart_as_the_committed_unit():
    """install-boot-pull.sh's heredoc-written unit must not drift from
    infrastructure/systemd/boot-pull.service — both hardcode ExecStart and
    a silent divergence would mean the installed unit on a freshly
    provisioned box differs from what's reviewed in this repo."""
    installer_src = _INSTALL_SCRIPT.read_text()
    heredoc_start = installer_src.index("cat > \"$SERVICE_FILE\"")
    heredoc = installer_src[heredoc_start:]
    installed_exec_start = _exec_start_line(heredoc)

    committed_exec_start = _exec_start_line(_SERVICE_FILE.read_text())
    assert installed_exec_start == committed_exec_start

    path = installed_exec_start.split("=", 1)[1].strip()
    assert not path.startswith(_SYNCED_REPO_PREFIX)


def test_install_boot_pull_installs_the_launcher_outside_the_repo():
    """install-boot-pull.sh must copy the launcher to /usr/local/sbin BEFORE
    the unit that ExecStarts it is written — a fresh box with the unit
    installed but no launcher copied would hard-fail every boot."""
    src = _INSTALL_SCRIPT.read_text()
    assert "/usr/local/sbin/boot-pull-launcher.sh" in src
    install_pos = src.index("install ")
    unit_write_pos = src.index("cat > \"$SERVICE_FILE\"")
    assert install_pos < unit_write_pos, (
        "the launcher must be installed to /usr/local/sbin BEFORE the "
        "systemd unit (which ExecStarts it) is written"
    )


def test_boot_pull_launcher_never_execs_the_in_repo_path_directly():
    """The launcher's whole purpose is to snapshot boot-pull.sh OUTSIDE the
    synced tree before exec'ing it. If it ever `exec`d the in-repo path
    directly, the byte-offset hazard this issue exists to close would still
    be live."""
    src = _LAUNCHER.read_text()
    assert 'SNAPSHOT="/home/ec2-user/.boot-pull-snapshot.sh"' in src
    assert 'exec "$SNAPSHOT"' in src
    assert 'exec "$SRC"' not in src


def test_boot_pull_launcher_snapshot_path_is_outside_the_synced_repo():
    src = _LAUNCHER.read_text()
    snapshot_line = next(
        line for line in src.splitlines() if line.strip().startswith("SNAPSHOT=")
    )
    snapshot_path = snapshot_line.split("=", 1)[1].strip().strip('"')
    assert not snapshot_path.startswith(_SYNCED_REPO_PREFIX), (
        f"launcher snapshot path {snapshot_path} is inside the synced repo — "
        "a concurrent git reset could still race the snapshot copy itself."
    )


def test_boot_pull_launcher_runs_and_execs_a_real_snapshot(tmp_path):
    """End-to-end: point the launcher at a fake repo + a fake HOME and
    confirm it copies boot-pull.sh to a snapshot outside the repo tree and
    execs it, rather than the in-repo file."""
    fake_repo = tmp_path / "alpha-engine" / "infrastructure"
    fake_repo.mkdir(parents=True)
    fake_boot_pull = fake_repo / "boot-pull.sh"
    fake_boot_pull.write_text("#!/bin/bash\necho RAN_FROM_SNAPSHOT\n")
    fake_boot_pull.chmod(0o755)

    fake_home = tmp_path / "home"
    fake_home.mkdir()

    launcher_src = _LAUNCHER.read_text()
    # Rebind the two hardcoded paths to the sandbox for this test only —
    # the production script hardcodes /home/ec2-user paths deliberately, so
    # the test substitutes tmp_path equivalents rather than parametrizing
    # the script itself.
    launcher_src = launcher_src.replace(
        'SRC="/home/ec2-user/alpha-engine/infrastructure/boot-pull.sh"',
        f'SRC="{fake_boot_pull}"',
    ).replace(
        'SNAPSHOT="/home/ec2-user/.boot-pull-snapshot.sh"',
        f'SNAPSHOT="{fake_home}/.boot-pull-snapshot.sh"',
    )
    sandboxed_launcher = tmp_path / "boot-pull-launcher.sh"
    sandboxed_launcher.write_text(launcher_src)
    sandboxed_launcher.chmod(0o755)

    result = subprocess.run(
        ["bash", str(sandboxed_launcher)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "RAN_FROM_SNAPSHOT" in result.stdout
    snapshot_path = fake_home / ".boot-pull-snapshot.sh"
    assert snapshot_path.is_file(), "launcher did not create the snapshot"
    assert not str(snapshot_path).startswith(str(fake_repo)), (
        "snapshot must live outside the repo tree"
    )
