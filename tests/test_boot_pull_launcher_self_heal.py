"""Regression: boot-pull.sh must self-heal its own launcher binary.

alpha-engine-config-I9444. boot-pull.sh already self-heals the unit file at
/etc/systemd/system/boot-pull.service on every boot (sync_systemd_units_from
below re-copies it from the repo). It had NO equivalent self-heal for the
launcher BINARY at /usr/local/sbin/boot-pull-launcher.sh — a two-artifact
install (unit file + out-of-tree exec target, config-I8734) where only one
artifact self-healed.

Measured 2026-08-31 on the trading box (i-018eb3307a21329bf): the launcher
was never provisioned there when crucible-executor#495 shipped, and
boot-pull.service FAILed at EVERY boot from 2026-08-28 through 2026-08-31
with `status=203/EXEC: Failed to locate executable
/usr/local/sbin/boot-pull-launcher.sh` — invisible except two levels
downstream, via a systemd-unit-drift alert on unrelated units. Remediated
live via the existing install-boot-pull.sh; this test pins the fix so a
future occurrence of the same partial-apply failure repairs itself on the
box's own next boot instead of waiting on a human to notice.

Source-text assertions, consistent with test_boot_pull_timer_reconcile.py
and test_trading_box_boot_pull.py in this directory — a full functional
exercise of this block would require sudo and a real /usr/local/sbin write,
which is exactly the surface boot-pull.sh itself only ever runs on the box.
"""
from __future__ import annotations

from pathlib import Path

_BOOT_PULL = Path(__file__).parent.parent / "infrastructure" / "boot-pull.sh"
_INSTALL_SCRIPT = Path(__file__).parent.parent / "infrastructure" / "install-boot-pull.sh"


def _source() -> str:
    return _BOOT_PULL.read_text()


def test_boot_pull_exists():
    assert _BOOT_PULL.exists(), f"boot-pull.sh missing at {_BOOT_PULL}"


def test_launcher_self_heal_block_exists():
    """boot-pull.sh must verify+repair the launcher on every run."""
    src = _source()
    assert 'LAUNCHER_SRC="/home/ec2-user/alpha-engine/infrastructure/boot-pull-launcher.sh"' in src, (
        "launcher self-heal must read from the in-repo launcher source, "
        "mirroring install-boot-pull.sh's LAUNCHER_SRC."
    )
    assert 'LAUNCHER_DST="/usr/local/sbin/boot-pull-launcher.sh"' in src, (
        "launcher self-heal must target /usr/local/sbin/boot-pull-launcher.sh "
        "— the exact path boot-pull.service's ExecStart points at "
        "(config-I8734) and install-boot-pull.sh's LAUNCHER_DST."
    )


def test_launcher_self_heal_checks_hash_not_just_existence():
    """A stale (edited-in-place, or older-commit) launcher must also be
    repaired, not only a missing one — a bare `[ ! -f "$LAUNCHER_DST" ]`
    check would silently leave a mismatched launcher in place forever."""
    src = _source()
    assert 'cmp -s "$LAUNCHER_SRC" "$LAUNCHER_DST"' in src, (
        "launcher self-heal must byte-compare the installed launcher "
        "against the repo copy (cmp -s), not merely check it exists."
    )


def test_launcher_self_heal_installs_with_matching_mode_and_owner():
    """The repair must use the SAME mode/owner as install-boot-pull.sh's
    one-time install, or a box that only ever gets healed by boot-pull.sh
    (never runs install-boot-pull.sh again) ends up with a launcher whose
    permissions have silently drifted from the documented contract."""
    src = _source()
    assert 'install -m 0755 -o root -g root "$LAUNCHER_SRC" "$LAUNCHER_DST"' in src, (
        "launcher repair must use `install -m 0755 -o root -g root "
        '"$LAUNCHER_SRC" "$LAUNCHER_DST"` — identical mode/owner to '
        "install-boot-pull.sh's own install(1) call."
    )


def test_launcher_self_heal_counts_failure_loudly():
    """A repair failure must be counted in PULL_FAILURES / FAILED_REPOS —
    per ~/Development/CLAUDE.md's fail-loud rule, not a silent WARN that
    leaves the box's only reconciliation path broken with no signal."""
    src = _source()
    heal_start = src.index('LAUNCHER_SRC="/home/ec2-user/alpha-engine/infrastructure/boot-pull-launcher.sh"')
    heal_end = src.index("# Sync systemd service files from repo")
    heal_block = src[heal_start:heal_end]
    assert "PULL_FAILURES=$((PULL_FAILURES + 1))" in heal_block, (
        "a failed launcher repair must increment PULL_FAILURES so it "
        "surfaces in boot-pull's end-of-run loud failure summary."
    )
    assert "FAILED_REPOS+=(" in heal_block, (
        "a failed launcher repair must be named in FAILED_REPOS."
    )


def test_launcher_self_heal_runs_unconditionally_every_boot():
    """Must NOT be gated on SYNC_OK or a code-changed check (PREV_SHA !=
    NEW_SHA) — those variables belong to the per-repo pull loop above and
    are out of scope by the time this block runs; the launcher can drift
    (or never have been installed) independently of whether THIS boot's
    pull changed anything, so the check must run every single boot."""
    src = _source()
    heal_start = src.index('LAUNCHER_SRC="/home/ec2-user/alpha-engine/infrastructure/boot-pull-launcher.sh"')
    heal_end = src.index("# Sync systemd service files from repo")
    heal_block = src[heal_start:heal_end]
    assert "SYNC_OK" not in heal_block
    assert "PREV_SHA" not in heal_block
    assert "NEW_SHA" not in heal_block


def test_launcher_self_heal_precedes_systemd_unit_sync():
    """Order the launcher (exec target) heal alongside the unit-file heal
    it mirrors, so a reader sees boot-pull's own two-artifact reconciliation
    as one unit rather than the launcher fix landing somewhere unrelated."""
    src = _source()
    heal_pos = src.index('LAUNCHER_DST="/usr/local/sbin/boot-pull-launcher.sh"')
    systemd_sync_call_pos = src.index(
        'sync_systemd_units_from "/home/ec2-user/alpha-engine/infrastructure/systemd"'
    )
    assert heal_pos < systemd_sync_call_pos


def test_install_boot_pull_launcher_paths_match_self_heal():
    """install-boot-pull.sh's LAUNCHER_SRC/LAUNCHER_DST must be the exact
    same paths boot-pull.sh's self-heal uses, or the one-time install and
    the every-boot repair converge on different targets."""
    boot_pull_src = _source()
    installer_src = _INSTALL_SCRIPT.read_text()
    assert 'LAUNCHER_SRC="/home/ec2-user/alpha-engine/infrastructure/boot-pull-launcher.sh"' in installer_src
    assert 'LAUNCHER_DST="/usr/local/sbin/boot-pull-launcher.sh"' in installer_src
    assert 'LAUNCHER_SRC="/home/ec2-user/alpha-engine/infrastructure/boot-pull-launcher.sh"' in boot_pull_src
    assert 'LAUNCHER_DST="/usr/local/sbin/boot-pull-launcher.sh"' in boot_pull_src
