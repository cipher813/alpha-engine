"""alpha-engine-config-I9832 / I9829: the launcher must snapshot boot-pull.sh
from origin/main, not from the working tree.

Root cause this closes, measured 2026-09-02 on the trading box. The launcher
copied `infrastructure/boot-pull.sh` out of the checkout and exec'd it, and
the copied script then synced that checkout. So the code that ran on boot N
was whatever the tree held at the end of boot N-1 — every boot-pull change
was live one boot late, by construction, with no signal saying so:

    12:15:58 OK   ~/.netrc refreshed from SSM /alpha-engine/GITHUB_TOKEN
    12:16:07 FAIL /home/ec2-user/alpha-engine-config — sync rc=10
    12:16:09 OK   /home/ec2-user/alpha-engine — 045a0e6 fix(boot-pull): stop
                  hydrating the netrc file, assert the credential helper

crucible-executor-PR532 had merged nine hours earlier. It was pulled into the
tree two seconds after the failure it fixes, and `ne-preopen-trading-pipeline`
lost the session on the same 403.

The tests below run the real launcher against sandbox git repos, so they
exercise the fetch/show/fallback control flow rather than asserting over the
script's source text. `AE_LAUNCHER_*` exist for exactly this; production sets
none of them.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_LAUNCHER = Path(__file__).parent.parent / "infrastructure" / "boot-pull-launcher.sh"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _make_origin(tmp_path: Path, body: str) -> Path:
    """A real non-bare repo used as the fetch origin."""
    origin = tmp_path / "origin"
    (origin / "infrastructure").mkdir(parents=True)
    (origin / "infrastructure" / "boot-pull.sh").write_text(body)
    _git(origin, "init", "--quiet", "--initial-branch=main")
    _git(origin, "config", "user.email", "test@example.invalid")
    _git(origin, "config", "user.name", "test")
    _git(origin, "add", "-A")
    _git(origin, "commit", "--quiet", "-m", "initial")
    return origin


def _make_clone(tmp_path: Path, origin: Path) -> Path:
    clone = tmp_path / "alpha-engine"
    subprocess.run(
        ["git", "clone", "--quiet", str(origin), str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )
    return clone


def _run_launcher(tmp_path: Path, repo: Path, snapshot: Path, env_extra=None):
    """Run the real launcher with its two hardcoded paths rebound.

    `run_git` shells out to `sudo -u ... flock ... git`, which cannot run
    unprivileged in CI, so AE_LAUNCHER_RUN_AS is neutralised by replacing the
    sudo/flock prefix with a direct git call. The fetch/show/compare/fallback
    logic under test is untouched by that substitution.
    """
    src = _LAUNCHER.read_text()
    src = src.replace(
        'SRC="/home/ec2-user/alpha-engine/infrastructure/boot-pull.sh"',
        f'SRC="{repo}/infrastructure/boot-pull.sh"',
    ).replace(
        'SNAPSHOT="/home/ec2-user/.boot-pull-snapshot.sh"',
        f'SNAPSHOT="{snapshot}"',
    ).replace(
        'sudo -u "$RUN_AS" -H flock -w 150 "$SYNC_LOCK" git -C "$REPO" "$@"',
        'git -C "$REPO" "$@"',
    )
    sandboxed = tmp_path / "boot-pull-launcher.sh"
    sandboxed.write_text(src)
    sandboxed.chmod(0o755)

    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
        "AE_LAUNCHER_REPO": str(repo),
    }
    if env_extra:
        env.update(env_extra)

    return subprocess.run(
        ["bash", str(sandboxed)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_launcher_runs_the_origin_version_not_the_stale_working_tree(tmp_path):
    """The regression itself: tree holds the old script, origin/main holds the
    fix, and the launcher must exec the fix on THIS boot."""
    origin = _make_origin(tmp_path, "#!/bin/bash\necho RAN_STALE\n")
    repo = _make_clone(tmp_path, origin)

    # main advances; the clone's working tree deliberately does not.
    (origin / "infrastructure" / "boot-pull.sh").write_text("#!/bin/bash\necho RAN_FIXED\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "--quiet", "-m", "the fix")

    snapshot = tmp_path / "snapshot.sh"
    result = _run_launcher(tmp_path, repo, snapshot)

    assert result.returncode == 0, result.stderr
    assert "RAN_FIXED" in result.stdout, (
        "launcher ran the working-tree copy; the one-boot lag is still open"
    )
    assert "RAN_STALE" not in result.stdout


def test_launcher_names_the_staleness_it_corrected(tmp_path):
    """Correcting silently is half a fix — the log must say the tree was behind."""
    origin = _make_origin(tmp_path, "#!/bin/bash\necho RAN_STALE\n")
    repo = _make_clone(tmp_path, origin)
    (origin / "infrastructure" / "boot-pull.sh").write_text("#!/bin/bash\necho RAN_FIXED\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "--quiet", "-m", "the fix")

    result = _run_launcher(tmp_path, repo, tmp_path / "snapshot.sh")

    assert "SNAPSHOT WAS STALE" in result.stderr, result.stderr


def test_launcher_is_quiet_when_the_tree_already_matches_origin(tmp_path):
    """No staleness line on a healthy boot, so the line means something."""
    origin = _make_origin(tmp_path, "#!/bin/bash\necho RAN_CURRENT\n")
    repo = _make_clone(tmp_path, origin)

    result = _run_launcher(tmp_path, repo, tmp_path / "snapshot.sh")

    assert result.returncode == 0, result.stderr
    assert "RAN_CURRENT" in result.stdout
    assert "SNAPSHOT WAS STALE" not in result.stderr


def test_launcher_falls_back_to_the_on_disk_copy_when_the_fetch_fails(tmp_path):
    """No network at boot must never block the box — it degrades to the old
    behaviour and says so."""
    origin = _make_origin(tmp_path, "#!/bin/bash\necho RAN_FROM_DISK\n")
    repo = _make_clone(tmp_path, origin)
    # Point origin at a path that does not exist, so `git fetch` fails the way
    # an unreachable network does.
    _git(repo, "remote", "set-url", "origin", str(tmp_path / "no-such-origin"))

    result = _run_launcher(tmp_path, repo, tmp_path / "snapshot.sh")

    assert result.returncode == 0, result.stderr
    assert "RAN_FROM_DISK" in result.stdout
    assert "falling back to the on-disk copy" in result.stderr


def test_launcher_never_mutates_the_working_tree(tmp_path):
    """Load-bearing: boot-pull.sh captures PREV_SHA *after* the launcher runs
    and its deploy gate rolls back to that SHA when the import smoke test
    fails. A launcher that reset the tree to origin/main would make PREV_SHA
    equal NEW_SHA and turn the rollback into a no-op — the deploy gate would
    restore the exact commit it was rejecting."""
    origin = _make_origin(tmp_path, "#!/bin/bash\necho RAN\n")
    repo = _make_clone(tmp_path, origin)
    head_before = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    (origin / "infrastructure" / "boot-pull.sh").write_text("#!/bin/bash\necho RAN_FIXED\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "--quiet", "-m", "the fix")

    result = _run_launcher(tmp_path, repo, tmp_path / "snapshot.sh")
    assert result.returncode == 0, result.stderr

    head_after = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head_before == head_after, (
        "launcher moved the checkout's HEAD; boot-pull's deploy-gate rollback "
        "target is destroyed by that"
    )


def test_launcher_rejects_an_empty_blob_from_origin(tmp_path):
    """An empty snapshot execs as a no-op and reports success — boot-pull would
    appear to have run and done nothing at all. Worse than running stale."""
    origin = _make_origin(tmp_path, "#!/bin/bash\necho RAN_FROM_DISK\n")
    repo = _make_clone(tmp_path, origin)
    (origin / "infrastructure" / "boot-pull.sh").write_text("")
    _git(origin, "add", "-A")
    _git(origin, "commit", "--quiet", "-m", "truncate")

    result = _run_launcher(tmp_path, repo, tmp_path / "snapshot.sh")

    assert result.returncode == 0, result.stderr
    assert "is empty" in result.stderr
    assert "RAN_FROM_DISK" in result.stdout


def test_launcher_source_reads_the_blob_rather_than_resetting(tmp_path):
    """Source-level pin for the property the runtime test above cannot see if
    someone later 'simplifies' the fetch into a reset."""
    src = _LAUNCHER.read_text()
    assert "git show" in src or 'show "origin/main' in src
    assert "reset --hard" not in src, (
        "the launcher must not reset the working tree — see "
        "test_launcher_never_mutates_the_working_tree"
    )
