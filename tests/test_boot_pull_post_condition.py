"""boot-pull's git-sync decision, exercised by RUNNING it (config-I4978).

Every other guard on this logic is a regex over `infrastructure/boot-pull.sh`'s
source text, and that is exactly how the defect these tests cover shipped and
then survived a review: a substring assertion cannot see control flow. These
tests build real git repositories, drive `sync_repo_to_main` against them, and
assert on its return code.

The failure being pinned. `git fetch` exits non-zero when ANY ref fails to
update. boot-pull chained `git fetch ... && git checkout -f main && git reset
--hard origin/main`, so a ref-update failure skipped the self-heal and reported
"the executor may be running stale code" — on every weekday boot from 07-22
through 07-28. The ref observed failing its compare-and-swap was
`refs/remotes/origin/main` itself, while the same fetch had already
fast-forwarded it, so narrowing the refspec does not close this. Only judging
the post-condition does.

`test_stale_checkout_with_unreachable_remote_fails` is the counterpart that
keeps the fix honest: it pins that the guard still FAILS when the box really is
on stale code, which is the case a fix that merely stopped complaining would
break. Per I4978's closes-when, the guard must be verified to fail without the
fix, not merely to pass with it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_BOOT_PULL = Path(__file__).parent.parent / "infrastructure" / "boot-pull.sh"

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _git(*args: str, cwd: Path) -> str:
    """Run a real git command, raising on failure."""
    res = subprocess.run(
        ["git", *args], cwd=cwd, env={**os.environ, **_GIT_ENV},
        capture_output=True, text=True, check=True,
    )
    return res.stdout.strip()


@pytest.fixture
def box(tmp_path: Path):
    """A remote at two commits and a `box` checkout sitting on the FIRST.

    Mirrors the trading box mid-boot: the remote has moved on, and the boot-pull
    run under test is what is supposed to close the gap.
    """
    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git("init", "--bare", "--initial-branch=main", ".", cwd=remote)

    seed = tmp_path / "seed"
    seed.mkdir()
    _git("init", "--initial-branch=main", ".", cwd=seed)
    (seed / "f.txt").write_text("v1\n")
    _git("add", "f.txt", cwd=seed)
    _git("commit", "-m", "v1", cwd=seed)
    _git("remote", "add", "origin", str(remote), cwd=seed)
    _git("push", "origin", "main", cwd=seed)

    checkout = tmp_path / "box"
    _git("clone", str(remote), str(checkout), cwd=tmp_path)
    old_sha = _git("rev-parse", "HEAD", cwd=checkout)

    # The remote moves on, so `box` is now one commit stale.
    (seed / "f.txt").write_text("v2\n")
    _git("commit", "-am", "v2", cwd=seed)
    _git("push", "origin", "main", cwd=seed)
    new_sha = _git("rev-parse", "HEAD", cwd=seed)

    assert old_sha != new_sha
    return {"checkout": checkout, "old_sha": old_sha, "new_sha": new_sha,
            "remote": remote, "tmp": tmp_path}


def _shim_dir(tmp_path: Path, name: str) -> Path:
    """A PATH-prepended directory, carrying a `flock` stand-in where needed.

    `flock(1)` is util-linux and absent on macOS, where boot-pull's sync would
    otherwise die at 127 and every test below would pass or fail for the wrong
    reason. The stand-in drops `-w <secs>` and the lock path and execs the rest,
    so the sync's decision logic — the thing under test — runs identically on
    both platforms. It is written ONLY when the real binary is missing, so CI
    (ubuntu-latest) still exercises real flock and the serialization it provides.
    """
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    if shutil.which("flock") is None:
        (d / "flock").write_text(
            "#!/bin/sh\n"
            'while [ "$1" = "-w" ]; do shift 2; done\n'
            "shift\n"
            'exec "$@"\n'
        )
        (d / "flock").chmod(0o755)
    return d


def _install_git_shim(tmp_path: Path, *, fail_every_fetch: bool) -> Path:
    """A PATH-shadowing `git` that fails `fetch`, passing everything else through.

    This reproduces the observed production condition rather than simulating it
    at a distance: `git fetch` exits non-zero with the real ref-lock message, and
    the caller must decide what that means.
    """
    real_git = shutil.which("git")
    assert real_git, "git must be on PATH"
    shim_dir = _shim_dir(tmp_path, f"shim-{int(fail_every_fetch)}")
    counter = shim_dir / "fetch_count"
    # `fail_every_fetch` picks which real-world cause is being modelled, and the
    # distinction is the whole point of the retry:
    #   False — a ref compare-and-swap race. The FIRST fetch fails; the retry
    #           succeeds, because the racing writer has finished. Production
    #           behaviour observed on the trading box.
    #   True  — an unreachable remote (auth/network). EVERY fetch fails, and no
    #           number of retries changes that.
    retry_branch = ("    exit 1\n" if fail_every_fetch else
                    f'    exec "{real_git}" "$@"\n')
    (shim_dir / "git").write_text(
        "#!/bin/sh\n"
        f'COUNT_FILE="{counter}"\n'
        'case " $* " in\n'
        "*\" fetch \"*)\n"
        '    n=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)\n'
        '    n=$((n + 1))\n'
        '    echo "$n" > "$COUNT_FILE"\n'
        '    if [ "$n" -eq 1 ]; then\n'
        "        echo \"error: cannot lock ref 'refs/remotes/origin/main'\" >&2\n"
        "        exit 1\n"
        "    fi\n"
        f"{retry_branch}"
        "    ;;\n"
        "esac\n"
        f'exec "{real_git}" "$@"\n'
    )
    (shim_dir / "git").chmod(0o755)
    return shim_dir


def _sync(checkout: Path, tmp_path: Path, *, shim: Path | None = None):
    """Source boot-pull.sh in lib-only mode and call sync_repo_to_main."""
    log = tmp_path / "boot-pull.log"
    env = {
        **os.environ, **_GIT_ENV,
        "AE_BOOT_PULL_LIB_ONLY": "1",
        "AE_BOOT_PULL_LOG": str(log),
        "AE_GIT_SYNC_LOCK": str(tmp_path / "sync.lock"),
        "AE_GIT_SYNC_LOCK_WAIT": "10",
    }
    if shim is not None:
        env["PATH"] = f"{shim}{os.pathsep}{os.environ['PATH']}"
    res = subprocess.run(
        ["bash", "-c", f'. "{_BOOT_PULL}"; sync_repo_to_main "$1"', "_", str(checkout)],
        env=env, capture_output=True, text=True,
    )
    return res.returncode, (log.read_text() if log.exists() else "")


def test_clean_boot_fast_forwards_to_remote_main(box):
    """The baseline: a stale checkout is brought to the remote tip and passes."""
    rc, log = _sync(box["checkout"], box["tmp"],
                    shim=_shim_dir(box["tmp"], "shim-clean"))
    assert rc == 0, f"clean sync must succeed; log:\n{log}"
    assert _git("rev-parse", "HEAD", cwd=box["checkout"]) == box["new_sha"]
    assert "OK   " in log


def test_benign_ref_cas_failure_still_self_heals(box):
    """THE REGRESSION. A `git fetch` that fails a ref compare-and-swap must not
    suppress the reset or raise a stale-code failure — the retry gets main.

    Under the `&&` chain this returns non-zero with the checkout left on the old
    commit: a false "executor may be running stale code" alert on every weekday
    boot, and the self-heal genuinely skipped. Measured on the trading box
    2026-07-30, `refs/remotes/origin/main` itself was among the failing refs.
    """
    shim = _install_git_shim(box["tmp"], fail_every_fetch=False)
    rc, log = _sync(box["checkout"], box["tmp"], shim=shim)

    assert rc == 0, f"a ref-CAS race must not FAIL the repo; log:\n{log}"
    assert _git("rev-parse", "HEAD", cwd=box["checkout"]) == box["new_sha"], (
        "the self-heal must still run — this is the part that was skipped"
    )


def test_unreachable_remote_fails_even_though_head_matches_origin(box):
    """The false OK that matters most.

    When the fetch cannot reach the remote, `origin/main` is stale too, so
    `HEAD == origin/main` is satisfied by two equally-old LOCAL refs. A guard
    that stopped at the post-condition would report OK on a genuinely stale box
    — the precise case this alert exists for. Retrying and still failing is what
    makes it detectable: "cannot verify" must never read as "safe".
    """
    shim = _install_git_shim(box["tmp"], fail_every_fetch=True)
    rc, log = _sync(box["checkout"], box["tmp"], shim=shim)

    assert rc != 0, f"unprovable freshness must FAIL; log:\n{log}"
    assert "cannot be shown to be current" in log
    # The checkout is left on the OLD commit, and HEAD == origin/main holds —
    # which is exactly why the post-condition alone cannot catch this.
    assert _git("rev-parse", "HEAD", cwd=box["checkout"]) == box["old_sha"]
    assert (_git("rev-parse", "HEAD", cwd=box["checkout"])
            == _git("rev-parse", "origin/main", cwd=box["checkout"]))


def test_wrong_branch_fails_even_when_sync_reports_success(box):
    """The post-condition's other direction: a detached HEAD at the right commit
    is still not "on main", and boot policy is that the box tracks main.
    """
    checkout = box["checkout"]
    _git("fetch", "origin", "main", cwd=checkout)
    _git("checkout", "--detach", box["new_sha"], cwd=checkout)
    # A shim whose checkout is a no-op: the sync "succeeds" without moving HEAD
    # back onto main, which is the state the exit code cannot distinguish.
    real_git = shutil.which("git")
    shim_dir = _shim_dir(box["tmp"], "shim-nocheckout")
    (shim_dir / "git").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "checkout" ] || [ "$1" = "reset" ]; then exit 0; fi\n'
        f'exec "{real_git}" "$@"\n'
    )
    (shim_dir / "git").chmod(0o755)

    rc, log = _sync(checkout, box["tmp"], shim=shim_dir)
    assert rc != 0, f"a detached HEAD must FAIL the post-condition; log:\n{log}"
    assert "post-condition failed" in log
