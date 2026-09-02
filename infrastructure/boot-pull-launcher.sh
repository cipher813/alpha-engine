#!/bin/bash
# boot-pull-launcher.sh — stable launcher for boot-pull.sh, installed OUTSIDE
# the alpha-engine checkout (/usr/local/sbin, by install-boot-pull.sh) so
# systemd never execs a script from the tree that script itself rewrites.
#
# alpha-engine-config-I8734: boot-pull.service used to point ExecStart
# straight at infrastructure/boot-pull.sh INSIDE the repo boot-pull.sh
# synchronises. sync_repo_to_main() hard-resets that checkout to
# origin/main (and the deploy-gate rollback hard-resets it to $PREV_SHA)
# WHILE bash may still be executing boot-pull.sh from it. Bash does not
# read a script into memory up front — it reads incrementally and resumes
# at a byte offset after each command. If the file's bytes change
# underneath it, execution resumes at that offset in the REPLACED
# content, landing mid-token whenever the rewrite changes the file's
# length above that point. Silent, non-deterministic, depends on how far
# through the script the sync happens to land.
#
# This launcher is the fix: it is the ONLY thing systemd ever execs, it
# is installed to a path OUTSIDE any synced repo, and it never changes
# itself mid-run. On every boot it snapshots whatever version of
# boot-pull.sh currently lives in the repo to a private path also outside
# the synced tree, then execs THAT snapshot. Once the snapshot is taken,
# nothing sync_repo_to_main() or the rollback does to the repo's copy can
# reach the bytes bash is actually running.
#
# This file itself must NEVER be executed in place from the repo — it is
# copied to /usr/local/sbin/boot-pull-launcher.sh by
# infrastructure/install-boot-pull.sh, and boot-pull.service's ExecStart
# points there, not here. A test in tests/test_trading_box_boot_pull.py
# pins that both the unit file and the installer agree on that path.
#
# ── The snapshot is taken from origin/main, not from the working tree ─────
# alpha-engine-config-I9832 / I9829. Snapshotting whatever the tree happens
# to hold, and only then letting boot-pull.sh sync that tree, makes the code
# that runs on boot N the code the tree held at the end of boot N-1. Every
# boot-pull change therefore took effect ONE BOOT LATE, by construction, and
# nothing said so: the log recorded the pull that arrived too late as an OK.
#
# Measured 2026-09-02 on the trading box. crucible-executor-PR532 ("stop
# hydrating the netrc file, assert the credential helper", I9739) merged at
# 03:38 UTC. The 12:15 UTC boot ran the PRE-532 snapshot:
#
#   12:15:58 OK   ~/.netrc refreshed from SSM /alpha-engine/GITHUB_TOKEN
#   12:16:07 FAIL /home/ec2-user/alpha-engine-config — sync rc=10
#   12:16:09 OK   /home/ec2-user/alpha-engine — 045a0e6 fix(boot-pull): stop
#                 hydrating the netrc file, assert the credential helper
#
# The fix for the 12:16:07 failure was pulled in at 12:16:09, nine seconds
# too late to prevent it, and ne-preopen-trading-pipeline lost the session on
# the same 403 at 12:16:01. A fix that cannot land on the boot that pulls it
# is not deployed; it is queued.
#
# The fix reads the file to run straight out of origin/main and never touches
# the working tree:
#
#   git fetch origin main && git show origin/main:infrastructure/boot-pull.sh
#
# Reading rather than resetting is load-bearing, not a stylistic choice.
# boot-pull.sh captures `PREV_SHA=$(git rev-parse HEAD)` INSIDE its own sync
# loop, after this launcher has finished, and its deploy gate rolls the tree
# back to that SHA when the post-sync import smoke test fails. A launcher
# that hard-reset the tree to origin/main first would make PREV_SHA equal
# NEW_SHA, so the rollback would restore the very commit it was rejecting —
# silently converting the deploy gate into a no-op. Reading a blob out of the
# fetched ref changes no ref and no file, so every rollback target survives.
#
# Three properties make the fetch safe this early in boot:
#
#   1. This repo is PUBLIC, so the fetch needs no credential. That matters
#      precisely because the failure class this closes is a BROKEN credential
#      path — a self-repair gated on authentication cannot repair the boot
#      that broke authentication. It also runs BEFORE boot-pull.sh's own
#      credential-helper assertion, so it must not depend on it, and does not.
#   2. It runs under the same shared flock sync_repo_to_main() takes, so it
#      cannot interleave with boot-pull's own git on the same checkout.
#   3. It is best-effort in every branch. No network yet, not a git checkout,
#      no such user, an empty blob — each logs and falls through to the
#      on-disk copy, which is exactly the pre-I9832 behaviour. Boot-pull's own
#      sync still runs afterwards and its post-condition still reports a stale
#      tree. This must never be able to block a boot: a trading box that does
#      not come up costs a session, which is the cost this whole file exists
#      to avoid.
#
# It also emits the staleness signal that was missing on 2026-09-02. When the
# snapshot taken from origin/main differs from the on-disk copy, the log names
# both, so "the box was about to run code older than main" is one grep rather
# than an inference from adjacent timestamps.
set -euo pipefail

SRC="/home/ec2-user/alpha-engine/infrastructure/boot-pull.sh"
SNAPSHOT="/home/ec2-user/.boot-pull-snapshot.sh"
# Overridable ONLY so the test suite can point this at a sandbox checkout;
# production uses the defaults and the unit file sets none of them.
REPO="${AE_LAUNCHER_REPO:-/home/ec2-user/alpha-engine}"
REPO_PATH="${AE_LAUNCHER_REPO_PATH:-infrastructure/boot-pull.sh}"
SYNC_LOCK="${AE_GIT_SYNC_LOCK:-/home/ec2-user/.ae-git-sync.lock}"
RUN_AS="${AE_LAUNCHER_RUN_AS:-ec2-user}"

# git runs as the checkout's owner under the same lock boot-pull.sh takes, so
# this cannot race sync_repo_to_main() on the same tree. -w 150 matches
# boot-pull's own wait.
run_git() {
    sudo -u "$RUN_AS" -H flock -w 150 "$SYNC_LOCK" git -C "$REPO" "$@"
}

# Writes the origin/main copy of boot-pull.sh to $SNAPSHOT and returns 0, or
# returns non-zero having written nothing. Never exits the script: the caller
# falls back to the on-disk copy on any failure.
snapshot_from_origin() {
    local tmp
    tmp="${SNAPSHOT}.fetching.$$"

    run_git rev-parse --git-dir >/dev/null 2>&1 || {
        echo "boot-pull-launcher: $REPO is not a readable git checkout — falling back to the on-disk copy" >&2
        return 1
    }

    if ! run_git fetch --quiet origin main 2>/dev/null; then
        echo "boot-pull-launcher: fetch of origin/main failed (network not up yet?) — falling back to the on-disk copy; boot-pull's own sync retries and its post-condition still reports a stale tree" >&2
        return 1
    fi

    if ! run_git show "origin/main:${REPO_PATH}" > "$tmp" 2>/dev/null; then
        rm -f "$tmp"
        echo "boot-pull-launcher: origin/main has no ${REPO_PATH} — falling back to the on-disk copy" >&2
        return 1
    fi

    # A zero-length blob would exec as a no-op script and report success, which
    # is the one failure mode worse than running stale code: boot-pull would
    # appear to have run and done nothing at all.
    if [ ! -s "$tmp" ]; then
        rm -f "$tmp"
        echo "boot-pull-launcher: origin/main:${REPO_PATH} is empty — falling back to the on-disk copy" >&2
        return 1
    fi

    if [ -f "$SRC" ] && ! cmp -s "$tmp" "$SRC"; then
        # The staleness signal. Reaching this line means the pre-I9832 order
        # would have run the on-disk copy for this entire boot while a
        # different boot-pull.sh was already on main.
        echo "boot-pull-launcher: SNAPSHOT WAS STALE — origin/main:${REPO_PATH} differs from the on-disk copy at $SRC; running the origin/main version. The pre-I9832 order would have run the on-disk copy for this whole boot." >&2
    fi

    mv "$tmp" "$SNAPSHOT"
    return 0
}

if ! snapshot_from_origin; then
    if [ ! -f "$SRC" ]; then
        echo "boot-pull-launcher: $SRC not found and origin/main unreadable — alpha-engine checkout missing or not yet cloned" >&2
        exit 1
    fi
    cp "$SRC" "$SNAPSHOT"
fi

chmod 700 "$SNAPSHOT"
exec "$SNAPSHOT"
