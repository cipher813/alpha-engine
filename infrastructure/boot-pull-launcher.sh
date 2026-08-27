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
set -euo pipefail

SRC="/home/ec2-user/alpha-engine/infrastructure/boot-pull.sh"
SNAPSHOT="/home/ec2-user/.boot-pull-snapshot.sh"

if [ ! -f "$SRC" ]; then
    echo "boot-pull-launcher: $SRC not found — alpha-engine checkout missing or not yet cloned" >&2
    exit 1
fi

cp "$SRC" "$SNAPSHOT"
chmod 700 "$SNAPSHOT"
exec "$SNAPSHOT"
