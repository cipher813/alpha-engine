#!/bin/bash
# git-sync-lock.sh — single source of truth for the shared advisory flock
# EVERY git-writing script on a trading/dashboard EC2 box must acquire
# before touching a git checkout under /home/ec2-user (alpha-engine-config
# #1944).
#
# Source this file rather than hardcoding the lock path/wait constants —
# the mutex only serializes writers that flock the SAME inode, so a second
# copy of the literal is a second (silently non-cooperating) lock, not a
# harmless duplicate. See infrastructure/boot-pull.sh's header for the full
# incident history this lock exists to close:
#   - 2026-07-08 ne-preopen-trading FailExecution: boot-pull's
#     `git reset --hard` held alpha-engine-data/.git/index.lock while the
#     weekday SF's CodeFreshnessGate/ChronicGapSelfHeal (nousergon-data
#     infrastructure/step_function_daily.json) ran its own checkout/reset ->
#     "Another git process seems to be running" (exit 128) -> no orders
#     placed.
#   - 2026-07-28/07-30: even a bare `git fetch`'s own ref update to
#     refs/remotes/origin/main hit a compare-and-swap failure racing another
#     writer on this box — a fetch is a git WRITE (it mutates the
#     remote-tracking ref), not a read, and must take this lock too.
#   - 2026-08-27 20:07 UTC (crucible-dashboard sibling incident, same
#     class): two unsynchronised git writers on the dashboard box's
#     ~/metron collided on `refs/remotes/origin/main`, and the deploy died
#     before the deploy script even started — the commit sat undeployed for
#     five hours.
#
# Lock lives in ec2-user's HOME, not /var/lock: /var/lock -> /run/lock is
# root:root 0755, so a script running as ec2-user cannot create a lock file
# there, and some git writers on this box run via `sudo -u ec2-user`. Every
# actor flocks this path AS ec2-user, so opening it for the lock always
# succeeds regardless of which actor created the inode first.
#
# boot-pull.sh does NOT source this file — it deliberately keeps its own
# inline copy of these two lines. boot-pull.service execs a private
# snapshot of boot-pull.sh copied to /home/ec2-user/.boot-pull-snapshot.sh
# (infrastructure/boot-pull-launcher.sh, alpha-engine-config-I8734) so a
# concurrent `git reset --hard` can't rewrite the running script out from
# under bash; a `source` of a path relative to that snapshot's location
# would not resolve back into this repo tree. Every OTHER git-writing
# script in this repo (which run in place from the checkout, never
# snapshotted) sources this file so the lock path/wait stays pinned to one
# literal instead of three.
GIT_SYNC_LOCK="${AE_GIT_SYNC_LOCK:-/home/ec2-user/.ae-git-sync.lock}"
GIT_SYNC_LOCK_WAIT="${AE_GIT_SYNC_LOCK_WAIT:-150}"
