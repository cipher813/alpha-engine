#!/bin/bash
# One-shot validation: executor upstream artifact-freshness gate (config#1725 Phase A).
# Runs `executor/main.py --dry-run` on ae-trading after weekday MorningEnrich +
# PredictorInference have populated the three gated deliverables.
#
# Invoked by upstream-gate-dryrun-validation.timer (2026-07-07 14:15 UTC) or manually:
#   bash infrastructure/ops/upstream-gate-dryrun-validation.sh

set -eo pipefail

REPO="/home/ec2-user/alpha-engine"
LOG="/var/log/upstream-gate-validation.log"
exec >>"$LOG" 2>&1

echo "=== upstream gate dry-run validation $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

cd "$REPO"

# shellcheck disable=SC1091
source "$REPO/infrastructure/lib/git-sync-lock.sh"

# alpha-engine-config#1944 class, closed here for this script: this pull used
# to be a bare `git pull --ff-only origin main`, unlocked, while boot-pull.sh
# (systemd, on boot) and the weekday SF's CodeFreshnessGate/ChronicGapSelfHeal
# already serialize every OTHER git writer on this SAME
# /home/ec2-user/alpha-engine checkout behind $GIT_SYNC_LOCK. An unlocked
# writer can still race any of them on refs/remotes/origin/main or
# .git/index.lock — the same class measured on the sibling dashboard box
# 2026-08-27 20:07 UTC: two unsynchronised git writers collided on
# `refs/remotes/origin/main`, the deploy died before the deploy script even
# started, and the commit sat undeployed for five hours. This routes through
# the SAME lock inode boot-pull.sh already uses (never a second lock), and
# does not change what this script pulls or when it runs, only serializes it
# against every other writer on the box.
#
# Fail loud: a flock timeout or a failed pull both exit non-zero here (no
# `|| true`), which fails this systemd unit and shows up in `systemctl
# status` / the journal, plus this script's own $LOG (uploaded to S3 below).
if ! flock -w "$GIT_SYNC_LOCK_WAIT" "$GIT_SYNC_LOCK" git pull --ff-only origin main; then
    echo "FAIL upstream-gate-dryrun-validation — git-sync flock/pull failed on $GIT_SYNC_LOCK after ${GIT_SYNC_LOCK_WAIT}s (either a git writer is stuck, or the ff-only pull itself failed/diverged)" >&2
    exit 1
fi

export FLOW_DOCTOR_ENABLED=1
export ALPHA_ENGINE_DEPLOYED=1
export PYTHONPATH="$REPO"
export AWS_REGION="${AWS_REGION:-us-east-1}"

set -a
# shellcheck disable=SC1091
source /home/ec2-user/.alpha-engine.env
set +a

source "$REPO/.venv/bin/activate"
"$REPO/infrastructure/wait-for-ibgateway.sh"

python "$REPO/executor/main.py" --dry-run
rc=$?

s3_key="_ssm_logs/upstream-gate-validation/$(date -u +%Y-%m-%d)/$(hostname)-$(date -u +%H%M%SZ).log"
aws s3 cp "$LOG" "s3://alpha-engine-research/${s3_key}" --only-show-errors || true
echo "=== exit $rc (log s3://${s3_key}) ==="
exit "$rc"
