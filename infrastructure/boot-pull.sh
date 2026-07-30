#!/bin/bash
# boot-pull.sh — Pull latest code for all Alpha Engine repos on EC2 boot.
#
# Runs as a systemd oneshot service (boot-pull.service) before any cron jobs.
# Safe to run manually too.
#
# Install:
#   sudo bash infrastructure/install-boot-pull.sh

set -uo pipefail

# AE_BOOT_PULL_LOG exists so tests can source this file and exercise
# sync_repo_to_main() without needing write access to /var/log.
LOG="${AE_BOOT_PULL_LOG:-/var/log/boot-pull.log}"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

# ── Shared git-sync serialization (config#1944) ────────────────────────────
# boot-pull.service and the weekday Step Function's CodeFreshnessGate +
# ChronicGapSelfHeal (nousergon-data infrastructure/step_function_daily.json)
# all run `git fetch / checkout -f main / reset --hard / pull` on the SAME
# ec2-user checkouts on THIS trading box. They are independent git writers and
# raced on `.git/index.lock`: 2026-07-08 ne-preopen-trading FailExecution —
# CodeFreshnessGate's checkout/reset died with "Another git process seems to be
# running" (exit 128) because boot-pull's `git reset --hard` still held
# alpha-engine-data/.git/index.lock.
#
# A shared advisory flock is window-free (a kernel mutex — whoever acquires
# first runs, the other blocks) and auto-releases on process death, unlike a
# bare .git/index.lock which can strand a stale lock and deadlock. Every
# trading-box git-sync section acquires this SAME lock inode so the writers
# serialize instead of racing.
#
# Lock lives in ec2-user's HOME, NOT /var/lock: /var/lock -> /run/lock is
# root:root 0755, so this script (runs as ec2-user) cannot create a lock file
# there, and the SF gate runs its git as `sudo -u ec2-user`. Every actor
# flocks this path AS ec2-user, so opening it for the lock always succeeds
# regardless of which actor created the inode first. The nousergon-data gate
# MUST use this identical path — pinned by a guard test in each repo.
#
# FAIL-LOUD: a `flock -w` timeout is a genuinely stuck git writer, not a
# swallowable condition — flock returns non-zero, the per-repo `if` below takes
# its else branch, PULL_FAILURES increments, and the script exits 1 (surfaced
# via flow-doctor + the FAIL log lines). Never swallow a lock timeout.
GIT_SYNC_LOCK="${AE_GIT_SYNC_LOCK:-/home/ec2-user/.ae-git-sync.lock}"
GIT_SYNC_LOCK_WAIT="${AE_GIT_SYNC_LOCK_WAIT:-150}"

# ── Sync one checkout to origin/main, judged on the post-condition ──────────
# Returns 0 when the checkout provably sits on the remote's current main, 1
# otherwise. Extracted as a function (config-I4978) for two reasons: the main
# loop reads as one decision instead of four nested branches, and the decision
# becomes directly testable — every prior guard on this logic was a regex over
# this file's source text, which is exactly why the control-flow defect below
# shipped and then survived a review.
#
# THE DEFECT THIS CLOSES. `git fetch` exits non-zero when ANY ref fails to
# update, and the previous shape chained `git fetch origin ... && git checkout
# -f main && git reset --hard origin/main`. So a ref-update failure skipped the
# self-heal AND raised "the executor may be running stale code" — on every
# weekday boot for at least 7 days (07-22 through 07-28 confirmed in
# /var/log/boot-pull.log).
#
# Scoping the fetch to `origin main` narrows the blast radius but does NOT close
# it: the ref observed failing its compare-and-swap on 2026-07-28 was
# `refs/remotes/origin/main` ITSELF —
#
#   error: cannot lock ref 'refs/remotes/origin/main': is at 01a279f...
#          but expected a3b5971...
#    ! a3b5971..01a279f  main -> origin/main  (unable to update local ref)
#
# — while that same fetch had already fast-forwarded origin/main to 01a279f
# (`git reflog show refs/remotes/origin/main` confirms). The error was benign
# and the exit code still suppressed the self-heal. A concurrent git writer on
# this box wins that race whenever it interleaves, so the exit code cannot be
# the gate no matter how narrow the refspec is.
sync_repo_to_main() {
    local repo="$1"
    local rc=0

    # The fetch's status is captured SEPARATELY from the checkout/reset, and the
    # two are chained with `;` rather than `&&`, for a reason the group's single
    # exit code cannot express. Two distinct failures need two distinct answers:
    #
    #   fetch failed  -> origin/main may be STALE, so no comparison between two
    #                    LOCAL refs can prove the checkout is current. Needs an
    #                    authoritative remote read.
    #   reset failed  -> the tree is not where we put it. Post-condition catches
    #                    it directly.
    #
    # Collapsing them (the previous `&&` chain, and a plain `;` too) produces the
    # false OK that matters most: a fetch that cannot reach the remote, followed
    # by a reset onto the stale origin/main, exits 0 with HEAD == origin/main
    # satisfied by two equally-old refs. That is precisely the stale-code case
    # this whole alert exists to catch.
    #
    # A failed fetch is RETRIED ONCE, scoped to main, before it is believed.
    # This is what separates the two causes, and it does so by construction:
    #
    #   ref-CAS race        — transient. The retry finds the ref already at the
    #                         value it wanted, so it succeeds.
    #   unreachable remote  — auth/network. The retry fails the same way.
    #
    # Measured on the trading box 2026-07-30: `refs/remotes/origin/main` itself
    # was among the refs failing CAS ("is at 876b73b7 but expected 7aa716e6"),
    # i.e. something had already advanced origin/main between this fetch's ref
    # negotiation and its ref update. That is precisely the transient a retry
    # absorbs and an exit code cannot.
    #
    # A remote read (`git ls-remote`) was the obvious alternative and is WORSE
    # here: `alpha-engine-config` takes merges continuously, so comparing HEAD
    # against a freshly-read remote tip fails whenever main advances in the
    # second between the reset and the read — reporting a correctly-synced box
    # as stale. The retry asks the question that actually matters ("can we reach
    # the remote and get main?") without racing anything.
    #
    # Exit code contract for the inner script:
    #   0  fetch (possibly on retry) + checkout + reset all clean
    #   10 fetch failed TWICE — remote genuinely unreachable
    #   2  checkout/reset failed
    #   3  cd failed
    # flock's own `-w` timeout surfaces as 1, which is why 10 is used for the
    # fetch rather than 1 — the two must stay distinguishable.
    flock -w "$GIT_SYNC_LOCK_WAIT" "$GIT_SYNC_LOCK" bash -c '
        cd "$1" || exit 3
        fetch_rc=0
        if ! git fetch origin main --prune; then
            echo "boot-pull: fetch reported an error; retrying main once" >&2
            git fetch origin main || fetch_rc=10
        fi
        git checkout -f main && git reset --hard origin/main || exit 2
        exit $fetch_rc
    ' _ "$repo" >> "$LOG" 2>&1 || rc=$?

    # A flock timeout keeps its established fail-loud semantics (config#1944): a
    # stuck git writer on this box is worth an alert on its own, independent of
    # whether the code happens to be current.
    if [ "$rc" -eq 1 ]; then
        log "FAIL $repo — git-sync flock timed out after ${GIT_SYNC_LOCK_WAIT}s on $GIT_SYNC_LOCK (a git writer is stuck)"
        return 1
    fi

    local head_branch head_sha origin_main_sha
    head_branch=$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    head_sha=$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo "")
    origin_main_sha=$(git -C "$repo" rev-parse origin/main 2>/dev/null || echo "")

    # Judge on the POST-CONDITION, evaluated UNCONDITIONALLY. The invariant that
    # matters is "this checkout is on main at the remote's current main"; the
    # sync's exit code is only ever a proxy for it, and it is a proxy that is
    # wrong in both directions.
    if [ "$head_branch" != "main" ] || [ -z "$head_sha" ] \
            || [ "$head_sha" != "$origin_main_sha" ]; then
        log "FAIL $repo — post-condition failed: branch=$head_branch HEAD=$head_sha origin/main=$origin_main_sha (sync rc=$rc)"
        return 1
    fi

    # HEAD == origin/main is necessary but NOT sufficient. If the fetch never
    # reached the remote, origin/main is stale too and the comparison above is
    # satisfied by two equally-old LOCAL refs — a false OK in exactly the
    # stale-code case this alert exists for. rc != 0 here means the fetch failed
    # twice, so nothing local can be trusted as current.
    if [ "$rc" -ne 0 ]; then
        log "FAIL $repo — sync rc=$rc (fetch failed twice / checkout unusable); HEAD=$head_sha cannot be shown to be current"
        return 1
    fi

    log "OK   $repo — $(git -C "$repo" log --oneline -1)"
    return 0
}

# Tests source this file for sync_repo_to_main() and must not execute the boot
# sequence (SSM reads, sudo, systemctl). Everything above this line is pure
# definition; everything below has side effects.
if [ "${AE_BOOT_PULL_LIB_ONLY:-0}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

log "=== boot-pull started ==="

# ── Refresh the GitHub PAT in ~/.netrc from SSM ────────────────────────────
# alpha-engine-config is the only PRIVATE repo pulled below; git authenticates
# to it over HTTPS via the fine-grained PAT in ~/.netrc (libcurl reads ~/.netrc
# by default — see this repo's CLAUDE.md "GitHub access on EC2"). That token
# used to be hand-copied onto each box, so a PAT rotation silently broke every
# box's private-repo pull until someone re-pasted it. 2026-06-03 incident: the
# executor PAT was rotated; the sibling dashboard box's stale ~/.netrc started
# returning 401 and its boot-pull FAILed on alpha-engine-config.
#
# /alpha-engine/GITHUB_TOKEN (SecureString) is now the single source of truth.
# Hydrating ~/.netrc from it on every run means a future rotation only needs an
# SSM update — it auto-propagates to every box within one boot cycle. Mirrors
# the dashboard repo's boot-pull.sh (alpha-engine-dashboard/infrastructure/).
#
# Best-effort by design (per ~/Development/CLAUDE.md item 3 — fail-loud): a
# refresh failure here is WARN-only and MUST NOT clobber a working ~/.netrc,
# because the on-disk token may still be valid and we only overwrite when SSM
# hands back a non-empty token (a transient SSM blip can never wipe valid
# creds). NOTE: this script only WARN-logs a private-repo fetch failure to
# /var/log/boot-pull.log (no flow-doctor report like the dashboard box has) —
# tracked as a follow-up to add symmetric failure surfacing.
GH_USER="cipher813"
NETRC="/home/ec2-user/.netrc"
if GH_TOKEN=$(aws ssm get-parameter --name /alpha-engine/GITHUB_TOKEN \
        --with-decryption --query "Parameter.Value" --output text 2>>"$LOG") \
        && [ -n "$GH_TOKEN" ] && [ "$GH_TOKEN" != "None" ]; then
    NEW_NETRC="machine github.com login ${GH_USER} password ${GH_TOKEN}"
    if [ ! -f "$NETRC" ] || [ "$NEW_NETRC" != "$(cat "$NETRC" 2>/dev/null)" ]; then
        # umask 077 + atomic tmp→mv so the token never lands in a
        # world-readable or half-written file.
        ( umask 077; printf '%s\n' "$NEW_NETRC" > "${NETRC}.tmp.$$" )
        mv "${NETRC}.tmp.$$" "$NETRC"
        chmod 600 "$NETRC"
        log "OK   ~/.netrc refreshed from SSM /alpha-engine/GITHUB_TOKEN"
    else
        log "OK   ~/.netrc unchanged from SSM"
    fi
    unset GH_TOKEN NEW_NETRC
else
    log "WARN ~/.netrc refresh skipped — SSM /alpha-engine/GITHUB_TOKEN unreadable/empty; keeping existing ~/.netrc (private-repo pull will WARN below if the on-disk token is also stale)"
fi

# Weekday/EOD SF only SSM-invokes executor + data on this box; dashboard and
# backtester live on ae-dashboard / Saturday spots (see config#1767).
REPOS=(
    /home/ec2-user/alpha-engine-config
    /home/ec2-user/alpha-engine
    /home/ec2-user/alpha-engine-data
)

# Accumulate pull/pip failures so we can surface them LOUD at the end (per
# ~/Development/CLAUDE.md item 3 — fail-loud). Previously this script only
# WARN-logged a fetch/reset failure to /var/log/boot-pull.log and continued,
# so a private-repo auth break (e.g. the 2026-06-03 PAT-rotation incident,
# which silently failed THIS box's alpha-engine-config pull at 12:45 UTC with
# no badge/email) was invisible. The sibling dashboard box already reported via
# flow-doctor; this closes the asymmetry (ROADMAP L4490).
PULL_FAILURES=0
FAILED_REPOS=()

for repo in "${REPOS[@]}"; do
    if [ ! -d "$repo/.git" ]; then
        log "SKIP $repo (not cloned)"
        continue
    fi

    log "Pulling $repo ..."
    cd "$repo"

    # Ownership reclaim: a script inside the checkout run via sudo (e.g. a
    # timer-install step) can leave root-owned files/dirs in the ec2-user
    # checkout. `git reset --hard` then fails with "unable to unlink ...
    # Permission denied" and the box silently runs stale code until the
    # executor's deploy-drift preflight refuses — 40 min into the pipeline
    # (2026-07-06 incident, config#1811: infrastructure/ops/ was left
    # root-owned by upstream-gate-dryrun-validation's sudo install step,
    # failing every boot-pull that day). Detect-then-chown keeps the common
    # clean-boot path to a single find(1) scan.
    if [ -n "$(find "$repo" -not -user ec2-user -print -quit 2>/dev/null)" ]; then
        log "WARN $repo — foreign-owned files found in checkout; reclaiming (chown -R ec2-user)"
        sudo chown -R ec2-user:ec2-user "$repo" >> "$LOG" 2>&1 \
            || log "WARN $repo — chown reclaim failed; git reset may fail below"
    fi

    PREV_SHA=$(git rev-parse HEAD 2>/dev/null || echo "none")
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    if [ "$CURRENT_BRANCH" != "main" ]; then
        log "NOTE $repo — on branch '$CURRENT_BRANCH', will reset to origin/main (policy: boot always tracks main)"
    fi

    # Fix stale remote URL (post-org-transfer: cipher813/alpha-engine-config →
    # nousergon/alpha-engine-config). Only the config repo was cloned pre-transfer
    # and carries the old path; sibling repos point at nousergon/ directly.
    if [ "$repo" = "/home/ec2-user/alpha-engine-config" ]; then
        _CURRENT_URL=$(git config --get remote.origin.url 2>/dev/null || echo "")
        _EXPECTED_URL="https://github.com/nousergon/alpha-engine-config.git"
        if [ -n "$_CURRENT_URL" ] && [ "$_CURRENT_URL" != "$_EXPECTED_URL" ]; then
            log "FIX $repo — remote.origin.url corrected: $_CURRENT_URL -> $_EXPECTED_URL"
            git remote set-url origin "$_EXPECTED_URL" >> "$LOG" 2>&1 \
                || log "WARN $repo — remote set-url failed"
        fi
        unset _CURRENT_URL _EXPECTED_URL
    fi

    # The sync + its post-condition live in sync_repo_to_main() (defined at the
    # top of this file, where the full rationale is). It serializes the
    # index-mutating git ops behind the shared flock (config#1944) so this cannot
    # race the weekday CodeFreshnessGate / ChronicGapSelfHeal on .git/index.lock,
    # and it judges success on the post-condition rather than on git's exit code.
    # The ownership reclaim above runs OUTSIDE the lock deliberately (it is not a
    # git-index op; a test pins that the reclaim still precedes the sync).
    if sync_repo_to_main "$repo"; then
        NEW_SHA=$(git rev-parse HEAD 2>/dev/null || echo "none")

        # Deploy gate: import smoke test (catches syntax + transitive
        # ImportErrors; no IB Gateway connection needed). Imports pull the full
        # transitive module graph, so a broken dependency in any executor module
        # surfaces here pre-planner, not at runtime.
        if [ "$repo" = "/home/ec2-user/alpha-engine" ] && [ "$PREV_SHA" != "$NEW_SHA" ]; then
            if [ -f ".venv/bin/python" ] && [ -f "executor/main.py" ]; then
                log "GATE $repo — running import smoke test..."
                if .venv/bin/python -c "import executor.main, executor.daemon, executor.eod_reconcile" >> "$LOG" 2>&1; then
                    log "OK   $repo — import smoke test passed"
                else
                    log "FAIL $repo — import smoke test failed, rolling back to $PREV_SHA"
                    git reset --hard "$PREV_SHA" >> "$LOG" 2>&1
                    log "ROLLBACK $repo — reverted to $(git log --oneline -1)"
                fi
            fi
        fi
    else
        PULL_FAILURES=$((PULL_FAILURES + 1))
        FAILED_REPOS+=("$repo (git)")
    fi

    # Update pip deps if venv + requirements.txt exist.
    # flow-doctor is now pulled in transitively via
    # nousergon-lib[flow-doctor]; the previous bundled editable
    # install (pip install -e /home/ec2-user/flow-doctor) has been
    # removed so boot doesn't silently overwrite the lib-provided copy
    # with a local dev branch.
    #
    # alpha-engine-data-config#1768 Phase 1: trading no longer runs the heavy
    # weekday/EOD data phases at all (those moved to ephemeral EC2 spot boxes
    # in config#1767 Phase 2, nousergon-data#643) — the ONLY thing this box's
    # alpha-engine-data checkout still needs a working venv for is the
    # metron-intraday collector (systemd timer, see the sync_systemd_units_from
    # call below), plus a manual/emergency SSM fallback of morning_enrich /
    # daily_append if ever needed. nousergon-data's own
    # infrastructure/data-trading-requirements.txt is the traced minimal
    # subset for exactly that scope (excludes arcticdb's RAG/backfill/
    # weekly-feature-engineer-only siblings: voyageai, edgartools, embit,
    # jsonschema, beautifulsoup4, feedparser — see that file's header for the
    # full trace) — full requirements.txt (~1.5GB, RAG + backfill + weekly
    # feature_engineer deps this box never runs) stays reserved for repos
    # that actually need it. REQUIREMENTS_FILE keyed on $repo rather than a
    # generic lookup table since alpha-engine-data is (for now) the only repo
    # in $REPOS with a non-default requirements filename; extend this
    # if/elif (not a full map) only if a second repo needs the same
    # treatment.
    REQUIREMENTS_FILE="requirements.txt"
    if [ "$repo" = "/home/ec2-user/alpha-engine-data" ] && [ -f "infrastructure/data-trading-requirements.txt" ]; then
        REQUIREMENTS_FILE="infrastructure/data-trading-requirements.txt"
    fi
    if [ -f ".venv/bin/pip" ] && [ -f "$REQUIREMENTS_FILE" ]; then
        if .venv/bin/pip install --quiet -r "$REQUIREMENTS_FILE" >> "$LOG" 2>&1; then
            log "OK   $repo — deps updated (from $REQUIREMENTS_FILE)"
        else
            log "FAIL $repo — pip install failed (from $REQUIREMENTS_FILE)"
            PULL_FAILURES=$((PULL_FAILURES + 1))
            FAILED_REPOS+=("$repo (pip)")
        fi
    fi
done

# ── Restore trades.db from S3 if missing or empty ──────────────────────────
# The trading instance is stopped/started daily by EventBridge. If a new
# instance is launched (or EBS is recreated), trades.db will be missing and
# init_db() creates an empty one — losing all trade history and EOD data.
# This restores trades_latest.db from S3 as a safety net.
RISK_YAML="/home/ec2-user/alpha-engine/config/risk.yaml"
if [ -f "$RISK_YAML" ]; then
    # Parse db_path and trades_bucket from risk.yaml. `\s` is a GNU-sed-only
    # shorthand — the deploy target (Amazon Linux EC2) and CI (ubuntu-latest)
    # both use GNU sed so this masked the bug there, but BSD sed (macOS, every
    # local dev machine) treats `\s` as a literal "s", leaving a stray leading
    # space/trailing quote in the parsed value. `[[:space:]]` is the POSIX
    # class both sed implementations support identically.
    DB_PATH=$(grep -E '^\s*db_path:' "$RISK_YAML" | head -1 | sed 's/.*db_path:[[:space:]]*["]*\([^"]*\)["]*[[:space:]]*/\1/' | tr -d "'\"")
    TRADES_BUCKET=$(grep -E '^\s*trades_bucket:' "$RISK_YAML" | head -1 | sed 's/.*trades_bucket:[[:space:]]*["]*\([^"]*\)["]*[[:space:]]*/\1/' | tr -d "'\"")

    if [ -n "$DB_PATH" ] && [ -n "$TRADES_BUCKET" ]; then
        S3_KEY="trades/trades_latest.db"
        RESTORE_NEEDED=false

        if [ ! -f "$DB_PATH" ]; then
            log "trades.db missing — restoring from s3://${TRADES_BUCKET}/${S3_KEY}"
            RESTORE_NEEDED=true
        else
            DB_SIZE=$(stat -c%s "$DB_PATH" 2>/dev/null || stat -f%z "$DB_PATH" 2>/dev/null || echo 0)
            if [ "$DB_SIZE" -le 20480 ]; then
                log "trades.db empty or minimal (${DB_SIZE}B) — restoring from S3"
                RESTORE_NEEDED=true
            else
                # Check for staleness: compare local max(eod_pnl.date) against the
                # expected most-recent trading day, not just presence of any row.
                #
                # config#2356: the presence-only check ("does eod_pnl have ANY
                # row") cannot catch a snapshot-restored stale-but-large DB — a
                # box restored from a 3-week-old EBS snapshot has plenty of
                # non-null eod_pnl rows (just old ones) and would incorrectly
                # report "has recent data". Compute the expected floor date via
                # nousergon_lib.trading_calendar (same helper executor/main.py's
                # ArcticDB freshness gate and executor/eod_reconcile.py use) and
                # require LOCAL_MAX_DATE to be within 1 trading day of the last
                # closed session — that tolerance absorbs the case where today's
                # EOD ingestion simply hasn't run yet, without masking a
                # genuinely stale restore (2+ trading days behind triggers
                # restore).
                #
                # AE_VENV_PY must point at the alpha-engine venv set up earlier
                # in THIS script (the .venv/bin/pip install -r requirements.txt
                # step above, which runs before this block) so nousergon_lib is
                # importable — mirrors the FD_VENV precedent below in this same
                # file. Falls back to bare python3 (best-effort: sqlite parsing
                # still works without nousergon_lib, only the recency half of
                # the check is skipped) if the venv isn't present yet.
                AE_VENV_PY="/home/ec2-user/alpha-engine/.venv/bin/python"
                if [ ! -x "$AE_VENV_PY" ]; then
                    AE_VENV_PY="python3"
                fi

                LOCAL_MAX_DATE=$("$AE_VENV_PY" - "$DB_PATH" <<'PYEOF' 2>/dev/null || echo ""
import sqlite3
import sys

db_path = sys.argv[1]
try:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT MAX(date) FROM eod_pnl")
    result = c.fetchone()
    conn.close()
    print(result[0] if result and result[0] else "")
except Exception:
    print("", file=sys.stderr)
    sys.exit(1)
PYEOF
)

                if [ -z "$LOCAL_MAX_DATE" ]; then
                    log "trades.db has no eod_pnl data — restoring from S3"
                    RESTORE_NEEDED=true
                else
                    # Floor date: last_closed_trading_day() minus 1 trading day
                    # of tolerance. LOCAL_MAX_DATE older than this floor means
                    # the DB is missing at least 2 trading days of EOD data —
                    # the "stale-but-large" snapshot-restore scenario.
                    STALENESS_FLOOR=$("$AE_VENV_PY" <<'PYEOF' 2>/dev/null || echo ""
try:
    from nousergon_lib.trading_calendar import last_closed_trading_day, subtract_trading_days
    print(subtract_trading_days(last_closed_trading_day(), 1).isoformat())
except Exception:
    print("")
PYEOF
)
                    if [ -n "$STALENESS_FLOOR" ] && [ "$LOCAL_MAX_DATE" \< "$STALENESS_FLOOR" ]; then
                        log "trades.db eod_pnl is stale (max_date=$LOCAL_MAX_DATE < floor=$STALENESS_FLOOR) — restoring from S3"
                        RESTORE_NEEDED=true
                    elif [ -z "$STALENESS_FLOOR" ]; then
                        log "WARN could not compute staleness floor (nousergon_lib.trading_calendar unavailable) — falling back to presence-only check (max_date=$LOCAL_MAX_DATE)"
                    else
                        log "trades.db has recent data (max_date=$LOCAL_MAX_DATE, floor=$STALENESS_FLOOR) — no restore needed"
                    fi
                fi
            fi
        fi

        if [ "$RESTORE_NEEDED" = "true" ]; then
            if aws s3 cp "s3://${TRADES_BUCKET}/${S3_KEY}" "$DB_PATH" >> "$LOG" 2>&1; then
                NEW_SIZE=$(stat -c%s "$DB_PATH" 2>/dev/null || stat -f%z "$DB_PATH" 2>/dev/null || echo 0)
                log "OK   trades.db restored (${NEW_SIZE}B)"
            else
                log "WARN trades.db restore failed — executor will start with empty db"
                exit 1
            fi
        fi
    else
        log "WARN could not parse db_path/trades_bucket from risk.yaml"
    fi
else
    log "SKIP trades.db restore (no risk.yaml)"
fi

# Sync systemd service files from repo (if alpha-engine has them).
#
# Installs NEW units as well as updating existing ones. Prior versions of
# this block only touched units already in /etc/systemd/system/, which
# silently skipped newly-added repo units (e.g. alpha-engine-daily-data.timer
# added in PR #62 sat on disk for days before this was noticed). After a
# new .timer is copied and daemon-reload'd, it's enabled + started so the
# next boot picks it up and the current boot activates it immediately.
#
# sync_systemd_units_from() (config#2352): factored out so this same
# install/update/orphan/enable-reconcile logic can run a SECOND pass against
# nousergon-data's infrastructure/systemd/ below — that repo's
# metron-intraday.{service,timer} were version-tracked but boot-pull never
# looked at them, so a merged unit edit silently never took effect until an
# operator remembered to re-run install-metron-intraday.sh over SSM
# (2026-07-13 operator ruling: "queue on merge, apply on next boot" — THIS
# boot-pull pass is that queue's drain point for the trading box, which is
# off most of the day and can't reliably receive a merge-time SSM push).
# $1: source dir holding *.service/*.timer. $2: space-separated EXCLUDE list
# of exact basenames to skip entirely (install AND enable-reconcile) — for
# a source dir that ships MULTIPLE repos' unit families where one family no
# longer belongs on THIS box (config#1768 Phase 1: metron-intraday moved off
# trading onto ae-dashboard, but nousergon-data's infrastructure/systemd/
# still ships its unit files for ae-dashboard's OWN sync pass to pick up —
# the install loop below globs *.service/*.timer unconditionally, so
# without this exclude it would keep re-installing + re-enabling
# metron-intraday on trading every boot regardless of the orphan-prefix
# args, since the orphan pass only fires when a unit is ABSENT from
# $SYSTEMD_SRC, not merely unwanted on this box). Pass "" for no excludes.
# $3+: one or more glob prefixes for orphan reconciliation (keeps each
# repo's orphan sweep scoped to units IT owns — alpha-engine's sweep must
# never disable/remove a nousergon-data unit or vice versa; multiple
# prefixes let one source dir cover unit families that don't share a common
# prefix, e.g. nousergon-data ships both "metron-intraday.*" and
# "systemd-unit-drift-check.*").
sync_systemd_units_from() {
    local SYSTEMD_SRC="$1"
    shift
    local EXCLUDE_BASENAMES="$1"
    shift
    local ORPHAN_GLOB_PREFIXES=("$@")
    [ -d "$SYSTEMD_SRC" ] || return 0

    _sync_is_excluded() {
        local candidate="$1"
        local excl
        for excl in $EXCLUDE_BASENAMES; do
            [ "$candidate" = "$excl" ] && return 0
        done
        return 1
    }

    local CHANGED=false
    for unit in "$SYSTEMD_SRC"/*.service "$SYSTEMD_SRC"/*.timer; do
        [ -f "$unit" ] || continue
        local name target
        name=$(basename "$unit")
        if _sync_is_excluded "$name"; then
            continue
        fi
        target="/etc/systemd/system/$name"
        if [ ! -f "$target" ]; then
            sudo cp "$unit" /etc/systemd/system/
            log "OK   systemd: installed $name (new, src=$SYSTEMD_SRC)"
            CHANGED=true
        elif ! diff -q "$unit" "$target" >/dev/null 2>&1; then
            sudo cp "$unit" /etc/systemd/system/
            log "OK   systemd: updated $name (src=$SYSTEMD_SRC)"
            CHANGED=true
        fi
    done
    # Orphan reconciliation: any installed unit matching $ORPHAN_GLOB_PREFIX
    # without a corresponding source in $SYSTEMD_SRC was removed from that
    # repo and must be retired here. Disable + remove. Safety: only matches
    # the given prefix, never touches unrelated system units (including the
    # OTHER repo's tracked units — each call site passes its own prefix).
    #
    # 2026-04-28: closes the asymmetry where adding a new timer was
    # self-healing (install/update/enable handled) but retiring one was
    # not — the systemd file lingered on disk and continued firing even
    # after deletion from the repo. After this pass, removing a unit
    # file from `infrastructure/systemd/` is the canonical retirement
    # path: next boot disables + deletes it.
    for prefix in "${ORPHAN_GLOB_PREFIXES[@]}"; do
        for installed in /etc/systemd/system/${prefix}*.service /etc/systemd/system/${prefix}*.timer; do
            [ -f "$installed" ] || continue  # glob may match nothing
            local oname
            oname=$(basename "$installed")
            if [ ! -f "$SYSTEMD_SRC/$oname" ]; then
                sudo systemctl disable --now "$oname" >> "$LOG" 2>&1 || true
                sudo rm -f "$installed"
                log "OK   systemd: orphan removed $oname (no longer in $SYSTEMD_SRC)"
                CHANGED=true
            fi
        done
    done

    if $CHANGED; then
        sudo systemctl daemon-reload
        log "OK   systemd: daemon-reload"
    fi

    # Reconcile timer enable state. Every timer shipped in the repo
    # MUST be enabled. `systemctl enable` is idempotent on already-
    # enabled timers, and recreates the `timers.target.wants/` symlink
    # on any timer that was disabled or whose symlink was lost. Fixes
    # three failure modes the prior new-install-only enable path could
    # not recover from:
    #
    # 1. Manual `systemctl disable` (intentional debugging, accidental).
    # 2. EBS volume state where unit files exist on disk but
    #    `timers.target.wants/` is empty.
    # 3. New timers added to `infrastructure/systemd/` after the
    #    one-shot `setup-trading-ec2.sh` run already completed — they
    #    get `cp`'d by boot-pull but the previous "enable only on first
    #    install" branch missed them because the target file already
    #    existed.
    #
    # 2026-04-21 SNDK EOD incident: `alpha-engine-eod.timer` was
    # disabled for an unknown reason. boot-pull's previous logic only
    # enabled timers inside the `[ ! -f "$target" ]` branch, so the
    # disabled state persisted for two boots. EOD emails silently
    # stopped firing until manual SSM intervention re-enabled the
    # timer. This reconciliation pass makes every boot self-healing
    # for timer-enable drift.
    for unit in "$SYSTEMD_SRC"/*.timer; do
        [ -f "$unit" ] || continue
        local tname
        tname=$(basename "$unit")
        if _sync_is_excluded "$tname"; then
            continue
        fi
        if sudo systemctl enable --now "$tname" >> "$LOG" 2>&1; then
            log "OK   systemd: enable reconciled $tname"
        else
            log "WARN systemd: enable reconcile failed: $tname"
        fi
    done
    unset -f _sync_is_excluded
}

sync_systemd_units_from "/home/ec2-user/alpha-engine/infrastructure/systemd" "" "alpha-engine-"

# nousergon-data's systemd units (metron-intraday.{service,timer} +
# config#2352's own systemd-unit-drift-check.{service,timer}) — see
# sync_systemd_units_from's comment above for why this second pass exists.
# Two orphan prefixes since nousergon-data's unit names don't share a common
# "alpha-engine-*"-style prefix: glob on the two exact basenames this repo
# ships instead of a wildcard prefix, so a future unrelated unit hand-placed
# on this box for debugging is never swept as an "orphan" of this repo.
#
# config#1768 Phase 1 (2026-07-21): metron-intraday MOVED off trading onto
# ae-dashboard (config#1768 workstream 2) — ae-dashboard now runs it
# (intraday price alerts Lambda already covers the duplicate work this was
# doing here; see that issue). nousergon-data's infrastructure/systemd/ dir
# is shared by BOTH boxes' sync passes (ae-dashboard's own boot-pull now
# also points at this same source dir for its metron-intraday pull), so
# metron-intraday.{service,timer} must stay in the exclude list here
# PERMANENTLY, not just as a one-time cleanup — without it, the very next
# nousergon-data merge touching either unit file would re-install +
# re-enable it right back onto trading via the reconcile loop above
# (orphan-removal alone can't catch this: the unit is NOT absent from
# $SYSTEMD_SRC, it's merely unwanted on THIS box).
sync_systemd_units_from "/home/ec2-user/alpha-engine-data/infrastructure/systemd" "metron-intraday.service metron-intraday.timer" "systemd-unit-drift-check"

# One-time (idempotent) self-heal for boxes that already had metron-intraday
# installed+enabled from before the exclude above existed: the exclude only
# stops FUTURE install/enable/re-enable, it does not retroactively touch a
# unit already present in /etc/systemd/system/, since sync_systemd_units_from
# only acts on units it manages (installs, updates, or orphan-removes ones
# absent from $SYSTEMD_SRC — metron-intraday is neither, now that it's
# excluded rather than removed from the source dir). `|| true` on every step:
# this must never fail boot-pull itself, and each systemctl call is already
# a no-op if the unit doesn't exist / isn't loaded. Safe to leave in
# permanently (config#1768's own closes-when checks `systemctl is-active
# metron-intraday` is inactive/masked on ae-trading — this is that
# self-healing path for this environment, which cannot verify it live).
if systemctl list-unit-files metron-intraday.timer >> "$LOG" 2>&1 || systemctl list-unit-files metron-intraday.service >> "$LOG" 2>&1; then
    log "NOTE metron-intraday unit(s) found installed on trading — disabling (config#1768: moved to ae-dashboard)"
    sudo systemctl disable --now metron-intraday.timer metron-intraday.service 2>> "$LOG" || true
fi

# Config files are now in the alpha-engine-config private repo (pulled above).
# Each module's config loader searches ~/alpha-engine-config/ first.

# ── Report failures to flow-doctor if any occurred ──────────────────────────
# Mirrors the dashboard box's boot-pull.sh (alpha-engine-dashboard/
# infrastructure/) — closes ROADMAP L4490, the asymmetry where a private-repo
# fetch failure on THIS box was WARN-only-in-the-log (invisible) while the
# dashboard box raised a flow-doctor badge + email. flow-doctor.yaml here is
# flow_name=executor → reports to nousergon/crucible-executor + emails the owner.
# Fire-and-forget (`|| true`): if flow-doctor itself is broken, the FAIL lines
# in /var/log/boot-pull.log are the fallback signal, and we still exit 1.
if [ "$PULL_FAILURES" -gt 0 ]; then
    log "=== boot-pull completed with $PULL_FAILURES failure(s): ${FAILED_REPOS[*]} ==="

    # This used to construct flow-doctor by hand in a heredoc. It had been
    # BROKEN for an unknown length of time and nobody knew, because the thing
    # that was broken was the failure reporter itself
    # (alpha-engine-config-I4509). Two independent faults, either one fatal:
    #
    #   1. `flow_doctor.init()` does not exist. flow-doctor 0.8.7 exports
    #      FlowDoctor/FlowDoctorBuilder and no `init`; the call raised
    #      AttributeError every time, leaving one stderr line in a log nobody
    #      reads.
    #   2. The env hydration was incomplete anyway — flow-doctor.yaml
    #      references more ${VAR}s than the heredoc hydrated, so even with the
    #      API call fixed, construction fails with ConfigError.
    #
    # This matters more here than on the dashboard box: THIS is the trading
    # box, boot-pull is what guarantees it runs current code on a trading day,
    # and it runs once at boot on an instance that only exists during the
    # trading window — so a failure has a one-day blast radius and then the
    # evidence disappears with the instance. It was observed failing for 3+
    # days straight with zero signal.
    #
    # `krepis.alerts` is the canonical alert CLI (config#1649) and resolves its
    # own secrets, so there is no hydration list here to drift.
    ALERT_PY="/home/ec2-user/alpha-engine/.venv/bin/python"
    if [ -x "$ALERT_PY" ]; then
        _dkey="boot-pull-trading-$(printf '%s' "${FAILED_REPOS[*]}" | tr ' /' '__' | cut -c1-72)"
        "$ALERT_PY" -m krepis.alerts publish \
            --message "boot-pull FAILED on the TRADING box: ${PULL_FAILURES} repo(s) could not be updated — ${FAILED_REPOS[*]}. The executor may be running stale code for today's session. See /var/log/boot-pull.log." \
            --severity error \
            --source boot-pull \
            --dedup-key "$_dkey" \
            --dedup-window-min 1440 \
            || log "ALERT PUBLISH FAILED — boot-pull failure is UNREPORTED"
    else
        log "ALERT PUBLISH SKIPPED — $ALERT_PY missing; boot-pull failure is UNREPORTED"
    fi

    log "=== boot-pull complete (with failures) ==="
    exit 1
fi

log "=== boot-pull complete ==="
