#!/usr/bin/env bash
# SINGLE SOURCE OF TRUTH for the requirements.in -> requirements.txt compile.
#
# Sourced by exactly two call sites, which must never disagree:
#
#   scripts/check_lock_reproducible.sh   VERIFIES the committed lock is what
#                                        requirements.in compiles to.
#   .github/upgrade_lock.sh              PRODUCES that lock.
#
# A flag present in the verifier and absent in the producer makes every
# produced lockfile unverifiable by the guard — which is the shape of
# alpha-engine-config-I9060: Dependabot produced requirements.txt with no
# knowledge of requirements.in at all, so `lockfile-reproducible` failed on
# every pip PR it ever opened and the standing auto-merge exception for
# Dependabot could never fire in this repo. Keeping the flags in one file is
# what stops the replacement producer inheriting the same defect quietly.
#
# Defines: LOCK_REPO_ROOT, LOCK_PYVER, LOCK_COMPILE_FLAGS[], lockfile_compile().
# Sets no shell options — both callers set `-euo pipefail` themselves.

LOCK_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_PYVER="$(cat "$LOCK_REPO_ROOT/.python-version")"

# The SSoT interpreter, same source `lockfile-python-parity` uses. This repo
# deploys to one platform (the trading box) and its lock carries no
# platform-conditional dependency, so no --python-platform is pinned here —
# unlike crucible-dashboard, whose streamlit graph drops `watchdog` when
# resolved on macOS. Add one here the moment a conditional dep appears.
LOCK_COMPILE_FLAGS=(--python "$LOCK_PYVER")

# The exact command a human should run to regenerate the lock by hand, for
# the guard's failure message. Pure.
lockfile_compile_hint() {
    echo "uv pip compile requirements.in --output-file requirements.txt ${LOCK_COMPILE_FLAGS[*]}"
}

# lockfile_compile <output-file> [extra uv flags...]
#
# The caller owns the seeding decision: the verifier copies the committed
# lock into <output-file> first (hold current pins unless a constraint moves
# them); the producer passes --upgrade (move every pin to the newest release
# requirements.in still permits).
#
# UV_CUSTOM_COMPILE_COMMAND pins the provenance header uv writes into the
# lockfile. Without it the header records the ABSOLUTE paths of whatever
# checkout and temp file produced it, so requirements.txt would churn on
# every run and differ between a laptop and a runner — a diff that says
# nothing about dependencies. Pinned, the header is the exact command a
# human would run, and it is byte-stable.
lockfile_compile() {
    local out="$1"
    shift
    UV_CUSTOM_COMPILE_COMMAND="$(lockfile_compile_hint)" \
    uv pip compile "$LOCK_REPO_ROOT/requirements.in" \
        --output-file "$out" \
        "${LOCK_COMPILE_FLAGS[@]}" \
        "$@"
}
