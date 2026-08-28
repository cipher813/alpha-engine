#!/bin/bash
# check_deps_match_manifest.sh — assert the trading box's INSTALLED
# dependencies match origin/main's requirements.txt, read INDEPENDENTLY of
# the box's own checkout state.
#
# alpha-engine-config-I8709: no preflight anywhere asserted this, and every
# existing check is INSIDE the loop it validates —
#
#   CodeFreshnessGate's import smoke test  — runs against whatever venv the
#     box already reconciled; self-consistent and can be stale.
#   executor/preflight.py::check_deploy_drift — compares checkout SHA to
#     /home/ec2-user/.frozen_executor_sha, a pin the SAME run produced.
#   boot-pull.sh's own post-pip-install import gate (alpha-engine-config-
#     I8682) — correct now, but it is inside the sequence it validates and
#     its verdict is a rollback: a detector for its own failure, not an
#     independent assertion of box state.
#
# This script is the assertion made from OUTSIDE: it fetches origin/main's
# requirements.txt via `git show` (never the locally checked-out file
# boot-pull.sh may be mid-rewriting — see alpha-engine-config-I8734) and
# diffs it against the box's actual `pip freeze`. A mismatch is named:
# package, manifest version, installed version — "krepis 0.54.0 != 0.59.33"
# is the line that would have ended the 2026-08-26 incident on day one.
#
# SCOPE: this script performs the ASSERTION ONLY. It does not decide
# whether a mismatch halts the pipeline. That decision — including the
# mandatory sf-pipeline-policy.md §7a observe-mode staging (N observe
# cycles before the verdict can halt anything, a promotion criterion
# living in the guard's OWN module, loud while observing) — belongs to
# whatever SF preflight stage invokes this script (nousergon-data's
# sf_preflight.py, or an equivalent early weekday-SF gate state, per
# alpha-engine-config-I8709's own deliverable). That wiring is OUT OF
# SCOPE for crucible-executor and is not done by this commit.
#
# Usage:
#   check_deps_match_manifest.sh <repo-root> [venv-python]
#
# Exit codes:
#   0  every pinned package in origin/main's requirements.txt matches its
#      installed version
#   1  at least one mismatch — each printed as
#      "MISMATCH <pkg>: manifest=<v> installed=<v-or-<not installed>>"
#   2  usage error, or the manifest/venv could not be read (caller should
#      treat this the way check_deploy_drift treats an unreachable GitHub
#      API — an inconclusive read, not a passing check)
set -uo pipefail

_LIB_ONLY="${AE_DEPS_CHECK_LIB_ONLY:-0}"

# Normalize a package name per PEP 503 (case-insensitive, runs of -_. treated
# as equivalent) so "nousergon_lib" in one source and "nousergon-lib" in the
# other are recognized as the same package.
_normalize_name() {
    printf '%s' "$1" | tr 'A-Z' 'a-z' | sed -E 's/[-_.]+/-/g'
}

# Parse "pkg==version" pins out of a requirements.txt-format file (this repo's
# pip-compile output includes "    # via ..." continuation comments and
# blank lines, which this intentionally skips).
parse_pins() {
    local file="$1"
    local line name version
    while IFS= read -r line; do
        case "$line" in
            *'=='*)
                name="${line%%==*}"
                # Strip a leading comment marker / whitespace, if any.
                name="$(printf '%s' "$name" | sed -E 's/^[[:space:]]*//')"
                case "$name" in \#*) continue ;; esac
                version="${line#*==}"
                # Cut at the first whitespace or ';' (environment markers).
                version="$(printf '%s' "$version" | sed -E 's/[[:space:];].*$//')"
                [ -n "$name" ] && [ -n "$version" ] || continue
                printf '%s %s\n' "$(_normalize_name "$name")" "$version"
                ;;
        esac
    done < "$file"
}

# Pure comparison — testable without git/pip/network by sourcing this file
# with AE_DEPS_CHECK_LIB_ONLY=1 and calling this function directly against
# fixture files.
compare_manifest_to_installed() {
    local manifest_file="$1"
    local installed_file="$2"
    local mismatches=0
    local name version got

    declare -A installed_versions=()
    while read -r name version; do
        [ -n "$name" ] || continue
        installed_versions["$name"]="$version"
    done < <(parse_pins "$installed_file")

    while read -r name version; do
        [ -n "$name" ] || continue
        got="${installed_versions[$name]:-<not installed>}"
        if [ "$got" != "$version" ]; then
            echo "MISMATCH $name: manifest=$version installed=$got"
            mismatches=$((mismatches + 1))
        fi
    done < <(parse_pins "$manifest_file")

    if [ "$mismatches" -gt 0 ]; then
        echo "$mismatches package(s) mismatched between origin/main's requirements.txt and the installed venv"
        return 1
    fi
    echo "OK — installed venv matches every pin in origin/main's requirements.txt"
    return 0
}

# Tests source this file for parse_pins()/compare_manifest_to_installed()
# and must not execute the live git-fetch/pip-freeze path below.
if [ "$_LIB_ONLY" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

if [ "$#" -lt 1 ]; then
    echo "usage: check_deps_match_manifest.sh <repo-root> [venv-python]" >&2
    exit 2
fi
REPO_ROOT="$1"
VENV_PY="${2:-$REPO_ROOT/.venv/bin/python}"

if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "check_deps_match_manifest: $REPO_ROOT is not a git checkout" >&2
    exit 2
fi
if [ ! -x "$VENV_PY" ]; then
    echo "check_deps_match_manifest: venv python not found/executable at $VENV_PY" >&2
    exit 2
fi

# Independent read of origin/main's requirements.txt: fetch + `git show`,
# never the locally checked-out file.
#
# The fetch still takes the shared git-sync flock (alpha-engine-config#1944),
# even though this script does no checkout/reset. A previous version of this
# comment argued a bare fetch is "read-only" and cannot race a mutating git
# op — that reasoning does not hold: `git fetch` WRITES the remote-tracking
# ref (refs/remotes/origin/main), and boot-pull.sh's own header documents
# that ref update itself losing a compare-and-swap race against a concurrent
# writer on this box (2026-07-28/07-30, "cannot lock ref
# 'refs/remotes/origin/main'"). This script targets the same
# /home/ec2-user/alpha-engine checkout boot-pull.sh and the weekday SF's
# CodeFreshnessGate/ChronicGapSelfHeal already serialize behind
# $GIT_SYNC_LOCK, so its fetch must take the same lock rather than assume
# immunity from the class.
#
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib/git-sync-lock.sh"

if ! flock -w "$GIT_SYNC_LOCK_WAIT" "$GIT_SYNC_LOCK" git -C "$REPO_ROOT" fetch origin main --quiet; then
    echo "check_deps_match_manifest: git-sync flock/fetch failed on $GIT_SYNC_LOCK after ${GIT_SYNC_LOCK_WAIT}s — cannot verify independently" >&2
    exit 2
fi

MANIFEST_TMP="$(mktemp)"
INSTALLED_TMP="$(mktemp)"
trap 'rm -f "$MANIFEST_TMP" "$INSTALLED_TMP"' EXIT

if ! git -C "$REPO_ROOT" show origin/main:requirements.txt > "$MANIFEST_TMP" 2>/dev/null; then
    echo "check_deps_match_manifest: git show origin/main:requirements.txt failed" >&2
    exit 2
fi

if ! "$VENV_PY" -m pip freeze > "$INSTALLED_TMP" 2>/dev/null; then
    echo "check_deps_match_manifest: pip freeze failed against $VENV_PY" >&2
    exit 2
fi

compare_manifest_to_installed "$MANIFEST_TMP" "$INSTALLED_TMP"
