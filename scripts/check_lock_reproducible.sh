#!/usr/bin/env bash
# Fail when requirements.txt is not what `uv pip compile requirements.in`
# produces (alpha-engine-config-I8309).
#
# WHY THIS EXISTS. requirements.txt declares in its own header that it is
# compiled from requirements.in. Nothing enforced that. Three Dependabot PRs
# (#428 pandas majors, #457, #473) edited requirements.in ONLY, and the
# trading box kept installing the old lock — with CI green every time,
# because `lockfile-python-parity` installs the (unchanged) lock and proves
# a strictly weaker claim under a name that reads like the stronger one.
# By 2026-08-24 the .in floored krepis>=0.59.18 while the lock pinned
# krepis==0.54.0, and requirements.in had drifted into an outright
# UNSATISFIABLE state (pandas~=3.0 against arcticdb's pandas<3) that no
# check could see, because no check ever ran the compile.
#
# WHAT IS COMPARED. Every `name==version` pin, as a set. The
# `nousergon-lib @ git+...` line is compared by PRESENCE only: a compile
# always resolves the tag to a commit SHA while requirements.in carries the
# vX.Y.Z tag (tests/test_lib_pin_lockstep.py owns that boundary), so
# comparing its text would fail on a difference that is correct by design.
#
# THE RECOMPILE IS SEEDED WITH THE CURRENT LOCK, deliberately. `uv pip
# compile` treats an existing output file as preferred versions and holds
# them unless a constraint forces a move. Compiling into an EMPTY temp file
# instead resolves everything to newest-on-PyPI, so the check would go red
# every time any transitive dependency cut a release — a detector that fails
# for a reason nobody in this repo caused is a detector that gets ignored.
# Seeded, it answers the question that matters: do the pins we ship still
# satisfy the constraints we declare? That is exactly what was false here.
set -euo pipefail

cd "$(dirname "$0")/.."
PYVER="$(cat .python-version)"
FRESH="$(mktemp)"
trap 'rm -f "$FRESH"' EXIT

echo "Recompiling requirements.in under Python ${PYVER}..."
cp requirements.txt "$FRESH"     # seed: hold current pins unless a constraint moves them
if ! uv pip compile requirements.in --output-file "$FRESH" --python "$PYVER" --quiet 2>"$FRESH.err"; then
    echo "FAIL: requirements.in does not resolve at all under Python ${PYVER}."
    echo "      The lockfile cannot be regenerated, so every floor raised in"
    echo "      requirements.in is unreachable by the deployed environment."
    echo
    cat "$FRESH.err"
    exit 1
fi

pins() { grep -E '^[A-Za-z0-9_.-]+==' "$1" | sort; }

if diff_out="$(diff <(pins requirements.txt) <(pins "$FRESH"))"; then
    echo "OK: requirements.txt matches a fresh compile of requirements.in."
else
    echo "FAIL: requirements.txt is not reproducible from requirements.in."
    echo "      '<' is what is COMMITTED (and deployed); '>' is what"
    echo "      requirements.in actually resolves to today."
    echo
    echo "$diff_out"
    echo
    echo "Fix: uv pip compile requirements.in --output-file requirements.txt --python ${PYVER}"
    echo "     then run the suite against the resolved environment before pushing."
    exit 1
fi

if grep -q 'nousergon-lib @ git+\|nousergon-lib\[' requirements.txt; then
    echo "OK: the nousergon-lib git pin is present (its ref form is owned by tests/test_lib_pin_lockstep.py)."
else
    echo "FAIL: requirements.txt carries no nousergon-lib git pin."
    exit 1
fi
