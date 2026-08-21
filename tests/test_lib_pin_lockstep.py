"""Forbid a bare commit-SHA nousergon-lib pin in this repo's SOURCE files.

alpha-engine-config-I7966: crucible-executor pinned nousergon-lib by bare SHA
(`fb383a98...`, resolving to the pre-`54b2a80`/I7924 release) and was outside
every cross-repo drift check — `crucible-predictor/inference/lib_pin_drift.py`
covered only `_CO_INSTALL_PAIR` (backtester + predictor) and `_FLOOR_REPOS`
(data + research). The executor is now in that probe's `_FLOOR_REPOS`
(crucible-predictor-PR539), reading THIS repo's `requirements.in` — see the
"scope" note below for why not `requirements.txt`.

`crucible-predictor` and `nousergon-data` already guard the identical class
via their own `tests/test_lib_pin_lockstep.py` (alpha-engine-config-I7301).
This is the THIRD adoption of that pattern, which `policy-shared-code` reads
as a lift-into-`nousergon-lib` trigger rather than a third copy.

Deliberately copied here instead, with the rationale on record
(alpha-engine-config-I7966 permits this explicitly when written down):

  - Lifting requires a new `nousergon_lib.testing` module PLUS a released
    tag, before crucible-executor's own pin can even be bumped to consume
    it — the exact chicken-and-egg `nousergon-lib-PR345`'s body describes
    for the sibling secret-scan lift (`crucible-predictor-PR536` /
    `nousergon-data-PR1483` are both blocked on that PR shipping a release).
  - `nousergon-lib-PR345` is open, unmerged, and is about the secret scanner,
    not this test — there is no existing `lib_pin_lockstep` helper in
    `nousergon_lib.testing` to adopt yet, on main or on that branch.
  - Opening a `nousergon-lib` PR + release is out of scope for the two repos
    (`crucible-executor`, `crucible-predictor`) alpha-engine-config-I7966
    names, and would block this fix on a third repo's release cadence.

  Filed as the tracked follow-up: alpha-engine-config-I7976 — lift
  `test_lib_pin_lockstep.py`'s shared parts (the tag/SHA regexes + the
  read-and-classify helper) into `nousergon_lib.testing.lib_pin_lockstep`
  once `nousergon-lib-PR345` merges and a release is cut, and re-point all
  three call sites (predictor, data, executor) at it.

Scope note (unlike predictor/data, this is NOT a two-compiled-file
lockstep check): `requirements.txt` here is a `uv pip compile` LOCKFILE
(see `tests/test_requirements_lockstep.py`) that resolves the VCS ref to its
exact commit SHA by design, on every compile, regardless of what
`requirements.in` pins. A SHA in `requirements.txt` is therefore NOT the
defect this test exists to catch — it is normal, correct, reproducible-lock
behavior. `requirements.in` (hand-maintained) and `pyproject.toml`'s exact
`==` pin (source for `pip install -e .`) are the two files a human actually
edits, and those are what a re-pin-by-SHA regression would show up in.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Mirrors crucible-predictor/tests/test_lib_pin_lockstep.py and
# crucible-predictor/inference/lib_pin_drift.py — the pin format is
# `nousergon-lib[extras] @ git+https://.../nousergon-lib@vX.Y.Z`. Extras are
# OPTIONAL here (requirements.in carries them; pyproject.toml's dependency
# string does not use the git-pin form at all, so this regex is exercised
# only against requirements.in in practice, but stays permissive for
# robustness against a future direct pyproject VCS pin).
_LIB_PIN_RE = re.compile(
    r"nousergon-lib(?:\[[^\]]*\])?\s*@\s*git\+"
    r"https://github\.com/nousergon/nousergon-lib@"
    r"(v[0-9]+\.[0-9]+\.[0-9]+)"
)

# The same line ending in a raw commit SHA instead of a version tag.
_LIB_SHA_PIN_RE = re.compile(
    r"nousergon-lib(?:\[[^\]]*\])?\s*@\s*git\+"
    r"https://github\.com/nousergon/nousergon-lib@([0-9a-f]{7,40})\b"
)

# pyproject.toml's exact-equals pin, e.g. nousergon-lib[flow-doctor,contracts]==0.124.79
_PYPROJECT_EXACT_PIN_RE = re.compile(
    r"nousergon-lib(?:\[[^\]]*\])?==([0-9]+\.[0-9]+\.[0-9]+)"
)


def test_requirements_in_pins_the_lib_by_tag_not_sha():
    """requirements.in must carry a vX.Y.Z tag, never a bare commit SHA.

    A SHA pin here passes review invisibly (it still installs) but is
    permanently uncomparable to crucible-predictor/inference/lib_pin_drift.py's
    floor check, which reads THIS file for the executor precisely because
    requirements.txt always resolves to a SHA regardless. A SHA pin in both
    places would put the executor back outside every drift check it was just
    added to (alpha-engine-config-I7966).
    """
    text = (_REPO_ROOT / "requirements.in").read_text()
    sha_match = _LIB_SHA_PIN_RE.search(text)
    assert sha_match is None, (
        f"requirements.in pins nousergon-lib by commit SHA "
        f"({sha_match.group(1)[:12]}...), not a vX.Y.Z tag.\n\n"
        f"Pin by tag. requirements.in is the human-authored source of truth "
        f"crucible-predictor/inference/lib_pin_drift.py reads for this repo "
        f"(alpha-engine-config-I7966) — a SHA pin here is invisible to that "
        f"check just like it was before this repo was added to its scope."
    )
    tag_match = _LIB_PIN_RE.search(text)
    assert tag_match is not None, (
        "could not find a nousergon-lib pin in requirements.in — expected "
        "`nousergon-lib[extras] @ git+https://github.com/nousergon/"
        "nousergon-lib@vX.Y.Z`"
    )


def test_pyproject_exact_pin_matches_requirements_in_tag():
    """pyproject.toml's ==pin must track the same tag requirements.in pins.

    `pip install -e .` reads only pyproject.toml; a drifted exact pin there
    resolves a different nousergon-lib release than `pip install -r
    requirements.txt` does in production, invisibly (this is the same class
    `test_git_pinned_deps_match_pyproject_exactly` in
    tests/test_requirements_lockstep.py already guards from the other
    direction — this test guards it from pyproject.toml's side and adds the
    explicit no-SHA assertion that file never carries a git-pin form for).
    """
    req_text = (_REPO_ROOT / "requirements.in").read_text()
    pyproject_text = (_REPO_ROOT / "pyproject.toml").read_text()

    req_match = _LIB_PIN_RE.search(req_text)
    assert req_match is not None, "requirements.in has no recognisable nousergon-lib tag pin"
    req_version = req_match.group(1).lstrip("v")

    pyproject_match = _PYPROJECT_EXACT_PIN_RE.search(pyproject_text)
    assert pyproject_match is not None, (
        "pyproject.toml has no exact `==X.Y.Z` nousergon-lib pin — expected "
        "`nousergon-lib[extras]==X.Y.Z`"
    )
    pyproject_version = pyproject_match.group(1)

    assert pyproject_version == req_version, (
        f"nousergon-lib pin drift: requirements.in=v{req_version} but "
        f"pyproject.toml=={pyproject_version}. `pip install -e .` (dev) and "
        f"`pip install -r requirements.txt` (production) must resolve the "
        f"same nousergon-lib release."
    )


def test_the_pin_regexes_distinguish_tag_from_sha():
    """Pin the guard itself, mirroring crucible-predictor's equivalent test."""
    sha_line = (
        "nousergon-lib[flow-doctor,contracts] @ "
        "git+https://github.com/nousergon/nousergon-lib"
        "@fb383a98da36249c09aae3778c96b5ef92325ce1"
    )
    tag_line = (
        "nousergon-lib[flow-doctor,contracts] @ "
        "git+https://github.com/nousergon/nousergon-lib@v0.124.79"
    )
    assert _LIB_PIN_RE.search(sha_line) is None
    assert _LIB_SHA_PIN_RE.search(sha_line) is not None
    assert _LIB_PIN_RE.search(tag_line).group(1) == "v0.124.79"


def test_a_sha_pinned_requirements_in_fails_with_the_reason_not_a_parse_miss():
    """The assertion must name the SHA, not report a generic missing-pin error.

    Exercises the classification logic directly (the same regexes the real
    test above runs against the live repo file) rather than monkeypatching
    module globals, since these functions are pure over their input text.
    """
    text = (
        "numpy<3\n"
        "nousergon-lib[flow-doctor,contracts] @ git+https://github.com/"
        "nousergon/nousergon-lib@fb383a98da36249c09aae3778c96b5ef92325ce1\n"
    )
    sha_match = _LIB_SHA_PIN_RE.search(text)
    assert sha_match is not None
    assert sha_match.group(1) == "fb383a98da36249c09aae3778c96b5ef92325ce1"
    # And the tag regex must NOT also match the same line (else the SHA
    # would silently pass as a valid tag pin instead of failing loudly).
    assert _LIB_PIN_RE.search(text) is None
