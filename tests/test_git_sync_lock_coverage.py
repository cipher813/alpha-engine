"""Structural guard: every git write on a shared on-box checkout is flocked.

alpha-engine-config#1944 class. Measured incidents this guard exists to keep
closed:
  - 2026-07-08 ne-preopen-trading FailExecution: boot-pull's unlocked
    `git reset --hard` raced the weekday SF's CodeFreshnessGate on
    .git/index.lock.
  - 2026-07-28/07-30: an unlocked `git fetch`'s own ref update to
    refs/remotes/origin/main lost a compare-and-swap race against a
    concurrent writer on the same box — a fetch is a git WRITE, not a read.
  - 2026-08-27 20:07 UTC (sibling incident, same class, crucible-dashboard's
    ~/metron checkout): two unsynchronised git writers collided on
    `refs/remotes/origin/main` and the deploy died before it even started;
    the commit sat undeployed for five hours.
  - Prior to this fix, infrastructure/ops/upstream-gate-dryrun-validation.sh
    ran a bare `git pull --ff-only origin main` on
    /home/ec2-user/alpha-engine from a systemd timer, unlocked, while
    boot-pull.sh already serializes every other writer on that SAME checkout
    behind $GIT_SYNC_LOCK.

This walks every shell script under infrastructure/ (the only place on-box
git writers live in this repo — see AGENTS.md "Trading instance boot
sequence") and fails if it finds a git command that WRITES repository state
(pull / fetch / checkout -f / reset --hard / merge) on a line that is not
itself flock-guarded and is not inside a `flock ... bash -c '...'` block.

Demonstrated failing pre-fix: reverting infrastructure/ops/
upstream-gate-dryrun-validation.sh's `flock -w "$GIT_SYNC_LOCK_WAIT"
"$GIT_SYNC_LOCK" git pull --ff-only origin main` back to a bare
`git pull --ff-only origin main` makes
test_every_git_write_in_infrastructure_is_flocked fail on that exact line.
"""
from __future__ import annotations

import re
from pathlib import Path

_INFRA_DIR = Path(__file__).parent.parent / "infrastructure"

# Matches a git subcommand that WRITES repository/ref state. Read-only
# commands (log, show, rev-parse, status, diff, ls-remote, config --get) are
# deliberately excluded — this guard is about serializing writers, not about
# forbidding reads.
_GIT_WRITE_RE = re.compile(
    r"\bgit\b[^|;&]*\b(pull|fetch|merge|reset\s+--hard|checkout\s+-f)\b"
)

# Comment-only or documentation lines (this file's own docstring, or a
# script's header prose quoting a git command as an example) are not
# executable and must not be flagged. A line is treated as prose when the
# FIRST non-whitespace character is '#', or when it's inside a markdown
# code fence in a comment block — this repo's infra scripts don't use the
# latter, so '#'-prefix is sufficient.
def _is_comment_line(line: str) -> bool:
    return line.strip().startswith("#")


def _iter_shell_scripts():
    yield from sorted(_INFRA_DIR.rglob("*.sh"))


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Join backslash-continued shell lines into one logical line, keyed by
    the line number the continuation STARTED on (so a `flock ...\\` opener
    and its continuation `git ... reset --hard ...` are matched together —
    this is exactly boot-pull.sh's deploy-gate rollback shape)."""
    out: list[tuple[int, str]] = []
    pending_start: int | None = None
    pending_text = ""
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        if pending_start is None:
            pending_start = lineno
            pending_text = raw_line
        else:
            pending_text += " " + raw_line.strip()
        if raw_line.rstrip().endswith("\\"):
            pending_text = pending_text.rstrip()[:-1]
            continue
        out.append((pending_start, pending_text))
        pending_start = None
        pending_text = ""
    if pending_start is not None:
        out.append((pending_start, pending_text))
    return out


def _flock_violations(path: Path) -> list[tuple[int, str]]:
    violations: list[tuple[int, str]] = []
    in_flock_block = False
    for lineno, line in _logical_lines(path.read_text()):
        if _is_comment_line(line):
            continue

        stripped = line.rstrip()

        # Track multi-line `flock ... bash -c '...'` blocks (boot-pull.sh's
        # shape): every line between the opening `bash -c '` and the closing
        # line (starts with a bare `'`) is inside the lock.
        if in_flock_block:
            if stripped.lstrip().startswith("'"):
                in_flock_block = False
            # The opening/closing lines themselves are lock machinery, not
            # writes to flag independently — still scan them below in case
            # a write shares the closing line, which it currently doesn't.
        opens_block = "flock" in line and stripped.endswith("bash -c '")
        if opens_block:
            in_flock_block = True

        if _GIT_WRITE_RE.search(line):
            guarded = "flock" in line or in_flock_block
            if not guarded:
                violations.append((lineno, line.strip()))

    return violations


def test_every_git_write_in_infrastructure_is_flocked():
    all_violations: dict[str, list[tuple[int, str]]] = {}
    for path in _iter_shell_scripts():
        violations = _flock_violations(path)
        if violations:
            all_violations[str(path.relative_to(_INFRA_DIR.parent))] = violations

    assert not all_violations, (
        "unflocked git write(s) found on infrastructure scripts — every git "
        "pull/fetch/checkout -f/reset --hard/merge on a shared on-box "
        "checkout must run under the SAME $GIT_SYNC_LOCK flock boot-pull.sh "
        "uses (alpha-engine-config#1944), or a concurrent writer can lose a "
        "ref compare-and-swap race and leave the box undeployed. "
        f"Violations: {all_violations}"
    )


def test_git_sync_lock_shared_module_defines_the_canonical_lock_path():
    """infrastructure/lib/git-sync-lock.sh must exist and match the exact
    lock path/wait boot-pull.sh (the reference implementation) uses — every
    consumer sourcing it must flock the SAME inode boot-pull.sh does, not a
    second, non-cooperating lock."""
    shared = _INFRA_DIR / "lib" / "git-sync-lock.sh"
    assert shared.is_file(), "infrastructure/lib/git-sync-lock.sh is missing"
    src = shared.read_text()
    assert 'GIT_SYNC_LOCK="${AE_GIT_SYNC_LOCK:-/home/ec2-user/.ae-git-sync.lock}"' in src
    assert 'GIT_SYNC_LOCK_WAIT="${AE_GIT_SYNC_LOCK_WAIT:-150}"' in src

    boot_pull_src = (_INFRA_DIR / "boot-pull.sh").read_text()
    assert (
        'GIT_SYNC_LOCK="${AE_GIT_SYNC_LOCK:-/home/ec2-user/.ae-git-sync.lock}"'
        in boot_pull_src
    ), "boot-pull.sh's inline lock path constant has drifted from the shared module"
    assert (
        'GIT_SYNC_LOCK_WAIT="${AE_GIT_SYNC_LOCK_WAIT:-150}"' in boot_pull_src
    ), "boot-pull.sh's inline lock wait constant has drifted from the shared module"


def test_upstream_gate_dryrun_validation_sources_the_shared_lock_module():
    """Regression pin for the specific defect this PR closes: the dry-run
    validation script's git pull must route through the shared lock module,
    not redefine the lock path/wait as a second literal."""
    src = (_INFRA_DIR / "ops" / "upstream-gate-dryrun-validation.sh").read_text()
    assert "infrastructure/lib/git-sync-lock.sh" in src
    assert re.search(
        r'flock -w "\$GIT_SYNC_LOCK_WAIT" "\$GIT_SYNC_LOCK" git pull --ff-only origin main',
        src,
    ), "the dry-run validation's git pull must be wrapped in the shared flock"
