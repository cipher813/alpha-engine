"""One flow-doctor page per event (alpha-engine-config-I10049).

``setup_logging`` attaches ``FlowDoctorHandler`` to the ROOT logger at ERROR,
so every ``logger.error`` record becomes its own flow-doctor report. A site
that ALSO calls ``fd.report`` for the same condition therefore pages twice
and auto-files two issues for one event — measured 2026-09-04 21:11:09 UTC,
two NAV hard-gate reports 0.2 ms apart (I10018 + I10019).

Two guards:

1. ``_log_paged`` logs at WARNING (below the handler floor) and never at
   ERROR — the helper is the mechanism, so its level is the contract.
2. A source scan: no ``logger.error(`` within the window immediately
   preceding an ``fd.report(``, and none immediately preceding an
   ``integrity_breaches.append``/``extend`` (those feed the single
   ``fd.report`` at the end of the run). Message-regex exclusion on the
   handler is the alternative and is rejected: the handler's own docstring
   names it as prose that drifts (I7276).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

EXECUTOR_DIR = Path(__file__).resolve().parents[1] / "executor"
WINDOW_LINES = 15

_FD_REPORT = re.compile(r"\bfd\.report\(|_fd_early\.report\(")
_FEEDS_SINGLE_REPORT = re.compile(r"\bintegrity_breaches\.(append|extend)\(")
_LOGGER_ERROR = re.compile(r"\blogger\.error\(")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _paired_error_sites(path: Path) -> list[str]:
    """Walk upward from each report site through its OWN block only.

    The walk stops at the first non-blank line indented less than the site,
    except an ``if fd:`` guard, which is transparent: the log line and the
    report live in the same ``except``/``if`` body, so a ``logger.error`` in
    an enclosing or sibling block (a different condition) is not a pair.
    """
    lines = path.read_text().splitlines()
    hits: list[str] = []
    for idx, line in enumerate(lines):
        if not (_FD_REPORT.search(line) or _FEEDS_SINGLE_REPORT.search(line)):
            continue
        limit = _indent(line)
        for j in range(idx - 1, max(-1, idx - WINDOW_LINES - 1), -1):
            prev = lines[j]
            stripped = prev.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if _indent(prev) < limit:
                if stripped.startswith(("if fd", "if _fd")):
                    limit = _indent(prev)
                    continue
                break
            if _LOGGER_ERROR.search(prev):
                hits.append(f"{path.name}:{j + 1} logger.error precedes {path.name}:{idx + 1}")
    return hits


def test_log_paged_is_below_the_handler_floor(monkeypatch):
    from unittest.mock import MagicMock

    from executor import eod_reconcile

    fake = MagicMock()
    monkeypatch.setattr(eod_reconcile, "logger", fake)
    eod_reconcile._log_paged("NAV three-way reconcile BREACH [%s]: test", "x")
    fake.warning.assert_called_once_with("NAV three-way reconcile BREACH [%s]: test", "x")
    # The FlowDoctorHandler floor is ERROR (krepis.logging._attach_flow_doctor);
    # anything at or above it is a second page.
    fake.error.assert_not_called()
    fake.critical.assert_not_called()
    fake.exception.assert_not_called()


def test_no_logger_error_beside_a_flow_doctor_report():
    hits: list[str] = []
    for path in sorted(EXECUTOR_DIR.glob("*.py")):
        hits.extend(_paired_error_sites(path))
    assert not hits, (
        "logger.error next to fd.report / integrity_breaches pages the same "
        "event twice — use _log_paged (I10049):\n  " + "\n  ".join(hits)
    )


@pytest.mark.parametrize(
    "site",
    [
        "NAV three-way reconcile BREACH [%s]",
        "P&L INTEGRITY BREACH: %s",
        "CUSTODIAN MARK BREACH: %s",
        "TWR CLOSURE BREACH: %s",
        "EOD report artifact build/write failed: %s",
        "EOD email failed: %s",
    ],
)
def test_known_paired_sites_use_the_helper(site):
    src = (EXECUTOR_DIR / "eod_reconcile.py").read_text()
    assert f'_log_paged(\n                "{site}' in src or f'_log_paged("{site}' in src, site
    assert f'logger.error("{site}' not in src, site
