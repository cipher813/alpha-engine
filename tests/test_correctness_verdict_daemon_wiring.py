"""alpha-engine-config-I9466 — the withholding is verified AT THE MUTATION POINT.

`tests/test_correctness_verdict.py` proves the verdict is computed correctly.
That is not the same claim as "no order is placed". A gate that returns the
right dataclass and is wired to nothing looks identical from the unit tests,
and this fleet has shipped that exact shape before.

So these tests drive `executor.daemon._execute_entry` and assert against the
BROKER CLIENT: under a blocking verdict `place_market_order` is never called,
and under every verdict `_execute_exit`'s path is untouched.
"""

from __future__ import annotations

import contextlib

import pytest

from executor import correctness_verdict as cv
from executor import daemon


class _SpyIBKR:
    def __init__(self):
        self.orders = []

    def place_market_order(self, ticker, action, shares, **kw):
        self.orders.append((ticker, action, shares))
        return {"status": "filled", "filled_price": 100.0, "shares": shares,
                "attempts": []}


@pytest.fixture(autouse=True)
def _reset_gate():
    """The gate is cached per process — reset it around every test so one
    test's primed verdict cannot leak into the next."""
    daemon._correctness_gate_state = None
    daemon._correctness_gate_evaluated = False
    yield
    daemon._correctness_gate_state = None
    daemon._correctness_gate_evaluated = False


def _prime(verdict, mode):
    daemon._correctness_gate_state = cv.CorrectnessVerdictState(
        verdict=verdict,
        mode=mode,
        blocks_new_entries=(mode == cv.MODE_ENFORCE and verdict != cv.PASS),
        attested_run_date="2026-08-21",
        params_updated_at="2026-08-28",
        reason="pinned by test",
    )
    daemon._correctness_gate_evaluated = True
    return daemon._correctness_gate_state


def _entry():
    return {"ticker": "AAPL", "shares": 10, "score": 90}


def _call_execute_entry(ibkr):
    daemon._execute_entry(
        ibkr, None, None, _entry(), {"last": 100.0}, "test-trigger",
        "2026-08-31", {}, False,
    )


# ════════════════════════════════════════════════════════════════════════════
# BLOCK — verified at the broker, not at the dataclass
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("verdict", [cv.FAIL, cv.UNKNOWN, cv.STALE_BINDING])
def test_no_order_reaches_the_broker_under_a_blocking_verdict(verdict):
    _prime(verdict, cv.MODE_ENFORCE)
    ibkr = _SpyIBKR()
    _call_execute_entry(ibkr)
    assert ibkr.orders == [], (
        f"a {verdict} verdict must withhold the entry AT the broker call; "
        "returning the right verdict object while still placing the order is "
        "the failure mode this test exists for"
    )


def test_the_withheld_entry_is_logged_with_the_operator_message(caplog):
    import logging

    _prime(cv.UNKNOWN, cv.MODE_ENFORCE)
    with caplog.at_level(logging.ERROR):
        _call_execute_entry(_SpyIBKR())
    msgs = [r.getMessage() for r in caplog.records]
    assert any("ENTRY WITHHELD AAPL" in m for m in msgs)
    assert any("UNBLOCK:" in m for m in msgs)
    assert any("STILL RUNNING" in m for m in msgs), (
        "the operator must not read a withheld entry as an unmanaged book"
    )


# ════════════════════════════════════════════════════════════════════════════
# DO NOT BLOCK — the cases where an order MUST still be placed
# ════════════════════════════════════════════════════════════════════════════

def test_a_PASS_verdict_does_not_stand_in_the_way(monkeypatch):
    """A guard that blocks everything passes every block test and is useless.
    Under PASS the entry must reach the code past the gate."""
    _prime(cv.PASS, cv.MODE_ENFORCE)
    reached = []
    monkeypatch.setattr(daemon.logger, "info",
                        lambda *a, **k: reached.append(a))
    # Everything downstream of the gate needs a real sqlite conn and order
    # book, so execution raises once it is past the gate. Getting far enough to
    # raise IS the assertion, and the assertion below is what checks it — the
    # suppression is scoped to this one call, never to the check.
    with contextlib.suppress(Exception):
        _call_execute_entry(_SpyIBKR())
    assert any("BUY %s" in str(a[0]) for a in reached), (
        "under PASS, execution must proceed past the gate to the BUY path"
    )


@pytest.mark.parametrize("verdict", [cv.FAIL, cv.UNKNOWN, cv.STALE_BINDING])
def test_observe_mode_places_the_entry_under_every_bad_verdict(verdict, monkeypatch):
    """§7a. On the guard's first production runs a non-PASS verdict must change
    NOTHING about what the daemon does — the system is trading live paper."""
    _prime(verdict, cv.MODE_OBSERVE)
    reached = []
    monkeypatch.setattr(daemon.logger, "info", lambda *a, **k: reached.append(a))
    with contextlib.suppress(Exception):
        _call_execute_entry(_SpyIBKR())
    assert any("BUY %s" in str(a[0]) for a in reached), (
        "observe mode must not withhold — that is what makes it observe mode"
    )


# ════════════════════════════════════════════════════════════════════════════
# THE CARVE-OUT — no exit path may consult this gate
# ════════════════════════════════════════════════════════════════════════════

def test_the_exit_path_never_consults_the_gate():
    """Structural, deliberately. Blocking an exit would strip the risk
    management off a book sized by exactly the parameters in doubt — §1.2's
    cost-of-not-trading, and the 2026-08-05..07 lost sessions.

    Asserted against the source of `_execute_exit` rather than by driving it,
    because the claim is that the gate is ABSENT from that path — an absence a
    behavioural test can only sample, never establish.
    """
    import inspect

    src = inspect.getsource(daemon._execute_exit)
    assert "_correctness_gate" not in src
    assert "blocks_new_entries" not in src


def test_only_the_entry_path_consults_the_gate():
    """One consulting site, so the withheld set cannot grow silently — §2.3a
    rule 4: 'a gate named after a function grows to cover whatever that
    function later does'."""
    import inspect

    src = inspect.getsource(daemon)
    call_sites = src.count("_correctness_gate(strategy_config)")
    assert call_sites == 1, (
        f"expected exactly one gate consult site, found {call_sites} — every "
        "new one widens the withheld set without that widening being visible"
    )


def test_the_gate_is_primed_once_at_daemon_start():
    """A session with no entries must still record what the verdict was."""
    import inspect

    assert "_prime_correctness_gate(config)" in inspect.getsource(daemon)
