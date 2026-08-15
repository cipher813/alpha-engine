"""An unbounded diagnostic evicts the ERROR beside it (alpha-engine-config-I7396).

``decide_exits_and_reduces`` is called once per simulated DATE, and the exit signal set
spans the whole universe, so ``SKIP EXIT <TICKER> — not in portfolio`` at INFO
emitted hundreds of lines per date and tens of thousands per backtest.

On 2026-08-15 that consumed the entire SSM diagnostic window for a FAILED
``predictor-backtest`` stage: the ``stdout_tail`` in
``_spot_diagnostics/ae-predictor-backtest/2026-08-15.json`` was end-to-end
``SKIP EXIT`` roster lines and carried no trace of the exception that killed
the run. The artifact still existed, and still looked like evidence.

Same class as ``alpha-engine-config-I7021`` (2026-08-12), recurring on a new
emitter — which is why this is pinned by a test rather than by a comment.
"""

from __future__ import annotations

import logging

from executor.deciders import decide_exits_and_reduces


def _positions(tickers):
    return {t: {"shares": 100, "avg_cost": 10.0, "sector": "Tech"} for t in tickers}


def _exit_signals(tickers):
    return {"exit": [{"ticker": t, "reason": "research"} for t in tickers]}


def _plan(caplog, held, exiting, level=logging.INFO):
    with caplog.at_level(level, logger="executor.deciders"):
        decide_exits_and_reduces(
            signals=_exit_signals(exiting),
            strategy_exits=[],
            current_positions=_positions(held),
            prices_now=dict.fromkeys(held, 10.0),
            predictions_by_ticker={},
            config={},
            market_regime="neutral",
            portfolio_nav=100_000.0,
            run_date="2026-08-15",
            signals_date="2026-08-15",
            predictions_date="2026-08-15",
        )
    return caplog


def _skip_lines(caplog):
    return [r.getMessage() for r in caplog.records if "SKIP EXIT" in r.getMessage()]


class TestTheRosterIsNotTheDiagnostic:
    def test_a_hundred_unheld_names_produce_one_info_line(self, caplog):
        names = [f"TK{i:03d}" for i in range(100)]
        _plan(caplog, held=[], exiting=names)
        lines = _skip_lines(caplog)
        assert len(lines) == 1, (
            f"{len(lines)} SKIP EXIT lines at INFO for 100 unheld names — the "
            "roster is back and will evict the next real error beside it"
        )

    def test_that_line_carries_the_count(self, caplog):
        names = [f"TK{i:03d}" for i in range(100)]
        _plan(caplog, held=[], exiting=names)
        line = _skip_lines(caplog)[0]
        assert "100 of 100" in line

    def test_the_named_tickers_are_capped(self, caplog):
        """The summary must not become the chatter it replaces."""
        names = [f"TK{i:03d}" for i in range(100)]
        _plan(caplog, held=[], exiting=names)
        line = _skip_lines(caplog)[0]
        assert line.count("TK") <= 8
        assert "+92 more" in line

    def test_no_summary_line_when_every_exit_is_held(self, caplog):
        _plan(caplog, held=["AAA", "BBB"], exiting=["AAA", "BBB"])
        assert _skip_lines(caplog) == []

    def test_a_small_skip_set_is_named_in_full(self, caplog):
        _plan(caplog, held=["AAA"], exiting=["AAA", "BBB", "CCC"])
        line = _skip_lines(caplog)[0]
        assert "BBB" in line and "CCC" in line
        assert "more" not in line

    def test_per_ticker_detail_survives_at_debug(self, caplog):
        """Reconciling one name is a real need; doing it 40,000 times is not."""
        _plan(caplog, held=[], exiting=["AAA", "BBB"], level=logging.DEBUG)
        debug = [r.getMessage() for r in caplog.records
                 if r.levelno == logging.DEBUG and "SKIP EXIT" in r.getMessage()]
        assert any("AAA" in m for m in debug)
        assert any("BBB" in m for m in debug)


def test_the_exit_plan_itself_is_unchanged_by_the_logging_change(caplog):
    """Guard against 'fixed the log, moved the behaviour'."""
    plan = None
    with caplog.at_level(logging.INFO, logger="executor.deciders"):
        plan = decide_exits_and_reduces(
            signals=_exit_signals(["AAA", "GHOST"]),
            strategy_exits=[],
            current_positions=_positions(["AAA"]),
            prices_now={"AAA": 10.0},
            predictions_by_ticker={},
            config={},
            market_regime="neutral",
            portfolio_nav=100_000.0,
            run_date="2026-08-15",
            signals_date="2026-08-15",
            predictions_date="2026-08-15",
        )
    exited = [o["ticker"] for o in plan.urgent_exits_with_meta]
    assert exited == ["AAA"], "the unheld name must still be skipped, not ordered"
