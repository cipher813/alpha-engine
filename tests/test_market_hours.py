"""alpha-engine-config-I7111 — ``executor.market_hours`` owns no NYSE calendar.

Until 2026-08-13 this module carried a second hand-maintained copy of the NYSE
holiday table, byte-for-byte identical to ``krepis.trading_calendar``'s only by
luck. The retired ``sf-watch-market-hours-toggler`` Lambda recorded the
duplication as an owed follow-up in its own source and was deleted without
paying it.

It matters more now than it did then: ``ne-preopen-trading-pipeline`` and
``ne-postclose-trading-pipeline`` gate their first state on the same session
predicate (nousergon-data ``MarketHoursGate``), sourced from krepis. A
divergent table here would mean the daemon and the pipelines disagree about
whether the market is open — the daemon placing orders on a day the pipelines
call closed, or refusing a day they call open.

These assertions make re-introducing a local copy a test failure rather than a
silent drift.
"""

from __future__ import annotations

import importlib
import inspect
import re
from datetime import date, datetime, time

import krepis.trading_calendar as krepis_calendar
import pytz

from executor import market_hours
from executor.market_hours import NYSE_HOLIDAYS, is_market_hours, is_trading_day

_ET = pytz.timezone("US/Eastern")


def _et(y, m, d, hh, mm, ss=0) -> datetime:
    return _ET.localize(datetime(y, m, d, hh, mm, ss))


class TestSingleCalendarSource:
    def test_holiday_table_is_the_krepis_object_not_a_copy(self):
        # Identity, not equality: an equal-but-separate set is exactly the
        # state this test exists to forbid.
        assert NYSE_HOLIDAYS is krepis_calendar.NYSE_HOLIDAYS

    def test_is_trading_day_is_the_krepis_function_not_a_reimplementation(self):
        assert is_trading_day is krepis_calendar.is_trading_day

    def test_module_source_declares_no_holiday_literals(self):
        src = inspect.getsource(market_hours)
        # A re-introduced table would show up as repeated ``date(YYYY, M, D)``
        # constructor literals. One or two would be a doctest-ish example;
        # a calendar is dozens.
        literals = re.findall(r"\bdate\(\s*20\d\d\s*,", src)
        assert len(literals) == 0, (
            f"executor/market_hours.py declares {len(literals)} date literals — "
            "the NYSE calendar must come from krepis.trading_calendar"
        )


class TestSessionBoundary:
    def test_midsession_is_open(self):
        assert is_market_hours(_et(2026, 8, 12, 12, 0)) is True

    def test_open_instant_is_inclusive(self):
        assert is_market_hours(_et(2026, 8, 12, 9, 30, 0)) is True

    def test_close_instant_is_exclusive(self):
        # daemon.py triggers ne-postclose-trading-pipeline only once this
        # returns False. A close-inclusive boundary would delay the EOD
        # chain — and the SF-side gate would refuse the run it starts.
        assert is_market_hours(_et(2026, 8, 12, 16, 0, 0)) is False

    def test_one_second_inside_each_edge(self):
        assert is_market_hours(_et(2026, 8, 12, 9, 29, 59)) is False
        assert is_market_hours(_et(2026, 8, 12, 15, 59, 59)) is True

    def test_weekend_wall_clock_inside_the_window_is_closed(self):
        assert is_market_hours(_et(2026, 8, 15, 12, 0)) is False

    def test_holiday_wall_clock_inside_the_window_is_closed(self):
        # Good Friday 2026 — resolved through the krepis table.
        assert date(2026, 4, 3) in NYSE_HOLIDAYS
        assert is_market_hours(_et(2026, 4, 3, 12, 0)) is False

    def test_naive_input_is_read_as_eastern(self):
        assert is_market_hours(datetime(2026, 8, 12, 12, 0)) is True
        assert is_market_hours(datetime(2026, 8, 12, 8, 0)) is False

    def test_non_eastern_input_is_converted(self):
        import datetime as _dt

        utc_noon_et = datetime(2026, 8, 12, 16, 0, tzinfo=_dt.UTC)
        assert is_market_hours(utc_noon_et) is True


class TestCloseOverride:
    def test_close_override_env_is_honoured(self, monkeypatch):
        # config#1610: MARKET_CLOSE_HOUR/MINUTE exist so the daemon's poll
        # window can be pinned to krepis' session axis. Read at import, so
        # the module has to be reloaded — asserted here rather than assumed.
        import importlib

        monkeypatch.setenv("MARKET_CLOSE_HOUR", "15")
        monkeypatch.setenv("MARKET_CLOSE_MINUTE", "30")
        reloaded = importlib.reload(market_hours)
        try:
            assert reloaded.is_market_hours(_et(2026, 8, 12, 15, 45)) is False
            assert reloaded.is_market_hours(_et(2026, 8, 12, 15, 0)) is True
            assert reloaded._MARKET_CLOSE == time(15, 30)
        finally:
            monkeypatch.undo()
            importlib.reload(market_hours)


class TestDelegatesToKrepis:
    """Replaces ``TestCollapseTrigger``, which fired and was deleted.

    The trigger's job was to make the duplicate impossible to forget. These
    two make it impossible to reintroduce: one asserts the session predicate
    is krepis's, the other asserts the one thing this module still owns — the
    close override — actually reaches it. A reimplementation that happened to
    agree with krepis today would pass neither.
    """

    def test_session_predicate_is_krepis_not_a_local_reimplementation(self, monkeypatch):
        calls = []

        def fake(now=None, *, open_et=None, close_et=None):
            calls.append({"now": now, "open_et": open_et, "close_et": close_et})
            return True

        monkeypatch.setattr(market_hours, "_krepis_is_market_hours", fake)
        assert market_hours.is_market_hours(datetime(2026, 8, 24, 11, 0)) is True
        assert len(calls) == 1, "is_market_hours did not go through krepis"

    def test_the_close_override_reaches_krepis(self, monkeypatch):
        """MARKET_CLOSE_HOUR/MINUTE is the ONE thing krepis cannot know."""
        monkeypatch.setenv("MARKET_CLOSE_HOUR", "15")
        monkeypatch.setenv("MARKET_CLOSE_MINUTE", "30")
        reloaded = importlib.reload(market_hours)
        try:
            calls = []

            def fake(now=None, *, open_et=None, close_et=None):
                calls.append(close_et)
                return False

            monkeypatch.setattr(reloaded, "_krepis_is_market_hours", fake)
            reloaded.is_market_hours(datetime(2026, 8, 24, 15, 45))
            assert calls == [time(15, 30)]
        finally:
            monkeypatch.undo()
            importlib.reload(market_hours)
