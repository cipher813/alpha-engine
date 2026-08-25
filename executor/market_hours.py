"""
Market hours validation — prevents order placement outside regular trading hours.

NYSE regular session: 9:30 AM – 4:00 PM Eastern, weekdays only.

The NYSE holiday calendar is NOT maintained here. ``NYSE_HOLIDAYS`` and
``is_trading_day`` are re-exported from ``krepis.trading_calendar``, which is
the fleet's single owner of the NYSE calendar (alpha-engine-config-I7111).

Until 2026-08-13 this module carried its own hand-maintained copy of the
holiday table — byte-for-byte identical to krepis's at the time of the
merge, and drifting only by luck. The retired ``sf-watch-market-hours-toggler``
Lambda recorded that duplication as an owed follow-up in its own source and
was then deleted without paying it; two Step Functions pipelines now gate on
this predicate (``ne-preopen-trading-pipeline`` and
``ne-postclose-trading-pipeline``, nousergon-data), so a divergent table
would mean the daemon and the pipelines disagree about whether the market
is open.

``is_market_hours`` was the ONE remaining local copy of logic that also exists
as ``krepis.trading_calendar.is_market_hours``. It was collapsed into a thin
delegation on 2026-08-24 when the lockfile recompile moved the pin from
``krepis==0.54.0`` to ``0.59.33`` (alpha-engine-config-I7149, I8309). The
collapse landed in the same PR as the bump because
``tests/test_market_hours.py::TestCollapseTrigger`` was wired to the condition
rather than a date and failed the moment the pinned krepis exposed the name —
which is exactly what it was for, and it is deleted now that it has fired.

What stays local is the ONE thing krepis cannot know: the
``MARKET_CLOSE_HOUR`` / ``MARKET_CLOSE_MINUTE`` environment override, read at
IMPORT time (unchanged — ``tests/test_market_hours.py`` reloads the module to
exercise it), and passed through as ``close_et``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, time

import pytz

# Single source of the NYSE calendar. Do NOT re-declare either name here.
from krepis.trading_calendar import (
    NYSE_HOLIDAYS,
    is_trading_day,
)
from krepis.trading_calendar import (
    is_market_hours as _krepis_is_market_hours,
)

logger = logging.getLogger(__name__)

__all__ = ["NYSE_HOLIDAYS", "is_market_hours", "is_trading_day"]

_ET = pytz.timezone("US/Eastern")
_MARKET_OPEN = time(9, 30)
# Default 16:00 ET — must match krepis ``session_date`` / ``assert_within_session``
# (config#1610). A later close (the old 16:15 default) left the daemon polling
# for 15 minutes after the session axis rolled, spamming nav_series guard
# ERRORs into flow-doctor #ops-health every tick.
_MARKET_CLOSE = time(
    int(os.environ.get("MARKET_CLOSE_HOUR", "16")),
    int(os.environ.get("MARKET_CLOSE_MINUTE", "0")),
)


def is_market_hours(now: datetime | None = None) -> bool:
    """
    Return True if the current time is during NYSE regular trading hours.

    Delegates to ``krepis.trading_calendar.is_market_hours`` — one holiday
    table, one session window, one definition of "the market is open". The
    only thing passed through is this repo's environment-driven close
    override; the open is 09:30 ET and is krepis's own default, stated here
    rather than implied.

    The close is EXCLUSIVE, in krepis as it was here. ``daemon.py`` reads that
    boundary directly: it triggers ``ne-postclose-trading-pipeline`` only once
    this returns False (``daemon.py`` — ``market_opened and not
    is_market_hours()``), which is why every observed ``eod-*`` execution
    starts at 16:00:0x ET rather than inside the session. A close-inclusive
    boundary would delay that chain by a poll interval every single day.

    The INFO line on a closed market is kept: it is the daemon's own record of
    why it did not act on a given tick, and krepis (a library) does not log it.
    """
    if now is None:
        now = datetime.now(_ET)
    elif now.tzinfo is None:
        now = _ET.localize(now)
    else:
        now = now.astimezone(_ET)

    open_now = _krepis_is_market_hours(
        now, open_et=_MARKET_OPEN, close_et=_MARKET_CLOSE,
    )
    if not open_now:
        logger.info(
            "Market closed: %s ET is outside the NYSE regular session %s-%s, "
            "or is not a trading day",
            now.strftime("%Y-%m-%d %H:%M"),
            _MARKET_OPEN.strftime("%H:%M"),
            _MARKET_CLOSE.strftime("%H:%M"),
        )
    return open_now
