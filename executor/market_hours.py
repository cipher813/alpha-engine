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

``is_market_hours`` below is the ONE remaining local copy of logic that also
exists as ``krepis.trading_calendar.is_market_hours`` (krepis >= 0.55.0). It
stays local only because this repo's uv lockfile pins ``krepis==0.16.2`` and
re-resolving it is a lockfile-wide change on the trading daemon, unrelated to
the boundary this ships. Collapse this into a re-export at the next krepis
pin bump — tracked as alpha-engine-config-I7149. The two implementations are
kept from outliving the pin by
``tests/test_market_hours.py::TestCollapseTrigger``, which fails the moment the
pinned krepis exposes ``is_market_hours``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, time

import pytz

# Single source of the NYSE calendar. Do NOT re-declare either name here.
from krepis.trading_calendar import NYSE_HOLIDAYS, is_trading_day

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

    Checks: weekday AND not a holiday AND between 9:30 AM – 4:00 PM Eastern.

    The close is EXCLUSIVE. ``daemon.py`` reads that boundary directly: it
    triggers ``ne-postclose-trading-pipeline`` only once this returns False
    (``daemon.py`` — ``market_opened and not is_market_hours()``), which is
    why every observed ``eod-*`` execution starts at 16:00:0x ET rather than
    inside the session.
    """
    if now is None:
        now = datetime.now(_ET)
    elif now.tzinfo is None:
        now = _ET.localize(now)
    else:
        now = now.astimezone(_ET)

    if not is_trading_day(now.date()):
        logger.info("Market closed: %s is not an NYSE trading day", now.date())
        return False

    current_time = now.time()
    if current_time < _MARKET_OPEN or current_time >= _MARKET_CLOSE:
        logger.info(
            "Market closed: current time %s ET is outside %s-%s",
            current_time.strftime("%H:%M"),
            _MARKET_OPEN.strftime("%H:%M"),
            _MARKET_CLOSE.strftime("%H:%M"),
        )
        return False

    return True
