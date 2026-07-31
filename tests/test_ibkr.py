"""Unit tests for executor/ibkr.py — IB Gateway wrapper helpers.

IB connection logic is mocked; these tests cover the parsing layer only.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from executor.ibkr import IBKRClient


def _client_with_account_values(values):
    """Build an IBKRClient with a mocked ib.accountValues() response."""
    client = IBKRClient.__new__(IBKRClient)
    client.ib = MagicMock()
    client.ib.isConnected.return_value = True
    client.ib.accountValues.return_value = values
    return client


class TestAccruedDividendsBySymbol:
    def test_empty_account_values(self):
        client = _client_with_account_values([])
        assert client.get_accrued_dividends_by_symbol() == {}

    def test_parses_per_symbol_accruals(self):
        client = _client_with_account_values(
            [
                SimpleNamespace(
                    tag="AccruedDividend", value="12.50", currency="USD", modelCode="AAPL", account="DU123"
                ),
                SimpleNamespace(
                    tag="DividendAccruals", value="7.25", currency="USD", modelCode="MSFT", account="DU123"
                ),
                # No modelCode — a total-level entry, must be ignored here
                SimpleNamespace(tag="AccruedDividend", value="19.75", currency="USD", modelCode="", account="DU123"),
                # Unrelated tag — must be ignored
                SimpleNamespace(tag="NetLiquidation", value="100000", currency="USD", modelCode="", account="DU123"),
            ]
        )
        result = client.get_accrued_dividends_by_symbol()
        assert result == {"AAPL": 12.50, "MSFT": 7.25}

    def test_skips_zero_and_non_numeric(self):
        client = _client_with_account_values(
            [
                SimpleNamespace(tag="AccruedDividend", value="0", currency="USD", modelCode="AAPL", account="DU"),
                SimpleNamespace(
                    tag="AccruedDividend", value="not-a-number", currency="USD", modelCode="MSFT", account="DU"
                ),
                SimpleNamespace(tag="AccruedDividend", value="5.00", currency="USD", modelCode="GOOG", account="DU"),
            ]
        )
        result = client.get_accrued_dividends_by_symbol()
        assert result == {"GOOG": 5.00}

    def test_sums_multiple_entries_for_same_symbol(self):
        """IB sometimes splits a symbol across multiple AccountValue rows."""
        client = _client_with_account_values(
            [
                SimpleNamespace(tag="AccruedDividend", value="3.00", currency="USD", modelCode="AAPL", account="DU"),
                SimpleNamespace(tag="DividendAccruals", value="4.50", currency="USD", modelCode="AAPL", account="DU"),
            ]
        )
        result = client.get_accrued_dividends_by_symbol()
        assert result == {"AAPL": 7.50}


class TestInitialConnectRetry:
    """The constructor must retry a transient connect failure, not hard-fail.

    Regression for the 2026-06-05 weekday-SF failure: the morning planner's
    only IB touchpoint is ``IBKRClient.__init__``, which used to do a single
    bare ``connect()``. An IB Gateway ``reqExecutions`` stall mid-handshake
    raised ``TimeoutError`` and nuked the whole pipeline.
    """

    def _fake_ib(self, monkeypatch, connect_side_effect):
        import executor.ibkr as ibkr_mod
        import executor.retry as retry_mod

        fake_ib = MagicMock()
        fake_ib.connect.side_effect = connect_side_effect
        monkeypatch.setattr(ibkr_mod, "IB", lambda: fake_ib)
        monkeypatch.setattr(retry_mod.time, "sleep", lambda _s: None)  # no real backoff
        return fake_ib

    def test_retries_then_succeeds(self, monkeypatch):
        state = {"connected": False, "calls": 0}

        def connect_side(*_a, **_k):
            state["calls"] += 1
            if state["calls"] == 1:
                raise TimeoutError("reqExecutions stalled mid-handshake")
            state["connected"] = True

        fake_ib = self._fake_ib(monkeypatch, connect_side)
        fake_ib.isConnected.side_effect = lambda: state["connected"]

        client = IBKRClient()  # must not raise

        assert state["calls"] == 2  # one transient failure, one success
        assert client.ib.isConnected()
        # half-open socket / stale clientId cleared before the retry
        assert fake_ib.disconnect.called

    def test_raises_after_exhausting_attempts(self, monkeypatch):
        fake_ib = self._fake_ib(monkeypatch, TimeoutError("gateway down"))
        fake_ib.isConnected.return_value = False

        with pytest.raises(TimeoutError):
            IBKRClient(reconnect_attempts=2)

        assert fake_ib.connect.call_count == 2  # honors reconnect_attempts, then raises loud


class TestGetCurrentPricePolling:
    """get_current_price must poll for a tick, not read once after a fixed
    sleep.

    Regression for 2026-06-29: a momentary IB data-farm delay returned nan
    for every ticker within the old single ``sleep(1)`` window, producing a
    0-entry order book and silently dropping every optimizer allocation
    (GE's 8% target among them). Bounded polling absorbs the cold-start /
    hiccup window; a genuinely unpriceable contract still returns None.
    """

    def _client(self):
        client = IBKRClient.__new__(IBKRClient)
        client.ib = MagicMock()
        client.ib.isConnected.return_value = True
        return client

    def _wire_ticker(self, client, tick_values):
        """tick_values: list of (last, close) revealed on successive sleeps.

        The Ticker starts nan/nan and is updated in place each ib.sleep(),
        mirroring how ib_insync mutates the live Ticker as ticks arrive.
        """
        ticker = SimpleNamespace(last=float("nan"), close=float("nan"))
        client.ib.reqMktData.return_value = ticker
        seq = list(tick_values)

        def _sleep(_interval):
            if seq:
                last, close = seq.pop(0)
                ticker.last, ticker.close = last, close

        client.ib.sleep.side_effect = _sleep
        return ticker

    def test_returns_price_once_tick_arrives(self):
        client = self._client()
        # nan for the first two polls, then a valid last price.
        self._wire_ticker(client, [(float("nan"), float("nan")), (float("nan"), float("nan")), (231.5, 230.0)])
        price = client.get_current_price("GE", max_wait=6.0, poll_interval=0.5)
        assert price == 231.5
        assert client.ib.sleep.call_count == 3  # stopped as soon as valid
        client.ib.cancelMktData.assert_called_once()  # subscription released

    def test_falls_back_to_close_when_no_last(self):
        client = self._client()
        self._wire_ticker(client, [(float("nan"), 99.0)])
        assert client.get_current_price("SPY", max_wait=2.0, poll_interval=0.5) == 99.0

    def test_returns_none_after_deadline_when_never_priced(self):
        client = self._client()
        self._wire_ticker(client, [])  # never reveals a price
        price = client.get_current_price("XYZ", max_wait=1.0, poll_interval=0.5)
        assert price is None
        # polled to the deadline (1.0 / 0.5 = 2 polls), then gave up
        assert client.ib.sleep.call_count == 2
        client.ib.cancelMktData.assert_called_once()  # released even on failure

    def test_rejects_nonpositive_price(self):
        client = self._client()
        self._wire_ticker(client, [(0.0, -1.0)])
        assert client.get_current_price("BAD", max_wait=0.5, poll_interval=0.5) is None


class TestAccountSummaryStallGuard:
    """``reqAccountSummary`` must be bounded, not an indefinite await.

    Regression for 2026-07-24 and 2026-07-27: ``ne-preopen-trading-pipeline``
    failed at ``MorningPlannerPollTimeout`` both days. The planner connected,
    logged "Connected to IB Gateway", and then emitted nothing but
    ``updatePortfolio`` until SSM killed it at the 600s ``executionTimeout``
    — ``ib_insync``'s ``accountSummary()`` awaits ``accountSummaryEnd`` with
    no timeout, so a gateway that accepts the socket but never answers the
    request hangs the planner silently. No order book, no daemon, no trading.
    """

    def _client(self, monkeypatch, summary_side_effect):
        import executor.ibkr as ibkr_mod

        client = IBKRClient.__new__(IBKRClient)
        client.ib = MagicMock()
        client.ib.isConnected.return_value = True
        client.ib.run.side_effect = summary_side_effect
        client.ib.accountSummary.return_value = [
            SimpleNamespace(tag="NetLiquidation", value="1000000.00"),
        ]
        client._account_summary_primed = False
        client._host, client._port = "127.0.0.1", 4002
        client._client_id, client._reconnect_attempts = 1, 3
        monkeypatch.setattr(ibkr_mod, "ACCOUNT_SUMMARY_ATTEMPTS", 3)
        return client

    def test_bounded_wait_is_applied(self, monkeypatch):
        """The request is awaited under a timeout, never bare."""
        import executor.ibkr as ibkr_mod

        seen = {}

        def _run(awaitable):
            seen["awaited"] = awaitable
            awaitable.close()  # don't leave the coroutine un-awaited

        client = self._client(monkeypatch, _run)
        assert client.get_portfolio_nav() == 1000000.00
        # asyncio.wait_for() returns a coroutine wrapping the request —
        # a bare reqAccountSummaryAsync() would not be wrapped at all.
        assert seen["awaited"].__qualname__.startswith("wait_for")
        assert ibkr_mod.ACCOUNT_SUMMARY_TIMEOUT_SECONDS > 0

    def test_stall_then_success_reconnects_between_attempts(self, monkeypatch):
        """A stalled attempt reconnects — the stuck subscription is bound to
        this clientId, so retrying on the same socket inherits the stall."""
        calls = {"n": 0}

        def _run(awaitable):
            awaitable.close()
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("accountSummaryEnd never arrived")

        client = self._client(monkeypatch, _run)
        monkeypatch.setattr(client, "_connect", lambda: None)

        assert client.get_portfolio_nav() == 1000000.00
        assert calls["n"] == 2
        assert client.ib.disconnect.called

    def test_persistent_stall_raises_loud(self, monkeypatch):
        """Never degrade to a stale/assumed NAV — position sizing and the
        drawdown circuit breaker both read it."""

        def _run(awaitable):
            awaitable.close()
            raise TimeoutError("accountSummaryEnd never arrived")

        client = self._client(monkeypatch, _run)
        monkeypatch.setattr(client, "_connect", lambda: None)

        with pytest.raises(RuntimeError, match="account summary stalled"):
            client.get_portfolio_nav()

    def test_primed_once_then_served_from_cache(self, monkeypatch):
        """The handshake is one-shot: a second read must not re-request."""
        calls = {"n": 0}

        def _run(awaitable):
            awaitable.close()
            calls["n"] += 1

        client = self._client(monkeypatch, _run)
        client.get_portfolio_nav()
        client.get_account_snapshot()
        assert calls["n"] == 1

    def test_reconnect_clears_the_prime(self, monkeypatch):
        """ib_insync drops its cached summary on disconnect, so a reconnected
        client must re-issue rather than trust a stale prime."""
        import executor.ibkr as ibkr_mod
        import executor.retry as retry_mod

        fake_ib = MagicMock()
        fake_ib.isConnected.return_value = True
        monkeypatch.setattr(ibkr_mod, "IB", lambda: fake_ib)
        monkeypatch.setattr(retry_mod.time, "sleep", lambda _s: None)

        client = IBKRClient()
        client._account_summary_primed = True
        client._connect()
        assert client._account_summary_primed is False
