"""Tests for explicit ex-date dividend accrual and total-return SPY
(alpha-engine-config-I8188, defect 3).

The load-bearing test is
``TestAccrual::test_a_dividend_paying_holding_produces_a_non_zero_accrual`` —
``dividend_usd`` summed to exactly $0.00 across all 115 live sessions while the
book held seven payers, and nothing in the suite could have noticed.
"""

from __future__ import annotations

import pytest

from executor.dividends import (
    SPY_TICKER,
    accrue_position_dividends,
    fetch_ex_dividends,
    spy_total_return_pct,
)


class _FakeClient:
    """Minimal PolygonClient stand-in: ``get_dividends(ticker, start=...)``."""

    def __init__(self, by_ticker, fail=()):
        self._by_ticker = by_ticker
        self._fail = set(fail)
        self.calls = []

    def get_dividends(self, ticker, start=None, limit=1000):
        self.calls.append((ticker, start))
        if ticker in self._fail:
            raise RuntimeError("polygon 500")
        return self._by_ticker.get(ticker, [])


class TestFetch:
    def test_an_ex_date_in_the_interval_is_picked_up(self):
        client = _FakeClient({
            "LMT": [{"ex_dividend_date": "2026-08-21", "cash_amount": 3.30}],
        })
        out, pay_dates, available, warning = fetch_ex_dividends(
            ["LMT"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert out == {"LMT": 3.30}
        assert pay_dates == {"LMT": None}
        assert available is True
        assert warning is None

    def test_the_pay_date_is_carried_for_the_receivable(self):
        """NAV is cash-basis on this account, so the accrual ledger has to know
        when the cash actually lands or the receivable never releases."""
        client = _FakeClient({
            "LMT": [{"ex_dividend_date": "2026-08-21", "cash_amount": 3.30,
                     "pay_date": "2026-09-26"}],
        })
        _out, pay_dates, _a, _w = fetch_ex_dividends(
            ["LMT"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert pay_dates["LMT"] == "2026-09-26"

    def test_two_distributions_in_one_interval_take_the_later_pay_date(self):
        client = _FakeClient({
            "LMT": [
                {"ex_dividend_date": "2026-08-21", "cash_amount": 3.30,
                 "pay_date": "2026-09-26"},
                {"ex_dividend_date": "2026-08-21", "cash_amount": 1.00,
                 "pay_date": "2026-10-02"},
            ],
        })
        out, pay_dates, _a, _w = fetch_ex_dividends(
            ["LMT"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert out["LMT"] == pytest.approx(4.30)
        assert pay_dates["LMT"] == "2026-10-02"

    def test_a_dividend_outside_the_interval_is_ignored(self):
        client = _FakeClient({
            "LMT": [
                {"ex_dividend_date": "2026-08-20", "cash_amount": 3.30},
                {"ex_dividend_date": "2026-08-25", "cash_amount": 3.30},
            ],
        })
        out, _pay, available, _ = fetch_ex_dividends(
            ["LMT"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert out == {}
        assert available is True

    def test_the_interval_spans_a_skipped_session(self):
        """The NAV baseline spans from the prior persisted eod_pnl row, so the
        dividend leg must span the same window or the two sides of the
        reconciliation measure different intervals."""
        client = _FakeClient({
            "CTAS": [{"ex_dividend_date": "2026-06-24", "cash_amount": 1.56}],
        })
        out, _pay, _, _ = fetch_ex_dividends(
            ["CTAS"], "2026-06-25", prior_date="2026-06-23", client=client,
        )
        assert out == {"CTAS": 1.56}

    def test_spy_is_always_fetched_for_the_benchmark_leg(self):
        client = _FakeClient({})
        fetch_ex_dividends(["LMT"], "2026-08-21", prior_date="2026-08-20",
                           client=client)
        assert SPY_TICKER in {t for t, _ in client.calls}

    def test_no_dividends_today_is_available_true(self):
        """An empty result and an unavailable feed are DIFFERENT facts —
        collapsing them is exactly the defect being fixed."""
        out, _pay, available, warning = fetch_ex_dividends(
            ["AMD"], "2026-08-21", prior_date="2026-08-20", client=_FakeClient({}),
        )
        assert out == {}
        assert available is True
        assert warning is None

    def test_total_feed_failure_is_available_false_with_a_warning(self):
        client = _FakeClient({}, fail={"AMD", SPY_TICKER})
        out, _pay, available, warning = fetch_ex_dividends(
            ["AMD"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert out == {}
        assert available is False
        assert "unattributed residual" in warning

    def test_partial_failure_is_named_not_swallowed(self):
        client = _FakeClient(
            {SPY_TICKER: [{"ex_dividend_date": "2026-08-21", "cash_amount": 1.80}]},
            fail={"AMD"},
        )
        out, _pay, available, warning = fetch_ex_dividends(
            ["AMD"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert out == {SPY_TICKER: 1.80}
        assert available is True
        assert "partial" in warning.lower()

    def test_malformed_rows_are_skipped(self):
        client = _FakeClient({
            "LMT": [
                {"ex_dividend_date": None, "cash_amount": 3.30},
                {"ex_dividend_date": "2026-08-21", "cash_amount": None},
                {"ex_dividend_date": "2026-08-21", "cash_amount": "x"},
                {"ex_dividend_date": "2026-08-21", "cash_amount": -1.0},
                {"ex_dividend_date": "2026-08-21", "cash_amount": 3.30},
            ],
        })
        out, _pay, _, _ = fetch_ex_dividends(
            ["LMT"], "2026-08-21", prior_date="2026-08-20", client=client,
        )
        assert out == {"LMT": 3.30}


class TestAccrual:
    def test_a_dividend_paying_holding_produces_a_non_zero_accrual(self):
        """The regression that could not previously be caught: LMT was held
        for 34 sessions and contributed exactly $0.00 of dividend for all of
        them."""
        positions = {"LMT": {"shares": 200, "prior_price": 500.0,
                             "daily_return_usd": 0.0, "daily_return_pct": 0.0}}
        accruals = accrue_position_dividends(positions, {"LMT": {"shares": 200}},
                                             {"LMT": 3.30})
        assert [a["ticker"] for a in accruals] == ["LMT"]
        assert accruals[0]["amount_usd"] == pytest.approx(660.0)
        assert positions["LMT"]["dividend_usd"] == pytest.approx(660.0)
        assert positions["LMT"]["daily_return_usd"] == pytest.approx(660.0)
        assert positions["LMT"]["daily_return_pct"] == pytest.approx(0.66)

    def test_entitlement_uses_the_prior_close_share_count(self):
        """A name bought ON the ex-date does not receive that dividend; a name
        sold on the ex-date still does."""
        positions = {"LMT": {"shares": 500, "prior_price": 500.0,
                             "daily_return_usd": 0.0, "daily_return_pct": 0.0}}
        accruals = accrue_position_dividends(positions, {"LMT": {"shares": 200}},
                                             {"LMT": 3.30})
        assert accruals[0]["amount_usd"] == pytest.approx(660.0)
        assert accruals[0]["shares"] == pytest.approx(200)

    def test_no_prior_snapshot_falls_back_to_todays_shares(self):
        positions = {"LMT": {"shares": 200, "prior_price": 500.0,
                             "daily_return_usd": 0.0, "daily_return_pct": 0.0}}
        accruals = accrue_position_dividends(positions, {}, {"LMT": 3.30})
        assert accruals[0]["amount_usd"] == pytest.approx(660.0)

    def test_broker_accrual_wins_and_is_not_double_counted(self):
        positions = {"LMT": {"shares": 200, "prior_price": 500.0,
                             "dividend_usd": 660.0,
                             "daily_return_usd": 660.0}}
        assert accrue_position_dividends(positions, {"LMT": {"shares": 200}},
                                         {"LMT": 3.30}) == []
        assert positions["LMT"]["dividend_usd"] == pytest.approx(660.0)

    def test_spy_is_accrued_like_any_other_holding(self):
        """SPY is the enhanced-index core position. Accruing its dividend on
        the BOOK's shares is not a double count against the BENCHMARK's total
        return — they are the two sides of the comparison."""
        positions = {SPY_TICKER: {"shares": 160, "prior_price": 762.60,
                                  "daily_return_usd": 0.0,
                                  "daily_return_pct": 0.0}}
        accruals = accrue_position_dividends(
            positions, {SPY_TICKER: {"shares": 160}}, {SPY_TICKER: 1.80})
        assert accruals[0]["amount_usd"] == pytest.approx(288.0)

    def test_a_name_with_no_dividend_is_untouched(self):
        positions = {"AMD": {"shares": 100, "prior_price": 100.0,
                             "daily_return_usd": 50.0}}
        assert accrue_position_dividends(positions, {}, {"LMT": 3.30}) == []
        assert "dividend_usd" not in positions["AMD"]

    def test_zero_share_position_accrues_nothing(self):
        positions = {"LMT": {"shares": 0, "prior_price": 500.0,
                             "daily_return_usd": 0.0}}
        assert accrue_position_dividends(positions, {"LMT": {"shares": 0}},
                                         {"LMT": 3.30}) == []


class TestSpyTotalReturn:
    def test_the_benchmark_is_total_return_not_price_return(self):
        price_only = spy_total_return_pct(
            spy_close=765.72, prior_spy_close=762.60, spy_dividend_per_share=0.0)
        total = spy_total_return_pct(
            spy_close=765.72, prior_spy_close=762.60, spy_dividend_per_share=1.80)
        assert total > price_only
        assert total - price_only == pytest.approx(1.80 / 762.60 * 100)

    def test_no_distribution_day_matches_price_return(self):
        assert spy_total_return_pct(
            spy_close=765.72, prior_spy_close=762.60
        ) == pytest.approx((765.72 / 762.60 - 1) * 100)

    def test_missing_leg_is_none_not_zero(self):
        assert spy_total_return_pct(spy_close=None, prior_spy_close=762.60) is None
        assert spy_total_return_pct(spy_close=765.72, prior_spy_close=None) is None
