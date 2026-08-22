"""Schema round-trip for the integrity + cost columns (alpha-engine-config-I8188).

The gates themselves are unit-tested in ``test_pnl_integrity.py``. What this
file protects is the boundary that actually failed in production: a value that
is computed, logged, and then never persisted. ``rotation_realized_usd`` and
``pricing_timing_usd`` were computed on every EOD run since config#2457 and
existed only in the log line — which is why eod_pnl.csv carried one
undifferentiated -$20,293 plug and the sleeve that explains -$20,815 of it was
invisible to every consumer.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from executor.pnl_integrity import plan_twr_self_heal, verify_twr_closes
from executor.trade_logger import (
    dividend_receivable_usd,
    init_db,
    log_eod,
    log_trade,
    record_dividend_accrual,
    settle_due_dividend_accruals,
)


@pytest.fixture()
def conn(tmp_path):
    c = init_db(str(tmp_path / "trades.db"))
    yield c
    c.close()


def _columns(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


class TestSchema:
    def test_every_new_eod_column_exists(self, conn):
        assert {
            "pricing_timing_usd", "rotation_realized_usd", "unattributed_true_usd",
            "commission_usd", "slippage_usd", "traded_notional_usd",
            "commission_available", "daily_return_net_pct",
            "daily_return_gross_pct", "dividend_accrual_available",
            "spy_dividend_per_share", "integrity_breach_json",
            "dividend_timing_usd", "dividend_receivable_usd",
        } <= _columns(conn, "eod_pnl")

    def test_trades_carries_a_commission_column(self, conn):
        assert "commission_usd" in _columns(conn, "trades")

    def test_migrations_are_idempotent(self, tmp_path):
        path = str(tmp_path / "twice.db")
        init_db(path).close()
        c = init_db(path)
        assert "commission_usd" in _columns(c, "eod_pnl")
        c.close()


class TestRoundTrip:
    def test_the_sleeves_survive_the_write(self, conn):
        """The exact failure mode: computed, logged, never persisted."""
        log_eod(conn, {
            "date": "2026-08-21", "portfolio_nav": 1_044_442.88,
            "daily_return_pct": 0.1, "spy_return_pct": 0.05,
            "nav_change_usd": -1_000.0, "position_pnl_usd": -500.0,
            "interest_usd": -1.0, "dividend_usd": 660.0,
            "unattributed_usd": -499.0,
            "pricing_timing_usd": -120.0,
            "rotation_realized_usd": -350.0,
            "unattributed_true_usd": -29.0,
            "commission_usd": 4.25, "slippage_usd": 172.37,
            "traded_notional_usd": 300_981.10,
            "commission_available": True,
            "daily_return_net_pct": 0.1, "daily_return_gross_pct": 0.117,
            "dividend_accrual_available": True,
            "spy_dividend_per_share": 1.80,
            "integrity_breach_json": None,
        })
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM eod_pnl WHERE date='2026-08-21'").fetchone()
        conn.row_factory = None
        assert row["rotation_realized_usd"] == pytest.approx(-350.0)
        assert row["pricing_timing_usd"] == pytest.approx(-120.0)
        assert row["unattributed_true_usd"] == pytest.approx(-29.0)
        assert row["commission_usd"] == pytest.approx(4.25)
        assert row["slippage_usd"] == pytest.approx(172.37)
        assert row["daily_return_gross_pct"] == pytest.approx(0.117)
        assert row["spy_dividend_per_share"] == pytest.approx(1.80)

    def test_availability_flags_are_tri_state(self, conn):
        """NULL ("not evaluated") must stay distinct from 0 ("evaluated, no").
        An absent measurement rendering as a measured negative is the whole
        class of defect this issue records."""
        log_eod(conn, {"date": "2026-08-19", "portfolio_nav": 1.0})
        log_eod(conn, {"date": "2026-08-20", "portfolio_nav": 1.0,
                       "commission_available": False,
                       "dividend_accrual_available": False})
        log_eod(conn, {"date": "2026-08-21", "portfolio_nav": 1.0,
                       "commission_available": True,
                       "dividend_accrual_available": True})
        got = {
            r[0]: (r[1], r[2]) for r in conn.execute(
                "SELECT date, commission_available, dividend_accrual_available "
                "FROM eod_pnl ORDER BY date")
        }
        assert got["2026-08-19"] == (None, None)
        assert got["2026-08-20"] == (0, 0)
        assert got["2026-08-21"] == (1, 1)

    def test_breach_detail_is_persisted_as_json(self, conn):
        breaches = [{"kind": "custodian_mark", "ticker": "AMD",
                     "error_usd": -5220.0, "message": "..."}]
        log_eod(conn, {"date": "2026-08-04", "portfolio_nav": 1_036_143.44,
                       "integrity_breach_json": json.dumps(breaches)})
        stored = conn.execute(
            "SELECT integrity_breach_json FROM eod_pnl WHERE date='2026-08-04'"
        ).fetchone()[0]
        assert json.loads(stored)[0]["ticker"] == "AMD"

    def test_trade_commission_round_trips_and_stays_nullable(self, conn):
        log_trade(conn, {"date": "2026-08-21", "ticker": "AMD",
                         "action": "ENTER", "shares": 10,
                         "commission_usd": 1.25})
        log_trade(conn, {"date": "2026-08-21", "ticker": "MU",
                         "action": "ENTER", "shares": 10})
        got = {
            r[0]: r[1] for r in conn.execute(
                "SELECT ticker, commission_usd FROM trades")
        }
        assert got["AMD"] == pytest.approx(1.25)
        assert got["MU"] is None


class TestTwrAgainstPersistedRows:
    def _rows(self, conn):
        return [
            {"date": r[0], "portfolio_nav": r[1], "daily_return_pct": r[2]}
            for r in conn.execute(
                "SELECT date, portfolio_nav, daily_return_pct FROM eod_pnl "
                "WHERE portfolio_nav IS NOT NULL ORDER BY date")
        ]

    def test_the_live_defect_is_reproduced_and_repaired_in_sqlite(self, conn):
        """The 2026-04-07 row, from the real NAVs: its stored return was
        computed against a 2026-04-06 NAV that was corrected afterwards."""
        navs = [("2026-04-03", 1_002_698.68, 0.021623),
                ("2026-04-06", 1_009_473.08, 0.675617),
                ("2026-04-07", 1_008_056.68, 0.026827)]
        for d, nav, pct in navs:
            log_eod(conn, {"date": d, "portfolio_nav": nav,
                           "daily_return_pct": pct, "spy_return_pct": 0.0})

        before = verify_twr_closes(self._rows(conn))
        assert before["closes"] is False
        assert [o["date"] for o in before["offenders"]] == ["2026-04-07"]

        plan = plan_twr_self_heal(self._rows(conn))
        assert plan["refused"] == []
        for c in plan["corrections"]:
            conn.execute(
                "UPDATE eod_pnl SET daily_return_pct = ?, "
                "daily_alpha_pct = CASE WHEN spy_return_pct IS NULL THEN NULL "
                "ELSE ? - spy_return_pct END WHERE date = ?",
                (c["to_pct"], c["to_pct"], c["date"]),
            )
        conn.commit()

        after = verify_twr_closes(self._rows(conn))
        assert after["closes"] is True
        healed = conn.execute(
            "SELECT daily_return_pct, daily_alpha_pct FROM eod_pnl "
            "WHERE date='2026-04-07'").fetchone()
        assert healed[0] == pytest.approx(-0.140311, abs=1e-5)
        assert healed[1] == pytest.approx(-0.140311, abs=1e-5)

    def test_a_clean_persisted_series_needs_no_repair(self, conn):
        prior = None
        for d, nav in [("2026-08-19", 1_055_842.43), ("2026-08-20", 1_050_000.0),
                       ("2026-08-21", 1_044_442.88)]:
            pct = 0.0 if prior is None else (nav / prior - 1) * 100
            log_eod(conn, {"date": d, "portfolio_nav": nav,
                           "daily_return_pct": pct})
            prior = nav
        assert plan_twr_self_heal(self._rows(conn)) == {
            "corrections": [], "refused": []}
        assert verify_twr_closes(self._rows(conn))["closes"] is True


class TestDividendAccrualLedger:
    """The receivable that keeps the reconciliation identity closed on BOTH
    the ex-date and the pay date.

    NAV on this account is CASH-BASIS: IB reports no per-symbol
    ``AccruedDividend`` and ``AccruedCash`` carries interest only (it summed to
    -$120 over the live window against ~$1,916 of dividends actually going ex).
    So accruing on the ex-date without a receivable would trade an attribution
    error for a timing one — the residual would read -dividend on the ex-date
    and +dividend on the pay date instead of simply +dividend once.
    """

    def test_an_accrual_is_recorded_and_outstanding(self, conn):
        assert record_dividend_accrual(
            conn, ticker="LMT", ex_date="2026-06-01", pay_date="2026-06-26",
            per_share=3.30, shares=170, amount_usd=561.0) is True
        assert dividend_receivable_usd(conn) == pytest.approx(561.0)

    def test_recording_is_idempotent_across_a_re_reconcile(self, conn):
        """Re-reconciling a past session is the canonical correction path and
        must not accrue the same distribution twice."""
        kwargs = {"ticker": "LMT", "ex_date": "2026-06-01",
                  "pay_date": "2026-06-26", "per_share": 3.30,
                  "shares": 170, "amount_usd": 561.0}
        assert record_dividend_accrual(conn, **kwargs) is True
        assert record_dividend_accrual(conn, **kwargs) is False
        assert dividend_receivable_usd(conn) == pytest.approx(561.0)

    def test_nothing_is_released_before_the_pay_date(self, conn):
        record_dividend_accrual(
            conn, ticker="LMT", ex_date="2026-06-01", pay_date="2026-06-26",
            per_share=3.30, shares=170, amount_usd=561.0)
        released, rows = settle_due_dividend_accruals(conn, "2026-06-25")
        assert released == 0.0 and rows == []
        assert dividend_receivable_usd(conn) == pytest.approx(561.0)

    def test_it_is_released_on_the_pay_date_and_only_once(self, conn):
        record_dividend_accrual(
            conn, ticker="LMT", ex_date="2026-06-01", pay_date="2026-06-26",
            per_share=3.30, shares=170, amount_usd=561.0)
        released, rows = settle_due_dividend_accruals(conn, "2026-06-26")
        assert released == pytest.approx(561.0)
        assert rows[0]["ticker"] == "LMT"
        assert dividend_receivable_usd(conn) == 0.0
        assert settle_due_dividend_accruals(conn, "2026-06-29") == (0.0, [])

    def test_a_missing_pay_date_still_releases(self, conn):
        """A distribution whose feed record carries no pay date still settles
        into cash. Stranding it would leave the residual permanently short by
        that amount — a silent, growing, one-directional error, which is the
        exact shape of the defect being fixed."""
        record_dividend_accrual(
            conn, ticker="LMT", ex_date="2026-06-01", pay_date=None,
            per_share=3.30, shares=170, amount_usd=561.0)
        assert settle_due_dividend_accruals(conn, "2026-06-20") == (0.0, [])
        released, rows = settle_due_dividend_accruals(conn, "2026-07-01")
        assert released == pytest.approx(561.0)
        assert rows[0]["pay_date"] is None
        assert rows[0]["effective_pay_date"] == "2026-07-01"

    def test_the_identity_closes_on_both_the_ex_date_and_the_pay_date(self, conn):
        """The load-bearing property. Simulated day: a 1.0% price fall on
        LMT of which 0.66pp is the dividend going ex, then the cash arriving
        three weeks later.

        Ex-date: NAV falls with the price and position P&L gains the accrual,
        so nav_change - position_pnl = -dividend; dividend_timing = +dividend.
        Pay date: NAV gains the cash and position P&L does not move, so
        nav_change - position_pnl = +dividend; dividend_timing = -dividend.
        Both net to a zero residual.
        """
        dividend = 561.0

        # --- ex-date -------------------------------------------------------
        record_dividend_accrual(
            conn, ticker="LMT", ex_date="2026-06-01", pay_date="2026-06-26",
            per_share=3.30, shares=170, amount_usd=dividend)
        released, _ = settle_due_dividend_accruals(conn, "2026-06-01")
        timing_ex = dividend - released
        nav_change_ex = -1_000.0            # price fall, dividend cash not yet in
        position_pnl_ex = -1_000.0 + dividend  # price fall plus the accrual
        residual_ex = nav_change_ex - position_pnl_ex - 0.0 + timing_ex
        assert residual_ex == pytest.approx(0.0)
        assert dividend_receivable_usd(conn) == pytest.approx(dividend)

        # --- pay date ------------------------------------------------------
        released, _ = settle_due_dividend_accruals(conn, "2026-06-26")
        timing_pay = 0.0 - released
        nav_change_pay = dividend           # the cash lands
        position_pnl_pay = 0.0              # no price move, no new accrual
        residual_pay = nav_change_pay - position_pnl_pay - 0.0 + timing_pay
        assert residual_pay == pytest.approx(0.0)
        assert dividend_receivable_usd(conn) == 0.0

    def test_without_the_receivable_the_identity_would_break_twice(self, conn):
        """Guards the reason the ledger exists: ex-date accrual against a
        cash-basis NAV, with no timing term, is worse than the defect it
        replaces — one signed error becomes two."""
        dividend = 561.0
        naive_ex = -1_000.0 - (-1_000.0 + dividend)
        naive_pay = dividend - 0.0
        assert naive_ex == pytest.approx(-dividend)
        assert naive_pay == pytest.approx(dividend)
