"""Tests for the EOD P&L integrity gates (alpha-engine-config-I8188).

Each test names the live measurement it encodes, so a future threshold change
has to argue with the data rather than with a number.
"""

from __future__ import annotations

import pytest

from executor.pnl_integrity import (
    MARK_HARD_MATERIALITY_NAV_BPS,
    RESIDUAL_HARD_PER_SESSION_NAV_BPS,
    TWR_SELF_HEAL_MAX_CORRECTION_PCT,
    check_custodian_marks,
    check_residual_bounds,
    gross_net_returns,
    mark_materiality_usd,
    nav_change_implied_returns,
    nav_implied_returns,
    plan_twr_self_heal,
    residual_cumulative_tolerance_usd,
    residual_per_session_tolerance_usd,
    session_costs,
    verify_nav_change_basis_closes,
    verify_twr_closes,
)

NAV = 1_030_000.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Residual bounds
# ─────────────────────────────────────────────────────────────────────────────

class TestResidualBounds:
    def test_the_identity_can_now_fail(self):
        """The whole defect: the reconciliation identity held on 114 of 114
        sessions because unattributed_usd was the remainder. A bound is the
        thing that makes a tautology falsifiable."""
        breaches = check_residual_bounds(
            unattributed_true_usd=-9_713.0, nav=NAV, run_date="2026-08-04",
        )
        kinds = {b["kind"] for b in breaches}
        assert "per_session" in kinds

    def test_ordinary_session_passes(self):
        """Measured median |residual| after lifting rotation out is $754 —
        7bp of NAV. The gate must not fire on the normal case."""
        assert check_residual_bounds(
            unattributed_true_usd=754.0, nav=NAV, run_date="2026-05-01",
        ) == []

    def test_p95_of_the_measured_distribution_passes(self):
        """p95 of the measured ex-rotation residual is $3,560 (35bp). The
        per-session bound sits at 50bp precisely so the observed band clears
        it — a hard gate set at the soft band's rate trains the operator to
        ignore it."""
        assert check_residual_bounds(
            unattributed_true_usd=3_560.0, nav=NAV, run_date="2026-05-02",
        ) == []

    def test_per_session_tolerance_is_nav_scaled_above_the_floor(self):
        assert residual_per_session_tolerance_usd(500_000.0) == 5_000.0
        assert residual_per_session_tolerance_usd(2_000_000.0) == pytest.approx(
            RESIDUAL_HARD_PER_SESSION_NAV_BPS / 10_000.0 * 2_000_000.0
        )

    def test_cumulative_drift_fires_where_no_single_session_would(self):
        """-$20,293 accumulated over the live window in daily increments that
        no per-session bound could ever see. That is the defect the cumulative
        leg exists for."""
        trailing = [-350.0] * 58  # -$20,300 total, each far inside the daily bound
        breaches = check_residual_bounds(
            unattributed_true_usd=-350.0,
            nav=NAV,
            trailing_residuals_usd=trailing,
            run_date="2026-08-21",
        )
        kinds = {b["kind"] for b in breaches}
        assert kinds == {"cumulative"}
        assert breaches[0]["value_usd"] == pytest.approx(-350.0 * 59)

    def test_measured_true_drift_does_not_fire_cumulatively(self):
        """After the sleeves are lifted, the measured cumulative residual over
        74 sessions is +$522. The bound is ~20x that and must not fire on the
        behaviour of a correctly-attributed book."""
        assert check_residual_bounds(
            unattributed_true_usd=7.0,
            nav=NAV,
            trailing_residuals_usd=[7.0] * 73,
            run_date="2026-08-21",
        ) == []

    def test_cumulative_window_is_bounded_to_a_quarter(self):
        """A residual that has already breached and been dealt with must age
        out, or the gate latches red forever."""
        trailing = [-5_000.0] * 30 + [0.0] * 62
        assert check_residual_bounds(
            unattributed_true_usd=0.0,
            nav=NAV,
            trailing_residuals_usd=trailing,
            run_date="2026-08-21",
        ) == []

    def test_none_residual_is_not_a_pass(self):
        """An absent measurement returns no breach, but the caller records
        dividend/sleeve availability separately — this asserts the gate does
        not invent a zero."""
        assert check_residual_bounds(
            unattributed_true_usd=None, nav=NAV, run_date="x") == []
        assert check_residual_bounds(
            unattributed_true_usd=1.0, nav=None, run_date="x") == []

    def test_cumulative_tolerance_scales(self):
        assert residual_cumulative_tolerance_usd(500_000.0) == 10_000.0
        assert residual_cumulative_tolerance_usd(3_000_000.0) == 30_000.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transaction costs
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionCosts:
    def test_absent_commission_is_not_a_measured_zero(self):
        """"Paper-account commissions are trivial" is how the cost line came
        not to exist. An absent figure and a reported $0.00 must be
        distinguishable."""
        absent = session_costs([
            {"action": "BUY", "shares": 100, "fill_price": 10.0,
             "price_at_order": 10.0},
        ])
        assert absent["commission_usd"] == 0.0
        assert absent["commission_available"] is False

        reported = session_costs([
            {"action": "BUY", "shares": 100, "fill_price": 10.0,
             "price_at_order": 10.0, "commission_usd": 0.0},
        ])
        assert reported["commission_usd"] == 0.0
        assert reported["commission_available"] is True

    def test_no_fills_is_available_not_missing(self):
        """A session with no trades has a known, complete cost picture."""
        assert session_costs([])["commission_available"] is True

    def test_slippage_is_signed_by_side(self):
        """Paying above arrival on a buy and selling below arrival are both
        costs. A sign error here would net two costs to zero."""
        out = session_costs([
            {"action": "BUY", "shares": 100, "fill_price": 10.10,
             "price_at_order": 10.00},
            {"action": "SELL", "shares": 100, "fill_price": 9.90,
             "price_at_order": 10.00},
        ])
        assert out["slippage_usd"] == pytest.approx(20.0)
        assert out["n_fills"] == 2

    def test_slippage_bps_matches_the_live_measurement_shape(self):
        """Live window: 468 fills, +6.4bp of traded notional."""
        out = session_costs([
            {"action": "BUY", "shares": 1000, "fill_price": 100.064,
             "price_at_order": 100.0},
        ])
        assert out["slippage_bps"] == pytest.approx(6.4, abs=0.01)

    def test_filled_shares_wins_over_ordered_shares(self):
        out = session_costs([
            {"action": "BUY", "shares": 100, "filled_shares": 40,
             "fill_price": 10.10, "price_at_order": 10.00},
        ])
        assert out["slippage_usd"] == pytest.approx(4.0)

    def test_unfilled_rows_are_skipped(self):
        out = session_costs([
            {"action": "BUY", "shares": 100, "fill_price": None,
             "price_at_order": 10.0},
            {"action": "BUY", "shares": 0, "fill_price": 10.0,
             "price_at_order": 10.0},
        ])
        assert out["n_fills"] == 0
        assert out["traded_notional_usd"] == 0.0

    def test_commission_is_normalised_to_a_positive_cost(self):
        out = session_costs([
            {"action": "BUY", "shares": 10, "fill_price": 10.0,
             "price_at_order": 10.0, "commission_usd": -1.25},
        ])
        assert out["commission_usd"] == pytest.approx(1.25)


class TestGrossNet:
    def test_gross_and_net_are_different_numbers(self):
        """Before this, gross and net performance were the same number and
        neither was labelled."""
        out = gross_net_returns(
            nav_change_usd=1_000.0, prior_nav=1_000_000.0,
            commission_usd=25.0, slippage_usd=200.0,
        )
        assert out["daily_return_net_pct"] == pytest.approx(0.1)
        assert out["daily_return_gross_pct"] == pytest.approx(0.1225)
        assert out["total_cost_usd"] == pytest.approx(225.0)

    def test_zero_cost_day_collapses_them(self):
        out = gross_net_returns(
            nav_change_usd=1_000.0, prior_nav=1_000_000.0,
            commission_usd=0.0, slippage_usd=0.0,
        )
        assert out["daily_return_net_pct"] == out["daily_return_gross_pct"]

    def test_first_session_has_neither(self):
        out = gross_net_returns(
            nav_change_usd=None, prior_nav=None,
            commission_usd=1.0, slippage_usd=2.0,
        )
        assert out["daily_return_net_pct"] is None
        assert out["daily_return_gross_pct"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. TWR closure
# ─────────────────────────────────────────────────────────────────────────────

def _series(navs, stored=None):
    rows = []
    prior = None
    for i, nav in enumerate(navs):
        pct = None if prior is None else (nav / prior - 1) * 100
        rows.append({
            "date": f"2026-04-{i + 1:02d}",
            "portfolio_nav": nav,
            "daily_return_pct": pct if pct is not None else 0.0,
        })
        prior = nav
    if stored:
        for idx, value in stored.items():
            rows[idx]["daily_return_pct"] = value
    return rows


class TestTwrClosure:
    def test_a_consistent_series_closes_exactly(self):
        result = verify_twr_closes(_series([1_000_000, 1_005_000, 1_002_000, 1_010_000]))
        assert result["closes"] is True
        assert abs(result["drift_bps"]) < 1e-6

    def test_the_live_defect_is_detected(self):
        """2026-04-07 stored +0.026827% where its own NAV series implies
        -0.140311%. That single row is 100% of the 17.4bp live drift."""
        rows = _series([1_001_658.39, 1_002_481.91, 1_002_698.68,
                        1_009_473.08, 1_008_056.68],
                       stored={4: 0.026827})
        result = verify_twr_closes(rows)
        assert result["closes"] is False
        assert [o["date"] for o in result["offenders"]] == ["2026-04-05"]
        assert result["drift_bps"] == pytest.approx(16.72, abs=0.2)

    def test_one_bp_is_the_tolerance(self):
        """Absent external flows the two figures are identically equal by
        construction, so the tolerance is numerical noise, not a band."""
        rows = _series([1_000_000, 1_005_000])
        rows[1]["daily_return_pct"] = 0.5 + 0.02  # 2bp off
        assert verify_twr_closes(rows)["closes"] is False
        rows[1]["daily_return_pct"] = 0.5 + 0.005  # 0.5bp off
        assert verify_twr_closes(rows)["closes"] is True

    def test_short_series_is_na_not_pass(self):
        assert verify_twr_closes([])["status"] == "n/a"
        assert verify_twr_closes(_series([1_000_000]))["status"] == "n/a"

    def test_missing_stored_return_is_na_not_pass(self):
        rows = _series([1_000_000, 1_005_000])
        rows[1]["daily_return_pct"] = None
        assert verify_twr_closes(rows)["status"] == "n/a"

    def test_nav_implied_returns_first_row_has_no_predecessor(self):
        detail = nav_implied_returns(_series([1_000_000, 1_005_000]))
        assert detail[0]["implied_pct"] is None
        assert detail[1]["implied_pct"] == pytest.approx(0.5)


class TestTwrSelfHeal:
    def test_the_live_defect_is_repaired_and_then_closes(self):
        rows = _series([1_001_658.39, 1_002_481.91, 1_002_698.68,
                        1_009_473.08, 1_008_056.68],
                       stored={4: 0.026827})
        plan = plan_twr_self_heal(rows)
        assert [c["date"] for c in plan["corrections"]] == ["2026-04-05"]
        assert plan["refused"] == []
        for correction in plan["corrections"]:
            for row in rows:
                if row["date"] == correction["date"]:
                    row["daily_return_pct"] = correction["to_pct"]
        assert verify_twr_closes(rows)["closes"] is True

    def test_a_clean_series_needs_no_repair(self):
        plan = plan_twr_self_heal(_series([1_000_000, 1_005_000, 1_002_000]))
        assert plan == {"corrections": [], "refused": []}

    def test_a_large_disagreement_is_refused_not_rewritten(self):
        """The self-heal must not be a licence to restate the track record.
        Past the ceiling the disagreement is a different NAV series — an
        external flow, a restated snapshot — and needs a ruling."""
        rows = _series([1_000_000, 1_005_000])
        rows[1]["daily_return_pct"] = 0.5 + TWR_SELF_HEAL_MAX_CORRECTION_PCT + 0.5
        plan = plan_twr_self_heal(rows)
        assert plan["corrections"] == []
        assert len(plan["refused"]) == 1
        assert "external flow" in plan["refused"][0]["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# 3b. TWR closure, nav_change_usd basis (alpha-engine-config-I9025)
# ─────────────────────────────────────────────────────────────────────────────

def _nc_series(navs, nav_change=None, stored=None, missing_nc_before=0):
    """Like ``_series`` but also carries ``nav_change_usd``.

    ``nav_change_usd`` defaults to the exact NAV delta (so the two bases agree
    unless overridden). ``missing_nc_before`` sets ``nav_change_usd=None`` on
    the first N rows — the I9025 day-set-mismatch shape.
    """
    rows = []
    prior = None
    for i, nav in enumerate(navs):
        pct = None if prior is None else (nav / prior - 1) * 100
        nc = None if prior is None else nav - prior
        rows.append({
            "date": f"2026-04-{i + 1:02d}",
            "portfolio_nav": nav,
            "daily_return_pct": pct if pct is not None else 0.0,
            "nav_change_usd": nc,
        })
        prior = nav
    if nav_change:
        for idx, value in nav_change.items():
            rows[idx]["nav_change_usd"] = value
    if stored:
        for idx, value in stored.items():
            rows[idx]["daily_return_pct"] = value
    for i in range(min(missing_nc_before, len(rows))):
        rows[i]["nav_change_usd"] = None
    return rows


class TestNavChangeBasis:
    def test_a_clean_series_closes_exactly(self):
        result = verify_nav_change_basis_closes(
            _nc_series([1_000_000, 1_005_000, 1_002_000, 1_010_000])
        )
        assert result["closes"] is True
        assert abs(result["drift_bps"]) < 1e-6
        assert result["coverage_gap_sessions"] == []

    def test_the_measured_live_cause_is_day_set_coverage_not_drift(self):
        """I9025: nav_change_usd is NULL on the early sessions (pre-PR490)
        while daily_return_pct is populated throughout. Those rows must be
        excluded from BOTH chains, not counted as drift — and the remaining
        rows, which persist a matching nav_change_usd, close exactly."""
        rows = _nc_series(
            [1_000_000, 1_001_000, 1_003_000, 998_000, 1_004_000],
            missing_nc_before=3,
        )
        result = verify_nav_change_basis_closes(rows)
        assert result["coverage_gap_sessions"] == ["2026-04-02", "2026-04-03"]
        assert result["n_sessions"] == 2
        assert result["closes"] is True
        assert result["offenders"] == []

    def test_a_disagreeing_row_is_an_offender_not_a_coverage_gap(self):
        rows = _nc_series([1_000_000, 1_005_000])
        rows[1]["nav_change_usd"] = 5_000 - 200  # 2bp off from the true delta
        result = verify_nav_change_basis_closes(rows)
        assert result["closes"] is False
        assert [o["date"] for o in result["offenders"]] == ["2026-04-02"]
        assert result["coverage_gap_sessions"] == []

    def test_one_bp_is_the_tolerance(self):
        rows = _nc_series([1_000_000, 1_005_000])
        rows[1]["nav_change_usd"] = 5_000 + 1_000_000 * 0.0002  # 2bp off
        assert verify_nav_change_basis_closes(rows)["closes"] is False
        rows[1]["nav_change_usd"] = 5_000 + 1_000_000 * 0.00005  # 0.5bp off
        assert verify_nav_change_basis_closes(rows)["closes"] is True

    def test_short_series_is_na_not_pass(self):
        assert verify_nav_change_basis_closes([])["status"] == "n/a"
        assert verify_nav_change_basis_closes(_nc_series([1_000_000]))["status"] == "n/a"

    def test_all_rows_missing_nav_change_usd_is_na_not_pass(self):
        """Every row a coverage gap (e.g. a book that has never persisted
        nav_change_usd) must not read as a clean close on an empty chain."""
        rows = _nc_series([1_000_000, 1_005_000, 1_002_000], missing_nc_before=3)
        result = verify_nav_change_basis_closes(rows)
        assert result["status"] == "n/a"
        assert result["coverage_gap_sessions"] == ["2026-04-02", "2026-04-03"]

    def test_nav_change_implied_returns_flags_coverage_gap_not_offender(self):
        rows = _nc_series([1_000_000, 1_005_000], missing_nc_before=2)
        detail = nav_change_implied_returns(rows)
        assert detail[1]["coverage_gap"] is True
        assert detail[1]["implied_pct"] is None
        assert detail[1]["delta_pct"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Custodian marks
# ─────────────────────────────────────────────────────────────────────────────

def _flag(ticker, mark, lo, hi, shares, error):
    return {"ticker": ticker, "ib_mark": mark, "day_low": lo, "day_high": hi,
            "shares": shares, "mark_error_usd": error}


class TestCustodianMarks:
    def test_the_three_material_live_breaches_raise(self):
        """AMD 2026-08-04 (-$5,220, 50.4bp), COIN 2026-07-30 (-$2,999,
        29.7bp), LNTH 2026-06-26 (-$2,532, 25.5bp) — all provably wrong marks,
        all material."""
        for ticker, mark, lo, hi, shares, error in [
            ("AMD", 479.00, 502.20, 530.13, 225, -5_220.00),
            ("COIN", 154.45, 159.31, 164.78, 617, -2_998.62),
            ("LNTH", 105.12, 108.51, 111.46, 747, -2_532.33),
        ]:
            breaches = check_custodian_marks(
                [_flag(ticker, mark, lo, hi, shares, error)],
                nav=NAV, run_date="2026-08-04",
            )
            assert len(breaches) == 1, ticker
            assert breaches[0]["ticker"] == ticker

    def test_the_five_immaterial_live_flags_do_not_raise(self):
        """Edge-of-range rounding: <=$584, <=5.9bp. They stay flags."""
        flags = [
            _flag("SPY", 728.41, 729.10, 742.68, 843, -583.74),
            _flag("COIN", 158.00, 158.68, 169.69, 617, -419.56),
            _flag("DECK", 91.76, 89.06, 91.68, 1273, 101.84),
            _flag("TWLO", 206.00, 206.47, 213.91, 188, -88.36),
            _flag("SPY", 772.42, 772.51, 776.78, 1015, -91.37),
        ]
        assert check_custodian_marks(flags, nav=NAV, run_date="2026-07-29") == []

    def test_materiality_is_nav_scaled_with_a_floor(self):
        assert mark_materiality_usd(1_000.0) == 500.0
        assert mark_materiality_usd(NAV) == pytest.approx(
            MARK_HARD_MATERIALITY_NAV_BPS / 10_000.0 * NAV
        )

    def test_no_flags_means_no_breach(self):
        assert check_custodian_marks([], nav=NAV) == []
        assert check_custodian_marks(None, nav=NAV) == []

    def test_absent_nav_cannot_be_graded(self):
        assert check_custodian_marks(
            [_flag("AMD", 479.0, 502.2, 530.13, 225, -5_220.0)], nav=None) == []
