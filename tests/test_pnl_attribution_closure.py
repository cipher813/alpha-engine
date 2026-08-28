"""Attribution closure — the check that replaces a tautology.

alpha-engine-config-I8188 (class sweep). ``eod_report.compute_alpha_attribution``
published ``ties_to_headline``, and ``eod_reconcile`` logged
"investigate before trusting per-position contributions" when it was False.
It could never be False: expanding the algebra leaves
``nav_change − position_pnl − interest − unattributed_usd``, and
``unattributed_usd`` is DEFINED as that remainder.

These tests (a) pin the tautology so nobody re-reads that field as a check,
and (b) exercise the replacement — the NAV mark-basis LEVEL, rebuilt from an
independent source, which can and does fail.
"""

from __future__ import annotations

import math

import pytest

from executor.eod_report import compute_alpha_attribution
from executor.pnl_integrity import (
    ATTRIBUTION_BASIS_HARD_NAV_BPS,
    attribution_basis_tolerance_usd,
    check_attribution_closure,
    nav_basis_level_usd,
)


def _attr(**over):
    kwargs = {
        "prior_nav": 1_000_000.0,
        "spy_return": 0.5,
        "positions": {
            "AAA": {
                "shares": 100, "daily_return_usd": 500.0,
                "closing_price": 110.0, "market_value": 11_000.0,
                "ib_market_value": 11_000.0,
            },
        },
        "prior_positions": {
            "AAA": {
                "shares": 100, "closing_price": 105.0,
                "market_value": 10_500.0, "ib_market_value": 10_500.0,
            },
        },
        "interest_usd": 10.0,
        "unattributed_usd": -250.0,
        "nav_change_usd": 260.0,
        "trades_today": [],
        "pricing_timing_usd": 0.0,
        "pricing_timing_available": True,
    }
    kwargs.update(over)
    return compute_alpha_attribution(**kwargs)


class TestTheIdentityIsTautological:
    """``ties_to_headline`` is True regardless of what the sleeves say."""

    def test_ties_to_headline_is_true_on_a_normal_day(self):
        a = _attr()
        assert a["ties_to_headline"] is True
        assert abs(a["residual_usd"]) < 1e-6

    @pytest.mark.parametrize(
        "over",
        [
            # A wildly wrong rotation input.
            {"trades_today": [
                {"action": "SELL", "ticker": "AAA", "shares": 40, "fill_price": 1.0},
            ]},
            # A wrong prior close on the only held name — every per-position
            # sleeve is now nonsense.
            {"prior_positions": {"AAA": {
                "shares": 100, "closing_price": 1.0,
                "market_value": 100.0, "ib_market_value": 100.0,
            }}},
            # A large unexplained plug.
            {"unattributed_usd": -50_000.0, "nav_change_usd": -49_490.0},
        ],
    )
    def test_ties_to_headline_still_true_when_the_attribution_is_garbage(self, over):
        """THE DEFECT. Each case corrupts the decomposition; the flag holds."""
        a = _attr(**over)
        assert a["ties_to_headline"] is True, (
            "the identity is closed by the plug's own definition — if this "
            "ever fails, the algebra changed and the honesty fields below "
            "must be revisited"
        )

    def test_the_artifact_says_so_out_loud(self):
        a = _attr()
        assert a["identity_is_tautological"] is True
        assert "plug" in a["identity_residual_note"]
        assert a["components_finite"] is True
        assert a["component_sum_usd"] == pytest.approx(
            sum(c["contrib_usd"] for c in a["components"])
        )


class TestComponentArithmetic:
    """A non-finite contribution CAN be caught — the identity never sees it."""

    def test_nan_component_is_named(self):
        breaches = check_attribution_closure(
            nav_basis={"available": False, "reason": "n/a"},
            nav=None,
            components=[
                {"label": "AAA", "kind": "position", "contrib_usd": 100.0},
                {"label": "BBB", "kind": "position", "contrib_usd": float("nan")},
            ],
            run_date="2026-08-27",
        )
        assert len(breaches) == 1
        assert breaches[0]["kind"] == "attribution_arithmetic"
        assert breaches[0]["label"] == "BBB"

    def test_absent_contribution_is_named(self):
        breaches = check_attribution_closure(
            nav_basis={"available": False, "reason": "n/a"},
            nav=None,
            components=[{"label": "AAA", "kind": "cash"}],
            run_date="2026-08-27",
        )
        assert [b["kind"] for b in breaches] == ["attribution_arithmetic"]

    def test_clean_components_produce_nothing(self):
        assert check_attribution_closure(
            nav_basis={"available": False, "reason": "n/a"},
            nav=None,
            components=[{"label": "AAA", "kind": "cash", "contrib_usd": 1.0}],
            run_date="2026-08-27",
        ) == []


class TestNavBasisLevel:
    def test_level_is_the_gap_between_broker_nav_and_the_settled_rebuild(self):
        got = nav_basis_level_usd(
            nav=1_000_000.0,
            total_cash=500_000.0,
            accrued_interest=100.0,
            positions={"AAA": {"shares": 1_000, "closing_price": 499.0}},
        )
        assert got["available"] is True
        assert got["settled_mv_usd"] == pytest.approx(499_000.0)
        assert got["nav_basis_usd"] == pytest.approx(900.0)

    def test_an_unpriced_name_makes_the_level_unavailable_not_zero(self):
        got = nav_basis_level_usd(
            nav=1_000_000.0,
            total_cash=500_000.0,
            accrued_interest=0.0,
            positions={
                "AAA": {"shares": 1_000, "closing_price": 499.0},
                "BBB": {"shares": 100},  # no settled close
            },
        )
        assert got["available"] is False
        assert got["nav_basis_usd"] is None
        assert "1 of 2" in got["reason"]

    def test_missing_broker_cash_makes_the_level_unavailable(self):
        got = nav_basis_level_usd(
            nav=1_000_000.0, total_cash=None, accrued_interest=None, positions={},
        )
        assert got["available"] is False


class TestBasisLevelGate:
    NAV = 1_000_000.0

    def _basis(self, level):
        return {
            "available": True, "nav_basis_usd": level,
            "settled_mv_usd": 500_000.0, "n_positions": 3,
            "n_positions_unpriced": 0, "reason": None,
        }

    def test_tolerance_is_the_greater_of_the_floor_and_the_nav_rate(self):
        assert attribution_basis_tolerance_usd(1.0) == 5_000.0
        assert attribution_basis_tolerance_usd(10_000_000.0) == pytest.approx(
            ATTRIBUTION_BASIS_HARD_NAV_BPS / 10_000.0 * 10_000_000.0
        )

    def test_inside_the_band_is_clean(self):
        assert check_attribution_closure(
            nav_basis=self._basis(2_723.0),  # measured p95 of the live window
            nav=self.NAV, run_date="2026-08-27",
        ) == []

    def test_the_2026_08_04_level_breaches(self):
        """The one live session the calibration says should fire: -$8,125."""
        breaches = check_attribution_closure(
            nav_basis=self._basis(-8_124.71), nav=1_036_140.0, run_date="2026-08-04",
        )
        assert [b["kind"] for b in breaches] == ["attribution_basis_level"]
        assert breaches[0]["severity"] == "breach"
        assert breaches[0]["nav_basis_usd"] == pytest.approx(-8_124.71)
        assert "invisible to it" in breaches[0]["message"]

    def test_a_constant_basis_error_breaches_the_level_gate(self):
        """The blindness this closes: a basis error identical on two days has
        a day-over-day delta of exactly zero, so the three-way gate sees
        nothing. The level gate fires on both days."""
        from executor.eod_reconcile import _check_nav_three_way_hard_gate

        constant = -20_000.0
        delta_gate = _check_nav_three_way_hard_gate(
            pricing_timing_usd=constant - constant,  # yesterday's identical error
            pricing_timing_available=True,
            nav=self.NAV,
            run_date="2026-08-27",
        )
        assert delta_gate is None, "the delta gate is blind to a constant error"

        level_gate = check_attribution_closure(
            nav_basis=self._basis(constant), nav=self.NAV, run_date="2026-08-27",
        )
        assert [b["kind"] for b in level_gate] == ["attribution_basis_level"]

    def test_unavailable_is_recorded_as_not_evaluated_never_as_clean(self):
        breaches = check_attribution_closure(
            nav_basis={"available": False, "reason": "2 of 5 name(s) unpriced"},
            nav=self.NAV, run_date="2026-08-27",
        )
        assert len(breaches) == 1
        assert breaches[0]["severity"] == "unevaluated"
        assert "NOT EVALUATED" in breaches[0]["message"]
        assert breaches[0]["nav_basis_usd"] is None

    def test_no_nav_short_circuits_without_claiming_health(self):
        assert check_attribution_closure(
            nav_basis=self._basis(999_999.0), nav=0.0, run_date="2026-08-27",
        ) == []


def test_math_import_is_used_for_the_finiteness_check():
    """Guards the helper against a silent `float('nan') == nan` regression."""
    from executor.eod_report import _is_finite

    assert _is_finite(1.0) is True
    assert _is_finite(float("nan")) is False
    assert _is_finite(float("inf")) is False
    assert _is_finite(None) is False
    assert _is_finite("x") is False
    assert math.isfinite(0.0)
