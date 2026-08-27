"""alpha-engine-config-I8753 — which constraint forced each unit of the
mandatory turnover floor.

``turnover_mandatory_floor`` has been emitted since the turnover constraint
shipped, and it answers "how much of today's trading was forced" — but not BY
WHAT, and the three causes have entirely different fixes.

Measured 2026-08-27 across the nine sessions since the v2 cutover
(``s3://alpha-engine-research/predictor/optimizer_shadow/{date}.json``,
read-only): cap cuts on held names mandated 0.278 of NAV in one-way selling,
~3.1% per session, entirely alpha-independent. On 2026-08-27 the floor was
0.105 against an executed one-way turnover of 0.170 — 62% of the day's trading
forced before the objective was consulted. That total was legible only by
reading nine artifacts and diffing ``stance_caps`` by hand.

These tests exercise the decomposition directly. It is pure numpy, so unlike
the rest of ``test_portfolio_optimizer.py`` it does not need ``cvxpy`` — which
is not installed on this laptop (48 of that file's 72 tests fail identically on
origin/main; see alpha-engine-config-I8739).
"""
from __future__ import annotations

import numpy as np
import pytest

from executor.portfolio_optimizer import (
    _decompose_mandatory_turnover,
    _mandatory_turnover_floor,
)

CFG = {"cash_sleeve_pct": 0.03}


def _run(w_prev, caps, eligibility, cash_idx):
    return _decompose_mandatory_turnover(
        np.array(w_prev, dtype=float),
        np.array(caps, dtype=float),
        np.array(eligibility, dtype=bool),
        cash_idx,
        CFG,
    )


def test_a_book_already_at_target_forces_nothing():
    #                A     B     CASH
    out = _run([0.50, 0.47, 0.03], [0.60, 0.60, 0.03], [True, True, True], 2)

    assert out["total"] == pytest.approx(0.0, abs=1e-12)
    assert out["position_cap"] == pytest.approx(0.0)
    assert out["ineligibility_pin"] == pytest.approx(0.0)
    assert out["n_names_over_cap"] == 0


def test_a_cap_cut_on_a_held_name_is_attributed_to_position_cap():
    """The live case. AVAV held at 0.108 with its cap flipped from 0.11 to
    0.04 on 2026-08-27 — a 6.8%-of-NAV sale mandated before the objective was
    consulted, on a name whose alpha nobody re-examined."""
    out = _run([0.108, 0.862, 0.03], [0.04, 0.90, 0.03], [True, True, True], 2)

    # 0.068 of L1 from the cap cut; the projection then sums to 0.932, so
    # 0.068 of residual must be absorbed by names with slack.
    assert out["position_cap"] == pytest.approx(0.068 / 2)
    assert out["renormalization"] == pytest.approx(0.068 / 2)
    assert out["ineligibility_pin"] == pytest.approx(0.0)
    assert out["n_names_over_cap"] == 1


def test_an_ineligible_held_name_is_NOT_attributed_to_the_cap():
    """A forced exit is the system working; a cap cut on an eligible name is a
    sizing artifact. Attributing them to one bucket would make the two
    indistinguishable, and their responses are opposite."""
    out = _run([0.30, 0.67, 0.03], [0.0, 0.90, 0.03], [False, True, True], 2)

    assert out["ineligibility_pin"] == pytest.approx(0.30 / 2)
    assert out["position_cap"] == pytest.approx(0.0)
    assert out["n_names_pinned_out"] == 1
    assert out["n_names_over_cap"] == 0


def test_the_cash_sleeve_pin_is_its_own_cause():
    """The sleeve is an equality pin; drift into or out of it is mandated every
    session and is not a defect. Folding it into either other bucket would put
    a permanent floor under a number read as 'avoidable forced trading'."""
    out = _run([0.60, 0.40, 0.00], [0.90, 0.90, 0.03], [True, True, True], 2)

    assert out["cash_sleeve_pin"] == pytest.approx(0.03 / 2)
    assert out["position_cap"] == pytest.approx(0.0)
    assert out["ineligibility_pin"] == pytest.approx(0.0)


def test_all_three_causes_separate_in_one_solve():
    #            over-cap  ineligible  fine  CASH
    out = _run(
        [0.20, 0.15, 0.65, 0.00],
        [0.10, 0.00, 0.90, 0.03],
        [True, False, True, True],
        3,
    )

    assert out["position_cap"] == pytest.approx(0.10 / 2)      # 0.20 -> 0.10
    assert out["ineligibility_pin"] == pytest.approx(0.15 / 2)  # 0.15 -> 0
    assert out["cash_sleeve_pin"] == pytest.approx(0.03 / 2)    # 0.00 -> 0.03
    assert out["n_names_over_cap"] == 1
    assert out["n_names_pinned_out"] == 1


@pytest.mark.parametrize(
    "w_prev,caps,elig,cash_idx",
    [
        ([0.50, 0.47, 0.03], [0.60, 0.60, 0.03], [True, True, True], 2),
        ([0.108, 0.862, 0.03], [0.04, 0.90, 0.03], [True, True, True], 2),
        ([0.30, 0.67, 0.03], [0.0, 0.90, 0.03], [False, True, True], 2),
        ([0.20, 0.15, 0.65, 0.00], [0.10, 0.00, 0.90, 0.03], [True, False, True, True], 3),
        ([0.11, 0.11, 0.10, 0.10, 0.55, 0.03],
         [0.04, 0.11, 0.04, 0.10, 0.90, 0.03],
         [True, True, False, True, True, True], 5),
    ],
)
def test_the_decomposition_reconciles_to_the_floor_it_explains(w_prev, caps, elig, cash_idx):
    """The load-bearing invariant: the parts sum to the whole.

    A decomposition that does not reconcile is worse than none — it reads as an
    explanation of a number it is not actually explaining.
    """
    out = _decompose_mandatory_turnover(
        np.array(w_prev, dtype=float), np.array(caps, dtype=float),
        np.array(elig, dtype=bool), cash_idx, CFG,
    )
    floor = _mandatory_turnover_floor(
        np.array(w_prev, dtype=float), np.array(caps, dtype=float), cash_idx, CFG,
    )

    assert out["total"] == pytest.approx(floor, abs=1e-12)
    parts = (
        out["cash_sleeve_pin"] + out["ineligibility_pin"]
        + out["position_cap"] + out["renormalization"]
    )
    assert parts == pytest.approx(out["total"], abs=1e-12)
