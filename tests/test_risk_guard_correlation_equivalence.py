"""`check_correlation`'s fast path must equal its pandas predecessor exactly.

Why this function was touched at all: py-spy stack sampling of the live
pit_parity walk-forward pass on 2026-08-13 (i-034c3cd083f586064, pid 27417) put
5 of 6 samples inside `check_correlation` — `Series.dropna`,
`Series.reset_index`, `numpy.corrcoef` via pandas `nancorr`. The lookahead pass
runs ONE simulation and finished in 922 s; the walk-forward pass runs
`param_sweep.sweep` -> `_run_combos`, one full simulation per combo, and blew
its 2700 s ceiling. The per-ticker trailing return series never depended on the
candidate, yet was recomputed for every held position on every candidate order.

`check_correlation` VETOES ENTRIES. A performance change here that moves a
single correlation by 1e-12 can flip a veto and silently change what the
strategy trades, so speed is worth nothing without exact agreement. These tests
re-implement the original pandas expression verbatim and assert the two agree —
over random inputs, and over the degenerate cases where "equal" is easy to get
wrong: constant series (zero variance -> NaN), a zero price (pct_change -> inf,
which `dropna` does NOT remove), unequal history lengths, and the `min_len < 10`
boundary.

The cache is keyed on the IDENTITY of the `price_histories` mapping, which
`_simulate_single_date` rebuilds per simulated date. The staleness test below is
the one that matters: a cache that outlives its date returns yesterday's
returns for today's decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from executor import risk_guard


# ── the original implementation, verbatim, as the oracle ────────────────────
def _original_returns(history: pd.DataFrame, lookback: int) -> pd.Series:
    return history["close"].iloc[-lookback:].pct_change().dropna()


def _original_corr(
    cand_hist: pd.DataFrame, held_hist: pd.DataFrame, lookback: int
) -> float:
    candidate_returns = _original_returns(cand_hist, lookback)
    held_returns = _original_returns(held_hist, lookback)
    min_len = min(len(candidate_returns), len(held_returns))
    if min_len < 10:
        return float("nan")
    cr = candidate_returns.iloc[-min_len:].reset_index(drop=True)
    hr = held_returns.iloc[-min_len:].reset_index(drop=True)
    return cr.corr(hr)


def _hist(closes) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": np.asarray(closes, dtype=np.float64)},
        index=pd.date_range("2026-01-01", periods=len(closes), freq="B"),
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    risk_guard._CORR_CACHE_OWNER = None
    risk_guard._CORR_CACHE = {}
    yield
    risk_guard._CORR_CACHE_OWNER = None
    risk_guard._CORR_CACHE = {}


# ── returns-array equivalence ───────────────────────────────────────────────
@pytest.mark.parametrize("seed", range(8))
def test_cached_returns_match_pandas_pct_change_dropna(seed) -> None:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 90)))
    hist = _hist(closes)
    ph = {"AAA": hist}

    got = risk_guard._cached_returns_tail(ph, "AAA", 60)
    want = _original_returns(hist, 60).to_numpy()

    np.testing.assert_array_equal(got, want)


def test_returns_keep_inf_exactly_as_dropna_does() -> None:
    """`pct_change` across a zero price yields inf. `dropna` removes NaN but
    NOT inf — the fast path must keep it, or a pair pandas would have DROPPED
    (via `pd.notna(corr)` on the resulting NaN) would instead be scored."""
    closes = [10.0] * 30 + [0.0] + [10.0] * 30
    hist = _hist(closes)
    want = _original_returns(hist, 60).to_numpy()
    got = risk_guard._cached_returns_tail({"AAA": hist}, "AAA", 60)

    assert np.isinf(want).any(), "fixture no longer produces an inf — test is moot"
    np.testing.assert_array_equal(got, want)


def test_short_history_returns_none() -> None:
    assert risk_guard._cached_returns_tail({"AAA": _hist([1.0] * 10)}, "AAA", 60) is None


def test_absent_ticker_returns_none() -> None:
    assert risk_guard._cached_returns_tail({}, "ZZZ", 60) is None


# ── correlation equivalence ─────────────────────────────────────────────────
@pytest.mark.parametrize("seed", range(12))
def test_pearson_matches_pandas_corr(seed) -> None:
    rng = np.random.default_rng(seed + 100)
    a = _hist(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 90))))
    b = _hist(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 90))))
    ph = {"A": a, "B": b}

    ar = risk_guard._cached_returns_tail(ph, "A", 60)
    br = risk_guard._cached_returns_tail(ph, "B", 60)
    n = min(ar.size, br.size)
    got = risk_guard._pearson(ar[-n:], br[-n:])
    want = _original_corr(a, b, 60)

    assert got == pytest.approx(want, rel=0, abs=1e-12)


def test_pearson_is_nan_on_zero_variance_like_pandas() -> None:
    """A constant price series has zero return variance. pandas gives NaN and
    the caller drops the pair; so must this."""
    a = _hist([100.0] * 90)
    b = _hist(100.0 + np.arange(90, dtype=np.float64))
    ph = {"A": a, "B": b}
    ar = risk_guard._cached_returns_tail(ph, "A", 60)
    br = risk_guard._cached_returns_tail(ph, "B", 60)
    n = min(ar.size, br.size)

    assert np.isnan(risk_guard._pearson(ar[-n:], br[-n:]))
    assert pd.isna(_original_corr(a, b, 60))


def test_pearson_is_nan_when_an_input_is_non_finite() -> None:
    a = np.array([np.inf] + [0.01] * 20)
    b = np.array([0.01] * 21)
    assert np.isnan(risk_guard._pearson(a, b))


def test_pearson_is_nan_below_two_points() -> None:
    assert np.isnan(risk_guard._pearson(np.array([0.1]), np.array([0.2])))


def test_perfect_correlation_is_exactly_one() -> None:
    r = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02, 0.0, 0.04, -0.03, 0.01])
    assert risk_guard._pearson(r, r) == pytest.approx(1.0, abs=1e-12)
    assert risk_guard._pearson(r, -r) == pytest.approx(-1.0, abs=1e-12)


# ── cache correctness ───────────────────────────────────────────────────────
def test_cache_is_scoped_to_the_price_histories_identity() -> None:
    """THE staleness test. `_simulate_single_date` rebuilds price_histories per
    date; a cache that survives the rebuild would serve one date's returns to
    another and silently change entry decisions."""
    day1 = {"AAA": _hist(100.0 + np.arange(90, dtype=np.float64))}
    day2 = {"AAA": _hist(200.0 + np.arange(90, dtype=np.float64) * 3.0)}

    r1 = risk_guard._cached_returns_tail(day1, "AAA", 60).copy()
    r2 = risk_guard._cached_returns_tail(day2, "AAA", 60)

    np.testing.assert_array_equal(r2, _original_returns(day2["AAA"], 60).to_numpy())
    assert not np.allclose(r1, r2), "fixture no longer distinguishes the dates"


def test_cache_hit_within_one_date_returns_the_same_object() -> None:
    """The whole point: every candidate on a date shares one computation."""
    ph = {"AAA": _hist(100.0 * np.exp(np.cumsum(np.full(90, 0.001))))}
    first = risk_guard._cached_returns_tail(ph, "AAA", 60)
    second = risk_guard._cached_returns_tail(ph, "AAA", 60)
    assert first is second


def test_cache_owner_is_held_by_strong_reference() -> None:
    """`is` on a freed dict is unsafe if nothing holds it: CPython can allocate
    a new dict at the same address, and the identity check would then pass for
    an unrelated date. The cache must keep the owner alive."""
    ph = {"AAA": _hist(100.0 + np.arange(90, dtype=np.float64))}
    risk_guard._cached_returns_tail(ph, "AAA", 60)
    assert risk_guard._CORR_CACHE_OWNER is ph


def test_different_lookbacks_do_not_collide() -> None:
    ph = {"AAA": _hist(100.0 + np.arange(90, dtype=np.float64))}
    r60 = risk_guard._cached_returns_tail(ph, "AAA", 60)
    r30 = risk_guard._cached_returns_tail(ph, "AAA", 30)
    assert r60.size != r30.size


# ── end-to-end through check_correlation ────────────────────────────────────
def _positions(tickers, sector="TECH"):
    return {t: {"sector": sector} for t in tickers}


@pytest.mark.parametrize("seed", range(6))
def test_check_correlation_verdict_matches_the_pandas_computation(seed) -> None:
    rng = np.random.default_rng(seed + 200)
    base = np.cumsum(rng.normal(0, 0.02, 90))
    ph = {
        "CAND": _hist(100.0 * np.exp(base)),
        # correlated and uncorrelated held names, so the mean is non-trivial
        "H1": _hist(100.0 * np.exp(base + rng.normal(0, 0.001, 90))),
        "H2": _hist(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 90)))),
    }
    positions = _positions(["CAND", "H1", "H2"])
    config = {"correlation_block_threshold": 0.8, "correlation_lookback_days": 60}

    approved, reason = risk_guard.check_correlation("CAND", positions, ph, config)

    expected = [
        _original_corr(ph["CAND"], ph[h], 60) for h in ("H1", "H2")
    ]
    expected = [c for c in expected if pd.notna(c)]
    expected_mean = float(sum(expected) / len(expected))

    assert approved is bool(expected_mean <= 0.8), reason
    if not approved:
        assert f"{expected_mean:.2f}" in reason


def test_disabled_flag_still_short_circuits() -> None:
    approved, reason = risk_guard.check_correlation(
        "CAND", {}, {}, {"correlation_block_enabled": False}
    )
    assert approved and "disabled" in reason


def test_insufficient_candidate_history_still_approves() -> None:
    ph = {"CAND": _hist([100.0] * 10)}
    approved, reason = risk_guard.check_correlation(
        "CAND", _positions(["CAND"]), ph, {"correlation_lookback_days": 60}
    )
    assert approved and "insufficient price history" in reason


def test_cross_sector_positions_are_still_ignored() -> None:
    rng = np.random.default_rng(7)
    base = np.cumsum(rng.normal(0, 0.02, 90))
    ph = {
        "CAND": _hist(100.0 * np.exp(base)),
        "OTHER": _hist(100.0 * np.exp(base)),  # perfectly correlated…
    }
    positions = {"CAND": {"sector": "TECH"}, "OTHER": {"sector": "ENERGY"}}
    approved, reason = risk_guard.check_correlation(
        "CAND", positions, ph,
        {"correlation_block_threshold": 0.1, "correlation_lookback_days": 60},
    )
    # …but in a different sector, so never compared.
    assert approved and "no same-sector positions" in reason
