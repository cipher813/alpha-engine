"""One alpha scale per solve — contract tests (alpha-engine-config-I7337).

Every test here is written against the MEASURED live numbers of 2026-08-14,
not invented ones:

* research-free cohort (2026-08-13, n=72): mean -0.2882, range
  -0.3184..-0.2094, ZERO positive, cross-sectional std 0.0177
* predictor predictions (2026-08-14, n=23): mean 4.3e-07, range
  -0.0400..+0.0948, 10/23 positive

Verified to FAIL on the pre-fix tree (policy-champion-challenger §7.4 — a
guard that cannot fail is worse than no guard, because it reads as coverage).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from executor.alpha_contract import (
    ANCHOR_FIELD,
    OPTIMIZER_ALPHA_ANCHOR,
    RAW_ALPHA_ANCHOR,
    AlphaAnchorError,
    assert_optimizer_anchor,
    center_to_market_relative,
)
from executor.optimizer_shadow import _build_alpha_hat

# ── Measured live constants (2026-08-14) ────────────────────────────────────
_COHORT_MEAN = -0.2882
_COHORT_STD = 0.0177
_COHORT_N = 72


def _research_free_cohort(n: int = _COHORT_N) -> list[float]:
    """A cohort with the measured live shape: every value negative, mean
    -0.2882, dispersion 0.0177. Deterministic."""
    rng = np.random.default_rng(7337)
    vals = rng.normal(_COHORT_MEAN, _COHORT_STD, n)
    # Re-center exactly onto the measured mean so the test asserts against a
    # known quantity rather than a sampling artifact.
    vals = vals - vals.mean() + _COHORT_MEAN
    assert (vals < 0).all(), "fixture must reproduce the all-negative cohort"
    return [float(v) for v in vals]


def _predictor_predictions() -> dict[str, dict]:
    """23 level-neutralized predictor records, mean ~0, straddling zero."""
    rng = np.random.default_rng(816)
    vals = rng.normal(0.0, 0.035, 23)
    vals = vals - vals.mean()
    return {
        f"P{i}": {
            "predicted_alpha": float(v),
            ANCHOR_FIELD: OPTIMIZER_ALPHA_ANCHOR,
        }
        for i, v in enumerate(vals)
    }


# ── center_to_market_relative ───────────────────────────────────────────────


def test_centering_moves_an_all_negative_cohort_onto_the_spy_anchor():
    """THE defect, in one assertion.

    Pre-fix the champion injected the raw column: 72 values, none of them
    positive, sitting ~29 points of 21d log alpha below the solve's SPY=0.0
    sentinel. No such name can ever win a mean-variance solve, whatever its
    merit. Post-centering the cohort straddles zero and its best names outrank
    the benchmark.
    """
    raw = _research_free_cohort()
    assert max(raw) < 0.0, "pre-condition: the raw cohort cannot beat SPY=0"

    centered, mean_removed = center_to_market_relative(raw)

    assert mean_removed == pytest.approx(_COHORT_MEAN, abs=1e-9)
    assert sum(centered) / len(centered) == pytest.approx(0.0, abs=1e-12)
    assert max(centered) > 0.0, "the best name must be able to beat SPY"
    assert min(centered) < 0.0
    # Dispersion is PRESERVED, not rescaled — see alpha_contract's rationale.
    assert float(np.std(centered)) == pytest.approx(float(np.std(raw)), rel=1e-9)


def test_centering_is_rank_preserving_so_the_arm_is_unchanged():
    """A constant shift cannot reorder. This is what makes the fix a units
    correction rather than a champion/challenger arm swap: the arm selects the
    identical candidate set before and after."""
    raw = _research_free_cohort()
    centered, _ = center_to_market_relative(raw)
    assert list(np.argsort(raw)) == list(np.argsort(centered))


def test_centering_refuses_a_degenerate_cross_section():
    with pytest.raises(AlphaAnchorError, match="cross-section of 1"):
        center_to_market_relative([-0.28])
    with pytest.raises(AlphaAnchorError, match="cross-section of 0"):
        center_to_market_relative([])


def test_centering_refuses_a_non_finite_value():
    with pytest.raises(AlphaAnchorError, match="non-finite"):
        center_to_market_relative([-0.28, float("nan"), -0.29])


# ── assert_optimizer_anchor ─────────────────────────────────────────────────


def _universe(preds: dict[str, dict]) -> tuple[list[str], int, int]:
    tickers = [*preds.keys(), "SPY", "CASH"]
    return tickers, len(tickers) - 2, len(tickers) - 1


def test_mixed_anchor_batch_is_refused_and_names_the_offenders():
    """The core guard: a market-relative batch plus ONE raw name must not
    solve. Pre-fix this mixture solved silently and flushed the book."""
    preds = _predictor_predictions()
    preds["RAWNAME"] = {
        "predicted_alpha": -0.2882,
        ANCHOR_FIELD: RAW_ALPHA_ANCHOR,
    }
    tickers, spy_idx, cash_idx = _universe(preds)

    with pytest.raises(AlphaAnchorError) as exc:
        assert_optimizer_anchor(
            tickers,
            preds,
            spy_idx=spy_idx,
            cash_idx=cash_idx,
        )
    msg = str(exc.value)
    assert "RAWNAME" in msg, "the failure must name the offending ticker"
    assert RAW_ALPHA_ANCHOR in msg, "and the anchor it declared"
    assert OPTIMIZER_ALPHA_ANCHOR in msg, "and the anchor required"


def test_undeclared_anchor_is_refused():
    """An unstamped numeric alpha is an UNKNOWN-anchor alpha. It is refused
    rather than assumed market-relative — the two are indistinguishable by
    value, which is precisely how this shipped."""
    preds = _predictor_predictions()
    preds["NOANCHOR"] = {"predicted_alpha": -0.2882}
    tickers, spy_idx, cash_idx = _universe(preds)

    with pytest.raises(AlphaAnchorError, match="NOANCHOR"):
        assert_optimizer_anchor(
            tickers,
            preds,
            spy_idx=spy_idx,
            cash_idx=cash_idx,
        )


def test_a_uniform_market_relative_batch_passes_and_publishes_its_count():
    preds = _predictor_predictions()
    tickers, spy_idx, cash_idx = _universe(preds)
    block = assert_optimizer_anchor(
        tickers,
        preds,
        spy_idx=spy_idx,
        cash_idx=cash_idx,
    )
    assert block["n_checked"] == 23
    assert block["single_anchor"] is True
    assert block["anchors"] == {OPTIMIZER_ALPHA_ANCHOR: 23}
    assert block["expected_anchor"] == OPTIMIZER_ALPHA_ANCHOR


def test_sentinels_and_opinionless_records_carry_no_anchor_obligation():
    """SPY/CASH are solver sentinels, and a `predicted_alpha: None` record
    (the thinktank arm's honest abstention) asserts no level. Neither may be
    forced to declare an anchor — that would make an abstention illegal."""
    preds = _predictor_predictions()
    preds["TT"] = {"predicted_alpha": None, "thinktank_coverage": True}
    preds["SPY"] = {"predicted_alpha": -99.0}  # unstamped sentinel
    preds["CASH"] = {"predicted_alpha": -99.0}
    tickers = [*_predictor_predictions().keys(), "TT", "SPY", "CASH"]
    block = assert_optimizer_anchor(
        tickers,
        preds,
        spy_idx=tickers.index("SPY"),
        cash_idx=tickers.index("CASH"),
    )
    assert block["n_checked"] == 23


# ── _build_alpha_hat ────────────────────────────────────────────────────────


def test_build_alpha_hat_refuses_the_live_pre_fix_mixture():
    """End to end at the boundary, with both live distributions in one dict —
    the exact vector the 2026-08-14 solve received."""
    preds = _predictor_predictions()
    for i, v in enumerate(_research_free_cohort(10)):
        preds[f"C{i}"] = {"predicted_alpha": v, ANCHOR_FIELD: RAW_ALPHA_ANCHOR}
    tickers, spy_idx, cash_idx = _universe(preds)

    with pytest.raises(AlphaAnchorError):
        _build_alpha_hat(tickers, preds, spy_idx, cash_idx)


def test_build_alpha_hat_solves_a_single_anchor_vector():
    preds = _predictor_predictions()
    centered, _ = center_to_market_relative(_research_free_cohort(10))
    for i, v in enumerate(centered):
        preds[f"C{i}"] = {"predicted_alpha": v, ANCHOR_FIELD: OPTIMIZER_ALPHA_ANCHOR}
    tickers, spy_idx, cash_idx = _universe(preds)

    alpha = _build_alpha_hat(tickers, preds, spy_idx, cash_idx)

    assert alpha[spy_idx] == 0.0
    # At least one champion-sourced name must be able to outrank SPY, or the
    # arm is structurally incapable of winning a solve and is not a challenger.
    champion_alphas = [alpha[tickers.index(f"C{i}")] for i in range(10)]
    assert max(champion_alphas) > alpha[spy_idx]


def test_an_exact_zero_alpha_is_an_opinion_not_a_missing_value():
    """Falsy-`or` regression. The pre-fix chain was
    ``pred.get("predicted_alpha") or pred.get("canonical_predicted_alpha") or 0.0``
    — an exact 0.0 is falsy, so a name whose model said "exactly neutral" fell
    through to a stale ``canonical_predicted_alpha``. Same present-and-zero
    class as the ``pullback_pct`` defect fixed in crucible-executor-PR477.
    """
    preds = {
        "Z": {
            "predicted_alpha": 0.0,
            "canonical_predicted_alpha": 0.09,  # must NOT be reached
            ANCHOR_FIELD: OPTIMIZER_ALPHA_ANCHOR,
        },
    }
    tickers, spy_idx, cash_idx = _universe(preds)
    alpha = _build_alpha_hat(tickers, preds, spy_idx, cash_idx)
    assert alpha[0] == 0.0


def test_non_finite_alpha_does_not_poison_the_vector():
    preds = {
        "N": {"predicted_alpha": float("nan"), ANCHOR_FIELD: OPTIMIZER_ALPHA_ANCHOR},
        "G": {"predicted_alpha": 0.05, ANCHOR_FIELD: OPTIMIZER_ALPHA_ANCHOR},
    }
    tickers, spy_idx, cash_idx = _universe(preds)
    alpha = _build_alpha_hat(tickers, preds, spy_idx, cash_idx)
    assert all(math.isfinite(x) for x in alpha)
    assert alpha[0] == 0.0
    assert alpha[1] == pytest.approx(0.05)


# ── champion adapter: the producing side ────────────────────────────────────


class TestChampionInjectsMarketRelativeAlpha:
    """The scanner_predictor_direct arm is the producer of the second alpha
    stream. These run it through the real S3 fakes from ``test_champion``."""

    @staticmethod
    def _run(cohort_alphas: list[float]):
        from executor.champion import (
            CHAMPION_POINTER_KEY,
            RESEARCH_FREE_PARQUET_KEY,
            apply_champion_selection,
        )
        from tests.test_champion import _FakeS3, _parquet_bytes, _pointer_bytes

        rows = [
            {
                "ticker": f"TKR{i:03d}",
                "prediction_date": "2026-08-13",
                "predicted_alpha": a,
                "n_research_features_missing": 4,
            }
            for i, a in enumerate(cohort_alphas)
        ]
        s3 = _FakeS3(
            {
                CHAMPION_POINTER_KEY: _pointer_bytes(champion="scanner_predictor_direct"),
                RESEARCH_FREE_PARQUET_KEY: _parquet_bytes(rows),
            }
        )
        return apply_champion_selection(
            {"date": "2026-08-14", "buy_candidates": [], "universe": []},
            {},
            bucket="test-bucket",
            run_date="2026-08-14",
            config={
                "champion_top_n_default": 10,
                "champion_score_floor": 60,
                "champion_score_ceiling": 95,
                "champion_freshness_max_days": 8,
            },
            sector_map={},
            s3_client=s3,
        )

    def test_the_all_negative_live_cohort_yields_a_positive_top_name(self):
        """THE operator-visible defect. Feed the measured 2026-08-13 cohort —
        72 rows, every one negative — and the top selected name must come out
        able to beat SPY. Pre-fix every injected alpha was <= -0.209."""
        out_signals, out_preds = self._run(_research_free_cohort())

        injected = {t: p for t, p in out_preds.items() if p.get("research_free")}
        assert injected, "the arm must have injected predictions"
        alphas = [p["predicted_alpha"] for p in injected.values()]
        assert max(alphas) > 0.0, f"the champion's best name must be able to outrank SPY=0.0; got max={max(alphas)}"

    def test_every_injected_prediction_declares_the_optimizer_anchor(self):
        _, out_preds = self._run(_research_free_cohort())
        injected = [p for p in out_preds.values() if p.get("research_free")]
        assert all(p[ANCHOR_FIELD] == OPTIMIZER_ALPHA_ANCHOR for p in injected)
        assert all(p["alpha_anchor_source"] == "champion_xsec_centered" for p in injected)

    def test_the_removed_common_mode_is_recorded_not_just_applied(self):
        """Measurability: the correction publishes its own magnitude, on the
        artifact and on each record. A silent correction is unmonitorable —
        a drifting common mode is the health signal for the producer-side
        defect this compensates for."""
        out_signals, out_preds = self._run(_research_free_cohort())
        cohort_block = out_signals["champion_cohort"]
        assert cohort_block["alpha_anchor"] == OPTIMIZER_ALPHA_ANCHOR
        assert cohort_block["alpha_xsec_mean_removed"] == pytest.approx(_COHORT_MEAN, abs=1e-9)
        injected = [p for p in out_preds.values() if p.get("research_free")]
        for p in injected:
            assert p["alpha_xsec_mean_removed"] == pytest.approx(_COHORT_MEAN, abs=1e-9)
            # The raw parquet value survives for forensics — the corrected
            # number must remain attributable to its source row.
            assert p["predicted_alpha_raw"] < 0.0
            assert p["predicted_alpha"] == pytest.approx(
                p["predicted_alpha_raw"] - p["alpha_xsec_mean_removed"], abs=1e-12
            )

    def test_the_selected_candidate_set_is_unchanged_by_the_correction(self):
        """Centering is rank-preserving, so this fix does NOT change which
        names the arm picks — only the level it reports them at. That is what
        keeps it a units correction rather than a champion/challenger arm
        swap requiring a promotion decision."""
        cohort = _research_free_cohort(20)
        out_raw_order = sorted(range(len(cohort)), key=lambda i: -cohort[i])[:10]
        expected = {f"TKR{i:03d}" for i in out_raw_order}

        out_signals, _ = self._run(cohort)
        got = {e["ticker"] for e in out_signals["buy_candidates"]}
        assert got == expected

    def test_the_injected_batch_passes_the_optimizer_contract(self):
        """Producer and consumer agree — the whole point."""
        _, out_preds = self._run(_research_free_cohort())
        tickers = [*out_preds.keys(), "SPY", "CASH"]
        block = assert_optimizer_anchor(
            tickers,
            out_preds,
            spy_idx=len(tickers) - 2,
            cash_idx=len(tickers) - 1,
        )
        assert block["single_anchor"] is True
        assert block["n_checked"] == 10


# ── signal_reader: the predictor side ───────────────────────────────────────


class TestReadPredictionsStampsTheDeclaredAnchor:
    """The anchor is read from the artifact's OWN `level_neutralization`
    block, never guessed from the field name or the sign of the values."""

    @staticmethod
    def _read(level_block, monkeypatch):
        import json

        from executor import signal_reader

        payload = {
            "date": "2026-08-14",
            "predictions": [
                {"ticker": "AAA", "predicted_alpha": 0.09},
                {"ticker": "BBB", "predicted_alpha": -0.04},
            ],
        }
        if level_block is not None:
            payload["level_neutralization"] = level_block

        class _S3:
            def get_object(self, Bucket, Key):  # noqa: N803
                class _B:
                    @staticmethod
                    def read():
                        return json.dumps(payload).encode()

                return {"Body": _B()}

        monkeypatch.setattr(signal_reader.boto3, "client", lambda *a, **k: _S3())
        return signal_reader.read_predictions("test-bucket")

    def test_applied_true_is_market_relative(self, monkeypatch):
        preds, _ = self._read({"enabled": True, "applied": True, "xsec_mean_removed": -0.011}, monkeypatch)
        assert all(p[ANCHOR_FIELD] == OPTIMIZER_ALPHA_ANCHOR for p in preds.values())

    def test_applied_false_is_raw_and_the_optimizer_refuses_it(self, monkeypatch):
        """Observe-mode (`XSEC_DEMEAN_ALPHA_ENABLED=False`) leaves the common
        mode in. The executor does NOT re-center it — crucible-predictor's
        level_neutralization declares centering a producer-side single source
        of truth with no per-consumer re-derivation — so the batch is stamped
        raw and the solve refuses it loudly."""
        preds, _ = self._read({"enabled": False, "applied": False, "xsec_mean_removed": -0.288}, monkeypatch)
        assert all(p[ANCHOR_FIELD] == RAW_ALPHA_ANCHOR for p in preds.values())

        tickers = ["AAA", "BBB", "SPY", "CASH"]
        with pytest.raises(AlphaAnchorError, match=RAW_ALPHA_ANCHOR):
            assert_optimizer_anchor(tickers, preds, spy_idx=2, cash_idx=3)

    def test_an_absent_block_is_unknown_not_assumed_neutralized(self, monkeypatch):
        """Predictions written before the block shipped (2026-06-01) carry no
        evidence either way. Absent is stamped raw — an unmeasured anchor is
        never rendered as the good one."""
        preds, _ = self._read(None, monkeypatch)
        assert all(p[ANCHOR_FIELD] == RAW_ALPHA_ANCHOR for p in preds.values())


def test_a_single_name_champion_cohort_is_refused_not_silently_injected():
    """A one-name cohort has no definable market level. Pre-I7337 its raw
    alpha was injected as-is, which is the whole defect in miniature; the
    alternative of centering it to itself would assert an exactly-neutral
    opinion the model never expressed. Both are wrong, so it raises.
    """
    with pytest.raises(AlphaAnchorError, match="cross-section of 1"):
        TestChampionInjectsMarketRelativeAlpha._run([-0.2882])
