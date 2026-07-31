"""Tests for executor.champion.assert_producer_champion_coherence
(config#5713) — the producer/champion coherence guard that refuses to start
a trading day when a producer whose ``buy_candidates`` is empty BY CONTRACT
is paired with a no-op champion arm (a pairing that guarantees no new entry
is ever proposed, silently).

All hermetic — the assertion is pure (no S3, no network).
"""

from __future__ import annotations

import pytest

from executor.champion import (
    ProducerChampionIncoherenceError,
    assert_producer_champion_coherence,
)


def _signals(producer: str | None = "signals_envelope") -> dict:
    return {"producer": producer, "date": "2026-07-31", "buy_candidates": []}


def _pointer(champion: str = "agentic") -> dict:
    return {"schema_version": 1, "champion": champion}


def test_signals_envelope_x_agentic_raises():
    """The issue's canonical pair: envelope producer (empty by contract) ×
    no-op agentic champion must raise, not return zero candidates."""
    with pytest.raises(ProducerChampionIncoherenceError, match="EMPTY BY CONTRACT"):
        assert_producer_champion_coherence(_signals(), _pointer("agentic"), {})


def test_signals_envelope_x_scanner_predictor_direct_ok():
    """The synthesizing arm fills the empty list — the legitimate pairing."""
    assert_producer_champion_coherence(
        _signals(), _pointer("scanner_predictor_direct"), {},
    )


def test_signals_envelope_x_thinktank_coverage_ok():
    """Second synthesizing arm — also legitimate."""
    assert_producer_champion_coherence(
        _signals(), _pointer("thinktank_coverage"), {},
    )


def test_unstamped_producer_exempt():
    """A signals.json with no ``producer`` stamp has no declared contract —
    nothing to enforce (legacy/foreign artifacts must not crash the day)."""
    assert_producer_champion_coherence(
        {"date": "2026-07-31", "buy_candidates": []}, _pointer("agentic"), {},
    )


def test_unrecognized_producer_exempt():
    """A producer not named in the matrix (e.g. the multi-agent producer,
    which populates buy_candidates) has no empty-by-contract declaration."""
    assert_producer_champion_coherence(
        _signals(producer="multi_agent"), _pointer("agentic"), {},
    )


def test_config_extends_empty_by_contract_set():
    """The config row declares NEW empty-by-contract producers; the baseline
    (signals_envelope) stays enforced alongside them (union semantics)."""
    config = {
        "producers_emitting_empty_buy_candidates_by_contract": ["other_producer"],
    }
    with pytest.raises(ProducerChampionIncoherenceError, match="other_producer"):
        assert_producer_champion_coherence(
            _signals(producer="other_producer"), _pointer("agentic"), config,
        )
    with pytest.raises(ProducerChampionIncoherenceError):
        assert_producer_champion_coherence(_signals(), _pointer("agentic"), config)


def test_config_empty_producer_list_still_enforces_baseline():
    """Fail-closed: an empty config row must not silently disable the guard."""
    config = {"producers_emitting_empty_buy_candidates_by_contract": []}
    with pytest.raises(ProducerChampionIncoherenceError):
        assert_producer_champion_coherence(_signals(), _pointer("agentic"), config)


def test_config_extends_noop_arms():
    """A future no-op arm declared in config is enforced the same way."""
    config = {"champion_noop_arms": ["agentic", "future_noop"]}
    with pytest.raises(ProducerChampionIncoherenceError):
        assert_producer_champion_coherence(_signals(), _pointer("future_noop"), config)


def test_config_empty_noop_list_still_enforces_baseline():
    """Fail-closed for the arm side too: agentic is always a no-op."""
    config = {"champion_noop_arms": []}
    with pytest.raises(ProducerChampionIncoherenceError):
        assert_producer_champion_coherence(_signals(), _pointer("agentic"), config)
