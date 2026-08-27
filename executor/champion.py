"""Champion candidate-source adapter (config#2364 / config#2366 /
alpha-engine-config-I2518 / I2515).

The champion-promotion loop lets the trading system switch its ENTRY
candidate source between arms without touching the exit/risk stack:

  * ``agentic``               — today's research pipeline (signals.json
                                 buy_candidates, unchanged).
  * ``scanner_predictor_direct`` — the "measured" arm: entries synthesized
                                 directly from the research-free predictor's
                                 outcome parquet, ranked by predicted_alpha.
  * ``thinktank_coverage``    — the Think Tank challenger arm (epic I2515):
                                 entries synthesized from the Think Tank
                                 challenger-selection artifact
                                 (``thinktank/challenger_selection/latest.json``,
                                 crucible-research PR#427), ranked by the
                                 analyst's own independent 0-100 rating.
                                 Brian's ruling (config-I2518, 2026-07-14):
                                 champion and challenger run side by side,
                                 whichever performs best in a given week is
                                 promoted at that time — this module is what
                                 lets the pointer actually EXECUTE when it
                                 flips to this arm.

Design correction (2026-07-11, config#2366): this is a PLANNER-LAYER
candidate-source adapter, NOT an ``executor/strategies/`` Slot-S plugin —
that contract is EXIT RULES ONLY (``ALLOWED_ACTIONS = ("EXIT", "REDUCE")``
in ``executor/strategies/contract.py``; entries are explicitly out of
scope there). The champion switch instead rewrites
``signals_raw["buy_candidates"]`` before the existing universe/coverage
gates run, so synthesized entries are subject to the SAME risk-rule path
as agentic ones.

Fail-loud posture: the ONLY silent default in this module is the S3 404
on the champion pointer itself (pre-bootstrap state, unambiguous — no
promotion has ever been written). Every other ambiguous condition
(malformed pointer JSON, unknown champion value, stale predictor cohort,
missing/incomplete/stale challenger-selection artifact) raises. No trading
day should start — or silently mis-trade — on an ambiguous champion
selection; the pointer is customer-visible via Metron's Showcase Portfolio.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import date

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from executor.alpha_contract import (
    ANCHOR_FIELD,
    ANCHOR_SOURCE_FIELD,
    OPTIMIZER_ALPHA_ANCHOR,
    AlphaAnchorError,
    _numeric_alpha,
    center_to_market_relative,
)

logger = logging.getLogger(__name__)

CHAMPION_POINTER_KEY = "config/producer_champion.json"
RESEARCH_FREE_PARQUET_KEY = "predictor/research_free_backfill/predictor_outcomes_research_free.parquet"
# Think Tank's challenger-arm submission (crucible-research thinktank/__init__.py
# CHALLENGER_SELECTION_LATEST_KEY — kept as a literal here rather than an
# import to avoid a cross-repo package dependency from crucible-executor on
# crucible-research; the key is a stable S3 contract, not shared code).
CHALLENGER_SELECTION_LATEST_KEY = "thinktank/challenger_selection/latest.json"

VALID_CHAMPIONS = (
    "agentic",
    "scanner_predictor_direct",
    "scanner_top20_predictor",
    "thinktank_coverage",
)

# alpha-engine-config-I8755 — the entry-selection arm the pipeline was already
# shaped for.
#
# Brian, 2026-08-27: "Im saying top 20 attractiveness from scanner, which is
# evaluated weekly, gets passed to the predictor as an arm to research's
# champion/challenger" — and, on the predictor's output: "the predictor's live
# daily output is downstream of what the scanner provides, correct? The scanner
# should be providing the top 20 to predictor, weekly."
#
# It is, and that settles what this arm reads. The live chain is:
#
#   Scanner (weekly) -> universe_membership/{date}/membership.json
#                       predictor_universe_cut: attractiveness_top_20
#                                  |
#   Predictor (daily) -> predictor/predictions/{date}.json
#                        canonical/predicted_alpha per name, already stamped
#                        alpha_anchor = market_relative_21d_log
#                                  |
#   Executor          -> this arm
#
# `scanner_top20_predictor` selects the top N by the PREDICTOR'S OWN alpha,
# restricted to the weekly cut the predictor resolves from. There is no second
# ranking to choose between: the predictor's output IS the scanner's top-20,
# scored.
#
# It therefore reads NOTHING that `_apply_scanner_predictor_direct` reads. That
# arm consumes `predictor/research_free_backfill/...parquet`, a parallel weekly
# feed of the meta-model with the four research features zeroed, whose cohort is
# the scanner's 60 and which never touches the predictor's cut. Measured
# 2026-08-27 on the live artifacts: the research-free parquet covered 8 of the
# 20 cut members, the predictor's live output covered 20 of 20, and the two
# alphas ranked their 12 overlapping names at Spearman 0.587. They are two
# different rules, which is why both run as arms.
#
# Count-matching (champion-challenger-policy.md §4) is at the OUTPUT: every arm
# in this slot emits `champion_top_n_default` = 10 entries. The differing input
# widths are the treatment under test, not an unmatched comparison.
#
# Nothing is injected into `predictions_by_ticker`. The other two arms must
# synthesize prediction records because their picks come from outside the
# predictor's output; this arm's picks ARE that output, already on the declared
# anchor, so `assert_predictions_cover_buy_candidates` passes by construction
# rather than by fabrication.
_MEMBERSHIP_LATEST_KEY = "universe_membership/latest.json"
_MEMBERSHIP_MAX_AGE_DAYS = 10

# Pre-bootstrap default: no promotion has ever been written, so the pointer
# key legitimately does not exist yet. This is the one unambiguous silent
# default in the whole module — every other failure mode below raises.
_DEFAULT_POINTER = {
    "schema_version": 1,
    "champion": "agentic",
    "promotion_source": "default_pre_bootstrap",
}


class ChampionPointerError(RuntimeError):
    """Raised when the champion pointer is present but unreadable/ambiguous.

    Deliberately NOT raised on a clean S3 404 (see ``_DEFAULT_POINTER``) —
    only on malformed JSON, an unknown ``champion`` value, or any other
    S3 error besides "the key doesn't exist yet".
    """


class StaleChampionFeedError(RuntimeError):
    """Raised when the research-free predictor cohort is older than the
    configured freshness window. A stale champion feed must not trade
    silently on data that no longer reflects current market state."""


class ProducerChampionIncoherenceError(ChampionPointerError):
    """Raised when a producer whose ``buy_candidates`` is EMPTY BY CONTRACT
    is paired with a champion arm that cannot synthesize entries (a no-op
    arm). Such a pairing guarantees zero new entries are ever proposed — a
    configuration incoherence (a rollback or bad promotion of the champion
    pointer), never a market condition (config#5713)."""


# Producers whose signals.json ``buy_candidates`` is EMPTY BY CONTRACT —
# they never propose entries themselves and rely on a champion arm that
# synthesizes them (today: ``signals_envelope``, see crucible-research
# scoring/signals_envelope.py's docstring caveat). These baselines mirror
# the declared producer/champion compatibility matrix on the private side
# (executor risk.yaml keys
# ``producers_emitting_empty_buy_candidates_by_contract`` /
# ``champion_noop_arms`` and crucible-research producers/registry.py,
# config#5713): config values EXTEND the baselines (union), they can never
# disable them — a stale/missing/partial config row must not silently turn
# the coherence guard off.
DEFAULT_EMPTY_BUY_CANDIDATES_PRODUCERS = frozenset({"signals_envelope"})

# Champion arms that are no-op passthroughs in ``apply_champion_selection``
# — they neither read nor fill ``buy_candidates``. Today exactly one: the
# ``agentic`` arm. Config-extensible via ``champion_noop_arms`` for the
# same fail-closed reason as the producer set.
DEFAULT_NOOP_CHAMPION_ARMS = frozenset({"agentic"})


def load_champion_pointer(bucket: str, s3_client=None) -> dict:
    """Read ``s3://{bucket}/config/producer_champion.json``.

    Semantics:
      * S3 404 / NoSuchKey → ``_DEFAULT_POINTER`` (agentic, pre-bootstrap).
        This is the one legitimate "pointer doesn't exist yet" case.
      * Any other S3 read error, malformed JSON, or an unknown ``champion``
        value → raise ``ChampionPointerError``. An ambiguous champion must
        never resolve to a silent default — that would risk starting a
        trading day (or worse, silently mixing arms) on a corrupt pointer.

    Pointer schema (written independently by config#2367 in
    crucible-backtester): ``{schema_version: 1, champion: "agentic" |
    "scanner_predictor_direct" | "thinktank_coverage", promoted_at:
    <iso8601>, promotion_source: <str>}``.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=CHAMPION_POINTER_KEY)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            logger.info(
                "No champion pointer at s3://%s/%s — defaulting to agentic (pre-bootstrap state)",
                bucket,
                CHAMPION_POINTER_KEY,
            )
            return dict(_DEFAULT_POINTER)
        raise ChampionPointerError(f"Failed to read champion pointer s3://{bucket}/{CHAMPION_POINTER_KEY}: {e}") from e

    try:
        raw = obj["Body"].read()
        pointer = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, KeyError) as e:
        raise ChampionPointerError(f"Champion pointer s3://{bucket}/{CHAMPION_POINTER_KEY} is malformed: {e}") from e

    if not isinstance(pointer, dict):
        raise ChampionPointerError(
            f"Champion pointer s3://{bucket}/{CHAMPION_POINTER_KEY} did not "
            f"parse to a JSON object (got {type(pointer).__name__})"
        )

    champion = pointer.get("champion")
    if champion not in VALID_CHAMPIONS:
        raise ChampionPointerError(
            f"Champion pointer s3://{bucket}/{CHAMPION_POINTER_KEY} has "
            f"unknown champion={champion!r} — expected one of {VALID_CHAMPIONS}. "
            "Refusing to start a trading day on an ambiguous champion."
        )

    return pointer


def assert_producer_champion_coherence(
    signals_raw: dict,
    pointer: dict,
    config: dict,
) -> None:
    """Raise when a producer whose ``buy_candidates`` is EMPTY BY CONTRACT
    is paired with a champion arm that cannot fill it (config#5713).

    The read path is the only place both facts are visible: the producer
    stamps ``signals_raw["producer"]`` at write time, the champion pointer
    resolves at read time, and neither side can see the other. An empty
    ``buy_candidates`` under the ``agentic`` no-op arm means NO new entry
    will ever be proposed — a configuration incoherence, not a market
    condition, and exactly the failure a rollback or bad promotion of the
    champion pointer produces silently (see crucible-research
    scoring/signals_envelope.py's docstring caveat).

    The producer set and the no-op arm set are DECLARED, not hardcoded in
    the branch: the module baselines mirror the compatibility matrix in the
    private config repo (executor risk.yaml keys
    ``producers_emitting_empty_buy_candidates_by_contract`` /
    ``champion_noop_arms``) and crucible-research's ``producers/registry.py``
    — config values EXTEND the baselines (union semantics); a
    missing/empty/partial config row still enforces the baselines, so the
    guard can never silently disable itself.

    Producers that do not stamp ``producer`` are exempt — no declared
    contract to enforce. Unrecognized producers are exempt — the matrix
    only names producers whose emptiness is a contract, not a value.
    """
    producer = signals_raw.get("producer")
    if not producer:
        return
    champion = pointer.get("champion")
    empty_by_contract = DEFAULT_EMPTY_BUY_CANDIDATES_PRODUCERS | frozenset(
        config.get("producers_emitting_empty_buy_candidates_by_contract") or ()
    )
    noop_arms = DEFAULT_NOOP_CHAMPION_ARMS | frozenset(config.get("champion_noop_arms") or ())
    if producer in empty_by_contract and champion in noop_arms:
        raise ProducerChampionIncoherenceError(
            f"Producer/champion configuration incoherence (config#5713): "
            f"signals.json was written by producer={producer!r}, whose "
            f"buy_candidates is EMPTY BY CONTRACT, but the resolved champion "
            f"arm is {champion!r} — a no-op passthrough that never "
            "synthesizes entries. No new entry will ever be proposed while "
            "this pairing is live (the book trades down and never up). "
            "Promote the champion pointer "
            f"({CHAMPION_POINTER_KEY}) to a synthesizing arm "
            "(scanner_predictor_direct / thinktank_coverage) or switch the "
            "signals producer. Compatibility matrix declared in the "
            "executor risk.yaml + crucible-research producers/registry.py. "
            "Refusing to start a trading day on a configuration that "
            "guarantees zero entries."
        )


def _rank_to_score(rank_fraction: float, floor: float, ceiling: float) -> float:
    """Map a within-cohort rank fraction in [0, 1] (0 = best) onto
    ``[floor, ceiling]``, best rank → ceiling, worst rank → floor.

    Monotonic and deterministic — preserves the predictor's relative
    ordering through ``decide_entries``' score gates (``min_score_to_enter``
    etc.) without hand-tuning a score per name.
    """
    if ceiling <= floor:
        raise ValueError(f"champion_score_ceiling ({ceiling}) must exceed champion_score_floor ({floor})")
    rank_fraction = min(max(rank_fraction, 0.0), 1.0)
    return ceiling - rank_fraction * (ceiling - floor)


def _load_research_free_cohort(bucket: str, s3_client=None) -> pd.DataFrame:
    """Read the research-free predictor outcomes parquet and return the
    LATEST ``prediction_date`` cohort as a DataFrame.

    Schema (crucible-backtester PR#486/#482, already live): ``ticker,
    prediction_date, predicted_alpha, n_research_features_missing``.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=RESEARCH_FREE_PARQUET_KEY)
        body = obj["Body"].read()
    except ClientError as e:
        raise ChampionPointerError(
            f"scanner_predictor_direct champion selected but research-free "
            f"parquet s3://{bucket}/{RESEARCH_FREE_PARQUET_KEY} is unreadable: {e}. "
            "Refusing to trade on a missing champion feed."
        ) from e

    try:
        df = pd.read_parquet(io.BytesIO(body))
    except Exception as e:  # noqa: BLE001 — any parse failure must raise, not silently no-op
        raise ChampionPointerError(
            f"Failed to parse research-free parquet s3://{bucket}/{RESEARCH_FREE_PARQUET_KEY}: {e}"
        ) from e

    required_cols = {"ticker", "prediction_date", "predicted_alpha"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ChampionPointerError(
            f"research-free parquet missing required column(s) {sorted(missing_cols)} "
            f"— got columns {sorted(df.columns)}"
        )

    if df.empty:
        raise ChampionPointerError(
            "research-free parquet is empty — scanner_predictor_direct champion has no candidates to select from."
        )

    latest_date = df["prediction_date"].max()
    cohort = df[df["prediction_date"] == latest_date].copy()
    return cohort


def _load_challenger_selection(bucket: str, s3_client=None) -> dict:
    """Read the Think Tank challenger-selection artifact
    (``thinktank/challenger_selection/latest.json``, crucible-research
    PR#427) and return it as a validated dict.

    Mirrors ``_load_research_free_cohort``'s failure-mode convention
    exactly: missing/unreadable artifact or malformed JSON raises
    ``ChampionPointerError`` — same degrade path, same loudness as the
    scanner_predictor_direct arm's missing-parquet case (no fallback to
    the raw signals.json candidates).

    Schema (``thinktank.schemas.ChallengerSelection``, read here as a
    plain dict — no cross-repo import of crucible-research's pydantic
    model, only the stable field-name contract): ``schema_version, arm,
    trading_day, calendar_date, run_id, mode, board_date, coverage_complete,
    uncovered_count, selections: [{ticker, rating, stance, conviction,
    thesis_version, attractiveness_rank}]``.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=CHALLENGER_SELECTION_LATEST_KEY)
        body = obj["Body"].read()
    except ClientError as e:
        raise ChampionPointerError(
            f"thinktank_coverage champion selected but challenger-selection "
            f"artifact s3://{bucket}/{CHALLENGER_SELECTION_LATEST_KEY} is "
            f"unreadable: {e}. Refusing to trade on a missing champion feed."
        ) from e

    try:
        selection = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as e:
        raise ChampionPointerError(
            f"Challenger-selection artifact s3://{bucket}/{CHALLENGER_SELECTION_LATEST_KEY} is malformed: {e}"
        ) from e

    if not isinstance(selection, dict):
        raise ChampionPointerError(
            f"Challenger-selection artifact s3://{bucket}/"
            f"{CHALLENGER_SELECTION_LATEST_KEY} did not parse to a JSON "
            f"object (got {type(selection).__name__})"
        )

    required_top_level = {"trading_day", "coverage_complete", "selections"}
    missing_top_level = required_top_level - set(selection.keys())
    if missing_top_level:
        raise ChampionPointerError(
            f"Challenger-selection artifact s3://{bucket}/"
            f"{CHALLENGER_SELECTION_LATEST_KEY} missing required top-level "
            f"key(s) {sorted(missing_top_level)} — got keys "
            f"{sorted(selection.keys())}"
        )

    selections = selection.get("selections")
    if not isinstance(selections, list):
        raise ChampionPointerError(
            f"Challenger-selection artifact s3://{bucket}/"
            f"{CHALLENGER_SELECTION_LATEST_KEY} 'selections' is not a list "
            f"(got {type(selections).__name__})"
        )

    required_row_keys = {"ticker", "rating"}
    for i, row in enumerate(selections):
        if not isinstance(row, dict):
            raise ChampionPointerError(
                f"Challenger-selection artifact s3://{bucket}/"
                f"{CHALLENGER_SELECTION_LATEST_KEY} selections[{i}] is not "
                f"an object (got {type(row).__name__})"
            )
        missing_row_keys = required_row_keys - set(row.keys())
        if missing_row_keys:
            raise ChampionPointerError(
                f"Challenger-selection artifact s3://{bucket}/"
                f"{CHALLENGER_SELECTION_LATEST_KEY} selections[{i}] missing "
                f"required key(s) {sorted(missing_row_keys)} — got "
                f"{sorted(row.keys())}"
            )

    return selection


#: Calendar-day age at or below which a champion cohort is considered current.
#: 3 allows the legitimate Friday-cohort-used-Monday case and nothing looser —
#: the producer refreshes every trading day, so anything beyond a weekend gap
#: means the producer has stopped. Deliberately NOT the hard-fail bound: see
#: the I7216 comment in ``_apply_scanner_predictor_direct``.
COHORT_FRESH_MAX_DAYS = 3

#: Hard freshness bound for the ``thinktank_coverage`` arm's
#: challenger-selection artifact, INDEPENDENT of ``champion_freshness_max_days``
#: (alpha-engine-config-I7232, Brian ruling 2026-08-14). A freshness bound is
#: a property of the PRODUCER's cadence, not a fleet-wide constant: the Think
#: Tank is a DAILY producer, while ``champion_freshness_max_days`` (default 8)
#: is shared with the ``scanner_predictor_direct`` arm, whose producer runs
#: WEEKLY. Sharing the 8-day bound is exactly what let a 3-day-frozen
#: challenger-selection pointer pass silently into a live champion feed.
#: Set to 3 to match ``POINTER_LAG_ERROR_DAYS`` in crucible-research
#: (thinktank/challenger_selection.py, crucible-research-PR630) — the
#: producer-side ERROR and this consumer-side refusal now agree on what
#: "stale" means for this arm. Accepted delta: the arm refuses to trade
#: sooner during a Think Tank outage than it did before; that is the intent.
#: This constant is deliberately NOT read from ``config`` — the whole point
#: of the ruling is that this bound must not silently move if
#: ``champion_freshness_max_days`` is retuned for the scanner arm.
THINKTANK_COVERAGE_FRESHNESS_MAX_DAYS = 3


def evaluate_cohort_staleness(prediction_date, run_date: str, config: dict) -> dict:
    """Classify the champion cohort's age. Never raises; never blocks.

    Returns a dict carrying the cohort date, its age in calendar days, and
    whether that age is within the fresh window. It is emitted on every run —
    including the healthy one — because a field that only appears when
    something is wrong cannot be trended, and its absence reads as "fine"
    rather than "not measured" (``principles.md`` §2.7).

    alpha-engine-config-I7216. The hard ``_check_freshness`` bound stays where
    it is; this is the signal that was missing between "current" and "so old
    we refuse to trade".
    """
    fresh_max = int(config.get("champion_cohort_fresh_max_days", COHORT_FRESH_MAX_DAYS))
    pred_d = pd.Timestamp(prediction_date).date()
    run_d = date.fromisoformat(run_date)
    age_days = (run_d - pred_d).days
    is_stale = age_days > fresh_max
    record = {
        "cohort_prediction_date": pred_d.isoformat(),
        "run_date": run_date,
        "age_days": age_days,
        "fresh_max_days": fresh_max,
        "is_stale": is_stale,
    }
    if is_stale:
        # WARNING, not INFO: this run will trade, and the orders it places are
        # derived from a candidate ranking this many days old. That belongs in
        # the degraded-run path, not buried in a success.
        logger.warning(
            "[champion] STALE COHORT: prediction_date=%s is %d calendar day(s) "
            "before run_date=%s (fresh window %d). Entries this run are drawn "
            "from a frozen candidate pool — check the producer "
            "(predictor/research_free_backfill/). alpha-engine-config-I7216.",
            pred_d,
            age_days,
            run_date,
            fresh_max,
        )
    else:
        logger.info(
            "[champion] cohort age OK: prediction_date=%s is %d day(s) before run_date=%s (fresh window %d)",
            pred_d,
            age_days,
            run_date,
            fresh_max,
        )
    return record


def _check_freshness(
    prediction_date, run_date: str, max_days: int, *, feed_label: str = "scanner_predictor_direct champion cohort"
) -> None:
    """Raise ``StaleChampionFeedError`` if ``prediction_date`` is more than
    ``max_days`` calendar days older than ``run_date``.

    ``feed_label`` names the feed in the raised message so the same
    calendar-day-diff check (technique mirrors ``main._warn_if_stale``'s
    knowledge-day age computation, but with HARD-FAIL severity — a
    champion-arm feed must never trade silently on stale data per this
    module's fail-loud posture) reads correctly for whichever arm calls it.
    """
    pred_d = pd.Timestamp(prediction_date).date()
    run_d = date.fromisoformat(run_date)
    age_days = (run_d - pred_d).days
    if age_days > max_days:
        raise StaleChampionFeedError(
            f"{feed_label} is stale: "
            f"prediction_date={pred_d} is {age_days} calendar day(s) before "
            f"run_date={run_date} (max allowed {max_days}). "
            "A stale champion feed must not trade silently."
        )
    if age_days < 0:
        raise StaleChampionFeedError(
            f"{feed_label} prediction_date="
            f"{pred_d} is AFTER run_date={run_date} ({-age_days} day(s) in "
            "the future) — refusing to trade on an inconsistent artifact."
        )


def apply_champion_selection(
    signals_raw: dict,
    predictions_by_ticker: dict,
    *,
    bucket: str,
    run_date: str,
    config: dict,
    sector_map: dict[str, str] | None,
    s3_client=None,
    pointer: dict | None = None,
) -> tuple[dict, dict]:
    """Apply the champion candidate-source switch to ``signals_raw``.

    No-op passthrough when the resolved champion is ``agentic`` (including
    the pre-bootstrap default) — returns ``(signals_raw, predictions_by_ticker)``
    unchanged, same objects, zero mutation.

    When ``scanner_predictor_direct``:
      * Loads the latest research-free predictor cohort (freshness-gated:
        ``run_date - prediction_date <= champion_freshness_max_days``, else
        raises — a stale champion feed must not trade silently).
      * Selects top-N by ``predicted_alpha`` where N = the current
        ``buy_candidates`` count (count-match preserves entry-budget parity
        with the measured counterfactual) when that count is > 0, else the
        ``champion_top_n_default`` config knob.
      * Replaces ``signals_raw["buy_candidates"]`` with synthesized ENTER
        entries. The ``universe`` list (held/EXIT/REDUCE population) is left
        untouched — exits stay managed for all holdings regardless of
        champion.
      * Injects research-free predictions into ``predictions_by_ticker`` for
        the selected tickers (deliberately ``prediction_confidence: 0.0`` —
        keeps the high-confidence-DOWN veto and the hold-book dispersion
        gate neutral on injected entries) so
        ``assert_predictions_cover_buy_candidates`` passes downstream.

    When ``thinktank_coverage`` (epic I2515 / config-I2518): same shape,
    entries synthesized from the Think Tank challenger-selection artifact
    instead of the research-free predictor cohort — see
    ``_apply_thinktank_coverage`` for the arm-specific validity gates
    (coverage-completeness + trading_day freshness) and rank→score mapping.

    ``pointer``: pass an already-resolved pointer dict (from a prior
    ``load_champion_pointer`` call) to avoid a second S3 round-trip on the
    same key — callers that need to branch on the champion arm BEFORE
    calling this function (e.g. to decide whether to pay for a sector-map
    load) already have the pointer in hand. When omitted, this function
    resolves it itself (single-read convenience for simpler callers/tests).

    Caller contract (config#2366 ordering constraint): this must run BEFORE
    ``filter_buy_candidates_to_universe`` / ``filter_buy_candidates_by_coverage``
    / ``assert_predictions_cover_buy_candidates`` so synthesized candidates
    flow through the same gates as agentic ones — wired inside
    ``executor.main._read_signals``.
    """
    if pointer is None:
        pointer = load_champion_pointer(bucket, s3_client=s3_client)
    champion = pointer["champion"]

    if champion == "agentic":
        return signals_raw, predictions_by_ticker

    if champion == "scanner_predictor_direct":
        return _apply_scanner_predictor_direct(
            signals_raw,
            predictions_by_ticker,
            bucket=bucket,
            run_date=run_date,
            config=config,
            sector_map=sector_map,
            s3_client=s3_client,
            pointer=pointer,
        )

    if champion == "scanner_top20_predictor":
        return _apply_scanner_top20_predictor(
            signals_raw,
            predictions_by_ticker,
            bucket=bucket,
            run_date=run_date,
            config=config,
            sector_map=sector_map,
            s3_client=s3_client,
            pointer=pointer,
        )

    if champion == "thinktank_coverage":
        return _apply_thinktank_coverage(
            signals_raw,
            predictions_by_ticker,
            bucket=bucket,
            run_date=run_date,
            config=config,
            sector_map=sector_map,
            s3_client=s3_client,
            pointer=pointer,
        )

    # Unreachable in practice — load_champion_pointer already validates
    # against VALID_CHAMPIONS — but fail loud rather than silently
    # falling through if a new champion value is ever added to the
    # pointer schema without a matching branch here.
    raise ChampionPointerError(f"apply_champion_selection has no handling for champion={champion!r}")


def _resolve_predictor_cut_pool(
    bucket: str, run_date: str, *, s3_client,
) -> tuple[set[str], str]:
    """``(tickers, cut_name)`` for the cut the PREDICTOR resolves from.

    Reads ``universe_membership`` — ``latest.json`` first, then dated keys
    walked back up to ``_MEMBERSHIP_MAX_AGE_DAYS`` — and returns the members of
    whichever cut that artifact's own ``predictor_universe_cut`` field names.

    The cut NAME is never hardcoded here (champion-challenger-policy.md §7.5).
    When ``crucible-research`` moves ``PREDICTOR_UNIVERSE_CUT``, this arm
    follows it with no edit, and the arm's picks stay honest about which pool
    they came from. A literal would go stale exactly the way
    ``watchlist_source: "scanner_candidate"`` did after the champion moved to an
    attractiveness cut.

    RAISES rather than degrading. This is the opposite posture from the
    producer-side union in ``crucible-backtester`` — there the same read only
    WIDENS a backfill work list, so failing soft costs coverage. Here it
    DEFINES which names may be entered, and an unresolvable pool would silently
    fall back to the champion's whole cohort: the arm would trade under its own
    name while being a duplicate of the arm it is supposed to be measured
    against. That is a vacuous comparison presented as a real one, and §4
    requires competing arms to actually differ.
    """
    from datetime import date as _date
    from datetime import timedelta as _td

    candidates: list[tuple[str, dict]] = []
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=_MEMBERSHIP_LATEST_KEY)
        candidates.append((_MEMBERSHIP_LATEST_KEY, json.loads(obj["Body"].read())))
    except ClientError as exc:
        logger.warning(
            "[champion] universe_membership/latest.json unreadable (%s) — "
            "falling back to the dated walk", exc,
        )

    try:
        start = _date.fromisoformat(run_date)
    except ValueError:
        start = None
    if start is not None:
        for days_back in range(_MEMBERSHIP_MAX_AGE_DAYS + 1):
            key = f"universe_membership/{start - _td(days=days_back)}/membership.json"
            try:
                obj = s3_client.get_object(Bucket=bucket, Key=key)
            except ClientError:
                continue
            candidates.append((key, json.loads(obj["Body"].read())))
            break

    # Freshness is judged on the artifact's OWN run_date, never on S3
    # LastModified — a re-upload of an old cycle must not read as fresh. Same
    # rule as crucible-predictor's _read_membership.
    fresh: tuple[str, dict] | None = None
    for key, doc in candidates:
        doc_date = doc.get("run_date")
        if not doc_date:
            continue
        try:
            age = (_date.fromisoformat(run_date) - _date.fromisoformat(doc_date)).days
        except ValueError:
            continue
        if age > _MEMBERSHIP_MAX_AGE_DAYS:
            logger.warning(
                "[champion] universe_membership %s is %dd old (limit %d) — rejecting",
                doc_date, age, _MEMBERSHIP_MAX_AGE_DAYS,
            )
            continue
        if fresh is None or doc_date > fresh[1].get("run_date", ""):
            fresh = (key, doc)

    if fresh is None:
        raise ChampionPointerError(
            "scanner_top20_predictor champion selected but no universe "
            f"membership artifact resolved within {_MEMBERSHIP_MAX_AGE_DAYS} "
            f"days of {run_date} (looked at {_MEMBERSHIP_LATEST_KEY} and dated "
            "keys). Refusing to select from an unknown pool — falling back to "
            "the full cohort would make this arm a silent duplicate of "
            "scanner_predictor_direct."
        )

    key, doc = fresh
    cut_name = doc.get("predictor_universe_cut")
    cut = (doc.get("cuts") or {}).get(cut_name) or {}
    tickers = {str(t).upper() for t in (cut.get("tickers") or []) if t}
    if not tickers:
        raise ChampionPointerError(
            f"universe membership {doc.get('run_date')} (s3://{bucket}/{key}) "
            f"names predictor_universe_cut={cut_name!r} but that cut is empty "
            "or absent — scanner_top20_predictor has no pool to select from."
        )
    logger.info(
        "[champion] scanner_top20_predictor pool: cut=%s %d name(s) from %s "
        "(membership run_date=%s)",
        cut_name, len(tickers), key, doc.get("run_date"),
    )
    return tickers, str(cut_name)


def _apply_scanner_predictor_direct(
    signals_raw: dict,
    predictions_by_ticker: dict,
    *,
    bucket: str,
    run_date: str,
    config: dict,
    sector_map: dict[str, str] | None,
    s3_client,
    pointer: dict,
) -> tuple[dict, dict]:
    """``scanner_predictor_direct`` arm — see ``apply_champion_selection``
    docstring for the full contract. Extracted verbatim (config-I2518) so
    ``apply_champion_selection`` can dispatch across multiple arms without a
    single function growing without bound.

    This arm's pool is the research-free parquet's whole scored cohort — the
    scanner-passing set, 60 names on 2026-08-21. It reads nothing the predictor
    produced. The arm that goes through the predictor is
    ``_apply_scanner_top20_predictor``; the two are separate functions on
    purpose, because they share no input.
    """
    max_days = int(config.get("champion_freshness_max_days", 8))
    cohort = _load_research_free_cohort(bucket, s3_client=s3_client)
    latest_date = cohort["prediction_date"].iloc[0]
    _check_freshness(latest_date, run_date, max_days)
    # alpha-engine-config-I7216: the hard bound above is deliberately loose
    # (8 calendar days) because halting a trading day is itself expensive —
    # sf-pipeline-policy.md §1.2 makes the cost of NOT trading a first-class
    # input. But 8 days spans a full trading week plus a weekend, so between
    # "yesterday's cohort" and "last Thursday's" the hard bound says nothing,
    # and the run completes as a clean success either way.
    #
    # Measured 2026-08-13: the cohort had been frozen at prediction_date
    # 2026-08-07 since its producer (the weekly pipeline's PredictorBacktest)
    # started failing. Six days stale, under the bound, so every trading day
    # drew its 10 entry candidates from the same frozen pool. Distinct names
    # newly entered fell from ~20/month to 3, and the only surface carrying
    # the cohort date at all was an INFO line in /var/log/executor.log on the
    # trading box. The operator noticed before any detector did.
    #
    # So: keep trading, and make the staleness LOUD. This is the honest-
    # degradation split (sf-pipeline-policy §2.3) — visibility to a human and
    # propagation to a machine are separate properties, and a stale entry feed
    # needs both.
    staleness = evaluate_cohort_staleness(latest_date, run_date, config)

    n_buy_candidates = len(signals_raw.get("buy_candidates") or [])
    n = n_buy_candidates if n_buy_candidates > 0 else int(config.get("champion_top_n_default", 10))

    cohort_sorted = cohort.sort_values("predicted_alpha", ascending=False).reset_index(drop=True)

    # ── One alpha scale per solve (alpha-engine-config-I7337, layer 3) ──────
    # The parquet carries the RAW MetaModel.predict_single output
    # (crucible-backtester/analysis/scanner_predictor_research_free_backfill.py
    # `alpha = float(mm.predict_single(feats))`), which is never
    # level-neutralized and so still carries the meta-L2's common-mode macro
    # level. Measured 2026-08-14 on the 2026-08-13 cohort: n=72, mean -0.2882,
    # range -0.3184..-0.2094, ZERO positive. The predictor's own
    # `predicted_alpha` IS level-neutralized (mean 4.3e-07, range
    # -0.0400..+0.0948), and `optimizer_shadow._build_alpha_hat` sums both into
    # one vector solved against a SPY=0.0 anchor — so every injected name sat
    # ~29 points of 21d log alpha below SPY as a pure artifact of anchoring and
    # could never win a solve. Center over the FULL cohort (the same
    # cross-section a producer-side transform would use) before injecting.
    #
    # Centering is a constant shift, so `cohort_sorted`'s ORDER and therefore
    # the selected candidate set are byte-identical. This is a units fix, not
    # an arm swap: `policy-champion-challenger` promotion/retirement is
    # untouched. Deliberately NOT rescaled to the predictor's dispersion —
    # this arm's spread is genuinely narrower (std 0.0177) because it zeroes
    # four research meta-features, and widening it would fabricate conviction.
    centered, xsec_mean_removed = center_to_market_relative(cohort_sorted["predicted_alpha"].tolist())
    cohort_sorted["predicted_alpha_market_relative"] = centered

    top_n = cohort_sorted.head(n)

    score_floor = float(config.get("champion_score_floor", 60))
    score_ceiling = float(config.get("champion_score_ceiling", 95))
    cohort_size = len(cohort_sorted)
    sector_map = sector_map or {}

    synthesized: list[dict] = []
    injected_predictions: dict[str, dict] = {}
    for rank, row in top_n.iterrows():
        ticker = row["ticker"]
        predicted_alpha_raw = float(row["predicted_alpha"])
        predicted_alpha = float(row["predicted_alpha_market_relative"])
        # rank_fraction: 0.0 for the best name (rank 0), approaching 1.0 for
        # the worst — computed against the FULL cohort size so the score
        # band reflects the name's standing in the whole scored universe,
        # not just within the top-N cut.
        rank_fraction = rank / max(cohort_size - 1, 1)
        score = _rank_to_score(rank_fraction, score_floor, score_ceiling)
        predicted_direction = "up" if predicted_alpha >= 0 else "down"

        entry = {
            "signal": "ENTER",
            "ticker": ticker,
            "date": run_date,
            "sector": sector_map.get(ticker, "Unknown"),
            "score": score,
            "conviction": "medium",
            "stance": None,
            "price_target_upside": None,
            "catalyst_date": None,
            "thesis_summary": "research-free predictor champion (config#2364)",
            "champion_arm": "scanner_predictor_direct",
        }
        synthesized.append(entry)

        injected_predictions[ticker] = {
            "predicted_alpha": predicted_alpha,
            # The optimizer's alpha input contract (I7337). Declared, not
            # inferred: this adapter KNOWS it just centered the batch.
            ANCHOR_FIELD: OPTIMIZER_ALPHA_ANCHOR,
            ANCHOR_SOURCE_FIELD: "champion_xsec_centered",
            # Forensics: what the parquet actually said, and by how much the
            # cohort's common mode moved it. Without these the corrected
            # value is unattributable to its source row.
            "predicted_alpha_raw": predicted_alpha_raw,
            "alpha_xsec_mean_removed": xsec_mean_removed,
            "predicted_direction": predicted_direction,
            # Deliberately neutral: the high-confidence-DOWN veto and the
            # hold-book alpha-dispersion gate must not fire off an
            # arbitrarily-assigned confidence for injected entries.
            "prediction_confidence": 0.0,
            "research_free": True,
        }

    logger.info(
        "[champion] scanner_predictor_direct selected %d/%d candidate(s) from "
        "cohort=%s age=%dd (n_buy_candidates=%d, cohort_size=%d)",
        len(synthesized), n, latest_date, staleness["age_days"],
        n_buy_candidates, cohort_size,
    )

    new_signals_raw = dict(signals_raw)
    new_signals_raw["buy_candidates"] = synthesized
    new_signals_raw["champion"] = "scanner_predictor_direct"
    new_signals_raw["promotion_source"] = pointer.get("promotion_source")
    # alpha-engine-config-I7216: carried on the artifact, not just logged, so a
    # consumer can render it and a stale feed is machine-visible. Emitted on
    # every run, healthy included — an absent field is unmeasured, not fine.
    new_signals_raw["champion_cohort"] = {
        **staleness,
        "cohort_size": cohort_size,
        "n_selected": len(synthesized),
        # alpha-engine-config-I8755: the pool this arm drew from, named on the
        # artifact rather than inferred from the arm's name — every arm in the
        # slot emits this block, so a reader compares like with like.
        "pool_source": "research_free_parquet",
        "pool_size": cohort_size,
        # alpha-engine-config-I7337: the common-mode level this adapter removed
        # to put its alphas on the optimizer's anchor. Carried on the artifact
        # and emitted on every run — a large or drifting value is the health
        # signal for the producer-side defect this compensates for, and an
        # ABSENT field would mean the correction never ran.
        "alpha_anchor": OPTIMIZER_ALPHA_ANCHOR,
        "alpha_xsec_mean_removed": xsec_mean_removed,
    }

    new_predictions_by_ticker = dict(predictions_by_ticker)
    new_predictions_by_ticker.update(injected_predictions)

    return new_signals_raw, new_predictions_by_ticker


def _apply_scanner_top20_predictor(
    signals_raw: dict,
    predictions_by_ticker: dict,
    *,
    bucket: str,
    run_date: str,
    config: dict,
    sector_map: dict[str, str] | None,
    s3_client,
    pointer: dict,
) -> tuple[dict, dict]:
    """``scanner_top20_predictor`` arm — the scanner's weekly cut, scored by the
    predictor (alpha-engine-config-I8755).

    Brian, 2026-08-27: *"The scanner should be providing the top 20 to
    predictor, weekly."* It does, and the predictor's daily output is that cut
    scored — so this arm's ranking is simply the predictor's own
    ``predicted_alpha`` over the members of ``predictor_universe_cut``.

    What makes this arm SHORTER than the other two, rather than a variation on
    them: its picks come from ``predictions_by_ticker``, which already exists,
    is already on the declared ``market_relative_21d_log`` anchor, and is
    already the vector ``optimizer_shadow._build_alpha_hat`` will solve over.
    So there is nothing to synthesize, nothing to centre, and no anchor to
    declare on this arm's behalf. ``scanner_predictor_direct`` and
    ``thinktank_coverage`` must inject prediction records because their picks
    come from outside the predictor's output; this arm's picks ARE that output.
    ``assert_predictions_cover_buy_candidates`` therefore passes by
    construction rather than by fabrication.

    Two refusals, both deliberate:

    * an unresolvable cut RAISES (in ``_resolve_predictor_cut_pool``) rather
      than falling back to a wider set — a silent widening would make this arm
      a duplicate of one it is measured against, which is a vacuous comparison
      presented as a real one (champion-challenger-policy.md §4);
    * a cut whose members carry no usable prediction RAISES rather than
      synthesizing zero candidates under a healthy-looking arm.

    A PARTIAL cut is reported, not refused: the arm still selects from what it
    has, and ``champion_cohort`` carries how short the pool was, because an arm
    scored on a short pool must not read as an arm that chose badly.
    """
    pool, cut_name = _resolve_predictor_cut_pool(
        bucket, run_date, s3_client=s3_client or boto3.client("s3"),
    )

    # The predictor's cut ∩ the names it actually scored today. A held name
    # outside the cut is deliberately NOT a candidate here: the predictor
    # unions holdings into its scoring universe so exits can be decided, and
    # this arm proposes ENTRIES.
    scored: list[tuple[str, float, dict]] = []
    for ticker in sorted(pool):
        pred = predictions_by_ticker.get(ticker)
        if not pred:
            continue
        alpha = _numeric_alpha(pred)
        if alpha is None:
            continue
        declared = pred.get(ANCHOR_FIELD)
        if declared is not None and declared != OPTIMIZER_ALPHA_ANCHOR:
            # Ranking one anchor against another orders names by where their
            # level was measured from, not by their alpha. Fatal for the same
            # reason `assert_optimizer_anchor` is fatal downstream.
            raise AlphaAnchorError(
                f"scanner_top20_predictor: {ticker} declares "
                f"{ANCHOR_FIELD}={declared!r}, not {OPTIMIZER_ALPHA_ANCHOR!r}. "
                "Refusing to rank a mixed-anchor batch."
            )
        scored.append((ticker, alpha, pred))

    n_pool = len(pool)
    n_unscored = n_pool - len(scored)
    if not scored:
        raise ChampionPointerError(
            f"scanner_top20_predictor: none of the {n_pool} name(s) in cut "
            f"{cut_name!r} carries a usable predicted_alpha in this run's "
            f"predictions ({len(predictions_by_ticker)} record(s)) — refusing "
            "to synthesize zero candidates while reporting a healthy arm."
        )
    if n_unscored:
        logger.warning(
            "[champion] scanner_top20_predictor: %d of %d cut member(s) carry "
            "no usable predicted_alpha — selecting from a SHORT pool, and the "
            "arm's score reflects that rather than its rule",
            n_unscored, n_pool,
        )

    scored.sort(key=lambda row: row[1], reverse=True)

    n_buy_candidates = len(signals_raw.get("buy_candidates") or [])
    n = n_buy_candidates if n_buy_candidates > 0 else int(
        config.get("champion_top_n_default", 10)
    )
    top_n = scored[:n]

    score_floor = float(config.get("champion_score_floor", 60))
    score_ceiling = float(config.get("champion_score_ceiling", 95))
    # Rank fraction is over the arm's OWN scored pool. The band feeds
    # `min_score`, so ranking a 20-name arm against a wider cross-section would
    # push most of its picks under the gate — the arm would look like it
    # selected nothing rather than like it selected differently.
    pool_size = len(scored)
    sector_map = sector_map or {}

    synthesized: list[dict] = []
    for rank, (ticker, alpha, pred) in enumerate(top_n):
        rank_fraction = rank / max(pool_size - 1, 1)
        synthesized.append({
            "signal": "ENTER",
            "ticker": ticker,
            "date": run_date,
            "sector": sector_map.get(ticker, "Unknown"),
            "score": _rank_to_score(rank_fraction, score_floor, score_ceiling),
            "conviction": "medium",
            # The predictor emits a stance per name; carry it rather than
            # dropping it to None. It is what sizes the position downstream
            # (max_position_pct x stance_multiplier), so discarding it would
            # silently move every pick to the default cap.
            "stance": pred.get("stance"),
            "price_target_upside": None,
            "catalyst_date": pred.get("catalyst_date"),
            "thesis_summary": (
                "scanner attractiveness top-20 (weekly), ranked by the "
                "predictor's own alpha (alpha-engine-config-I8755)"
            ),
            "champion_arm": "scanner_top20_predictor",
            "predicted_alpha": alpha,
        })

    logger.info(
        "[champion] scanner_top20_predictor selected %d/%d from cut=%s "
        "(pool=%d scored, %d unscored, n_buy_candidates=%d)",
        len(synthesized), n, cut_name, pool_size, n_unscored, n_buy_candidates,
    )

    new_signals_raw = dict(signals_raw)
    new_signals_raw["buy_candidates"] = synthesized
    new_signals_raw["champion"] = "scanner_top20_predictor"
    new_signals_raw["promotion_source"] = pointer.get("promotion_source")
    new_signals_raw["champion_cohort"] = {
        # No staleness block: this arm reads no separate feed. Its freshness IS
        # the predictions file the rest of the run is already built on, and the
        # cut's own age is gated in `_resolve_predictor_cut_pool`.
        "pool_source": "predictor_predictions",
        "pool_cut": cut_name,
        "pool_declared_size": n_pool,
        "pool_size": pool_size,
        "n_cut_members_unscored": n_unscored,
        "n_selected": len(synthesized),
        "alpha_anchor": OPTIMIZER_ALPHA_ANCHOR,
    }

    # predictions_by_ticker is returned UNCHANGED and that is the point — see
    # the docstring. Nothing is injected because nothing needs to be.
    return new_signals_raw, predictions_by_ticker


def _apply_thinktank_coverage(
    signals_raw: dict,
    predictions_by_ticker: dict,
    *,
    bucket: str,
    run_date: str,
    config: dict,
    sector_map: dict[str, str] | None,
    s3_client,
    pointer: dict,
) -> tuple[dict, dict]:
    """``thinktank_coverage`` arm (epic I2515 / config-I2518) — entries
    synthesized from the Think Tank challenger-selection artifact
    (``thinktank/challenger_selection/latest.json``, crucible-research
    PR#427) instead of the research-free predictor cohort.

    HARD VALIDITY GATES (config#1580 — a champion feed must never trade
    silently on stale/invalid data; same fail-loud posture and same
    missing-artifact degrade path as ``_apply_scanner_predictor_direct``'s
    missing-parquet case — no fallback to raw signals.json candidates):

      * ``coverage_complete`` must be True. Brian's ruling (config#1580):
        the selection only counts as valid champion-arm evidence once the
        ENTIRE current-scan top-N coverage window is covered — an
        incomplete-coverage selection raises ``ChampionPointerError``
        rather than trading a partial/unrepresentative pool.
      * ``trading_day`` must be within ``THINKTANK_COVERAGE_FRESHNESS_MAX_DAYS``
        (3 calendar days) of ``run_date`` — the same calendar-day-diff
        technique as the scanner arm's ``_check_freshness``, but its OWN
        bound (alpha-engine-config-I7232, Brian ruling 2026-08-14), not the
        shared ``champion_freshness_max_days`` (default 8) used by the
        ``scanner_predictor_direct`` arm below. The Think Tank is a daily
        producer; the scanner's cohort is a weekly one — sharing one
        fleet-wide constant between a daily and a weekly producer is exactly
        what let a 3-day-frozen challenger-selection pointer pass silently
        into a live champion feed. 3 matches ``POINTER_LAG_ERROR_DAYS`` in
        crucible-research (crucible-research-PR630), so the producer-side
        ERROR and this consumer-side refusal agree on what "stale" means.
        Note this is a HARD gate, distinct from the artifact's own
        ``board_date`` (which the producer deliberately never hard-fails on
        — the daily Think Tank cadence legitimately reads a stale universe
        board all week, config#1580) — ``trading_day`` is the run identity
        of the challenger-selection artifact itself, the analogue of the
        scanner arm's ``prediction_date``.

    Rank → score: ``selections`` arrives PRE-SORTED best-rating-first from
    the producer (``thinktank.challenger_selection.write_challenger_selection``
    sorts by rating descending before truncating to its own top-N), but this
    is defensively re-sorted here rather than trusted, mirroring the scanner
    arm's own defensive ``sort_values`` on its cohort. rank_fraction is
    computed WITHIN the selection itself (denominator = the number of names
    Think Tank actually submitted, up to its own ``CHALLENGER_TOP_N``) —
    unlike the scanner arm, there is no larger scored population to rank
    against; the challenger-selection artifact only ever contains its own
    top-N, so "within the selection" is the correct (and only available)
    cohort for the rank-fraction denominator.

    Deliberately-neutral injected prediction fields — same intent as the
    scanner arm (keep the high-confidence-DOWN veto and hold-book dispersion
    gate authoritative, not skewed by champion-injected values), same
    ``prediction_confidence: 0.0``. ``predicted_alpha``/``predicted_direction``
    are explicitly ``None`` rather than a fabricated numeric value: Think
    Tank's rating is a subjective 0-100 score, not a log-alpha estimate, and
    inventing a fake numeric alpha would misrepresent the hold-book
    dispersion calc (``main._should_hold_book``) rather than keep it neutral
    — ``None`` is excluded from that calc's cross-sectional stdev entirely
    (its `isinstance(a, (int, float))` guard), the honest way to contribute
    zero signal-magnitude opinion.
    """
    selection = _load_challenger_selection(bucket, s3_client=s3_client)

    if not selection.get("coverage_complete"):
        raise ChampionPointerError(
            f"thinktank_coverage champion selected but the challenger-"
            f"selection artifact's coverage_complete=False "
            f"(uncovered_count={selection.get('uncovered_count')!r}, "
            f"trading_day={selection.get('trading_day')!r}) — refusing to "
            "trade on an incomplete-coverage selection (config#1580)."
        )

    # alpha-engine-config-I7232 (Brian ruling 2026-08-14): this arm's own
    # freshness bound, NOT the shared `champion_freshness_max_days` used by
    # the scanner arm below — see THINKTANK_COVERAGE_FRESHNESS_MAX_DAYS and
    # the docstring above.
    trading_day = selection["trading_day"]
    _check_freshness(
        trading_day,
        run_date,
        THINKTANK_COVERAGE_FRESHNESS_MAX_DAYS,
        feed_label="thinktank_coverage challenger-selection artifact",
    )

    rows = list(selection.get("selections") or [])
    if not rows:
        raise ChampionPointerError(
            "thinktank_coverage challenger-selection artifact has no "
            "selections — champion has no candidates to select from."
        )

    rows_sorted = sorted(rows, key=lambda r: r["rating"], reverse=True)
    selection_size = len(rows_sorted)

    n_buy_candidates = len(signals_raw.get("buy_candidates") or [])
    n = n_buy_candidates if n_buy_candidates > 0 else int(config.get("champion_top_n_default", 10))
    top_n = rows_sorted[:n]

    score_floor = float(config.get("champion_score_floor", 60))
    score_ceiling = float(config.get("champion_score_ceiling", 95))
    sector_map = sector_map or {}

    synthesized: list[dict] = []
    injected_predictions: dict[str, dict] = {}
    for rank, row in enumerate(top_n):
        ticker = row["ticker"]
        # rank_fraction: 0.0 for the best-rated name, approaching 1.0 for
        # the worst — computed WITHIN the selection (see docstring: there is
        # no larger scored population to rank against for this arm).
        rank_fraction = rank / max(selection_size - 1, 1)
        score = _rank_to_score(rank_fraction, score_floor, score_ceiling)

        entry = {
            "signal": "ENTER",
            "ticker": ticker,
            "date": run_date,
            "sector": sector_map.get(ticker, "Unknown"),
            "score": score,
            "conviction": "medium",
            "stance": None,
            "price_target_upside": None,
            "catalyst_date": None,
            "thesis_summary": "thinktank_coverage challenger champion (config-I2518 / I2515)",
            "champion_arm": "thinktank_coverage",
        }
        synthesized.append(entry)

        injected_predictions[ticker] = {
            # Deliberately None, not a fabricated numeric alpha — see
            # docstring. Keeps this entry OUT of main._should_hold_book's
            # cross-sectional dispersion calc entirely.
            #
            # No `alpha_anchor` is stamped, and that is correct rather than an
            # omission (alpha-engine-config-I7337): an anchor declares where a
            # LEVEL was measured from, and this record asserts no level at all.
            # `alpha_contract.assert_optimizer_anchor` skips records with no
            # numeric alpha for exactly this reason. Stamping one would be a
            # claim about a number that does not exist.
            #
            # Consequence worth naming: such a name lands at alpha_hat 0.0,
            # identical to SPY, so it can tie but never beat the benchmark on
            # the alpha term. That is an honest representation of a subjective
            # 0-100 rating, not a defect of this contract — but it does mean
            # the thinktank arm cannot win a solve on alpha alone. Tracked
            # separately from this fix.
            "predicted_alpha": None,
            "predicted_direction": None,
            # Same neutral value as the scanner arm — keeps the
            # high-confidence-DOWN veto and hold-book dispersion gate
            # authoritative, not skewed by champion-injected entries.
            "prediction_confidence": 0.0,
            "thinktank_coverage": True,
        }

    logger.info(
        "[champion] thinktank_coverage selected %d/%d candidate(s) from "
        "challenger-selection trading_day=%s (n_buy_candidates=%d, "
        "selection_size=%d, uncovered_count=%d)",
        len(synthesized),
        n,
        trading_day,
        n_buy_candidates,
        selection_size,
        selection.get("uncovered_count", 0),
    )

    new_signals_raw = dict(signals_raw)
    new_signals_raw["buy_candidates"] = synthesized
    new_signals_raw["champion"] = "thinktank_coverage"
    new_signals_raw["promotion_source"] = pointer.get("promotion_source")

    new_predictions_by_ticker = dict(predictions_by_ticker)
    new_predictions_by_ticker.update(injected_predictions)

    return new_signals_raw, new_predictions_by_ticker
