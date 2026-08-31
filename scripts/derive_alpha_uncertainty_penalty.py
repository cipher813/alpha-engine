#!/usr/bin/env python
"""Derive ``alpha_uncertainty_penalty`` (gamma) by replaying stored solves.

alpha-engine-config-I9452. The Garlappi-Uppal-Wang term's Omega is built from
``predicted_alpha_std_epistemic``; gamma must therefore be derived against THAT
scale, which is one to two orders of magnitude below the total predictive std
the term was originally sized against. A value picked to "look right" for the
new magnitude is not a derivation — this script is the derivation.

What it does, end to end:

1. Reads every ``predictor/optimizer_shadow/{date}.json`` in the bucket. Each
   carries the complete input set of one live morning solve: universe, alpha
   vector, prior weights, caps, eligibility, the daily covariance matrix, ADV
   and the exact optimizer config that ran.
2. Recovers the per-session epistemic vector. Artifacts written before
   crucible-predictor PR596 deployed (2026-08-31) carry only the TOTAL, so the
   split is reconstructed exactly the way the producer computes it:

       sigma_eps^2 = sigma_total^2 - sigma_n^2,   sigma_n^2 = 1/alpha_hat

   with ``alpha_`` read off the BayesianRidge inside the champion pickle that
   served that session (``predictor/registry/{version_id}/meta_model.pkl``).
   The champion timeline comes from each registry manifest's
   ``served_version``/``served_date`` pair plus ``champion_version_id`` on the
   predictions artifact, and every assignment is FALSIFIED against the identity
   sigma_n < min_i(sigma_total_i), which holds only for the true champion.
   ``--epistemic-from-artifact`` skips all of this once enough artifacts carry
   the emitted field.
3. Re-solves each session across a gamma grid, with the conviction budget gate
   both enabled and disabled, so the two signal-quality knobs are measured
   jointly rather than each in isolation.
4. Grades each grid point on the realised market-relative return of the solved
   book over the canonical 21-day horizon (and 5-day as a diagnostic), net of a
   configurable per-side cost, using closes from the ArcticDB universe library.

It writes a JSON result and prints the table. It changes nothing: the derived
value is applied by hand, in ``config/risk.yaml.example`` and the deployed
``risk.yaml``, with the sweep recorded on the tracking issue.

Read-only; needs AWS credentials with the research bucket and s3:ListBucket
(``AWS_PROFILE=ne-admin`` on the laptop). Run from the repo root:

    python scripts/derive_alpha_uncertainty_penalty.py --bucket alpha-engine-research
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import re
import statistics as st
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("derive_gamma")

DEFAULT_GAMMAS = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 3000.0, 10000.0)


def _load_optimizer():
    """Import portfolio_optimizer BY FILE SPEC.

    ``executor/__init__.py`` imports arcticdb for its macOS symbol-priming side
    effect; going through the package would make this script depend on it at
    import time for no reason.
    """
    spec = importlib.util.spec_from_file_location(
        "_po_for_sweep", str(REPO_ROOT / "executor" / "portfolio_optimizer.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_po_for_sweep"] = mod
    spec.loader.exec_module(mod)
    return mod


def _list_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            return keys
        token = resp["NextContinuationToken"]


def _get_json(s3, bucket: str, key: str):
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def _champion_noise_table(s3, bucket: str, cache: Path) -> tuple[list[tuple[str, str]], dict]:
    """(champion timeline, {version_id: sigma_n}) from the model registry.

    ``sigma_n = sqrt(1/alpha_)`` is read from the estimator bytes, which is
    where it travels — the ``.pkl.meta.json`` sidecar the loader used to read it
    from does not exist under the registry prefix.
    """
    import joblib

    cache.mkdir(parents=True, exist_ok=True)
    versions = sorted({
        k.split("/")[2] for k in _list_keys(s3, bucket, "predictor/registry/")
        if k.count("/") >= 3
    })
    sigma_n, observations = {}, []
    for v in versions:
        pkl = cache / f"{v}.pkl"
        if not pkl.exists():
            try:
                s3.download_file(bucket, f"predictor/registry/{v}/meta_model.pkl", str(pkl))
            except Exception as exc:
                logger.warning("%s: no meta_model.pkl in the registry (%s)", v, exc)
                continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                alpha_ = _find_alpha(joblib.load(pkl))
        except Exception:
            alpha_ = None
        if alpha_:
            sigma_n[v] = math.sqrt(1.0 / alpha_)
        try:
            man = _get_json(s3, bucket, f"predictor/registry/{v}/manifest.json")
        except Exception as exc:
            logger.warning("%s: no registry manifest (%s) — no timeline entry", v, exc)
            continue
        if man.get("served_version") and man.get("date"):
            # "at the time THIS candidate was trained, X was serving" — the only
            # durable record of the champion timeline the registry keeps.
            observations.append((man["date"], man["served_version"]))
    timeline = sorted(set(observations))
    return timeline, sigma_n


def _find_alpha(obj, depth: int = 0):
    if depth > 6:
        return None
    a = getattr(obj, "alpha_", None)
    if a is not None:
        return float(a)
    if isinstance(obj, dict):
        for v in obj.values():
            r = _find_alpha(v, depth + 1)
            if r:
                return r
    for attr in ("_model", "model", "estimator"):
        if hasattr(obj, attr):
            r = _find_alpha(getattr(obj, attr), depth + 1)
            if r:
                return r
    return None


def _resolve_sigma_n(date: str, totals: list[float], timeline, sigma_n) -> tuple[str | None, float | None, int]:
    """Champion sigma_n for one session, falsified against sigma_n < min(total).

    The timeline is keyed on TRAINING date; a promotion takes effect one or more
    sessions later. sigma_total^2 = sigma_n^2 + sigma_eps^2 with sigma_eps >= 0
    makes ``sigma_n < min_i(sigma_total_i)`` a hard identity, so when the
    nominal entry violates it we step back one promotion until it holds. An
    assignment that never satisfies it is REPORTED, never guessed at.
    """
    cands = [i for i, (d, _) in enumerate(timeline) if d <= date]
    if not cands:
        return (None, None, 0)
    mn = min(totals)
    for lag, i in enumerate(range(max(cands), -1, -1)):
        v = timeline[i][1]
        s = sigma_n.get(v)
        if s is not None and s < mn:
            return (v, s, lag)
    return (None, None, 0)


def load_sessions(s3, bucket: str, cache: Path, from_artifact: bool) -> list[dict]:
    keys = [
        k for k in _list_keys(s3, bucket, "predictor/optimizer_shadow/")
        if re.fullmatch(r"predictor/optimizer_shadow/\d{4}-\d{2}-\d{2}\.json", k)
    ]
    timeline, sigma_n = ((), {}) if from_artifact else _champion_noise_table(s3, bucket, cache)
    if not from_artifact:
        logger.info("champion timeline: %d observations, %d models with alpha_",
                    len(timeline), len(sigma_n))
    out = []
    for k in sorted(keys):
        date = k.split("/")[-1][:-5]
        d = _get_json(s3, bucket, k)
        if d.get("shadow_status") != "ok":
            continue
        au = d.get("alpha_uncertainty")
        if not au:
            continue
        totals = [x for x in au if x is not None and x > 0]
        if len(totals) < 3:
            continue
        tot = np.array([np.nan if x is None else float(x) for x in au], dtype=float)
        if from_artifact:
            epi_raw = d.get("alpha_uncertainty_epistemic")
            if not epi_raw:
                continue
            epi = np.array([np.nan if x is None else float(x) for x in epi_raw], dtype=float)
            champ, lag = "artifact", 0
        else:
            champ, s, lag = _resolve_sigma_n(date, totals, timeline, sigma_n)
            if s is None:
                logger.warning("%s: no champion satisfies sigma_n < min(total) — SKIPPED", date)
                continue
            epi = np.where(
                np.isfinite(tot) & (tot > 0),
                np.sqrt(np.maximum(tot ** 2 - s ** 2, 0.0)),
                np.where(tot == 0.0, 0.0, np.nan),
            )
        d.update({"_date": date, "_total": tot, "_epistemic": epi,
                  "_champion": champ, "_lag": lag})
        out.append(d)
    return out


def load_closes(bucket: str, tickers: set[str], start: str):
    import pandas as pd
    from nousergon_lib.arcticdb import open_universe_lib

    lib = open_universe_lib(bucket)
    cols = {}
    for t in sorted(tickers | {"SPY"}):
        if t == "CASH":
            continue
        try:
            cols[t] = lib.read(t, columns=["Close"]).data["Close"]
        except Exception:
            logger.warning("no ArcticDB series for %s — its forward return is NaN", t)
    return pd.DataFrame(cols).loc[start:]


def forward_alpha(px, date: str, tickers: list[str], horizon: int):
    """Market-relative forward log return per name, or None when censored."""
    import pandas as pd

    idx = px.index
    d = pd.Timestamp(date)
    pos = idx.searchsorted(d)
    if pos >= len(idx) or idx[pos] != d or pos + horizon >= len(idx):
        return None
    p0, p1 = px.iloc[pos], px.iloc[pos + horizon]
    spy = np.log(p1["SPY"] / p0["SPY"])
    vals = []
    for t in tickers:
        if t == "CASH":
            vals.append(-spy)
        elif t in px.columns and np.isfinite(p0.get(t, np.nan)) and np.isfinite(p1.get(t, np.nan)):
            vals.append(np.log(p1[t] / p0[t]) - spy)
        else:
            vals.append(np.nan)
    return np.array(vals)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default="alpha-engine-research")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(DEFAULT_GAMMAS))
    ap.add_argument("--cost-bps-per-side", type=float, default=10.0)
    ap.add_argument("--horizons", type=int, nargs="+", default=[21, 5])
    ap.add_argument("--epistemic-from-artifact", action="store_true",
                    help="use the emitted predicted_alpha_std_epistemic instead of "
                         "reconstructing it from the champion pickle (use once enough "
                         "artifacts postdate crucible-predictor PR596)")
    ap.add_argument("--cache", default=".gamma_sweep_cache")
    ap.add_argument("--out", default="gamma_sweep.json")
    args = ap.parse_args()

    import boto3
    s3 = boto3.client("s3")
    po = _load_optimizer()
    warnings.filterwarnings("ignore")

    sessions = load_sessions(s3, args.bucket, Path(args.cache), args.epistemic_from_artifact)
    if not sessions:
        logger.error("no usable stored solves — nothing to derive from")
        return 1
    logger.info("%d sessions %s .. %s", len(sessions), sessions[0]["_date"], sessions[-1]["_date"])

    universe: set[str] = set()
    for d in sessions:
        universe.update(d["tickers"])
    px = load_closes(args.bucket, universe, min(d["_date"] for d in sessions))

    results = []
    for gamma in args.gammas:
        for gate in (True, False):
            per_session = []
            for d in sessions:
                tickers = d["tickers"]
                cfg = {**d["optimizer_cfg"],
                       "alpha_uncertainty_penalty": float(gamma),
                       "conviction_budget_gate_enabled": gate}
                n = len(tickers)
                adv = d.get("adv_usd") or [None] * n
                try:
                    r = po.solve_target_weights(
                        tickers=tickers,
                        alpha_hat=np.array(d["alpha_hat"], dtype=float),
                        returns_panel=None,
                        w_prev=np.array(d["current_weights"], dtype=float),
                        sectors=d["sectors"],
                        stance_caps=np.array(d["stance_caps"], dtype=float),
                        eligibility=np.array(d["eligibility"], dtype=bool),
                        spy_idx=tickers.index("SPY"),
                        cash_idx=tickers.index("CASH"),
                        cfg=cfg,
                        alpha_uncertainty=d["_total"],
                        alpha_uncertainty_epistemic=d["_epistemic"],
                        covariance=np.array(d["covariance_daily"], dtype=float),
                        adv_usd=np.array([np.nan if x is None else float(x) for x in adv], dtype=float),
                        portfolio_notional=d.get("portfolio_nav"),
                    )
                except Exception as exc:  # a failed solve is DATA, not a crash
                    logger.warning("%s gamma=%g gate=%s: solve failed: %s",
                                   d["_date"], gamma, gate, exc)
                    continue
                w = np.asarray(r.weights, dtype=float)
                row = {"date": d["_date"], "turnover": r.diagnostics.get("turnover_one_way"),
                       "operative": bool(r.diagnostics.get("alpha_uncertainty_penalty_used")),
                       "reason": r.diagnostics.get("alpha_uncertainty_inoperative_reason"),
                       "epistemic_cv": r.diagnostics.get("alpha_uncertainty_epistemic_cv"),
                       "conviction_q": r.diagnostics.get("conviction_budget_multiplier")}
                for h in args.horizons:
                    fa = forward_alpha(px, d["_date"], tickers, h)
                    if fa is None:
                        row[f"alpha_{h}d"] = None
                        continue
                    m = np.isfinite(fa)
                    gross = float(np.sum(w[m] * fa[m]))
                    cost = 2.0 * float(row["turnover"] or 0.0) * args.cost_bps_per_side / 1e4
                    row[f"alpha_{h}d"] = gross - cost
                per_session.append(row)
            results.append({"gamma": gamma, "conviction_gate": gate, "sessions": per_session})
            logger.info("swept gamma=%g gate=%s", gamma, gate)

    Path(args.out).write_text(json.dumps(
        {"bucket": args.bucket, "cost_bps_per_side": args.cost_bps_per_side,
         "n_sessions": len(sessions), "results": results}, indent=1))

    hdr = f"{'gamma':>8}{'gate':>6}{'oper':>5}{'medTO':>8}"
    for h in args.horizons:
        hdr += f"{f'a{h}d_bps':>10}{f'n{h}':>5}"
    print(hdr)
    for blk in results:
        rows = blk["sessions"]
        line = (f"{blk['gamma']:8.0f}{str(blk['conviction_gate']):>6}"
                f"{sum(r['operative'] for r in rows):5d}"
                f"{st.median([r['turnover'] for r in rows]):8.4f}")
        for h in args.horizons:
            v = [r[f"alpha_{h}d"] for r in rows if r.get(f"alpha_{h}d") is not None]
            line += f"{(st.mean(v) * 1e4 if v else float('nan')):10.1f}{len(v):5d}"
        print(line)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
