"""Regime-stratified conformal calibration + drift-recalibration FAR (revision, GPT-P4).

Two questions a reviewer will ask after seeing the per-regime FAR violation (0.075 /
0.005 / 0.054 at alpha = 0.05 under one pooled threshold):

  A) STRATIFIED: does calibrating a separate split-conformal threshold PER REGIME
     restore per-regime FAR to the nominal level? (The manuscript recommends
     regime-stratified calibration in deployment; this experiment demonstrates it.)

  B) RECALIBRATION: after a distribution shift, how bad is FAR with the stale
     pre-drift threshold, and does drift-triggered recalibration on fresh post-drift
     normal data restore FAR to nominal? (The manuscript ships a drift detector whose
     purpose is to trigger exactly this; this experiment quantifies before vs after.)

Setup mirrors conformal_conditional.py: normal data as a mixture of 3 operating modes
(affine-shifted synthetic streams); IsolationForest detector; split-conformal thresholds.

Output (analysis/results/): conformal_stratified_recalib.csv
    part A columns: part='stratified', alpha, rep, mode, scheme in {pooled, stratified},
                    n_test, n_flagged, far
    part B columns: part='recalib', alpha, rep, phase in {pre-drift, post-drift-stale,
                    post-drift-recalibrated}, n_test, n_flagged, far
plus printed pooled summaries with Wilson 95% CIs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ai_cta.anomaly_detector import IsolationForestDetector
from ai_cta.data import generate_synthetic_stream
from ai_cta.risk_model import ConformalThresholdCalibrator

RES = Path(__file__).resolve().parent / "results"
OUT = RES / "conformal_stratified_recalib.csv"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mode_stream(n_samples, seed, scale, offset):
    s = generate_synthetic_stream(n_samples=n_samples, random_state=seed)
    cols = [c for c in s.columns if c != "timestamp"]
    s = s[cols] * scale + offset
    s.columns = [f"f{i}" for i in range(s.shape[1])]
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-reps", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--alphas", type=str, default="0.05,0.10")
    ap.add_argument("--modes", type=int, default=3)
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    RES.mkdir(parents=True, exist_ok=True)
    mode_params = [(1.0 + 0.15 * m, 0.4 * m) for m in range(args.modes)]

    rows = []
    poolA = {}  # (alpha, scheme, mode) -> [flagged, n]
    poolB = {}  # (alpha, phase) -> [flagged, n]

    for rep in range(args.n_reps):
        # ---------- Part A: pooled vs stratified thresholds ----------
        train = pd.concat([mode_stream(args.n_samples // 2, 100 + rep * 10 + m, sc, off)
                           for m, (sc, off) in enumerate(mode_params)], ignore_index=True)
        det = IsolationForestDetector(window_size=64, stride=16, use_spectral=False,
                                      random_state=rep)
        det.fit(train)

        cal_scores = {m: det.decision_function(
                          mode_stream(args.n_samples // 2, 500 + rep * 10 + m, sc, off))
                      for m, (sc, off) in enumerate(mode_params)}
        test_scores = {m: det.decision_function(
                           mode_stream(args.n_samples, 900 + rep * 10 + m, sc, off))
                       for m, (sc, off) in enumerate(mode_params)}
        pooled_cal = np.concatenate(list(cal_scores.values()))

        for a in alphas:
            thr_pooled = ConformalThresholdCalibrator(alpha=a).calibrate(pooled_cal).threshold_
            for m in range(args.modes):
                thr_strat = ConformalThresholdCalibrator(alpha=a).calibrate(
                    cal_scores[m]).threshold_
                for scheme, thr in (("pooled", thr_pooled), ("stratified", thr_strat)):
                    n = len(test_scores[m])
                    k = int((test_scores[m] > thr).sum())
                    rows.append({"part": "stratified", "alpha": a, "rep": rep, "mode": m,
                                 "scheme": scheme, "phase": "", "n_test": n,
                                 "n_flagged": k, "far": round(k / n, 5)})
                    key = (a, scheme, m)
                    poolA.setdefault(key, [0, 0])
                    poolA[key][0] += k
                    poolA[key][1] += n

        # ---------- Part B: FAR before/after drift-triggered recalibration ----------
        # Pre-drift world: mode-0 params; post-drift world: +2-sigma-style mean shift.
        sc0, off0 = mode_params[0]
        drift_off = off0 + 2.0
        det_b = IsolationForestDetector(window_size=64, stride=16, use_spectral=False,
                                        random_state=1000 + rep)
        det_b.fit(mode_stream(args.n_samples, 2000 + rep, sc0, off0))
        cal_pre = det_b.decision_function(
            mode_stream(args.n_samples // 2, 2500 + rep, sc0, off0))
        test_pre = det_b.decision_function(
            mode_stream(args.n_samples, 3000 + rep, sc0, off0))
        # Post-drift NORMAL data (the new normal after the shift).
        cal_post = det_b.decision_function(
            mode_stream(args.n_samples // 2, 3500 + rep, sc0, drift_off))
        test_post = det_b.decision_function(
            mode_stream(args.n_samples, 4000 + rep, sc0, drift_off))

        for a in alphas:
            thr_pre = ConformalThresholdCalibrator(alpha=a).calibrate(cal_pre).threshold_
            thr_re = ConformalThresholdCalibrator(alpha=a).calibrate(cal_post).threshold_
            for phase, scores, thr in (("pre-drift", test_pre, thr_pre),
                                       ("post-drift-stale", test_post, thr_pre),
                                       ("post-drift-recalibrated", test_post, thr_re)):
                n = len(scores)
                k = int((scores > thr).sum())
                rows.append({"part": "recalib", "alpha": a, "rep": rep, "mode": -1,
                             "scheme": "", "phase": phase, "n_test": n,
                             "n_flagged": k, "far": round(k / n, 5)})
                key = (a, phase)
                poolB.setdefault(key, [0, 0])
                poolB[key][0] += k
                poolB[key][1] += n

        if (rep + 1) % 10 == 0:
            print(f"  ...{rep + 1}/{args.n_reps}", flush=True)

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Part A: per-mode FAR, pooled vs regime-stratified thresholds ===")
    print(f"{'alpha':>6} {'scheme':>11} {'mode':>5} {'FAR':>9} {'Wilson 95% CI':>22}")
    for a in alphas:
        for scheme in ("pooled", "stratified"):
            fars = []
            for m in range(args.modes):
                k, n = poolA[(a, scheme, m)]
                lo, hi = wilson(k, n)
                fars.append(k / n)
                print(f"{a:>6.2f} {scheme:>11} {m:>5} {k/n:>9.4f}  [{lo:.4f}, {hi:.4f}]")
            print(f"       {scheme:>11} spread across modes: "
                  f"{max(fars) - min(fars):.4f} (nominal {a})")

    print("\n=== Part B: FAR before/after drift-triggered recalibration ===")
    print(f"{'alpha':>6} {'phase':>26} {'FAR':>9} {'Wilson 95% CI':>22}")
    for a in alphas:
        for phase in ("pre-drift", "post-drift-stale", "post-drift-recalibrated"):
            k, n = poolB[(a, phase)]
            lo, hi = wilson(k, n)
            print(f"{a:>6.2f} {phase:>26} {k/n:>9.4f}  [{lo:.4f}, {hi:.4f}]")

    print(f"\nWrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
