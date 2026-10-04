"""Conditional (subgroup) conformal coverage — critical-zone diagnostic (council Tier-1).

The headline coverage result (conformal_coverage.py) shows the empirical false-alarm rate
(FAR) stays at or below the nominal alpha MARGINALLY. For a safety framing, a reviewer
rightly asks whether that marginal control hides a worse-behaving operating regime. This
experiment tests exactly that: normal data is generated as a mixture of M distinct
operating modes (different mean/scale offsets); the detector is fit and the split-conformal
threshold is calibrated on the POOLED normal data (mixing all modes), then the empirical
FAR is measured separately WITHIN each mode. If per-mode FAR stays near alpha, marginal
coverage is approximately conditional across regimes; if one mode's FAR markedly exceeds
alpha, the marginal guarantee is hiding a regime — which we then disclose honestly.

Output (analysis/results/): conformal_conditional.csv
    columns: alpha, rep, mode, n_test, n_flagged, far
plus a printed per-mode summary with pooled FAR and Wilson 95% CI.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ai_cta.anomaly_detector import IsolationForestDetector
from ai_cta.data import generate_synthetic_stream
from ai_cta.risk_model import ConformalThresholdCalibrator

RES = Path(__file__).resolve().parent / "results"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mode_stream(n_samples, seed, scale, offset):
    """A normal stream for one operating mode: affine-shifted to make modes distinct."""
    s = generate_synthetic_stream(n_samples=n_samples, random_state=seed)
    cols = [c for c in s.columns if c != "timestamp"]
    s[cols] = s[cols] * scale + offset
    return s[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--alphas", type=str, default="0.05,0.10")
    ap.add_argument("--modes", type=int, default=3)
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    RES.mkdir(parents=True, exist_ok=True)

    # Distinct operating modes: mode m has scale (1 + 0.15 m) and offset 0.4 m.
    mode_params = [(1.0 + 0.15 * m, 0.4 * m) for m in range(args.modes)]

    rows = []
    pooled = {(a, m): [0, 0] for a in alphas for m in range(args.modes)}
    for rep in range(args.n_reps):
        # Train + calibration: pooled normal across all modes.
        train = np.vstack([mode_stream(args.n_samples // 2, 100 + rep * 10 + m, sc, off).to_numpy()
                           for m, (sc, off) in enumerate(mode_params)])
        import pandas as pd
        train_df = pd.DataFrame(train, columns=[f"f{i}" for i in range(train.shape[1])])
        det = IsolationForestDetector(window_size=64, stride=16, use_spectral=False, random_state=rep)
        det.fit(train_df)

        # Pooled calibration scores (fresh normal, all modes).
        cal_parts = []
        for m, (sc, off) in enumerate(mode_params):
            s = mode_stream(args.n_samples // 2, 500 + rep * 10 + m, sc, off)
            s.columns = [f"f{i}" for i in range(s.shape[1])]
            cal_parts.append(det.decision_function(s))
        cal_scores = np.concatenate(cal_parts)

        # Per-mode test scores.
        test_scores = {}
        for m, (sc, off) in enumerate(mode_params):
            s = mode_stream(args.n_samples, 900 + rep * 10 + m, sc, off)
            s.columns = [f"f{i}" for i in range(s.shape[1])]
            test_scores[m] = det.decision_function(s)

        for a in alphas:
            if len(cal_scores) < int(1 / a):
                continue
            thr = ConformalThresholdCalibrator(alpha=a).calibrate(cal_scores).threshold_
            for m in range(args.modes):
                flagged = int((test_scores[m] > thr).sum())
                n = len(test_scores[m])
                rows.append({"alpha": a, "rep": rep, "mode": m, "n_test": n,
                             "n_flagged": flagged, "far": round(flagged / n, 5)})
                pooled[(a, m)][0] += flagged
                pooled[(a, m)][1] += n
        if (rep + 1) % 10 == 0:
            print(f"  ...{rep + 1}/{args.n_reps}")

    out = RES / "conformal_conditional.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Per-mode conditional FAR (pooled over reps) ===")
    print(f"{'alpha':>6} {'mode':>5} {'pooled FAR':>11} {'Wilson 95% CI':>22}")
    for a in alphas:
        fars = []
        for m in range(args.modes):
            k, n = pooled[(a, m)]
            if n == 0:
                continue
            lo, hi = wilson(k, n)
            fars.append(k / n)
            print(f"{a:>6.2f} {m:>5} {k/n:>11.4f}  [{lo:.4f}, {hi:.4f}]")
        if fars:
            print(f"       -> across modes: min {min(fars):.4f}, max {max(fars):.4f}, "
                  f"spread {max(fars)-min(fars):.4f} (nominal {a})")
    print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
