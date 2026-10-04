"""Conformal coverage validation: empirical false-alarm rate vs nominal alpha
(Phase 2.3 additional analysis).

Validates the marginal false-alarm guarantee of ai_cta.ConformalThresholdCalibrator
under the exchangeability assumption it requires. Protocol per repetition:

  1. Generate an i.i.d.-normal training stream and fit an IsolationForest detector.
  2. Generate a SEPARATE normal evaluation stream and score every window. Because the
     detector is fixed and the evaluation windows are exchangeable, their anomaly
     scores are exchangeable -- the setting in which split-conformal coverage holds.
  3. Randomly partition the evaluation scores into a calibration half and a test half.
  4. For each nominal alpha, calibrate the threshold on the calibration half and
     measure the empirical false-alarm rate (fraction of TEST normal scores flagged).

Aggregated over many repetitions, the mean empirical FAR should sit at or below each
nominal alpha (the finite-sample-corrected quantile is slightly conservative). We
report the mean empirical FAR with a Wilson 95% binomial confidence interval over the
pooled test decisions, which is what Fig. 5 plots.

Output (analysis/results/): conformal_coverage.csv
    columns: alpha, rep, n_calib, n_test, n_flagged, empirical_far
plus a printed summary table with pooled empirical FAR and Wilson 95% CI per alpha.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ai_cta.anomaly_detector import IsolationForestDetector
from ai_cta.data import generate_synthetic_stream
from ai_cta.risk_model import ConformalThresholdCalibrator

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-reps", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=6000)
    ap.add_argument("--alphas", type=str, default="0.01,0.05,0.10")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    pooled = {a: [0, 0] for a in alphas}  # alpha -> [n_flagged, n_test]
    for rep in range(args.n_reps):
        train = generate_synthetic_stream(n_samples=args.n_samples // 2, random_state=10_000 + rep)
        eval_ = generate_synthetic_stream(n_samples=args.n_samples, random_state=20_000 + rep)
        det = IsolationForestDetector(window_size=64, stride=16, use_spectral=False,
                                      random_state=rep)
        det.fit(train.drop(columns=["timestamp"]))
        scores = det.decision_function(eval_.drop(columns=["timestamp"]))

        rng = np.random.default_rng(30_000 + rep)
        perm = rng.permutation(len(scores))
        half = len(scores) // 2
        cal_scores = scores[perm[:half]]
        test_scores = scores[perm[half:]]

        for a in alphas:
            if len(cal_scores) < int(1 / a):
                continue
            cal = ConformalThresholdCalibrator(alpha=a).calibrate(cal_scores)
            flagged = int((test_scores > cal.threshold_).sum())
            far = flagged / len(test_scores)
            rows.append({"alpha": a, "rep": rep, "n_calib": len(cal_scores),
                         "n_test": len(test_scores), "n_flagged": flagged,
                         "empirical_far": round(far, 5)})
            pooled[a][0] += flagged
            pooled[a][1] += len(test_scores)
        if (rep + 1) % 10 == 0:
            print(f"  ...{rep + 1}/{args.n_reps} reps")

    out = RESULTS_DIR / "conformal_coverage.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Conformal coverage: empirical FAR vs nominal alpha ===")
    print(f"{'nominal a':>10} {'mean FAR':>10} {'pooled FAR':>11} {'Wilson 95% CI':>22} {'<= a?':>7}")
    for a in alphas:
        k, n = pooled[a]
        if n == 0:
            continue
        per_rep = [r["empirical_far"] for r in rows if r["alpha"] == a]
        mean_far = float(np.mean(per_rep))
        lo, hi = wilson_ci(k, n)
        ok = "yes" if hi <= a + 1e-9 or (k / n) <= a else "check"
        print(f"{a:>10.2f} {mean_far:>10.4f} {k/n:>11.4f}  [{lo:.4f}, {hi:.4f}]   {ok:>7}")
    print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
