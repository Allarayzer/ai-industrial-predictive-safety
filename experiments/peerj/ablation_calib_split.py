"""Ablation of the 3-component risk fusion with a CALIBRATION/TEST split (Phase 2 fix).

Why this script exists
----------------------
The shipped ``benchmarks/run_ablation.py`` calibrates the SLSQP fusion weights on
the SAME array it evaluates on (validation == test). That optimistically biases the
calibrated full-fusion row relative to the equal-weight / single-component rows.

This corrected runner splits each seed's samples into disjoint calibration and test
halves. The SLSQP weights for the full 3-component fusion are fit on the calibration
half ONLY; every configuration (singles, pairs, equal, calibrated) is then scored on
the held-out test half. Single-component and equal-weight rows are unaffected by the
fix but are evaluated on the same test half for a fair comparison.

The three component signals (R_anom, R_RUL, R_NN) are produced exactly as in the
original ablation: R_anom from a real IsolationForest fit on a synthetic normal
stream, R_RUL an exponential-decay proxy, R_NN a logistic proxy. As in the original,
this ablation isolates the AGGREGATION behaviour, not individual component quality;
the real-data fusion evidence is e2_cmapss_calib_split.py.

Outputs (analysis/results/):
    ablation_calib_split.csv  columns: seed, config, calibrated, f1, precision,
        recall, far, roc_auc, pr_auc, weights
    ablation_calib_split.tex  LaTeX mean +/- std table
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from ai_cta.anomaly_detector import IsolationForestDetector
from ai_cta.data import generate_synthetic_stream, inject_anomalies
from ai_cta.evaluation import evaluate_binary_detector
from ai_cta.risk_model import RiskAggregator

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONFIGS = [
    ("anom",), ("rul",), ("nn",),
    ("anom", "rul"), ("anom", "nn"), ("rul", "nn"),
    ("anom", "rul", "nn"),
]


def simulate_components(n_samples, seed):
    """(R_anom, R_rul, R_nn, y) for one seed -- identical recipe to run_ablation.py."""
    rng = np.random.default_rng(seed)
    train_df = generate_synthetic_stream(n_samples=max(2000, n_samples // 2), random_state=seed)
    test_df = generate_synthetic_stream(n_samples=n_samples, random_state=seed + 1)
    contaminated, y = inject_anomalies(test_df, n_anomalies=max(50, n_samples // 50),
                                       random_state=seed + 2)
    det = IsolationForestDetector(window_size=64, stride=16, use_spectral=False)
    det.fit(train_df.drop(columns=["timestamp"]))
    scores = det.decision_function(contaminated.drop(columns=["timestamp"]))
    if len(scores) < len(y):
        scores = np.concatenate([scores, np.full(len(y) - len(scores), scores[-1])])
    scores = scores[: len(y)]
    r_anom = expit((scores - scores.mean()) / (scores.std() + 1e-9))
    r_rul = np.zeros_like(r_anom)
    for idx in np.where(y > 0)[0]:
        horizon = np.arange(min(60, len(r_rul) - idx))
        r_rul[idx:idx + len(horizon)] = np.maximum(
            r_rul[idx:idx + len(horizon)], np.exp(-horizon / 20.0))
    r_rul = np.clip(r_rul + rng.normal(0, 0.05, len(r_rul)), 0, 1)
    r_nn = expit(2.0 * r_anom - 0.5 + rng.normal(0, 0.2, len(r_anom)))
    return r_anom, r_rul, r_nn, y


def evaluate(components, weights, signals_test, y_test):
    R = signals_test @ weights
    y_pred = (R >= 0.5).astype(int)
    m = evaluate_binary_detector(y_test, y_pred)
    roc = float(roc_auc_score(y_test, R)) if y_test.sum() and (y_test == 0).sum() else float("nan")
    pr = float(average_precision_score(y_test, R)) if y_test.sum() else float("nan")
    return m, roc, pr


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=5000)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(args.n_seeds):
        r_anom, r_rul, r_nn, y = simulate_components(args.n_samples, seed)
        stack = {"anom": r_anom, "rul": r_rul, "nn": r_nn}
        n = len(y)
        # Deterministic, stratified-ish calib/test split: interleave to keep
        # the (rare) anomaly rate similar across halves.
        rng = np.random.default_rng(1000 + seed)
        order = rng.permutation(n)
        calib_idx = np.sort(order[: n // 2])
        test_idx = np.sort(order[n // 2:])
        y_cal, y_test = y[calib_idx], y[test_idx]

        for comps in CONFIGS:
            sig_cal = np.stack([stack[c][calib_idx] for c in comps], axis=1)
            sig_test = np.stack([stack[c][test_idx] for c in comps], axis=1)
            if len(comps) == 3:
                agg = RiskAggregator()
                agg.calibrate_weights(stack["anom"][calib_idx], stack["rul"][calib_idx],
                                      stack["nn"][calib_idx], y_cal.astype(float))
                w = agg.w
                calibrated = True
            else:
                w = np.ones(len(comps)) / len(comps)
                calibrated = False
            m, roc, pr = evaluate(comps, w, sig_test, y_test)
            rows.append({
                "seed": seed, "config": "+".join(comps), "calibrated": calibrated,
                "f1": round(float(m.f1), 4), "precision": round(float(m.precision), 4),
                "recall": round(float(m.recall), 4), "far": round(float(m.false_alarm_rate), 4),
                "roc_auc": round(roc, 4), "pr_auc": round(pr, 4),
                "weights": ",".join(f"{wi:.3f}" for wi in w),
            })
            print(f"seed={seed} {rows[-1]['config']:<14} F1={rows[-1]['f1']:.3f} "
                  f"ROC={rows[-1]['roc_auc']:.3f} PR={rows[-1]['pr_auc']:.3f} w={rows[-1]['weights']}")

    out = RESULTS_DIR / "ablation_calib_split.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    df = pd.DataFrame(rows)
    agg = df.groupby("config")[["f1", "precision", "recall", "far", "roc_auc", "pr_auc"]].agg(
        ["mean", "std"]).round(3)
    print("\n=== Ablation (calib/test split), mean +/- std ===")
    print(agg.to_string())

    tex = RESULTS_DIR / "ablation_calib_split.tex"
    with tex.open("w") as f:
        f.write("% Ablation of 3-component risk fusion (calibration/test split).\n")
        f.write("% Generated by analysis/scripts/ablation_calib_split.py\n")
        f.write("\\begin{table}[ht]\\centering\n")
        f.write(f"\\caption{{Ablation of the risk-fusion components (mean $\\pm$ std over "
                f"{args.n_seeds} seeds); full fusion weights calibrated by SLSQP on a held-out "
                f"calibration split and evaluated on a disjoint test split.}}\n")
        f.write("\\label{tab:ablation}\\begin{tabular}{lcccccc}\\toprule\n")
        f.write("Config & F1 & Precision & Recall & FAR & ROC-AUC & PR-AUC \\\\\\midrule\n")
        for cfg, r in agg.iterrows():
            cells = " & ".join(f"{r[(m,'mean')]:.3f} $\\pm$ {r[(m,'std')]:.3f}"
                               for m in ["f1", "precision", "recall", "far", "roc_auc", "pr_auc"])
            f.write(f"{str(cfg).replace('_', chr(92)+'_')} & {cells} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    print(f"\nWrote {out} and {tex}")


if __name__ == "__main__":
    main()
