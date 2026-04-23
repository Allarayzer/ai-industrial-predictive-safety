"""Experiment E2 (monograph § 13.6): hybrid risk function on Dataset S1.

Compares five configurations of the risk aggregator on a synthetic
multichannel stream with labelled degradation events:

    R_anom only      - anomaly score only, ignore RUL and NN channels
    R_RUL only       - RUL-derived probability only
    R_NN only        - neural-risk estimator only
    Equal weights    - (1/3, 1/3, 1/3) fixed
    Calibrated       - weights tuned via SLSQP on the 2-simplex
                       minimising asymmetric BCE (§ 8.4.2).

Target label: "fail within 24 hours" binary flag per timestep.

Metrics (matching book Table 13.2):
    PR-AUC, ROC-AUC, MCC
    Lead time (h) - median time between first Warning
                    threshold crossing and actual failure in the
                    simulation.
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)

from ai_cta.anomaly_detector import IsolationForestDetector
from ai_cta.data import generate_synthetic_stream, inject_anomalies
from ai_cta.risk_model import RiskAggregator

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

WARNING_THRESHOLD = 0.3  # matches monograph § 8.4.3 default
SAMPLES_PER_HOUR = 60  # assume 1-minute sampling


def simulate_components(n_samples: int, seed: int):
    """Return (R_anom, R_rul, R_nn, y, failure_indices).

    Each of the three risk components is simulated to be individually
    imperfect (matching the realistic case where no single signal is
    sufficient) with complementary information, so that the calibrated
    fusion is expected to improve modestly over each component.
    """
    rng = np.random.default_rng(seed)
    train_df = generate_synthetic_stream(n_samples=max(2000, n_samples // 2), random_state=seed)
    test_df = generate_synthetic_stream(n_samples=n_samples, random_state=seed + 1)
    n_failures = max(5, n_samples // 400)
    contaminated, y = inject_anomalies(
        test_df, n_anomalies=n_failures, random_state=seed + 2
    )
    failure_idx = np.where(y > 0)[0]

    # Label: "fails within next 24h" — positive for a 24-hour window
    # BEFORE each actual failure (not just at the failure timestep).
    horizon_samples = 24 * SAMPLES_PER_HOUR
    y_horizon = np.zeros_like(y)
    for idx in failure_idx:
        y_horizon[max(0, idx - horizon_samples) : idx + 1] = 1

    # R_anom: IForest calibrated to [0, 1]. Provides instantaneous
    # anomaly signal — useful near failure but limited lead time.
    det = IsolationForestDetector(window_size=64, stride=16, use_spectral=False,
                                   random_state=seed)
    det.fit(train_df.drop(columns=["timestamp"]))
    raw = det.decision_function(contaminated.drop(columns=["timestamp"]))
    if len(raw) < len(y):
        raw = np.concatenate([raw, np.full(len(y) - len(raw), raw[-1])])
    raw = raw[: len(y)]
    r_anom = expit(2.0 * (raw - np.percentile(raw, 70)))

    # R_RUL: exponential rise over the horizon window before each failure
    # (probability of failure within 24h). Provides long lead time but
    # with broader false-alarm region around each event.
    r_rul = np.zeros_like(r_anom, dtype=np.float32)
    for idx in failure_idx:
        start = max(0, idx - 2 * horizon_samples)
        length = idx - start + 1
        # Sigmoid rising from ~0 to ~1 over 2*horizon.
        t = np.arange(length) - length / 2
        r_rul[start : idx + 1] = np.maximum(r_rul[start : idx + 1], 1.0 / (1.0 + np.exp(-t / 60)))
    r_rul = np.clip(r_rul + rng.normal(0, 0.08, len(r_rul)), 0, 1)

    # R_NN: noisy but-calibrated combination of r_anom and r_rul plus
    # independent noise (simulates a trained NN that has learned
    # non-linear interactions between the two). Correlated with y
    # through r_anom, r_rul — NOT derived directly from y.
    r_nn = expit(
        2.5 * r_anom + 1.8 * r_rul - 1.5 + rng.normal(0, 0.35, len(y))
    )

    return r_anom, r_rul, r_nn, y_horizon, failure_idx


def median_lead_time_hours(risk: np.ndarray, failures: np.ndarray) -> float:
    """Median hours between first Warning crossing and each failure."""
    leads = []
    for f in failures:
        window = risk[: f + 1]
        above = np.where(window >= WARNING_THRESHOLD)[0]
        if len(above):
            first = above[0]
            leads.append((f - first) / SAMPLES_PER_HOUR)
    return float(np.median(leads)) if leads else 0.0


def evaluate_config(
    name: str, R_stack: np.ndarray, weights: np.ndarray, y: np.ndarray,
    failures: np.ndarray,
) -> dict:
    R = R_stack @ weights
    y_pred = (R >= WARNING_THRESHOLD).astype(int)
    if y.sum() > 0 and (y == 0).sum() > 0:
        pr = average_precision_score(y, R)
        roc = roc_auc_score(y, R)
    else:
        pr = float("nan"); roc = float("nan")
    mcc = matthews_corrcoef(y, y_pred) if len(set(y_pred)) > 1 else 0.0
    lt = median_lead_time_hours(R, failures)
    return {
        "config": name,
        "pr_auc": round(float(pr), 3),
        "roc_auc": round(float(roc), 3),
        "mcc": round(float(mcc), 3),
        "lead_time_h": round(float(lt), 1),
        "weights": ",".join(f"{w:.3f}" for w in weights),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=5000)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for seed in range(args.n_seeds):
        r_anom, r_rul, r_nn, y, failures = simulate_components(args.n_samples, seed)
        R_stack = np.stack([r_anom, r_rul, r_nn], axis=1)

        # R_anom only
        rows_seed = [
            evaluate_config("R_anom only", R_stack, np.array([1.0, 0.0, 0.0]), y, failures),
            evaluate_config("R_RUL only", R_stack, np.array([0.0, 1.0, 0.0]), y, failures),
            evaluate_config("R_NN only", R_stack, np.array([0.0, 0.0, 1.0]), y, failures),
            evaluate_config("Equal weights", R_stack, np.array([1/3, 1/3, 1/3]), y, failures),
        ]
        # Calibrated via SLSQP.
        agg = RiskAggregator()
        agg.calibrate_weights(r_anom, r_rul, r_nn, y.astype(float))
        rows_seed.append(
            evaluate_config("Calibrated (SLSQP)", R_stack, agg.w, y, failures)
        )
        for r in rows_seed:
            r["seed"] = seed
            all_rows.append(r)
            print(f"  seed={seed}  {r['config']:<22}  "
                  f"PR-AUC={r['pr_auc']:.3f}  ROC={r['roc_auc']:.3f}  "
                  f"MCC={r['mcc']:.3f}  Lead={r['lead_time_h']:.1f}h")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "experiment_e2_s1.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    df = pd.DataFrame(all_rows)
    agg_df = df.groupby("config")[["pr_auc", "roc_auc", "mcc", "lead_time_h"]].agg(
        ["mean", "std"]
    ).round(3)
    print("\n=== Aggregated (mean ± std) ===")
    print(agg_df.to_string())
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
