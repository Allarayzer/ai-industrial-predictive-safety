"""Experiment E4 (monograph § 13.8): robustness to noise and missingness.

Measures F1 degradation of the Cascade detector as Gaussian noise
amplitude sigma and fraction of missing data increase.

Two sweeps:
    1. Noise only (missingness = 0) at sigma in {0.00, 0.01, 0.05, 0.10, 0.20}
    2. Missingness only (sigma = 0.05) at fraction in {0.00, 0.01, 0.05, 0.10}

Dataset: the synthetic 3-channel stream from data.py with 5% injected
anomalies. Cascade = HybridDetector (IsolationForest + LSTM).
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time

import numpy as np
import pandas as pd

from ai_cta.anomaly_detector import HybridDetector
from ai_cta.data import generate_synthetic_stream, inject_anomalies
from ai_cta.evaluation import evaluate_binary_detector

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def corrupt(test: pd.DataFrame, sigma: float, missing: float, seed: int) -> pd.DataFrame:
    """Apply additive Gaussian noise and random missingness to sensor channels."""
    rng = np.random.default_rng(seed)
    cols = [c for c in test.columns if c != "timestamp"]
    out = test.copy()
    if sigma > 0:
        noise = rng.normal(0.0, sigma * test[cols].std().values, size=(len(test), len(cols)))
        out[cols] = out[cols].values + noise
    if missing > 0:
        # Replace with last-known value (forward fill) for simplicity.
        mask = rng.random(size=(len(test), len(cols))) < missing
        vals = out[cols].values.copy()
        vals[mask] = np.nan
        df_tmp = pd.DataFrame(vals, columns=cols)
        df_tmp = df_tmp.ffill().bfill().fillna(0)
        out[cols] = df_tmp.values
    return out


def evaluate_cascade(
    train: pd.DataFrame, test: pd.DataFrame, labels: np.ndarray, seed: int
) -> float:
    """Fit Cascade, threshold at 95th percentile, return F1."""
    det = HybridDetector(
        if_weight=0.5,
        if_params={"window_size": 64, "stride": 32, "use_spectral": False, "random_state": seed},
        lstm_params={"window_size": 32, "lstm_units": (32, 16), "epochs": 8, "random_state": seed},
    )
    det.fit(train.drop(columns=["timestamp"]))
    scores = det.decision_function(test.drop(columns=["timestamp"]))
    # Pad to labels length.
    if len(scores) < len(labels):
        scores = np.concatenate([np.full(len(labels) - len(scores), scores[0]), scores])
    else:
        scores = scores[:len(labels)]
    threshold = float(np.percentile(scores, 95))
    y_pred = (scores >= threshold).astype(int)
    return float(evaluate_binary_detector(labels, y_pred).f1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=4000)
    args = parser.parse_args()

    # Generate one clean dataset per seed.
    datasets = []
    for s in range(args.n_seeds):
        train = generate_synthetic_stream(n_samples=int(args.n_samples * 0.7), random_state=s)
        test_base = generate_synthetic_stream(n_samples=int(args.n_samples * 0.3), random_state=s + 1)
        test, labels = inject_anomalies(test_base, n_anomalies=int(len(test_base) * 0.05), random_state=s + 2)
        datasets.append((train, test, labels))

    results: list[dict] = []

    # Sweep 1: noise only.
    for sigma in [0.00, 0.01, 0.05, 0.10, 0.20]:
        for seed, (train, test, labels) in enumerate(datasets):
            corrupted = corrupt(test, sigma=sigma, missing=0.0, seed=seed + 100)
            t0 = time.perf_counter()
            f1 = evaluate_cascade(train, corrupted, labels, seed)
            print(f"  sigma={sigma:<5} missing=0.00  seed={seed}  F1={f1:.3f}  ({time.perf_counter()-t0:.1f}s)")
            results.append({
                "sweep": "noise", "sigma": sigma, "missing": 0.0,
                "seed": seed, "f1": round(f1, 3),
            })

    # Sweep 2: missingness at fixed noise sigma=0.05.
    for missing in [0.00, 0.01, 0.05, 0.10]:
        for seed, (train, test, labels) in enumerate(datasets):
            corrupted = corrupt(test, sigma=0.05, missing=missing, seed=seed + 200)
            t0 = time.perf_counter()
            f1 = evaluate_cascade(train, corrupted, labels, seed)
            print(f"  sigma=0.05  missing={missing:<5}  seed={seed}  F1={f1:.3f}  ({time.perf_counter()-t0:.1f}s)")
            results.append({
                "sweep": "missing", "sigma": 0.05, "missing": missing,
                "seed": seed, "f1": round(f1, 3),
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "experiment_e4_robustness.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # Aggregated.
    df = pd.DataFrame(results)
    print("\n=== Noise sweep (missing=0) ===")
    print(df[df["sweep"] == "noise"].groupby("sigma")["f1"].agg(["mean", "std"]).round(3).to_string())
    print("\n=== Missingness sweep (sigma=0.05) ===")
    print(df[df["sweep"] == "missing"].groupby("missing")["f1"].agg(["mean", "std"]).round(3).to_string())
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
