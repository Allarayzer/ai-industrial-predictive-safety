"""Experiment E1 (monograph § 13.5): anomaly detection on Dataset S1.

Reproduces Table 13.1: four detector configurations compared on a
synthetic multichannel industrial stream with injected anomalies.

    Threshold 3σ (baseline) - per-channel static threshold, no ML
    IF only                 - IsolationForest over engineered features
    AE only                 - LSTM forecaster with prediction residuals
    Cascade (IF + AE)       - HybridDetector combining both scores

Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC, false-alarm rate
(per hour of the emulated stream, assuming 1-minute sampling).

Runs with 5 random seeds by default; reports mean ± std.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ai_cta.anomaly_detector import (
    HybridDetector,
    IsolationForestDetector,
    LSTMDetector,
)
from ai_cta.data import generate_synthetic_stream, inject_anomalies

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def build_dataset(seed: int, n_samples: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Generate Dataset S1: 3-channel industrial stream with ~5% anomalies.

    Split: 70% train (no anomalies) / 30% test (anomalies injected).
    """
    train = generate_synthetic_stream(n_samples=int(n_samples * 0.7), random_state=seed)
    test_base = generate_synthetic_stream(n_samples=int(n_samples * 0.3), random_state=seed + 1)
    n_anom = max(20, int(len(test_base) * 0.05))
    test, labels = inject_anomalies(test_base, n_anomalies=n_anom, random_state=seed + 2)
    return train, test, labels


def threshold_3sigma(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Per-channel 3σ threshold. Returns anomaly score in [0, 1] per timestep."""
    cols = [c for c in train.columns if c != "timestamp"]
    mu = train[cols].mean().values
    sigma = train[cols].std().values + 1e-9
    X = test[cols].values
    z = np.abs((X - mu) / sigma)
    # Binary flag if any channel exceeds 3σ.
    return z.max(axis=1)


def iforest_scores(
    train: pd.DataFrame, test: pd.DataFrame, seed: int, window: int = 64
) -> np.ndarray:
    det = IsolationForestDetector(
        window_size=window, stride=window // 2, use_spectral=False, random_state=seed
    )
    det.fit(train.drop(columns=["timestamp"]))
    s = det.decision_function(test.drop(columns=["timestamp"]))
    return _pad_to_length(s, len(test))


def lstm_scores(
    train: pd.DataFrame, test: pd.DataFrame, seed: int, window: int = 32
) -> np.ndarray:
    det = LSTMDetector(
        window_size=window, lstm_units=(32, 16),
        epochs=10, batch_size=64, random_state=seed,
    )
    det.fit(train.drop(columns=["timestamp"]))
    s = det.decision_function(test.drop(columns=["timestamp"]))
    return _pad_to_length(s, len(test))


def cascade_scores(
    train: pd.DataFrame, test: pd.DataFrame, seed: int, window: int = 32
) -> np.ndarray:
    det = HybridDetector(
        if_weight=0.5,
        if_params={"window_size": window, "stride": window // 2,
                   "use_spectral": False, "random_state": seed},
        lstm_params={"window_size": window, "lstm_units": (32, 16),
                     "epochs": 10, "random_state": seed},
    )
    det.fit(train.drop(columns=["timestamp"]))
    s = det.decision_function(test.drop(columns=["timestamp"]))
    return _pad_to_length(s, len(test))


def _pad_to_length(scores: np.ndarray, length: int) -> np.ndarray:
    """Pad scores to match label length by repeating the first value."""
    if len(scores) >= length:
        return scores[:length]
    pad = np.full(length - len(scores), scores[0] if len(scores) > 0 else 0.0)
    return np.concatenate([pad, scores])


def score_binary(
    scores: np.ndarray, labels: np.ndarray, target_far_per_hour: float = 2.0
) -> dict[str, float]:
    """Compute metrics; threshold chosen for target false-alarm rate.

    target_far_per_hour: assumes 1-minute sampling, so 60 samples/hour.
    Threshold set at the 1 - (target_far / 60)-th percentile of scores on
    negative labels.
    """
    neg_mask = labels == 0
    if neg_mask.sum() > 0:
        threshold = float(np.percentile(scores[neg_mask], 100 * (1 - target_far_per_hour / 60)))
    else:  # pragma: no cover
        threshold = float(np.median(scores))
    y_pred = (scores >= threshold).astype(int)

    f1 = f1_score(labels, y_pred, zero_division=0)
    p = precision_score(labels, y_pred, zero_division=0)
    r = recall_score(labels, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(labels, scores)
        pr = average_precision_score(labels, scores)
    except ValueError:  # pragma: no cover - degenerate labels
        roc = float("nan")
        pr = float("nan")
    tn, fp, _, _ = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()
    # FAR per hour: false positives per hour (60 min/sample).
    hours = len(labels) / 60.0
    far_per_hour = fp / hours if hours > 0 else 0.0

    return {
        "precision": round(float(p), 3),
        "recall": round(float(r), 3),
        "f1": round(float(f1), 3),
        "roc_auc": round(float(roc), 3),
        "pr_auc": round(float(pr), 3),
        "far_per_hour": round(float(far_per_hour), 2),
    }


METHODS = {
    "Threshold 3sigma": threshold_3sigma,
    "IF only": iforest_scores,
    "AE only": lstm_scores,
    "Cascade (IF+AE)": cascade_scores,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=5000)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for seed in range(args.n_seeds):
        print(f"\n=== seed {seed} ===")
        train, test, labels = build_dataset(seed, args.n_samples)
        for name, fn in METHODS.items():
            t0 = time.perf_counter()
            if name == "Threshold 3sigma":
                scores = fn(train, test)
            else:
                scores = fn(train, test, seed)
            elapsed = time.perf_counter() - t0
            metrics = score_binary(scores, labels)
            row = {"seed": seed, "method": name, **metrics, "time_sec": round(elapsed, 1)}
            print(f"  {name:<22} F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
                  f"R={metrics['recall']:.3f}  ROC={metrics['roc_auc']:.3f}  "
                  f"PR={metrics['pr_auc']:.3f}  FAR/h={metrics['far_per_hour']:.2f}")
            all_rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "experiment_e1_s1.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # Aggregated.
    df = pd.DataFrame(all_rows)
    agg = df.groupby("method")[["precision", "recall", "f1", "roc_auc", "pr_auc", "far_per_hour"]].agg(
        ["mean", "std"]
    ).round(3)
    print("\n=== Aggregated (mean ± std) ===")
    print(agg.to_string())
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
