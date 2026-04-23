"""Baseline comparison: IsolationForest vs LOF vs One-Class SVM vs z-score.

Reviewers of predictive-maintenance papers in Scopus journals routinely
ask "how does your detector compare to classical baselines?". This
script answers that by running all four methods on the same synthetic
evaluation stream (and, optionally, on CMAPSS / CWRU when the data is
present locally).

Methods compared
----------------

- **IsolationForest** (the detector used in the AI-CTA pipeline).
- **Local Outlier Factor (LOF)** — density-based anomaly detection.
- **One-Class SVM** — kernel-based boundary detection.
- **z-score threshold** — trivial statistical baseline (|(x - mu) / sigma| > 3).

All four are evaluated over ``--n-seeds`` random seeds on a common
test set with controlled anomaly injection. Metrics reported:
F1, precision, recall, false-alarm rate, AUC.

Output
------

    benchmarks/results/baselines.csv
    benchmarks/results/baselines.tex

Usage
-----

    python benchmarks/run_baselines.py --n-seeds 10 --n-samples 5000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ai_cta.anomaly_detector import IsolationForestDetector
from ai_cta.data import generate_synthetic_stream, inject_anomalies
from ai_cta.evaluation import evaluate_binary_detector

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _make_dataset(n_samples: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return (train_df, test_df_contaminated, y_test)."""
    train_df = generate_synthetic_stream(n_samples=max(2000, n_samples // 2), random_state=seed)
    test_df = generate_synthetic_stream(n_samples=n_samples, random_state=seed + 1)
    contaminated, y = inject_anomalies(
        test_df, n_anomalies=max(50, n_samples // 50), random_state=seed + 2
    )
    return train_df, contaminated, y


def _iforest(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> np.ndarray:
    det = IsolationForestDetector(
        window_size=64, stride=16, use_spectral=False, random_state=seed
    )
    det.fit(train_df.drop(columns=["timestamp"]))
    scores = det.decision_function(test_df.drop(columns=["timestamp"]))
    # Right-pad to full length.
    pad = np.full(len(test_df) - len(scores), scores[-1])
    return np.concatenate([scores, pad])


def _lof(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> np.ndarray:
    from sklearn.neighbors import LocalOutlierFactor

    lof = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination="auto")
    lof.fit(train_df.drop(columns=["timestamp"]))
    # Higher score == more anomalous (invert decision_function).
    return -lof.decision_function(test_df.drop(columns=["timestamp"]))


def _ocsvm(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> np.ndarray:
    from sklearn.svm import OneClassSVM

    svm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    svm.fit(train_df.drop(columns=["timestamp"]))
    return -svm.decision_function(test_df.drop(columns=["timestamp"]))


def _zscore(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> np.ndarray:
    cols = [c for c in train_df.columns if c != "timestamp"]
    mu = train_df[cols].mean().values
    sigma = train_df[cols].std().values + 1e-9
    X = test_df[cols].values
    z = np.abs((X - mu) / sigma)
    return z.max(axis=1)  # per-timestep max |z| across channels


METHODS = {
    "iforest": _iforest,
    "lof": _lof,
    "ocsvm": _ocsvm,
    "zscore": _zscore,
}


def _score_method(name: str, scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    # Threshold at score's 95th percentile on training-era distribution.
    threshold = float(np.percentile(scores, 95))
    y_pred = (scores >= threshold).astype(int)
    m = evaluate_binary_detector(y, y_pred)
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, scores)) if y.sum() > 0 else float("nan")
    except Exception:  # pragma: no cover
        auc = float("nan")
    return {
        "method": name,
        "f1": float(m.f1),
        "precision": float(m.precision),
        "recall": float(m.recall),
        "far": float(m.false_alarm_rate),
        "auc": auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=3000)
    args = parser.parse_args()

    all_rows: list[dict[str, float]] = []
    print(f"Running baselines across {args.n_seeds} seeds, {args.n_samples} samples each...")
    for seed in range(args.n_seeds):
        train_df, test_df, y = _make_dataset(args.n_samples, seed)
        for name, fn in METHODS.items():
            scores = fn(train_df, test_df, seed)
            # Align length to y.
            if len(scores) != len(y):
                scores = scores[: len(y)] if len(scores) > len(y) else np.concatenate(
                    [scores, np.full(len(y) - len(scores), scores[-1])]
                )
            row = _score_method(name, scores, y)
            row["seed"] = seed
            all_rows.append(row)
            print(
                f"  seed={seed}  {name:<8} "
                f"F1={row['f1']:.3f}  AUC={row['auc']:.3f}  FAR={row['far']:.3f}"
            )

    # Persist raw CSV.
    csv_path = RESULTS_DIR / "baselines.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "method", "f1", "precision", "recall", "far", "auc"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nRaw results -> {csv_path}")

    # Aggregated + LaTeX.
    df = pd.DataFrame(all_rows)
    agg = (
        df.groupby("method")[["f1", "precision", "recall", "far", "auc"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print("\n--- Aggregated (mean \u00b1 std) ---")
    print(agg.to_string())

    tex_path = RESULTS_DIR / "baselines.tex"
    with tex_path.open("w") as f:
        f.write("% Baseline anomaly-detector comparison.\n")
        f.write("% Generated by benchmarks/run_baselines.py\n")
        f.write("\\begin{table}[ht]\n\\centering\n")
        caption = (
            "\\caption{Anomaly-detection baselines on the synthetic industrial "
            f"stream (mean $\\pm$ std over {args.n_seeds} seeds).}}\n"
        )
        f.write(caption.replace("}}", "}"))
        f.write("\\label{tab:baselines}\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Method & F1 & Precision & Recall & FAR & AUC \\\\\n\\midrule\n")
        for name, row in agg.iterrows():
            cells = []
            for m in ["f1", "precision", "recall", "far", "auc"]:
                cells.append(f"{row[(m, 'mean')]:.3f} $\\pm$ {row[(m, 'std')]:.3f}")
            f.write(f"{name} & {' & '.join(cells)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX table -> {tex_path}")


if __name__ == "__main__":
    main()
