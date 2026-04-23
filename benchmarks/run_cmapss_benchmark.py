"""C-MAPSS FD001 benchmark runner.
Loads the NASA Turbofan Engine Degradation Simulation dataset, trains
three detector configurations, and writes a CSV of per-method metrics
to benchmarks/results/.
Task formulation
----------------
For each training engine we label the last `degradation_horizon` cycles
as "degraded" (1) and the rest as "normal" (0). The detectors are fit on
the first 60% of each engine's trajectory (assumed healthy) and
evaluated on the concatenation of all engines' full trajectories.
This is a simplification compared to the remaining-useful-life task
traditionally run on C-MAPSS, but it aligns with the anomaly-detection
framing of the rest of the package and makes the benchmark cheap
enough for quick iteration on a laptop.
"""
from __future__ import annotations
import argparse
import csv
import pathlib
import time
import numpy as np
import pandas as pd
from ai_cta.anomaly_detector import (
    HybridDetector,
    IsolationForestDetector,
)
from ai_cta.evaluation import evaluate_binary_detector

DATA_DIR = pathlib.Path(__file__).parent / "data" / "cmapss"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"
COLUMN_NAMES = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
# Sensors that the CMAPSS literature considers informative for FD001.
SELECTED_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

def load_cmapss_fd001(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )
    return df

def make_labels(df: pd.DataFrame, degradation_horizon: int = 30) -> np.ndarray:
    """Label the last `degradation_horizon` cycles of each unit as degraded."""
    labels = np.zeros(len(df), dtype=int)
    for unit, group in df.groupby("unit"):
        n = len(group)
        threshold_cycle = max(n - degradation_horizon, 0)
        labels[group.index[threshold_cycle:]] = 1
    return labels

def run_one_method(
    name: str,
    detector_factory,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_labels: np.ndarray,
) -> dict:
    detector = detector_factory()
    t0 = time.perf_counter()
    detector.fit(train_df)
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    scores = detector.decision_function(test_df)
    predict_time = time.perf_counter() - t1
    # Align scores to the label array by nearest-neighbor downsampling.
    n_scores = len(scores)
    if n_scores != len(test_labels):
        idx = np.linspace(0, len(test_labels) - 1, n_scores).astype(int)
        aligned_labels = test_labels[idx]
    else:
        aligned_labels = test_labels
    preds = (scores >= 0.5).astype(int)
    metrics = evaluate_binary_detector(aligned_labels, preds)
    return {
        "method": name,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "false_alarm_rate": metrics.false_alarm_rate,
        "true_positive": metrics.true_positive,
        "false_positive": metrics.false_positive,
        "true_negative": metrics.true_negative,
        "false_negative": metrics.false_negative,
        "fit_time_sec": round(fit_time, 3),
        "predict_time_sec": round(predict_time, 3),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--degradation-horizon", type=int, default=30,
        help="Cycles labelled as degraded at the end of each engine.",
    )
    parser.add_argument(
        "--window-size", type=int, default=32,
        help="Analysis window length for the detectors.",
    )
    parser.add_argument(
        "--skip-lstm", action="store_true",
        help="Skip LSTM-based methods (useful without tensorflow).",
    )
    args = parser.parse_args()
    train_path = DATA_DIR / "train_FD001.txt"
    if not train_path.exists():
        print(f"Dataset not found at {train_path}. See docs/benchmarks.md.")
        return 1
    print(f"Loading C-MAPSS FD001 from {train_path}...")
    df = load_cmapss_fd001(train_path)
    print(f"Loaded {len(df)} rows covering {df['unit'].nunique()} engine units.")
    labels = make_labels(df, degradation_horizon=args.degradation_horizon)
    print(
        f"Positive labels: {int(labels.sum())}/{len(labels)} "
        f"({100 * labels.mean():.1f}%)"
    )
    feature_cols = SELECTED_SENSORS
    train_mask = np.zeros(len(df), dtype=bool)
    for _, group in df.groupby("unit"):
        cutoff = int(len(group) * 0.6)
        train_mask[group.index[:cutoff]] = True
    train_df = df.loc[train_mask, feature_cols].reset_index(drop=True)
    test_df = df.loc[:, feature_cols].reset_index(drop=True)
    test_labels = labels
    print(f"Training windows drawn from {len(train_df)} healthy rows.")
    print(f"Evaluating on {len(test_df)} rows total.")
    rows: list[dict] = []
    rows.append(
        run_one_method(
            "IsolationForest",
            lambda: IsolationForestDetector(
                window_size=args.window_size,
                stride=args.window_size // 2,
                use_spectral=False,
            ),
            train_df, test_df, test_labels,
        )
    )
    rows.append(
        run_one_method(
            "IsolationForest+Spectral",
            lambda: IsolationForestDetector(
                window_size=args.window_size,
                stride=args.window_size // 2,
                use_spectral=True,
                sampling_rate=1.0,
            ),
            train_df, test_df, test_labels,
        )
    )
    if not args.skip_lstm:
        try:
            rows.append(
                run_one_method(
                    "Hybrid(IF+LSTM)",
                    lambda: HybridDetector(
                        if_weight=0.5,
                        if_params={
                            "window_size": args.window_size,
                            "stride": args.window_size // 2,
                            "use_spectral": False,
                        },
                        lstm_params={"window_size": args.window_size, "epochs": 10},
                    ),
                    train_df, test_df, test_labels,
                )
            )
        except ImportError as exc:
            print(f"Skipping LSTM/hybrid: {exc}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cmapss_fd001_results.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out_path}\n")
    for row in rows:
        print(
            f"  {row['method']:<30s} "
            f"F1={row['f1']:.3f}  "
            f"prec={row['precision']:.3f}  "
            f"rec={row['recall']:.3f}  "
            f"FAR={row['false_alarm_rate']:.3f}"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
