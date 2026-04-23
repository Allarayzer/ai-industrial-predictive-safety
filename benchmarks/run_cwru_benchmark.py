"""CWRU bearing-fault benchmark runner.
Loads .mat files from benchmarks/data/cwru/{normal,fault}, segments each
recording into fixed-length windows, extracts engineered features, and
trains a detector fit only on the normal-baseline windows. The detector
is then scored on the full mixed set, with a binary label indicating
which window originated from a faulty recording.
CWRU recordings are named by the fault-severity (e.g. 97.mat, 105.mat,
118.mat); this script is agnostic to naming and treats every file in
the `fault/` subdirectory as anomalous.
"""
from __future__ import annotations
import argparse
import csv
import pathlib
import time
import numpy as np
import pandas as pd
from ai_cta.detection import (
    HybridDetector,
    IsolationForestDetector,
)
from ai_cta.utils.evaluation import evaluate_binary_detector

DATA_DIR = pathlib.Path(__file__).parent / "data" / "cwru"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

def load_mat_signal(path: pathlib.Path) -> np.ndarray:
    """Return the drive-end vibration signal from a CWRU .mat file."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("The CWRU benchmark requires scipy.") from exc
    mat = loadmat(path)
    # Variables of interest end with _DE_time (drive end).
    de_keys = [k for k in mat.keys() if k.endswith("_DE_time")]
    if not de_keys:
        # Fall back to the first numeric array > 1000 samples.
        for k, v in mat.items():
            if isinstance(v, np.ndarray) and v.size > 1000 and v.dtype.kind == "f":
                return v.flatten().astype(np.float64)
        raise ValueError(f"No drive-end vibration channel found in {path}.")
    return mat[de_keys[0]].flatten().astype(np.float64)

def segment(signal: np.ndarray, window: int, stride: int) -> list[np.ndarray]:
    """Split a 1-D signal into overlapping windows."""
    segments = []
    for start in range(0, len(signal) - window + 1, stride):
        segments.append(signal[start : start + window])
    return segments

def build_dataset(
    normal_paths: list[pathlib.Path],
    fault_paths: list[pathlib.Path],
    window: int,
    stride: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Concatenate windows from all files into a single DataFrame.
    The resulting DataFrame has `window` numeric columns (one per sample
    in the window). Labels: 0 = normal, 1 = fault.
    """
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for path in normal_paths:
        sig = load_mat_signal(path)
        for w in segment(sig, window, stride):
            rows.append(w)
            labels.append(0)
    for path in fault_paths:
        sig = load_mat_signal(path)
        for w in segment(sig, window, stride):
            rows.append(w)
            labels.append(1)
    df = pd.DataFrame(
        np.vstack(rows),
        columns=[f"s{i}" for i in range(window)],
    )
    return df, np.asarray(labels, dtype=int)

def run_one_method(
    name: str,
    detector_factory,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_labels: np.ndarray,
) -> dict:
    detector = detector_factory()
    t0 = time.perf_counter()
    detector.fit(train_df)
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    scores = detector.decision_function(eval_df)
    predict_time = time.perf_counter() - t1
    if len(scores) != len(eval_labels):
        idx = np.linspace(0, len(eval_labels) - 1, len(scores)).astype(int)
        aligned = eval_labels[idx]
    else:
        aligned = eval_labels
    preds = (scores >= 0.5).astype(int)
    metrics = evaluate_binary_detector(aligned, preds)
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
    parser.add_argument("--window", type=int, default=1024, help="Samples per window.")
    parser.add_argument("--stride", type=int, default=512, help="Window stride.")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument(
        "--sampling-rate", type=float, default=12000.0,
        help="Sampling rate in Hz (12000 or 48000 for CWRU).",
    )
    args = parser.parse_args()
    normal_paths = sorted((DATA_DIR / "normal").glob("*.mat"))
    fault_paths = sorted((DATA_DIR / "fault").glob("*.mat"))
    if not normal_paths or not fault_paths:
        print("CWRU data not found. See docs/benchmarks.md.")
        return 1
    print(f"Normal recordings:  {len(normal_paths)}")
    print(f"Fault recordings:   {len(fault_paths)}")
    df, labels = build_dataset(normal_paths, fault_paths, args.window, args.stride)
    print(f"Built {len(df)} windows of {args.window} samples each.")
    print(f"Positive labels: {int(labels.sum())}/{len(labels)}")
    # Train only on the normal windows.
    train_df = df.loc[labels == 0].reset_index(drop=True)
    eval_df = df
    eval_labels = labels
    rows: list[dict] = []
    rows.append(
        run_one_method(
            "IsolationForest(stat)",
            lambda: IsolationForestDetector(
                window_size=args.window,
                stride=args.window,
                use_spectral=False,
                sampling_rate=args.sampling_rate,
            ),
            train_df, eval_df, eval_labels,
        )
    )
    rows.append(
        run_one_method(
            "IsolationForest(stat+spectral)",
            lambda: IsolationForestDetector(
                window_size=args.window,
                stride=args.window,
                use_spectral=True,
                sampling_rate=args.sampling_rate,
            ),
            train_df, eval_df, eval_labels,
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
                            "window_size": args.window,
                            "stride": args.window,
                            "use_spectral": True,
                            "sampling_rate": args.sampling_rate,
                        },
                        lstm_params={"window_size": 64, "epochs": 5},
                    ),
                    train_df, eval_df, eval_labels,
                )
            )
        except ImportError as exc:
            print(f"Skipping LSTM/hybrid: {exc}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cwru_results.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out_path}\n")
    for row in rows:
        print(
            f"  {row['method']:<34s} "
            f"F1={row['f1']:.3f}  "
            f"prec={row['precision']:.3f}  "
            f"rec={row['recall']:.3f}  "
            f"FAR={row['false_alarm_rate']:.3f}"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
