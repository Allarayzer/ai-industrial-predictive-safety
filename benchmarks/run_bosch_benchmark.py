"""Bosch CNC Machining Dataset benchmark.

Reference: Tnani, M., Feil, M., Diepold, K. (2022).
"Smart Data Collection System for Brownfield CNC Milling Machines:
A New Benchmark Dataset for Data-Driven Machine Monitoring."
Procedia CIRP 107, 131–136.

Source: https://github.com/boschresearch/CNC_Machining
Licence: CC-BY-4.0 (dataset), BSD-3-Clause (code).

Dataset structure:
    data/M0N/OPNN/{good,bad}/M0N_<timeframe>_OPNN_xxx.h5
    Each .h5 has vibration_data of shape (~268k, 3) at 2 kHz (X/Y/Z axes).
    1632 normal + 70 anomalous recordings across 3 machines × 15 operations.

This benchmark evaluates IsolationForest + engineered features
(statistical + spectral) for tool-wear / process-anomaly detection.
The train/test split is:
    * train: earlier calendar months of NORMAL recordings, near a 70% split
    * test : later months of NORMAL + anomalous recordings in those months

Metrics: F1, Precision, Recall, FAR at 95th-percentile train threshold.
Per-machine breakdown + pooled result.

Outputs:
    experiments/maintained/results/bosch_results.csv
    experiments/maintained/results/bosch_pooled.csv

Usage:
    python benchmarks/run_bosch_benchmark.py
    python benchmarks/run_bosch_benchmark.py --machines M01,M02
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import time
from datetime import date

import numpy as np
from scipy import stats as scipy_stats

DATA_DIR = pathlib.Path(__file__).parent / "data" / "bosch" / "data"
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "maintained" / "results"


def load_h5(path: pathlib.Path) -> np.ndarray:
    """Load a Bosch .h5 file and return its vibration_data array (T, 3)."""
    import h5py
    with h5py.File(str(path), "r") as f:
        return f["vibration_data"][:].astype(np.float64)


def extract_features(signal: np.ndarray) -> np.ndarray:
    """Extract 27 features from a 3-axis vibration signal.

    Per axis (3): mean, std, RMS, crest factor, peak-to-peak,
                  dominant-freq-bin, spectral entropy, skew, kurtosis.
    Total: 3 × 9 = 27 features.
    """
    feats: list[float] = []
    for c in range(signal.shape[1]):
        w = signal[:, c]
        feats.append(float(w.mean()))
        feats.append(float(w.std()))
        rms = float(np.sqrt(np.mean(w ** 2)))
        feats.append(rms)
        feats.append(float(np.max(np.abs(w)) / (rms + 1e-9)))  # crest
        feats.append(float(np.max(w) - np.min(w)))  # peak-to-peak
        # Spectral features
        fft = np.abs(np.fft.rfft(w))[1:]  # drop DC
        if fft.sum() > 0:
            dom_bin = int(np.argmax(fft))
            p = fft / (fft.sum() + 1e-12)
            sent = float(-np.sum(p * np.log(p + 1e-12)))
        else:
            dom_bin = 0
            sent = 0.0
        feats.append(float(dom_bin))
        feats.append(sent)
        feats.append(float(scipy_stats.skew(w)))
        feats.append(float(scipy_stats.kurtosis(w)))
    return np.asarray(feats, dtype=np.float32)


def collect_files(machine: str) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """Return (normal_files, anomalous_files) for a given machine dir."""
    m_dir = DATA_DIR / machine
    normal = sorted(m_dir.glob("OP*/good/*.h5"))
    anomaly = sorted(m_dir.glob("OP*/bad/*.h5"))
    return normal, anomaly


def recording_month(path: pathlib.Path) -> date:
    """Read the documented English month/year tokens; never guess chronology."""
    match = re.fullmatch(r"M\d+_([A-Za-z]{3})_(\d{4})_OP\d+_\d+", path.stem)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if match is None or match[1] not in months:
        raise ValueError(f"Unknown recording date/month: {path}")
    return date(int(match[2]), months.index(match[1]) + 1, 1)


def chronological_split(normal_files, anomaly_files, train_fraction=0.7):
    """Reserve whole later months; older fault files are excluded from test.

    The nominal fraction selects the first test month; ties stay together.
    Recording order within a month is not asserted to be chronological.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    dated_normal = sorted((recording_month(p), p) for p in normal_files)
    dated_fault = sorted((recording_month(p), p) for p in anomaly_files)
    months = sorted({month for month, _ in dated_normal})
    if len(months) < 2:
        raise ValueError("A chronological split needs at least two calendar months.")
    cutoff = dated_normal[min(int(len(dated_normal) * train_fraction), len(dated_normal) - 1)][0]
    if cutoff == months[0]:
        cutoff = months[1]
    return (
        [p for month, p in dated_normal if month < cutoff],
        [p for month, p in dated_normal if month >= cutoff],
        [p for month, p in dated_fault if month >= cutoff],
    )


def build_feature_matrix(files: list[pathlib.Path]) -> np.ndarray:
    arrs = []
    for p in files:
        try:
            sig = load_h5(p)
            arrs.append(extract_features(sig))
        except Exception as exc:
            print(f"  skipping {p.name}: {exc}")
    return np.stack(arrs, axis=0) if arrs else np.zeros((0, 27), dtype=np.float32)


def evaluate_machine(
    machine: str, seed: int = 42, train_fraction: float = 0.7
) -> dict | None:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler

    normal_files, anomaly_files = collect_files(machine)
    if len(normal_files) < 20:
        print(f"  [{machine}] too few normal files ({len(normal_files)}), skip")
        return None
    print(f"\n=== {machine}: {len(normal_files)} normal, {len(anomaly_files)} anomalous ===")

    # Feature extraction
    t0 = time.perf_counter()
    train_files, test_normal_files, test_anomaly_files = chronological_split(
        normal_files, anomaly_files, train_fraction
    )
    X_train = build_feature_matrix(train_files)
    X_test_normal = build_feature_matrix(test_normal_files)
    X_test_anom = build_feature_matrix(test_anomaly_files)
    n_train = len(X_train)
    print(f"  Features: {n_train} train normal, {len(X_test_normal)} test normal, "
          f"{len(X_test_anom)} test faults ({time.perf_counter()-t0:.1f}s)")
    y_test = np.concatenate([
        np.zeros(len(X_test_normal), dtype=int),
        np.ones(len(X_test_anom), dtype=int),
    ])
    X_test = np.concatenate([X_test_normal, X_test_anom], axis=0)

    # Robust scaler on train only
    scaler = RobustScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # IsolationForest fit on normal only
    t0 = time.perf_counter()
    det = IsolationForest(
        n_estimators=200, contamination="auto",
        random_state=seed, n_jobs=-1,
    ).fit(X_train)
    train_scores = -det.decision_function(X_train)
    threshold = float(np.percentile(train_scores, 95))

    # Predict
    test_scores = -det.decision_function(X_test)
    y_pred = (test_scores >= threshold).astype(int)
    elapsed = time.perf_counter() - t0

    # Metrics
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0

    from sklearn.metrics import average_precision_score, roc_auc_score
    try:
        roc = float(roc_auc_score(y_test, test_scores)) if y_test.sum() else float("nan")
        pr = float(average_precision_score(y_test, test_scores)) if y_test.sum() else float("nan")
    except ValueError:
        roc = float("nan")
        pr = float("nan")

    result = {
        "machine": machine,
        "n_train_normal": int(n_train),
        "n_test_normal": int(len(X_test_normal)),
        "n_test_anom": int(len(X_test_anom)),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "far": round(far, 3),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "roc_auc": round(roc, 3),
        "pr_auc": round(pr, 3),
        "macro_roc_auc": float("nan"),
        "macro_pr_auc": float("nan"),
        "threshold": round(threshold, 4),
        "fit_time_sec": round(elapsed, 1),
    }
    print(f"  F1={f1:.3f}  P={precision:.3f}  R={recall:.3f}  "
          f"FAR={far:.3f}  ROC={roc:.3f}  PR={pr:.3f}  ({elapsed:.1f}s)")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--machines", default="M01,M02,M03",
                        help="Comma-separated list of machines.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"ERROR: Bosch data not found at {DATA_DIR}")
        return 1

    rows: list[dict] = []
    for m in args.machines.split(","):
        r = evaluate_machine(m.strip(), seed=args.seed)
        if r:
            rows.append(r)

    if not rows:
        print("No results produced.")
        return 1

    # Pooled metrics
    total_tp = sum(r["tp"] for r in rows)
    total_fn = sum(r["fn"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    total_tn = sum(r["tn"] for r in rows)
    p_pool = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    r_pool = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1_pool = 2 * p_pool * r_pool / (p_pool + r_pool) if (p_pool + r_pool) else 0
    far_pool = total_fp / (total_fp + total_tn) if (total_fp + total_tn) else 0

    pooled = {
        "machine": "POOLED",
        "n_train_normal": sum(r["n_train_normal"] for r in rows),
        "n_test_normal": sum(r["n_test_normal"] for r in rows),
        "n_test_anom": sum(r["n_test_anom"] for r in rows),
        "precision": round(p_pool, 3),
        "recall": round(r_pool, 3),
        "f1": round(f1_pool, 3),
        "far": round(far_pool, 3),
        "tp": total_tp, "fn": total_fn, "fp": total_fp, "tn": total_tn,
        # Per-machine score scales have not been calibrated to one common scale.
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "macro_roc_auc": round(float(np.mean([r["roc_auc"] for r in rows])), 3),
        "macro_pr_auc": round(float(np.mean([r["pr_auc"] for r in rows])), 3),
        "threshold": "(per-machine)",
        "fit_time_sec": round(sum(r["fit_time_sec"] for r in rows), 1),
    }
    rows.append(pooled)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "bosch_results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Summary (Bosch CNC Machining) ===")
    print(f"{'Machine':<10}{'F1':>8}{'P':>8}{'R':>8}{'FAR':>8}{'ROC-AUC':>10}{'PR-AUC':>10}")
    for r in rows:
        print(f"{r['machine']:<10}{r['f1']:>8.3f}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['far']:>8.3f}"
              f"{r['roc_auc']:>10.3f}{r['pr_auc']:>10.3f}")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
