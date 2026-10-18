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
The default split is lexicographic PATH order, not validated temporal order:
    * train: 70% of NORMAL recordings only
    * test : 30% of NORMAL + ALL anomalous recordings

Metrics: exact confusion counts and F1/precision/recall/FAR at a
95th-percentile train threshold. Machine ROC-AUC/AP and their macro means are
separate from pooled threshold-decision metrics. Raw scores from separately
fitted machine detectors are not assumed to share a ranking scale.

Outputs:
    benchmarks/results/bosch-corrected/bosch_results.csv
    benchmarks/results/bosch-corrected/bosch_predictions.csv
    benchmarks/results/bosch-corrected/bosch_split_metadata.json
    The output directory must be new; historical CSVs are not replaced.

Usage:
    python benchmarks/run_bosch_benchmark.py
    python benchmarks/run_bosch_benchmark.py --machines M01,M02
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time

import numpy as np
from scipy import stats as scipy_stats

DATA_DIR = pathlib.Path(__file__).parent / "data" / "bosch" / "data"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"


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
    """Return lexicographically ordered paths; OP/month/name order is not time."""
    m_dir = DATA_DIR / machine
    normal = sorted(m_dir.glob("OP*/good/*.h5"))
    anomaly = sorted(m_dir.glob("OP*/bad/*.h5"))
    return normal, anomaly


def build_feature_matrix(files: list[pathlib.Path]) -> np.ndarray:
    arrs = []
    for p in files:
        try:
            sig = load_h5(p)
            features = extract_features(sig)
        except Exception as exc:
            raise ValueError(f"Failed extracting recording {p}: {exc}") from exc
        if features.shape != (27,) or not np.isfinite(features).all():
            raise ValueError(f"Invalid shape or nonfinite features for recording {p}")
        arrs.append(features)
    return np.stack(arrs, axis=0) if arrs else np.zeros((0, 27), dtype=np.float32)


def count_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Compute decision metrics from counts, never from rounded rates."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "far": fp / (fp + tn) if fp + tn else 0.0}


def aggregate_results(rows: list[dict]) -> dict:
    """Pool exact decisions; report unweighted machine-mean ranking separately."""
    if not rows:
        raise ValueError("No machine results to aggregate.")
    for row in rows:
        for name in ("tp", "fp", "tn", "fn"):
            value = row[name]
            if not np.isfinite(value) or value < 0 or int(value) != value:
                raise ValueError(f"Invalid exact count: {row['machine']} {name}")
        if row["tp"] + row["fn"] != row["n_test_anom"] or row["fp"] + row["tn"] != row["n_test_normal"]:
            raise ValueError(f"Counts do not match test populations: {row['machine']}")
    pooled = {"machine": "POOLED", "metric_scope": "pooled_decisions_and_macro_ranking",
              "ordering_policy": "lexicographic",
              "n_train_normal": sum(r["n_train_normal"] for r in rows),
              "n_test_normal": sum(r["n_test_normal"] for r in rows),
              "n_test_anom": sum(r["n_test_anom"] for r in rows),
              **count_metrics(*(sum(int(r[k]) for r in rows) for k in ("tp", "fp", "tn", "fn"))),
              "roc_auc": None, "pr_auc": None, "threshold": "per-machine",
              "fit_time_sec": sum(r["fit_time_sec"] for r in rows)}
    for metric in ("roc_auc", "pr_auc"):
        values = [r[metric] for r in rows if r[metric] is not None and np.isfinite(r[metric])]
        pooled[f"macro_{metric}"] = float(np.mean(values)) if values else None
        pooled[f"macro_{metric}_machine_count"] = len(values)
    return pooled


def evaluate_machine(
    machine: str, seed: int = 42, train_fraction: float = 0.7,
    ordering_policy: str = "lexicographic",
) -> dict | None:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler

    if ordering_policy != "lexicographic":
        raise ValueError("Only lexicographic ordering is supported; validated acquisition timestamps are unavailable.")
    if not np.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be finite and strictly between zero and one.")
    normal_files, anomaly_files = collect_files(machine)
    if len(normal_files) < 20:
        print(f"  [{machine}] too few normal files ({len(normal_files)}), skip")
        return None
    print(f"\n=== {machine}: {len(normal_files)} normal, {len(anomaly_files)} anomalous ===")

    # Feature extraction
    t0 = time.perf_counter()
    X_norm = build_feature_matrix(normal_files)
    X_anom = build_feature_matrix(anomaly_files)
    print(f"  Features: {X_norm.shape[0]} normal x {X_norm.shape[1]} dims, "
          f"{X_anom.shape[0]} anomalous  ({time.perf_counter()-t0:.1f}s)")

    # Path-ordered holdout, not a temporal/prospective validation claim.
    n_train = int(len(X_norm) * train_fraction)
    if not 0 < n_train < len(X_norm):
        raise ValueError("The split must leave normal recordings in both roles.")
    X_train = X_norm[:n_train]
    X_test_normal = X_norm[n_train:]
    X_test_anom = X_anom
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
    if not np.isfinite(train_scores).all():
        raise ValueError("Nonfinite training scores.")
    threshold = float(np.percentile(train_scores, 95))

    # Predict
    test_scores = -det.decision_function(X_test)
    if not np.isfinite(test_scores).all():
        raise ValueError("Nonfinite test scores.")
    y_pred = (test_scores >= threshold).astype(int)
    elapsed = time.perf_counter() - t0

    # Metrics
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    decision_metrics = count_metrics(tp, fp, tn, fn)

    from sklearn.metrics import roc_auc_score, average_precision_score
    roc = float(roc_auc_score(y_test, test_scores)) if len(np.unique(y_test)) == 2 else None
    pr = float(average_precision_score(y_test, test_scores)) if len(np.unique(y_test)) == 2 else None

    def recording_id(path):
        return path.relative_to(DATA_DIR).as_posix()

    records = []
    for role, paths, labels, scores in (
        ("train", normal_files[:n_train], np.zeros(n_train, dtype=int), train_scores),
        ("test", normal_files[n_train:] + anomaly_files, y_test, test_scores),
    ):
        for path, label, score in zip(paths, labels, scores, strict=True):
            records.append({"recording_id": recording_id(path), "machine": machine,
                            "seed": seed, "role": role, "label": int(label),
                            "score": float(score), "threshold": threshold,
                            "prediction": int(score >= threshold)})

    result = {
        "machine": machine,
        "metric_scope": "single_machine",
        "ordering_policy": ordering_policy,
        "n_train_normal": int(n_train),
        "n_test_normal": int(len(X_test_normal)),
        "n_test_anom": int(len(X_test_anom)),
        **decision_metrics,
        "roc_auc": roc, "pr_auc": pr,
        "macro_roc_auc": None, "macro_pr_auc": None,
        "macro_roc_auc_machine_count": None, "macro_pr_auc_machine_count": None,
        "threshold": threshold,
        "fit_time_sec": elapsed,
        "_prediction_records": records,
        "_split_metadata": {"machine": machine, "seed": seed,
                            "ordering_policy": ordering_policy, "temporal_order_validated": False,
                            "requested_train_fraction": train_fraction,
                            "actual_train_fraction": n_train / len(normal_files),
                            "train_normal_ids": [recording_id(p) for p in normal_files[:n_train]],
                            "test_normal_ids": [recording_id(p) for p in normal_files[n_train:]],
                            "test_anomaly_ids": [recording_id(p) for p in anomaly_files],
                            "threshold": threshold, "threshold_source": "95th percentile of training-normal scores",
                            "threshold_comparison": ">="},
    }
    print(f"  F1={result['f1']:.3f}  P={result['precision']:.3f}  R={result['recall']:.3f}  "
          f"FAR={result['far']:.3f}  ROC={roc}  PR={pr}  ({elapsed:.1f}s)")
    return result


def main() -> int:
    global DATA_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--machines", default="M01,M02,M03",
                        help="Comma-separated list of machines.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=RESULTS_DIR / "bosch-corrected")
    parser.add_argument("--ordering-policy", choices=["lexicographic"], default="lexicographic",
                        help="Path order only; filename month/example number is not a validated acquisition timestamp.")
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    machines = [m.strip() for m in args.machines.split(",")]
    if any(not m for m in machines) or len(set(machines)) != len(machines):
        parser.error("Machine names must be nonempty and unique.")

    if not DATA_DIR.exists():
        print(f"ERROR: Bosch data not found at {DATA_DIR}")
        return 1
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("Use a new output directory; existing outputs are never overwritten.")

    rows: list[dict] = []
    skipped = []
    for m in machines:
        r = evaluate_machine(m, seed=args.seed, ordering_policy=args.ordering_policy)
        if r:
            rows.append(r)
        else:
            skipped.append(m)

    if not rows:
        print("No results produced.")
        return 1

    prediction_records = [record for row in rows for record in row["_prediction_records"]]
    split_metadata = {"seed": args.seed, "ordering_policy": args.ordering_policy,
                      "temporal_order_validated": False,
                      "ordering_note": "Lexicographic path order; no acquisition time is inferred from month or example number.",
                      "anomaly_selection": "All anomalous recordings held out; not a prospective temporal test.",
                      "recording_id_base": "relative to --data-dir",
                      "threshold_note": "In-sample normal 95th percentile, not an independent conformal calibration.",
                      "pooled_ranking_note": "Not computed: scores from separately fitted machine detectors have no validated common scale; macro AUC/AP are unweighted machine means.",
                      "machine_splits": [row["_split_metadata"] for row in rows],
                      "skipped_machines_fewer_than_20_normals": skipped}
    rows.append(aggregate_results(rows))
    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    out = args.output_dir / "bosch_results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in public_rows for k in row)))
        w.writeheader()
        w.writerows(public_rows)
    with (args.output_dir / "bosch_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prediction_records[0]))
        w.writeheader()
        w.writerows(prediction_records)
    (args.output_dir / "bosch_split_metadata.json").write_text(json.dumps(split_metadata, indent=2, allow_nan=False) + "\n")

    print(f"\n=== Summary (Bosch CNC Machining) ===")
    print(f"{'Machine':<10}{'F1':>8}{'P':>8}{'R':>8}{'FAR':>8}")
    for r in rows:
        print(f"{r['machine']:<10}{r['f1']:>8.3f}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['far']:>8.3f}")
    print("Ranking metrics: per-machine ROC-AUC/AP; POOLED has separate macro_* fields, no pooled raw-score AUC.")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
