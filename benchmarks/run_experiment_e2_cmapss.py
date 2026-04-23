"""Experiment E2 on REAL NASA C-MAPSS data (monograph § 13.6 alt).

Instead of the synthetic Dataset S1 used in run_experiment_e2.py, this
runner evaluates the three-component hybrid risk function on real
C-MAPSS FD001 data. The target label is "fails within 30 cycles"
(binary), which matches the operational scenario described in § 8.4
(alerts should precede the 24-hour threshold referenced there).

Components:
    R_anom : calibrated IsolationForestDetector decision score in [0, 1]
    R_RUL  : 1 - RUL_pred / RUL_max, from a median LSTM trained on the
             train split (RUL_pred < threshold => high R_RUL)
    R_NN   : logistic output of a feedforward network trained directly
             on the binary "fails within 30 cycles" target.

Configurations compared (same 5 as synthetic E2):
    R_anom only, R_RUL only, R_NN only, Equal weights, Calibrated (SLSQP)

Metrics: PR-AUC, ROC-AUC, MCC, Lead time (in cycles, not hours).
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
from sklearn.preprocessing import StandardScaler

from ai_cta.anomaly_detector import IsolationForestDetector
from ai_cta.risk_model import RiskAggregator

DATA_DIR = pathlib.Path(__file__).parent / "data" / "cmapss"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

COL_NAMES = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
SELECTED_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21",
]
RUL_MAX = 125.0
HORIZON = 30  # "fails within 30 cycles"
WARNING_THRESHOLD = 0.3


def load_and_split(seed: int) -> tuple:
    """Load FD001 train, split into (train_engines, val_engines) per seed."""
    df = pd.read_csv(
        DATA_DIR / "train_FD001.txt",
        sep=r"\s+", header=None, names=COL_NAMES, engine="python",
    )
    rng = np.random.default_rng(seed)
    units = df["unit"].unique()
    rng.shuffle(units)
    split = int(len(units) * 0.7)
    train_units, val_units = units[:split], units[split:]
    train = df[df["unit"].isin(train_units)].reset_index(drop=True)
    val = df[df["unit"].isin(val_units)].reset_index(drop=True)
    return train, val


def compute_rul(df: pd.DataFrame) -> np.ndarray:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    rul = (max_cycle - df["cycle"]).to_numpy(dtype=np.float32)
    return np.minimum(rul, RUL_MAX)


def compute_labels(rul: np.ndarray) -> np.ndarray:
    return (rul <= HORIZON).astype(np.int32)


def build_r_anom(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    """Fit IForest on train (all cycles as "normal"-ish window stream),
    return calibrated R_anom for val."""
    det = IsolationForestDetector(
        window_size=16, stride=1, use_spectral=False, random_state=0,
    )
    det.fit(train_df[SELECTED_SENSORS])
    raw = det.decision_function(val_df[SELECTED_SENSORS])
    if len(raw) < len(val_df):
        raw = np.concatenate([np.full(len(val_df) - len(raw), raw[0]), raw])
    # Calibrate to [0, 1] with a logistic squash around the 70th percentile
    # of training distribution (tunable, matches § 10.5 Platt-style calibration).
    ref = det.decision_function(train_df[SELECTED_SENSORS])
    pctl70 = float(np.percentile(ref, 70))
    scale = float(ref.std() + 1e-9)
    return expit(2.0 * (raw - pctl70) / scale)


def build_r_rul(train_df: pd.DataFrame, train_rul: np.ndarray,
                val_df: pd.DataFrame) -> np.ndarray:
    """LSTM RUL regressor using ai_cta.RULEstimator, single quantile 0.5.
    Returns 1 - clip(pred/RUL_MAX, 0, 1) -> higher => closer to failure.
    """
    import tensorflow as tf

    tf.random.set_seed(0)
    np.random.seed(0)

    # Simpler direct-keras LSTM for speed; same logic as run_cmapss_rul.py.
    X = train_df[SELECTED_SENSORS].to_numpy(dtype=np.float32)
    y = np.minimum(train_rul, RUL_MAX).astype(np.float32)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X).astype(np.float32)

    window = 16
    n_win = len(X) - window + 1
    Xs = np.stack([X[i:i + window] for i in range(n_win)], axis=0)
    ys = y[window - 1:]

    inp = tf.keras.Input(shape=(window, len(SELECTED_SENSORS)))
    x = tf.keras.layers.LSTM(32, dropout=0.1)(inp)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(
        Xs, ys, epochs=20, batch_size=64, validation_split=0.15,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0,
    )

    # Predict RUL for each val row (using a sliding window ending at that row).
    V = scaler.transform(val_df[SELECTED_SENSORS].to_numpy(dtype=np.float32)).astype(np.float32)
    preds = np.zeros(len(V), dtype=np.float32)
    for i in range(len(V)):
        start = max(0, i - window + 1)
        win = V[start:i + 1]
        if len(win) < window:
            pad = np.tile(win[0], (window - len(win), 1))
            win = np.concatenate([pad, win], axis=0)
        preds[i] = float(model.predict(win[None, ...], verbose=0).flatten()[0])
    preds = np.clip(preds, 0.0, RUL_MAX)
    # R_RUL: rises as RUL falls below horizon.
    return np.clip(1.0 - preds / RUL_MAX, 0, 1)


def build_r_nn(train_df: pd.DataFrame, train_labels: np.ndarray,
               val_df: pd.DataFrame) -> np.ndarray:
    """Simple feedforward network on raw sensor values with cost-asymmetric BCE.
    Trained directly on the binary 'fails within 30 cycles' label.
    """
    import tensorflow as tf

    tf.random.set_seed(1)
    np.random.seed(1)

    scaler = StandardScaler().fit(train_df[SELECTED_SENSORS])
    X_tr = scaler.transform(train_df[SELECTED_SENSORS]).astype(np.float32)
    X_va = scaler.transform(val_df[SELECTED_SENSORS]).astype(np.float32)

    inp = tf.keras.Input(shape=(len(SELECTED_SENSORS),))
    x = tf.keras.layers.Dense(64, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                   loss="binary_crossentropy")
    # Weighted: cost of FN is 10x cost of FP (per monograph § 8.4.2).
    class_weight = {0: 1.0, 1: 10.0}
    model.fit(
        X_tr, train_labels.astype(np.float32),
        epochs=20, batch_size=64, validation_split=0.15,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0,
    )
    return model.predict(X_va, verbose=0).flatten()


def median_lead_time_cycles(risk: np.ndarray, labels: np.ndarray,
                             units: np.ndarray) -> float:
    """Median cycles between first Warning (R >= 0.3) and failure per engine."""
    leads = []
    df = pd.DataFrame({"unit": units, "risk": risk, "label": labels})
    for u, sub in df.groupby("unit"):
        sub = sub.reset_index(drop=True)
        failure_idx = sub.index[sub["label"] == 1]
        if len(failure_idx) == 0:
            continue
        first_failure = int(failure_idx[0])
        above = sub.index[sub["risk"] >= WARNING_THRESHOLD]
        first_warning = above[above <= first_failure]
        if len(first_warning) > 0:
            leads.append(int(first_failure - first_warning[0]))
    return float(np.median(leads)) if leads else 0.0


def evaluate_config(name: str, R_stack: np.ndarray, w: np.ndarray,
                    y: np.ndarray, units: np.ndarray) -> dict:
    R = R_stack @ w
    y_pred = (R >= WARNING_THRESHOLD).astype(int)
    pr = average_precision_score(y, R) if y.sum() else float("nan")
    roc = roc_auc_score(y, R) if y.sum() and (y == 0).sum() else float("nan")
    mcc = matthews_corrcoef(y, y_pred) if len(set(y_pred)) > 1 else 0.0
    lt = median_lead_time_cycles(R, y, units)
    return {
        "config": name,
        "pr_auc": round(float(pr), 3),
        "roc_auc": round(float(roc), 3),
        "mcc": round(float(mcc), 3),
        "lead_time_cycles": round(float(lt), 1),
        "weights": ",".join(f"{wi:.2f}" for wi in w),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for seed in range(args.n_seeds):
        print(f"\n=== seed {seed} ===")
        train, val = load_and_split(seed)
        train_rul = compute_rul(train)
        val_rul = compute_rul(val)
        train_labels = compute_labels(train_rul)
        val_labels = compute_labels(val_rul)

        print(f"  Train rows: {len(train)}, val rows: {len(val)}")
        print(f"  Positive labels in val: {int(val_labels.sum())}/{len(val_labels)}")

        print("  Building R_anom...")
        r_anom = build_r_anom(train, val)
        r_anom = r_anom[:len(val)]
        print("  Building R_RUL (LSTM)...")
        r_rul = build_r_rul(train, train_rul, val)
        print("  Building R_NN (feedforward)...")
        r_nn = build_r_nn(train, train_labels, val)

        R_stack = np.stack([r_anom, r_rul, r_nn], axis=1)
        units = val["unit"].to_numpy()

        rows = [
            evaluate_config("R_anom only", R_stack, np.array([1.0, 0.0, 0.0]), val_labels, units),
            evaluate_config("R_RUL only", R_stack, np.array([0.0, 1.0, 0.0]), val_labels, units),
            evaluate_config("R_NN only", R_stack, np.array([0.0, 0.0, 1.0]), val_labels, units),
            evaluate_config("Equal weights", R_stack, np.array([1/3, 1/3, 1/3]), val_labels, units),
        ]
        # Calibrated
        agg = RiskAggregator()
        agg.calibrate_weights(r_anom, r_rul, r_nn, val_labels.astype(float))
        rows.append(evaluate_config("Calibrated (SLSQP)", R_stack, agg.w, val_labels, units))

        for r in rows:
            r["seed"] = seed
            all_rows.append(r)
            print(f"  seed={seed}  {r['config']:<22}  "
                  f"PR-AUC={r['pr_auc']:.3f}  ROC={r['roc_auc']:.3f}  "
                  f"MCC={r['mcc']:.3f}  Lead={r['lead_time_cycles']:.1f} cycles")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "experiment_e2_cmapss.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    df = pd.DataFrame(all_rows)
    agg_df = df.groupby("config")[["pr_auc", "roc_auc", "mcc", "lead_time_cycles"]].agg(
        ["mean", "std"]
    ).round(3)
    print("\n=== Aggregated on C-MAPSS FD001 (mean ± std) ===")
    print(agg_df.to_string())
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
