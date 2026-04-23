"""RUL regression on NASA C-MAPSS with operational-condition normalization.

Extends run_cmapss_rul.py with per-operational-cluster normalization
(the standard trick for multi-regime subsets FD002/FD004). The
algorithm follows Saxena et al. (2008) and the survey of Ramasso &
Saxena (2014):

    1. Cluster rows by (op_setting_1, op_setting_2, op_setting_3) using
       KMeans. Number of clusters = 1 for FD001/FD003 (single regime)
       and 6 for FD002/FD004.
    2. Fit a per-cluster StandardScaler on the sensor columns.
    3. At training / inference time, look up each row's cluster and
       apply the matching scaler. This removes the operational-regime
       offset that otherwise dominates the raw sensor distributions.
    4. Train / evaluate any downstream regressor on the normalized
       sensors as usual.

Re-evaluates the same 5 methods as run_cmapss_rul.py and writes
benchmarks/results/cmapss_rul_normalized_<subset>.csv plus
benchmarks/results/cmapss_rul_normalized_all_subsets.csv.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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
OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]
RUL_MAX = 125.0
SUBSET_N_CLUSTERS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def load_train_test(subset: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    train = pd.read_csv(DATA_DIR / f"train_{subset}.txt",
                        sep=r"\s+", header=None, names=COL_NAMES, engine="python")
    test = pd.read_csv(DATA_DIR / f"test_{subset}.txt",
                       sep=r"\s+", header=None, names=COL_NAMES, engine="python")
    rul_true = np.loadtxt(DATA_DIR / f"RUL_{subset}.txt").astype(np.float32)
    rul_true = np.minimum(rul_true, RUL_MAX)
    return train, test, rul_true


def compute_rul(df: pd.DataFrame) -> np.ndarray:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    rul = (max_cycle - df["cycle"]).to_numpy(dtype=np.float32)
    return np.minimum(rul, RUL_MAX)


def operational_normalize(
    train: pd.DataFrame, test: pd.DataFrame, n_clusters: int, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster rows by op_setting, then per-cluster z-score the sensors."""
    if n_clusters <= 1:
        # Single-regime subset: one global scaler.
        scaler = StandardScaler().fit(train[SELECTED_SENSORS])
        train_norm = train.copy()
        test_norm = test.copy()
        train_norm[SELECTED_SENSORS] = scaler.transform(train[SELECTED_SENSORS])
        test_norm[SELECTED_SENSORS] = scaler.transform(test[SELECTED_SENSORS])
        return train_norm, test_norm

    # Multi-regime: fit KMeans on op_settings using train, apply to both.
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(train[OP_SETTINGS])
    train_labels = km.predict(train[OP_SETTINGS])
    test_labels = km.predict(test[OP_SETTINGS])

    # Force float64 dtype on sensor columns so per-cluster scaled values fit.
    train_norm = train.copy()
    test_norm = test.copy()
    train_norm[SELECTED_SENSORS] = train_norm[SELECTED_SENSORS].astype(np.float64)
    test_norm[SELECTED_SENSORS] = test_norm[SELECTED_SENSORS].astype(np.float64)
    for c in range(n_clusters):
        mask_tr = train_labels == c
        if mask_tr.sum() < 2:
            continue
        scaler = StandardScaler().fit(train.loc[mask_tr, SELECTED_SENSORS])
        train_norm.loc[mask_tr, SELECTED_SENSORS] = scaler.transform(
            train.loc[mask_tr, SELECTED_SENSORS]
        )
        mask_te = test_labels == c
        if mask_te.sum() > 0:
            test_norm.loc[mask_te, SELECTED_SENSORS] = scaler.transform(
                test.loc[mask_te, SELECTED_SENSORS]
            )
    return train_norm, test_norm


def cmapss_score(d: np.ndarray) -> float:
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(s.sum())


def alpha_lambda_accuracy(pred: np.ndarray, true: np.ndarray, alpha: float = 0.2) -> float:
    if len(true) == 0:
        return float("nan")
    lower = (1.0 - alpha) * true
    upper = (1.0 + alpha) * true
    return float(((pred >= lower) & (pred <= upper)).mean())


# --------- pointwise (sklearn) predictors ---------
def fit_predict_pointwise(model_factory, train: pd.DataFrame, train_rul: np.ndarray,
                          test: pd.DataFrame) -> np.ndarray:
    X = train[SELECTED_SENSORS].to_numpy()
    model = model_factory().fit(X, train_rul)
    last_rows = test.groupby("unit").tail(1)[SELECTED_SENSORS].to_numpy()
    preds = model.predict(last_rows)
    return np.clip(preds, 0.0, RUL_MAX)


# --------- LSTM with MSE loss (the good baseline) ---------
def fit_predict_lstm_mse(train: pd.DataFrame, train_rul: np.ndarray,
                          test: pd.DataFrame, epochs: int, window_size: int) -> np.ndarray:
    import tensorflow as tf

    tf.random.set_seed(42); np.random.seed(42)
    X = train[SELECTED_SENSORS].to_numpy(dtype=np.float32)
    y = np.minimum(train_rul, RUL_MAX).astype(np.float32)

    n_win = len(X) - window_size + 1
    Xs = np.stack([X[i:i + window_size] for i in range(n_win)], axis=0)
    ys = y[window_size - 1:]

    d = len(SELECTED_SENSORS)
    inp = tf.keras.Input(shape=(window_size, d))
    x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inp)
    x = tf.keras.layers.LSTM(32, dropout=0.2)(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(Xs, ys, epochs=epochs, batch_size=64, validation_split=0.15,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
              verbose=0)

    # Predict last window per test engine.
    units = sorted(test["unit"].unique())
    preds = np.zeros(len(units), dtype=np.float32)
    for i, u in enumerate(units):
        traj = test[test["unit"] == u][SELECTED_SENSORS].to_numpy(dtype=np.float32)
        if len(traj) < window_size:
            pad = np.tile(traj[0], (window_size - len(traj), 1))
            traj = np.concatenate([pad, traj], axis=0)
        win = traj[-window_size:][None, ...]
        p = float(model.predict(win, verbose=0).flatten()[0])
        preds[i] = np.clip(p, 0.0, RUL_MAX)
    return preds


# --------- LSTM quantile ensemble ---------
def fit_predict_lstm_ensemble(train: pd.DataFrame, train_rul: np.ndarray,
                               test: pd.DataFrame, epochs: int, window_size: int) -> np.ndarray:
    from ai_cta.rul_estimator import RULEstimator

    est = RULEstimator(window_size=window_size, quantiles=(0.1, 0.5, 0.9),
                       rul_max=RUL_MAX, epochs=epochs)
    est.fit(train[SELECTED_SENSORS], train_rul)
    units = sorted(test["unit"].unique())
    preds = np.zeros(len(units), dtype=np.float32)
    for i, u in enumerate(units):
        traj = test[test["unit"] == u][SELECTED_SENSORS]
        if len(traj) < window_size:
            pad = pd.DataFrame(
                np.tile(traj.iloc[0].values, (window_size - len(traj), 1)),
                columns=SELECTED_SENSORS,
            )
            traj = pd.concat([pad, traj], ignore_index=True)
        q = est.predict_quantiles(traj)
        preds[i] = float(np.clip(q[0.5][-1], 0.0, RUL_MAX))
    return preds


METHODS = [
    ("Linear Regression", lambda tr, y, te, ep, ws: fit_predict_pointwise(
        lambda: LinearRegression(), tr, y, te)),
    ("Random Forest", lambda tr, y, te, ep, ws: fit_predict_pointwise(
        lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), tr, y, te)),
    ("Gradient Boosting", lambda tr, y, te, ep, ws: fit_predict_pointwise(
        lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42), tr, y, te)),
    ("LSTM (MSE)", fit_predict_lstm_mse),
    ("Proposed Ensemble (LSTM quantile)", fit_predict_lstm_ensemble),
]


def evaluate_subset(subset: str, args) -> list[dict]:
    print(f"\n===== {subset} (with op-cond normalization) =====")
    try:
        train, test, rul_true = load_train_test(subset)
    except FileNotFoundError as exc:
        print(f"  Skipping: {exc}")
        return []

    n_clusters = SUBSET_N_CLUSTERS.get(subset, 1)
    print(f"  {subset}: n_clusters={n_clusters}")
    train_norm, test_norm = operational_normalize(train, test, n_clusters)
    train_rul = compute_rul(train_norm)  # RUL depends on cycle count, not sensor values

    print(f"  Train: {len(train_norm)} rows, {train_norm['unit'].nunique()} engines")
    print(f"  Test: {len(test_norm)} rows, {len(rul_true)} engines")

    rows: list[dict] = []
    for name, fn in METHODS:
        print(f"  [{name}] ... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            preds = fn(train_norm, train_rul, test_norm, args.epochs, args.window_size)
        except Exception as exc:  # pragma: no cover
            print(f"FAILED ({exc})")
            continue
        elapsed = time.perf_counter() - t0

        d = preds - rul_true
        rmse = float(np.sqrt((d ** 2).mean()))
        mae = float(np.abs(d).mean())
        score = cmapss_score(d)
        al = alpha_lambda_accuracy(preds, rul_true, alpha=0.2)
        print(f"RMSE={rmse:.2f} MAE={mae:.2f} Score={score:.0f} "
              f"alpha-lambda={al:.2f} ({elapsed:.1f}s)")
        rows.append({
            "subset": subset, "method": name,
            "rmse": round(rmse, 2), "mae": round(mae, 2),
            "score": round(score, 1), "alpha_lambda": round(al, 3),
            "fit_predict_time_sec": round(elapsed, 1),
            "n_test_engines": len(rul_true),
            "n_op_clusters": n_clusters,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=32)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for subset in args.subsets.split(","):
        all_rows.extend(evaluate_subset(subset.strip(), args))

    if not all_rows:
        print("No results.")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for subset in set(r["subset"] for r in all_rows):
        subset_rows = [r for r in all_rows if r["subset"] == subset]
        out = RESULTS_DIR / f"cmapss_rul_normalized_{subset.lower()}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(subset_rows[0].keys()))
            w.writeheader()
            w.writerows(subset_rows)

    out_all = RESULTS_DIR / "cmapss_rul_normalized_all_subsets.csv"
    with out_all.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n=== Final table (operational-normalized) ===")
    print(f"{'Subset':<8}{'Method':<34}{'RMSE':>8}{'MAE':>8}{'Score':>10}{'alpha-lambda':>14}")
    for r in all_rows:
        print(f"{r['subset']:<8}{r['method']:<34}{r['rmse']:>8}{r['mae']:>8}"
              f"{r['score']:>10}{r['alpha_lambda']:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
