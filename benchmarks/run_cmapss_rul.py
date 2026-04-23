"""Experiment E3 (monograph § 13.7): RUL regression on NASA C-MAPSS.

Reproduces the five-method RUL comparison from Table 13.3:

    Linear Regression (baseline)
    Random Forest
    Gradient Boosting (median only, XGBoost-like)
    LSTM (median quantile only)
    Proposed Ensemble (LSTM multi-quantile via pinball loss)

Evaluated on all four subsets FD001-FD004. Metrics:

    RMSE        - root-mean-squared error (lower is better)
    MAE         - mean absolute error
    Score       - C-MAPSS asymmetric scoring function: penalises late
                  predictions (d > 0) more than early ones (d < 0)
    alpha_lambda - fraction of predictions within ±20% of true RUL on the
                  last 20% of engine life (Saxena & Goebel, 2010)

Usage
-----

    python benchmarks/run_cmapss_rul.py --subsets FD001,FD002
    python benchmarks/run_cmapss_rul.py --epochs 30   # finer LSTM training

Outputs
-------

    benchmarks/results/cmapss_rul_all_subsets.csv
    benchmarks/results/cmapss_rul_<subset>.csv  (per subset)
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ai_cta.rul_estimator import RULEstimator

DATA_DIR = pathlib.Path(__file__).parent / "data" / "cmapss"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

COL_NAMES = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Standard 14-sensor subset used in most C-MAPSS papers; drops the
# constant-valued channels that contribute only noise to learners.
SELECTED_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21",
]

RUL_MAX = 125.0  # Standard piecewise-linear RUL capping (Heimes 2008)


def load_train(subset: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a C-MAPSS training subset and compute per-timestep RUL."""
    path = DATA_DIR / f"train_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COL_NAMES, engine="python")
    # RUL = max cycle for this unit - current cycle, capped at RUL_MAX.
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    rul = (max_cycle - df["cycle"]).to_numpy(dtype=np.float32)
    rul = np.minimum(rul, RUL_MAX)
    return df, rul


def load_test(subset: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a C-MAPSS test subset and the ground-truth RUL at the last cycle."""
    df = pd.read_csv(
        DATA_DIR / f"test_{subset}.txt",
        sep=r"\s+", header=None, names=COL_NAMES, engine="python",
    )
    rul_true = np.loadtxt(DATA_DIR / f"RUL_{subset}.txt").astype(np.float32)
    rul_true = np.minimum(rul_true, RUL_MAX)
    return df, rul_true


def cmapss_score(d: np.ndarray) -> float:
    """Asymmetric C-MAPSS scoring function (Saxena et al., 2008).

    d = pred - true (per engine).
        d < 0  -> early prediction  -> exp(-d/13) - 1
        d >= 0 -> late prediction   -> exp(d/10) - 1
    """
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(s.sum())


def alpha_lambda_accuracy(
    pred: np.ndarray, true: np.ndarray, alpha: float = 0.2
) -> float:
    """Alpha-lambda accuracy at lambda=1 (last cycle).

    Fraction of predictions within [(1-alpha) * true, (1+alpha) * true].
    """
    if len(true) == 0:
        return float("nan")
    lower = (1.0 - alpha) * true
    upper = (1.0 + alpha) * true
    return float(((pred >= lower) & (pred <= upper)).mean())


# -------------------------------------------------------- classical baselines
def fit_predict_sklearn_pointwise(
    model_factory,
    train_df: pd.DataFrame,
    train_rul: np.ndarray,
    test_df: pd.DataFrame,
) -> np.ndarray:
    """Fit a pointwise (non-temporal) regressor, return per-engine last-cycle pred."""
    X_train = train_df[SELECTED_SENSORS].to_numpy()
    y_train = train_rul
    model = model_factory()
    model.fit(X_train, y_train)

    # For each test engine, predict on its last available row.
    last_rows = test_df.groupby("unit").tail(1)[SELECTED_SENSORS].to_numpy()
    preds = model.predict(last_rows)
    return np.clip(preds, 0.0, RUL_MAX)


def fit_predict_lstm_mse(
    train_df: pd.DataFrame,
    train_rul: np.ndarray,
    test_df: pd.DataFrame,
    epochs: int,
    window_size: int,
) -> np.ndarray:
    """Vanilla LSTM regressor trained with MSE loss — proper 'LSTM only'
    baseline. Uses Keras directly to avoid the multi-quantile
    RULEstimator machinery, so we get an apples-to-apples comparison
    of a standard deep-learning baseline against the proposed ensemble.
    """
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    tf.random.set_seed(42)
    np.random.seed(42)

    X = train_df[SELECTED_SENSORS].to_numpy(dtype=np.float32)
    y = np.minimum(train_rul, RUL_MAX).astype(np.float32)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X).astype(np.float32)

    # Build windows: (n_win, window_size, n_features), target = RUL at window end.
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
    model.fit(
        Xs, ys, epochs=epochs, batch_size=64, validation_split=0.15,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
        verbose=0,
    )

    # Predict last window per test engine.
    units = sorted(test_df["unit"].unique())
    preds = np.zeros(len(units), dtype=np.float32)
    for i, u in enumerate(units):
        traj = test_df[test_df["unit"] == u][SELECTED_SENSORS].to_numpy(dtype=np.float32)
        if len(traj) < window_size:
            pad = np.tile(traj[0], (window_size - len(traj), 1))
            traj = np.concatenate([pad, traj], axis=0)
        win = scaler.transform(traj[-window_size:])[None, ...]
        p = float(model.predict(win.astype(np.float32), verbose=0).flatten()[0])
        preds[i] = np.clip(p, 0.0, RUL_MAX)
    return preds


def fit_predict_lstm_ensemble(
    train_df: pd.DataFrame,
    train_rul: np.ndarray,
    test_df: pd.DataFrame,
    epochs: int,
    window_size: int,
) -> np.ndarray:
    """Proposed ensemble: LSTM trained with multi-quantile pinball loss.

    The 0.5 (median) quantile is used as the point estimate; the 0.1/0.9
    quantiles provide a confidence interval which the application layer
    can surface to operators (see monograph § 8.6).
    """
    est = RULEstimator(
        window_size=window_size,
        quantiles=(0.1, 0.5, 0.9),
        rul_max=RUL_MAX,
        epochs=epochs,
    )
    est.fit(train_df[SELECTED_SENSORS], train_rul)
    return _predict_last_per_engine(est, test_df, window_size)


def _predict_last_per_engine(
    est: RULEstimator, test_df: pd.DataFrame, window_size: int
) -> np.ndarray:
    """For each test engine, run RULEstimator over its trajectory and return
    the median prediction at the last available window.
    """
    units = sorted(test_df["unit"].unique())
    preds = np.zeros(len(units), dtype=np.float32)
    for i, u in enumerate(units):
        traj = test_df[test_df["unit"] == u][SELECTED_SENSORS]
        if len(traj) < window_size:
            # Too few cycles - pad with first row.
            pad = pd.DataFrame(
                np.tile(traj.iloc[0].values, (window_size - len(traj), 1)),
                columns=SELECTED_SENSORS,
            )
            traj = pd.concat([pad, traj], ignore_index=True)
        q = est.predict_quantiles(traj)
        median = q[0.5][-1]  # last window
        preds[i] = float(np.clip(median, 0.0, RUL_MAX))
    return preds


# ------------------------------------------------------------- main runner
METHODS = [
    ("Linear Regression", lambda td, tr, te, ep, ws: fit_predict_sklearn_pointwise(
        lambda: LinearRegression(), td, tr, te)),
    ("Random Forest", lambda td, tr, te, ep, ws: fit_predict_sklearn_pointwise(
        lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), td, tr, te)),
    ("Gradient Boosting (median)", lambda td, tr, te, ep, ws: fit_predict_sklearn_pointwise(
        lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42), td, tr, te)),
    ("LSTM (MSE)", fit_predict_lstm_mse),
    ("Proposed Ensemble (LSTM quantile)", fit_predict_lstm_ensemble),
]


def evaluate_subset(subset: str, args) -> list[dict]:
    print(f"\n===== {subset} =====")
    try:
        train_df, train_rul = load_train(subset)
        test_df, rul_true = load_test(subset)
    except FileNotFoundError as exc:
        print(f"  Skipping: {exc}")
        return []

    print(f"  Train: {len(train_df)} rows, {train_df['unit'].nunique()} engines")
    print(f"  Test : {len(test_df)} rows, {len(rul_true)} engines")

    results: list[dict] = []
    for name, fn in METHODS:
        print(f"  [{name}] ... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            preds = fn(train_df, train_rul, test_df, args.epochs, args.window_size)
        except Exception as exc:  # pragma: no cover
            print(f"FAILED ({exc})")
            continue
        elapsed = time.perf_counter() - t0

        d = preds - rul_true
        rmse = float(np.sqrt((d ** 2).mean()))
        mae = float(np.abs(d).mean())
        score = cmapss_score(d)
        al = alpha_lambda_accuracy(preds, rul_true, alpha=0.2)

        print(f"RMSE={rmse:.2f} MAE={mae:.2f} Score={score:.0f} alpha-lambda={al:.2f} ({elapsed:.1f}s)")
        results.append({
            "subset": subset, "method": name,
            "rmse": round(rmse, 2), "mae": round(mae, 2),
            "score": round(score, 1), "alpha_lambda": round(al, 3),
            "fit_predict_time_sec": round(elapsed, 1),
            "n_test_engines": len(rul_true),
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--subsets", default="FD001,FD002,FD003,FD004",
        help="Comma-separated list of C-MAPSS subsets.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=32)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for subset in args.subsets.split(","):
        all_rows.extend(evaluate_subset(subset.strip(), args))

    if not all_rows:
        print("No results.")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-subset.
    for subset in set(r["subset"] for r in all_rows):
        subset_rows = [r for r in all_rows if r["subset"] == subset]
        out = RESULTS_DIR / f"cmapss_rul_{subset.lower()}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(subset_rows[0].keys()))
            w.writeheader()
            w.writerows(subset_rows)

    # Combined.
    out_all = RESULTS_DIR / "cmapss_rul_all_subsets.csv"
    with out_all.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nSaved: {out_all}")
    print(f"\n{'Subset':<8}{'Method':<36}{'RMSE':>8}{'MAE':>8}{'Score':>10}{'alpha-lambda':>14}")
    for r in all_rows:
        print(f"{r['subset']:<8}{r['method']:<36}{r['rmse']:>8}{r['mae']:>8}{r['score']:>10}{r['alpha_lambda']:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
