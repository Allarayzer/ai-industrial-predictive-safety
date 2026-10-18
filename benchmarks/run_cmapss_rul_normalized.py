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
experiments/maintained/results/cmapss_rul_normalized_<subset>.csv plus
experiments/maintained/results/cmapss_rul_normalized_all_subsets.csv. The aggregate
includes all per-subset files in the selected output directory. Use
``--aggregate-only`` to rebuild it without loading data or evaluating models;
``--output-dir`` selects an isolated directory for a new run.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

DATA_DIR = pathlib.Path(__file__).parent / "data" / "cmapss"
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "maintained" / "results"

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
RESULT_FIELDS = [
    "subset", "method", "rmse", "mae", "score", "alpha_lambda",
    "fit_predict_time_sec", "n_test_engines", "n_op_clusters",
]


def load_train_test(
    subset: str, data_dir: pathlib.Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    data_dir = DATA_DIR if data_dir is None else pathlib.Path(data_dir)
    train = pd.read_csv(data_dir / f"train_{subset}.txt",
                        sep=r"\s+", header=None, names=COL_NAMES, engine="python")
    test = pd.read_csv(data_dir / f"test_{subset}.txt",
                       sep=r"\s+", header=None, names=COL_NAMES, engine="python")
    rul_true = np.loadtxt(data_dir / f"RUL_{subset}.txt").astype(np.float32)
    rul_true = np.minimum(rul_true, RUL_MAX)
    return train, test, rul_true


def compute_rul(df: pd.DataFrame) -> np.ndarray:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    rul = (max_cycle - df["cycle"]).to_numpy(dtype=np.float32)
    return np.minimum(rul, RUL_MAX)


def split_engine_validation(train, validation_fraction=0.15, seed=42):
    """Reserve whole engines before fitting preprocessing or making windows."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    units = train["unit"].unique().copy()
    if len(units) < 2:
        raise ValueError("Need at least two engines for disjoint validation.")
    np.random.default_rng(seed).shuffle(units)
    n_val = min(len(units) - 1, max(1, int(np.ceil(len(units) * validation_fraction))))
    mask = train["unit"].isin(units[:n_val])
    return train.loc[~mask].copy(), train.loc[mask].copy()


def make_engine_windows(frame, targets, window_size):
    """Build chronological windows and endpoint targets inside each unit."""
    targets = np.asarray(targets, dtype=np.float32)
    if targets.shape != (len(frame),):
        raise ValueError("targets must have one value per row.")
    values = frame[SELECTED_SENSORS].to_numpy(dtype=np.float32)
    windows, endpoints = [], []
    for positions in frame.groupby("unit", sort=False).indices.values():
        positions = positions[np.argsort(frame.iloc[positions]["cycle"].to_numpy())]
        for end in range(window_size, len(positions) + 1):
            window = positions[end - window_size:end]
            windows.append(values[window])
            endpoints.append(targets[window[-1]])
    if not windows:
        raise ValueError(f"No engine has at least {window_size} cycles.")
    return np.stack(windows), np.asarray(endpoints, dtype=np.float32)


def training_roles(train, train_rul, validation_data=None, seed=42):
    """Resolve disjoint early-stopping data while preserving target alignment."""
    if validation_data is not None:
        validation, val_rul = validation_data
        if set(train.unit) & set(validation.unit):
            raise ValueError("Training and validation engine IDs must be disjoint.")
        return train, np.asarray(train_rul), validation, np.asarray(val_rul)
    frame = train.reset_index(drop=True)
    fit, validation = split_engine_validation(frame, seed=seed)
    targets = np.asarray(train_rul)
    return fit, targets[fit.index], validation, targets[validation.index]


def normalize_roles(train, test, n_clusters, seed=42):
    """Fit normalization only on fitting engines; apply it to val and test."""
    fit, validation = split_engine_validation(train, seed=seed)
    held_out = pd.concat([validation, test], ignore_index=True)
    fit_norm, held_norm = operational_normalize(fit, held_out, n_clusters, seed)
    return (fit_norm, held_norm.iloc[:len(validation)].copy(),
            held_norm.iloc[len(validation):].copy())


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
def finite_clip_rul(predictions, rul_max=RUL_MAX):
    """Reject invalid raw model outputs before clipping finite RUL estimates."""
    values = np.asarray(predictions)
    if not np.isfinite(values).all():
        raise ValueError("Raw RUL predictions must be finite before clipping.")
    return np.clip(values, 0.0, rul_max)


def fit_predict_pointwise(model_factory, train: pd.DataFrame, train_rul: np.ndarray,
                          test: pd.DataFrame) -> np.ndarray:
    X = train[SELECTED_SENSORS].to_numpy()
    model = model_factory().fit(X, train_rul)
    last_rows = test.groupby("unit").tail(1)[SELECTED_SENSORS].to_numpy()
    preds = model.predict(last_rows)
    return finite_clip_rul(preds)


# --------- LSTM with MSE loss (the good baseline) ---------
def fit_predict_lstm_mse(train: pd.DataFrame, train_rul: np.ndarray,
                          test: pd.DataFrame, epochs: int, window_size: int,
                          validation_data=None, seed=42) -> np.ndarray:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    train, y, validation, val_y = training_roles(train, train_rul, validation_data, seed)
    Xs, ys = make_engine_windows(train, np.minimum(y, RUL_MAX), window_size)
    Xv, yv = make_engine_windows(validation, np.minimum(val_y, RUL_MAX), window_size)

    d = len(SELECTED_SENSORS)
    inp = tf.keras.Input(shape=(window_size, d))
    x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inp)
    x = tf.keras.layers.LSTM(32, dropout=0.2)(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(Xs, ys, epochs=epochs, batch_size=64, validation_data=(Xv, yv),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
              verbose=0)

    # Predict last window per test engine.
    units = sorted(test["unit"].unique())
    preds = np.zeros(len(units), dtype=np.float32)
    for i, u in enumerate(units):
        traj = test[test["unit"] == u].sort_values("cycle")[SELECTED_SENSORS].to_numpy(dtype=np.float32)
        if len(traj) < window_size:
            pad = np.tile(traj[0], (window_size - len(traj), 1))
            traj = np.concatenate([pad, traj], axis=0)
        win = traj[-window_size:][None, ...]
        p = float(model.predict(win, verbose=0).flatten()[0])
        preds[i] = finite_clip_rul(p)
    return preds


# --------- LSTM quantile ensemble ---------
def fit_predict_lstm_ensemble(train: pd.DataFrame, train_rul: np.ndarray,
                               test: pd.DataFrame, epochs: int, window_size: int,
                               validation_data=None, seed=42) -> np.ndarray:
    from ai_cta.rul_estimator import RULEstimator

    train, train_rul, validation, val_rul = training_roles(train, train_rul, validation_data, seed)
    est = RULEstimator(window_size=window_size, quantiles=(0.1, 0.5, 0.9),
                       rul_max=RUL_MAX, epochs=epochs, random_state=seed)
    est.fit(train[SELECTED_SENSORS], train_rul, groups=train["unit"].to_numpy(),
            validation_data=(validation[SELECTED_SENSORS], val_rul, validation["unit"].to_numpy()))
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
        preds[i] = float(finite_clip_rul(q[0.5][-1]))
    return preds


METHODS = [
    ("Linear Regression", lambda tr, y, te, ep, ws, **kw: fit_predict_pointwise(
        lambda: LinearRegression(), tr, y, te)),
    ("Random Forest", lambda tr, y, te, ep, ws, **kw: fit_predict_pointwise(
        lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), tr, y, te)),
    ("Gradient Boosting", lambda tr, y, te, ep, ws, **kw: fit_predict_pointwise(
        lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42), tr, y, te)),
    ("LSTM (MSE)", fit_predict_lstm_mse),
    ("Proposed Ensemble (LSTM quantile)", fit_predict_lstm_ensemble),
]


def evaluate_subset(subset: str, args) -> list[dict]:
    print(f"\n===== {subset} (with op-cond normalization) =====")
    try:
        train, test, rul_true = load_train_test(subset, getattr(args, "data_dir", None))
    except FileNotFoundError as exc:
        print(f"  Skipping: {exc}")
        return []

    n_clusters = SUBSET_N_CLUSTERS.get(subset, 1)
    print(f"  {subset}: n_clusters={n_clusters}")
    train_norm, validation_norm, test_norm = normalize_roles(train, test, n_clusters)
    train_rul = compute_rul(train_norm)  # RUL depends on cycle count, not sensor values
    val_rul = compute_rul(validation_norm)

    print(f"  Train: {len(train_norm)} rows, {train_norm['unit'].nunique()} engines")
    print(f"  Test: {len(test_norm)} rows, {len(rul_true)} engines")

    rows: list[dict] = []
    for name, fn in METHODS:
        print(f"  [{name}] ... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            preds = fn(train_norm, train_rul, test_norm, args.epochs, args.window_size,
                       validation_data=(validation_norm, val_rul))
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


def fit_predict_full_ensemble(
    train: pd.DataFrame, train_rul: np.ndarray, test: pd.DataFrame,
    epochs: int, window_size: int, validation_data=None, seed=42,
) -> np.ndarray:
    """Full 3-model ensemble per monograph § 9.6 (Models A + B + C).

    Model A — XGBoost multi-quantile (tabular features).
    Model B — LSTM quantile ensemble (sequential features).
    Model C — Weibull physics-guided fallback.

    Equal ensemble weights; disjoint validation engines control neural early
    stopping only. This harness does not perform ensemble-weight tuning.
    """
    from ai_cta.rul_ensemble import (
        PhysicsGuidedRUL,
        RULEnsemble,
        XGBoostQuantileRegressor,
    )
    from ai_cta.rul_estimator import RULEstimator

    train, train_rul, validation, val_rul = training_roles(train, train_rul, validation_data, seed)

    # ---- Feature engineering for Model A (flat tabular features) ----
    # Per engine, aggregate sensor rolling stats over last window_size cycles.
    def make_tabular(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_tabular, y_rul_per_sample, elapsed_cycles_per_sample)."""
        X_list, y_list, elapsed_list = [], [], []
        for _, g in df.groupby("unit"):
            g = g.sort_values("cycle").reset_index(drop=True)
            max_cyc = g["cycle"].max()
            for i in range(window_size, len(g) + 1):
                win = g.iloc[i - window_size : i][SELECTED_SENSORS].to_numpy()
                feats = np.concatenate([
                    win.mean(axis=0), win.std(axis=0),
                    win.min(axis=0), win.max(axis=0),
                    win[-1] - win[0],  # trend
                ])
                X_list.append(feats)
                cycle = g.iloc[i - 1]["cycle"]
                y_list.append(min(max_cyc - cycle, 125.0))
                elapsed_list.append(cycle)
        return np.array(X_list, dtype=np.float32), \
               np.array(y_list, dtype=np.float32), \
               np.array(elapsed_list, dtype=np.float32)

    print("    Preparing tabular features for Model A...")
    X_train_tab, y_train_tab, _ = make_tabular(train)

    # ---- Model A: XGBoost quantile ----
    print("    Fitting Model A (XGBoost quantile)...")
    model_a = XGBoostQuantileRegressor(
        quantiles=(0.1, 0.5, 0.9), n_estimators=300,
        max_depth=6, learning_rate=0.05,
    )
    model_a.fit(X_train_tab, y_train_tab)

    # ---- Model B: LSTM quantile (existing RULEstimator) ----
    print("    Fitting Model B (LSTM quantile)...")
    model_b = RULEstimator(
        window_size=window_size, quantiles=(0.1, 0.5, 0.9),
        rul_max=125.0, epochs=epochs, random_state=seed,
    )
    model_b.fit(train[SELECTED_SENSORS], train_rul, groups=train["unit"].to_numpy(),
                validation_data=(validation[SELECTED_SENSORS], val_rul, validation["unit"].to_numpy()))

    # ---- Model C: Weibull physics ----
    print("    Fitting Model C (Weibull physics)...")
    failure_times = train.groupby("unit")["cycle"].max().to_numpy()
    model_c = PhysicsGuidedRUL(rul_max=125.0)
    model_c.fit(failure_times)

    # ---- Ensemble (equal weights — no val tuning in this harness) ----
    print("    Assembling ensemble...")
    ensemble = RULEnsemble(
        quantiles=(0.1, 0.5, 0.9), rul_max=125.0, use_physics=True,
    )
    ensemble.fit(model_a, model_b, model_c)

    # ---- Per-engine last-window prediction on test ----
    units_test = sorted(test["unit"].unique())
    preds = np.zeros(len(units_test), dtype=np.float32)
    for i, u in enumerate(units_test):
        traj = test[test["unit"] == u].sort_values("cycle").reset_index(drop=True)
        if len(traj) < window_size:
            pad_rows = window_size - len(traj)
            pad = pd.DataFrame(
                np.tile(traj.iloc[0][SELECTED_SENSORS].to_numpy(), (pad_rows, 1)),
                columns=SELECTED_SENSORS,
            )
            tab_win = np.concatenate([pad.to_numpy(), traj[SELECTED_SENSORS].to_numpy()])
            sens_df = pd.concat([pad, traj[SELECTED_SENSORS]], ignore_index=True)
        else:
            tab_win = traj.iloc[-window_size:][SELECTED_SENSORS].to_numpy()
            sens_df = traj.iloc[-window_size:][SELECTED_SENSORS]
        feats = np.concatenate([
            tab_win.mean(axis=0), tab_win.std(axis=0),
            tab_win.min(axis=0), tab_win.max(axis=0),
            tab_win[-1] - tab_win[0],
        ])[None, ...]
        elapsed = float(traj["cycle"].max())
        q = ensemble.predict_quantiles(feats, sens_df, elapsed=np.array([elapsed]))
        preds[i] = float(finite_clip_rul(q[0.5][-1]))
    return preds


def _validate_result_rows(rows: list[dict], subset: str, source: str) -> None:
    """Reject malformed tables before replacing an existing result file."""
    if not rows:
        raise ValueError(f"{source}: no result rows.")
    seen = set()
    for line, row in enumerate(rows, start=2):
        where = f"{source}:{line}"
        if set(row) != set(RESULT_FIELDS) or any(value is None for value in row.values()):
            raise ValueError(f"{where}: malformed result row; expected {RESULT_FIELDS}.")
        if row["subset"] != subset:
            raise ValueError(f"{where}: subset must match filename ({subset}).")
        method = str(row["method"]).strip()
        if not method:
            raise ValueError(f"{where}: method is empty.")
        key = (subset, method)
        if key in seen:
            raise ValueError(f"{where}: duplicate subset/method {key}.")
        seen.add(key)
        for field in RESULT_FIELDS[2:]:
            try:
                value = float(row[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{where}: {field} must be numeric.") from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{where}: {field} must be finite and nonnegative.")
            if field == "alpha_lambda" and value > 1:
                raise ValueError(f"{where}: alpha_lambda must lie in [0, 1].")
            if field in ("n_test_engines", "n_op_clusters") and (value < 1 or not value.is_integer()):
                raise ValueError(f"{where}: {field} must be a positive integer.")


def _write_results_atomic(path: pathlib.Path, rows: list[dict]) -> None:
    """Replace one CSV only after the complete temporary file is closed."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = pathlib.Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def rebuild_aggregate(results_dir: pathlib.Path) -> pathlib.Path | None:
    """Validate and combine all per-subset CSVs without re-rounding values.

    A single-subset rerun replaces that subset's file, but the aggregate keeps
    the other subsets already on disk. Use a fresh output directory to keep
    independent runs separate. Validation failures leave the old aggregate
    untouched. Atomic replacement protects each file, not a multi-file run.
    """
    results_dir = pathlib.Path(results_dir)
    paths = sorted(results_dir.glob("cmapss_rul_normalized_fd*.csv"))
    if not paths:
        print("No per-subset normalized CSVs found; aggregate unchanged.")
        return None
    rows = []
    for path in paths:
        subset = path.stem.removeprefix("cmapss_rul_normalized_").upper()
        if subset not in SUBSET_N_CLUSTERS:
            raise ValueError(f"{path.name}: unrecognized subset filename.")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != RESULT_FIELDS:
                raise ValueError(f"{path.name}: schema mismatch; expected {RESULT_FIELDS}.")
            subset_rows = list(reader)
        _validate_result_rows(subset_rows, subset, path.name)
        rows.extend(subset_rows)
    output = results_dir / "cmapss_rul_normalized_all_subsets.csv"
    _write_results_atomic(output, rows)
    print(f"Aggregate: {len(rows)} rows from {len(paths)} subset files -> {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR,
                        help="Directory containing the C-MAPSS text files.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=RESULTS_DIR,
                        help="Per-subset and aggregate CSV directory; use a fresh directory for an independent run.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Validate and combine existing per-subset CSVs without model evaluation.")
    parser.add_argument("--full-ensemble", action="store_true",
                        help="Also evaluate the full 3-model ensemble from monograph § 9.6 "
                             "(XGBoost + LSTM + Weibull physics-guided).")
    args = parser.parse_args()

    if args.aggregate_only:
        try:
            return 0 if rebuild_aggregate(args.output_dir) is not None else 1
        except (ValueError, OSError) as exc:
            print(f"Aggregate failed: {exc}")
            return 1

    subsets = [subset.strip().upper() for subset in args.subsets.split(",")]
    if len(set(subsets)) != len(subsets) or any(subset not in SUBSET_N_CLUSTERS for subset in subsets):
        parser.error("--subsets must list distinct choices from FD001,FD002,FD003,FD004.")

    all_rows: list[dict] = []
    original_methods = METHODS.copy()
    try:
        if args.full_ensemble:
            METHODS.append(("Full Ensemble (A+B+C)", fit_predict_full_ensemble))
        for subset in subsets:
            all_rows.extend(evaluate_subset(subset, args))
    finally:
        METHODS[:] = original_methods

    if not all_rows:
        print("No results.")
        return 1

    subset_results = {subset: [row for row in all_rows if row["subset"] == subset]
                      for subset in sorted({row["subset"] for row in all_rows})}
    try:
        for subset, rows in subset_results.items():
            _validate_result_rows(rows, subset, f"{subset} evaluation")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for subset, rows in subset_results.items():
            _write_results_atomic(args.output_dir / f"cmapss_rul_normalized_{subset.lower()}.csv", rows)
        rebuild_aggregate(args.output_dir)
    except (ValueError, OSError) as exc:
        print(f"Writing results failed: {exc}")
        return 1

    print("\n=== Final table (operational-normalized) ===")
    print(f"{'Subset':<8}{'Method':<34}{'RMSE':>8}{'MAE':>8}{'Score':>10}{'alpha-lambda':>14}")
    for r in all_rows:
        print(f"{r['subset']:<8}{r['method']:<34}{r['rmse']:>8}{r['mae']:>8}"
              f"{r['score']:>10}{r['alpha_lambda']:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
