"""Maintained 2026-10-12 protocol, not a replay of frozen reference CSVs.

Real-data 3-component risk fusion on C-MAPSS FD001 with a TRAIN/CALIB/TEST
engine split (Phase 2 fix of run_experiment_e2_cmapss.py).

Why this script exists
----------------------
The shipped ``run_experiment_e2_cmapss.py`` calibrates the SLSQP fusion weights on
the same validation split it then evaluates on (val == test). This corrected runner
partitions FD001 engines into THREE disjoint groups, all split by engine unit
(leakage-safe at the engine level):

    train   (60% of engines): split again into 85% fit / 15% early-stop engines
    calib   (20% of engines): fit the SLSQP fusion weights ONLY
    test    (20% of engines): evaluate every configuration

The component models never fit preprocessing on early-stop/calib/test engines; the fusion
weights never see test engines. Label: "fails within 30 cycles" (RUL <= 30, RUL
capped at 125). Components:
    R_anom : calibrated IsolationForest decision score
    R_RUL  : 1 - clip(RUL_pred / 125, 0, 1) from an LSTM RUL regressor
    R_NN   : feedforward net, cost-asymmetric (FN:FP = 10:1) on the binary label

Configs compared: R_anom only, R_RUL only, R_NN only, Equal weights, Calibrated.
Metrics on the held-out TEST engines: PR-AUC, ROC-AUC, MCC, median lead time (cycles).

Outputs (experiments/maintained/results/): experiment_e2_cmapss_calib_split.csv
    columns: seed, config, pr_auc, roc_auc, mcc, lead_time_cycles, weights
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from ai_cta.preprocess import StatisticalFeatureExtractor
from ai_cta.risk_model import RiskAggregator

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import provenance  # noqa: E402
import run_cmapss_rul_normalized as rul_helpers  # noqa: E402

DATA_DIR = REPO / "benchmarks" / "data" / "cmapss"
RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"

COL_NAMES = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
SELECTED_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]
RUL_MAX = 125.0
HORIZON = 30
WARNING_THRESHOLD = 0.3


def load_split(seed):
    df = pd.read_csv(
        DATA_DIR / "train_FD001.txt", sep=r"\s+", header=None, names=COL_NAMES, engine="python"
    ).sort_values(["unit", "cycle"])
    rng = np.random.default_rng(seed)
    units = df["unit"].unique().copy()
    rng.shuffle(units)
    n = len(units)
    n_tr, n_cal = int(n * 0.6), int(n * 0.2)
    tr_u, cal_u, te_u = units[:n_tr], units[n_tr : n_tr + n_cal], units[n_tr + n_cal :]

    def g(us):
        return df[df["unit"].isin(us)].reset_index(drop=True)

    return g(tr_u), g(cal_u), g(te_u)


def compute_rul(df):
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    return np.minimum((max_cycle - df["cycle"]).to_numpy(dtype=np.float32), RUL_MAX)


def compute_labels(rul):
    return (rul <= HORIZON).astype(np.int32)


def window_inputs(df, window=16):
    """Return windows and their row endpoints; omit each engine's warm-up."""
    values = df[SELECTED_SENSORS].to_numpy(dtype=np.float32)
    windows, endpoints = [], []
    for positions in df.groupby("unit", sort=False).indices.values():
        positions = positions[np.argsort(df.iloc[positions]["cycle"].to_numpy())]
        for end in range(window, len(positions) + 1):
            indices = positions[end - window : end]
            windows.append(values[indices])
            endpoints.append(indices[-1])
    if not windows:
        raise ValueError(f"No engine has {window} cycles.")
    return np.stack(windows), np.asarray(endpoints, dtype=int)


def fit_r_anom(train_df, seed=0):
    stat = StatisticalFeatureExtractor(channels=SELECTED_SENSORS).fit(train_df[SELECTED_SENSORS])

    def features(df):
        windows, endpoints = window_inputs(df)
        matrix = np.vstack(
            [stat.transform(pd.DataFrame(w, columns=SELECTED_SENSORS)) for w in windows]
        )
        return matrix, endpoints

    matrix, _ = features(train_df)
    scaler = RobustScaler().fit(matrix)
    fitted = scaler.transform(matrix)
    detector = IsolationForest(n_estimators=200, contamination=0.05, random_state=seed).fit(fitted)
    raw = -detector.score_samples(fitted)
    mean, std = raw.mean(), raw.std() + 1e-9
    reference = expit((raw - mean) / std)
    pctl70, scale = np.percentile(reference, 70), reference.std() + 1e-9

    def predict(df):
        matrix, endpoints = features(df)
        raw = -detector.score_samples(scaler.transform(matrix))
        scores = expit((raw - mean) / std)
        result = np.full(len(df), np.nan)
        result[endpoints] = expit(2.0 * (scores - pctl70) / scale)
        return result

    return predict


def fit_r_rul(train_df, train_rul, validation_data, seed=0):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    validation, val_rul = validation_data
    if set(train_df.unit) & set(validation.unit):
        raise ValueError("Training and early-stopping engines must be disjoint.")
    scaler = StandardScaler().fit(train_df[SELECTED_SENSORS])

    def scaled_windows(df):
        scaled = df.copy()
        scaled[SELECTED_SENSORS] = scaler.transform(df[SELECTED_SENSORS])
        return window_inputs(scaled)

    Xw, endpoints = scaled_windows(train_df)
    Xv, val_endpoints = scaled_windows(validation)
    yw = np.minimum(train_rul, RUL_MAX)[endpoints]
    yv = np.minimum(val_rul, RUL_MAX)[val_endpoints]
    inp = tf.keras.Input(shape=(16, len(SELECTED_SENSORS)))
    x = tf.keras.layers.LSTM(32, dropout=0.1)(inp)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(
        Xw,
        yw,
        epochs=20,
        batch_size=64,
        validation_data=(Xv, yv),
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0,
    )

    def predict(df):
        wins, endpoints = scaled_windows(df)
        preds = rul_helpers.finite_clip_rul(
            model.predict(wins, batch_size=512, verbose=0).flatten(), RUL_MAX
        )
        result = np.full(len(df), np.nan)
        result[endpoints] = 1.0 - preds / RUL_MAX
        return result

    return predict


def fit_r_nn(train_df, train_labels, validation_data, seed=0):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    validation, val_labels = validation_data
    if set(train_df.unit) & set(validation.unit):
        raise ValueError("Training and early-stopping engines must be disjoint.")
    scaler = StandardScaler().fit(train_df[SELECTED_SENSORS])
    X_tr = scaler.transform(train_df[SELECTED_SENSORS]).astype(np.float32)
    X_val = scaler.transform(validation[SELECTED_SENSORS]).astype(np.float32)
    inp = tf.keras.Input(shape=(len(SELECTED_SENSORS),))
    x = tf.keras.layers.Dense(64, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    model.fit(
        X_tr,
        train_labels.astype(np.float32),
        epochs=20,
        batch_size=64,
        validation_data=(X_val, val_labels.astype(np.float32)),
        class_weight={0: 1.0, 1: 10.0},
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0,
    )

    def predict(df):
        Xv = scaler.transform(df[SELECTED_SENSORS]).astype(np.float32)
        return model.predict(Xv, batch_size=512, verbose=0).flatten()

    return predict


def median_lead_time(risk, labels, units, cycles=None):
    """Warning-to-observed-failure distance for complete run-to-failure units.

    The final observed cycle is the failure endpoint in train_FD001, not the
    first RUL<=30 label. This function is unsuitable for truncated test histories.
    Missing warnings are omitted; they must not be interpreted as zero lead.
    """
    leads = []
    d = pd.DataFrame({"unit": units, "risk": risk, "label": labels})
    d["cycle"] = d.groupby("unit").cumcount() if cycles is None else cycles
    for _, sub in d.groupby("unit"):
        failure_cycle = sub["cycle"].max()
        warning_cycles = sub.loc[sub["risk"] >= WARNING_THRESHOLD, "cycle"]
        if len(warning_cycles):
            leads.append(float(failure_cycle - warning_cycles.min()))
    return float(np.median(leads)) if leads else float("nan")


def valid_scores(stack, labels):
    stack, labels = np.asarray(stack, dtype=float), np.asarray(labels)
    if stack.ndim != 2 or stack.shape[1] != 3 or labels.shape != (len(stack),) or not len(stack):
        raise ValueError(
            "Risk predictions and labels must have aligned non-empty shapes (n,3)/(n,)."
        )
    if not np.isfinite(stack).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Risk predictions must be finite and labels binary.")
    if ((stack < 0) | (stack > 1)).any():
        raise ValueError("Risk predictions must lie in [0,1].")
    return stack, labels


def valid_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if (
        weights.shape != (3,)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError("Fusion weights must be three finite non-negative values summing to one.")
    return weights


def fit_fusion_weights(stack, labels):
    stack, labels = valid_scores(stack, labels)
    if len(np.unique(labels)) != 2:
        raise ValueError("Fusion calibration requires both label classes.")
    aggregator = RiskAggregator()
    aggregator.calibrate_weights(stack[:, 0], stack[:, 1], stack[:, 2], labels.astype(float))
    warning = getattr(aggregator, "last_optimization_warning_", None)
    if warning:
        raise RuntimeError(f"Fusion calibration optimization failed: {warning}")
    return valid_weights(aggregator.w)


def evaluate(name, R_stack, w, y, units, cycles=None):
    R_stack, y = valid_scores(R_stack, y)
    w = valid_weights(w)
    units = np.asarray(units)
    if units.shape != y.shape or (cycles is not None and np.asarray(cycles).shape != y.shape):
        raise ValueError("Engine IDs and cycles must align with predictions.")
    R = R_stack @ w
    y_pred = (R >= WARNING_THRESHOLD).astype(int)
    pr = average_precision_score(y, R) if y.sum() else float("nan")
    roc = roc_auc_score(y, R) if y.sum() and (y == 0).sum() else float("nan")
    mcc = matthews_corrcoef(y, y_pred) if len(set(y_pred)) > 1 else 0.0
    lt = median_lead_time(R, y, units, cycles)
    n_engines = len(np.unique(units))
    n_warned = len(np.unique(units[R >= WARNING_THRESHOLD]))
    return {
        "config": name,
        "pr_auc": float(pr),
        "roc_auc": float(roc),
        "mcc": float(mcc),
        "lead_time_cycles": float(lt),
        "weights": ",".join(repr(float(wi)) for wi in w),
        "n_engines": n_engines,
        "n_warned_engines": n_warned,
        "n_missed_engines": n_engines - n_warned,
        "warning_coverage": n_warned / n_engines,
    }


def main():
    global DATA_DIR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--output-dir", type=pathlib.Path, default=RESULTS_DIR)
    ap.add_argument("--data-dir", type=pathlib.Path, default=None)
    args = ap.parse_args()
    if args.n_seeds < 1:
        ap.error("--n-seeds must be positive")
    DATA_DIR = (args.data_dir or DATA_DIR).resolve()
    output_dir = args.output_dir.resolve()
    seeds = list(range(args.n_seeds))
    splits, roles = {}, {}
    for seed in seeds:
        train, calib, test = load_split(seed)
        train, early = rul_helpers.split_engine_validation(train, seed=seed)
        splits[seed] = (train, early, calib, test)
        roles[str(seed)] = {
            name: sorted(int(unit) for unit in frame.unit.unique())
            for name, frame in zip(("fit", "early", "calib", "test"), splits[seed], strict=True)
        }
        role_sets = [set(ids) for ids in roles[str(seed)].values()]
        if any(not ids for ids in role_sets) or sum(map(len, role_sets)) != len(
            set.union(*role_sets)
        ):
            raise ValueError(
                "Fit/early/calibration/test engine sets must be non-empty and disjoint."
            )
    name = "experiment_e2_cmapss_calib_split"
    provenance.prepare_run(
        output_dir,
        name,
        {**vars(args), "data_dir": str(DATA_DIR), "output_dir": str(output_dir)},
        {"train_FD001.txt": DATA_DIR / "train_FD001.txt"},
        provenance.source_files(REPO, __file__),
        seeds,
        roles,
    )
    out = output_dir / f"{name}.csv"
    fields = [
        "seed",
        "config",
        "pr_auc",
        "roc_auc",
        "mcc",
        "lead_time_cycles",
        "weights",
        "n_engines",
        "n_warned_engines",
        "n_missed_engines",
        "warning_coverage",
    ]
    configs = ["R_anom only", "R_RUL only", "R_NN only", "Equal weights", "Calibrated (SLSQP)"]
    all_rows = provenance.read_checkpoint(
        out,
        fields,
        ("seed", "config"),
        {(seed, config) for seed in seeds for config in configs},
        ("mcc", "n_engines", "n_warned_engines", "n_missed_engines", "warning_coverage"),
    )
    for seed in seeds:
        existing = [row for row in all_rows if row["seed"] == seed]
        if existing:
            if len(existing) != len(configs):
                raise ValueError("Partial E2 seed checkpoint; use a new --output-dir.")
            print(f"seed {seed}: completed under identical provenance, skipping")
            continue
        print(f"\n=== seed {seed} ===")
        train, early, calib, test = splits[seed]
        early_rul = compute_rul(early)
        tr_rul, cal_rul, te_rul = compute_rul(train), compute_rul(calib), compute_rul(test)
        tr_lab = compute_labels(tr_rul)
        cal_lab, te_lab = compute_labels(cal_rul), compute_labels(te_rul)
        print(
            f"  engines fit/early/calib/test = "
            f"{train['unit'].nunique()}/{early['unit'].nunique()}/"
            f"{calib['unit'].nunique()}/{test['unit'].nunique()}; "
            f"rows {len(train)}/{len(calib)}/{len(test)}; "
            f"test pos {int(te_lab.sum())}/{len(te_lab)}"
        )

        p_anom = fit_r_anom(train, seed=seed)
        p_rul = fit_r_rul(train, tr_rul, (early, early_rul), seed=seed)
        p_nn = fit_r_nn(train, tr_lab, (early, compute_labels(early_rul)), seed=seed)

        # Component predictions on calib (for weight fitting) and test (for eval).
        def predictions(frame, predictors=(p_anom, p_rul, p_nn)):
            columns = [np.asarray(predict(frame)) for predict in predictors]
            if any(column.shape != (len(frame),) for column in columns):
                raise ValueError("Each component must return one score per input row.")
            return np.column_stack(columns)

        cal_stack, te_stack = predictions(calib), predictions(test)
        # Compare every channel on the same available endpoints, after warm-up.
        cal_valid = calib.groupby("unit").cumcount().to_numpy() >= 15
        te_valid = test.groupby("unit").cumcount().to_numpy() >= 15
        if not np.isfinite(cal_stack[cal_valid]).all() or not np.isfinite(te_stack[te_valid]).all():
            raise ValueError("Non-finite predictions at an available evaluation endpoint.")
        cal_stack, cal_lab = cal_stack[cal_valid], cal_lab[cal_valid]
        te_stack, te_lab = te_stack[te_valid], te_lab[te_valid]
        te_units = test["unit"].to_numpy()[te_valid]
        te_cycles = test["cycle"].to_numpy()[te_valid]

        rows = [
            evaluate("R_anom only", te_stack, np.array([1.0, 0, 0]), te_lab, te_units, te_cycles),
            evaluate("R_RUL only", te_stack, np.array([0, 1.0, 0]), te_lab, te_units, te_cycles),
            evaluate("R_NN only", te_stack, np.array([0, 0, 1.0]), te_lab, te_units, te_cycles),
            evaluate(
                "Equal weights",
                te_stack,
                np.array([1 / 3, 1 / 3, 1 / 3]),
                te_lab,
                te_units,
                te_cycles,
            ),
        ]
        weights = fit_fusion_weights(cal_stack, cal_lab)
        rows.append(evaluate("Calibrated (SLSQP)", te_stack, weights, te_lab, te_units, te_cycles))
        provenance.write_json(
            output_dir / f"{name}.seed{seed}.settings.json",
            {
                "seed": seed,
                "component_seeds": {"isolation_forest": seed, "rul": seed, "nn": seed},
                "optimizer_status": "success",
                "fusion_weights": weights.tolist(),
                "warning_threshold": WARNING_THRESHOLD,
                "horizon": HORIZON,
            },
        )

        for r in rows:
            r["seed"] = seed
            all_rows.append(r)
            print(
                f"  {r['config']:<20} PR-AUC={r['pr_auc']:.3f} ROC={r['roc_auc']:.3f} "
                f"MCC={r['mcc']:.3f} Lead={r['lead_time_cycles']:.1f}  w={r['weights']}"
            )
        provenance.write_checkpoint(out, fields, all_rows)

    df = pd.DataFrame(all_rows)
    agg_df = (
        df.groupby("config")[["pr_auc", "roc_auc", "mcc", "lead_time_cycles", "warning_coverage"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print("\n=== FD001 fusion, held-out test engines (mean +/- std) ===")
    print(agg_df.to_string())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
