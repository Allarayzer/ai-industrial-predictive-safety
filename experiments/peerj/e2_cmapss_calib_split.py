"""Real-data 3-component risk fusion on C-MAPSS FD001 with a TRAIN/CALIB/TEST
engine split (Phase 2 fix of run_experiment_e2_cmapss.py).

Why this script exists
----------------------
The shipped ``run_experiment_e2_cmapss.py`` calibrates the SLSQP fusion weights on
the same validation split it then evaluates on (val == test). This corrected runner
partitions FD001 engines into THREE disjoint groups, all split by engine unit
(leakage-safe at the engine level):

    train   (60% of engines): fit IsolationForest, LSTM-RUL, and the neural risk net
    calib   (20% of engines): fit the SLSQP fusion weights ONLY
    test    (20% of engines): evaluate every configuration

The component models never see calib or test engines during fitting; the fusion
weights never see test engines. Label: "fails within 30 cycles" (RUL <= 30, RUL
capped at 125). Components:
    R_anom : calibrated IsolationForest decision score
    R_RUL  : 1 - clip(RUL_pred / 125, 0, 1) from an LSTM RUL regressor
    R_NN   : feedforward net, cost-asymmetric (FN:FP = 10:1) on the binary label

Configs compared: R_anom only, R_RUL only, R_NN only, Equal weights, Calibrated.
Metrics on the held-out TEST engines: PR-AUC, ROC-AUC, MCC, median lead time (cycles).

Outputs (analysis/results/): experiment_e2_cmapss_calib_split.csv
    columns: seed, config, pr_auc, roc_auc, mcc, lead_time_cycles, weights
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ai_cta.anomaly_detector import IsolationForestDetector
from ai_cta.risk_model import RiskAggregator

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "benchmarks" / "data" / "cmapss"
RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"

COL_NAMES = (["unit", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)]
             + [f"sensor_{i}" for i in range(1, 22)])
SELECTED_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
                    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
                    "sensor_15", "sensor_17", "sensor_20", "sensor_21"]
RUL_MAX = 125.0
HORIZON = 30
WARNING_THRESHOLD = 0.3


def load_split(seed):
    df = pd.read_csv(DATA_DIR / "train_FD001.txt", sep=r"\s+", header=None,
                     names=COL_NAMES, engine="python")
    rng = np.random.default_rng(seed)
    units = df["unit"].unique().copy()
    rng.shuffle(units)
    n = len(units)
    n_tr, n_cal = int(n * 0.6), int(n * 0.2)
    tr_u, cal_u, te_u = units[:n_tr], units[n_tr:n_tr + n_cal], units[n_tr + n_cal:]
    g = lambda us: df[df["unit"].isin(us)].reset_index(drop=True)
    return g(tr_u), g(cal_u), g(te_u)


def compute_rul(df):
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    return np.minimum((max_cycle - df["cycle"]).to_numpy(dtype=np.float32), RUL_MAX)


def compute_labels(rul):
    return (rul <= HORIZON).astype(np.int32)


def fit_r_anom(train_df):
    det = IsolationForestDetector(window_size=16, stride=1, use_spectral=False, random_state=0)
    det.fit(train_df[SELECTED_SENSORS])
    ref = det.decision_function(train_df[SELECTED_SENSORS])
    pctl70 = float(np.percentile(ref, 70))
    scale = float(ref.std() + 1e-9)

    def predict(df):
        raw = det.decision_function(df[SELECTED_SENSORS])
        if len(raw) < len(df):
            raw = np.concatenate([np.full(len(df) - len(raw), raw[0]), raw])
        return expit(2.0 * (raw[:len(df)] - pctl70) / scale)
    return predict


def fit_r_rul(train_df, train_rul):
    import tensorflow as tf
    tf.random.set_seed(0)
    np.random.seed(0)
    X = train_df[SELECTED_SENSORS].to_numpy(dtype=np.float32)
    y = np.minimum(train_rul, RUL_MAX).astype(np.float32)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    window = 16
    n_win = len(Xs) - window + 1
    Xw = np.stack([Xs[i:i + window] for i in range(n_win)], axis=0)
    yw = y[window - 1:]
    inp = tf.keras.Input(shape=(window, len(SELECTED_SENSORS)))
    x = tf.keras.layers.LSTM(32, dropout=0.1)(inp)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(Xw, yw, epochs=20, batch_size=64, validation_split=0.15,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
              verbose=0)

    def predict(df):
        V = scaler.transform(df[SELECTED_SENSORS].to_numpy(dtype=np.float32)).astype(np.float32)
        # Batched sliding-window prediction (pad the head).
        wins = np.empty((len(V), window, len(SELECTED_SENSORS)), dtype=np.float32)
        for i in range(len(V)):
            start = max(0, i - window + 1)
            win = V[start:i + 1]
            if len(win) < window:
                win = np.concatenate([np.tile(win[0], (window - len(win), 1)), win], axis=0)
            wins[i] = win
        preds = model.predict(wins, batch_size=512, verbose=0).flatten()
        preds = np.clip(preds, 0.0, RUL_MAX)
        return np.clip(1.0 - preds / RUL_MAX, 0, 1)
    return predict


def fit_r_nn(train_df, train_labels):
    import tensorflow as tf
    tf.random.set_seed(1)
    np.random.seed(1)
    scaler = StandardScaler().fit(train_df[SELECTED_SENSORS])
    X_tr = scaler.transform(train_df[SELECTED_SENSORS]).astype(np.float32)
    inp = tf.keras.Input(shape=(len(SELECTED_SENSORS),))
    x = tf.keras.layers.Dense(64, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    model.fit(X_tr, train_labels.astype(np.float32), epochs=20, batch_size=64,
              validation_split=0.15, class_weight={0: 1.0, 1: 10.0},
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
              verbose=0)

    def predict(df):
        Xv = scaler.transform(df[SELECTED_SENSORS]).astype(np.float32)
        return model.predict(Xv, batch_size=512, verbose=0).flatten()
    return predict


def median_lead_time(risk, labels, units):
    leads = []
    d = pd.DataFrame({"unit": units, "risk": risk, "label": labels})
    for _, sub in d.groupby("unit"):
        sub = sub.reset_index(drop=True)
        fidx = sub.index[sub["label"] == 1]
        if len(fidx) == 0:
            continue
        first_fail = int(fidx[0])
        above = sub.index[sub["risk"] >= WARNING_THRESHOLD]
        first_warn = above[above <= first_fail]
        if len(first_warn):
            leads.append(int(first_fail - first_warn[0]))
    return float(np.median(leads)) if leads else 0.0


def evaluate(name, R_stack, w, y, units):
    R = R_stack @ w
    y_pred = (R >= WARNING_THRESHOLD).astype(int)
    pr = average_precision_score(y, R) if y.sum() else float("nan")
    roc = roc_auc_score(y, R) if y.sum() and (y == 0).sum() else float("nan")
    mcc = matthews_corrcoef(y, y_pred) if len(set(y_pred)) > 1 else 0.0
    lt = median_lead_time(R, y, units)
    return {"config": name, "pr_auc": round(float(pr), 4), "roc_auc": round(float(roc), 4),
            "mcc": round(float(mcc), 4), "lead_time_cycles": round(float(lt), 1),
            "weights": ",".join(f"{wi:.2f}" for wi in w)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in range(args.n_seeds):
        print(f"\n=== seed {seed} ===")
        train, calib, test = load_split(seed)
        tr_rul, cal_rul, te_rul = compute_rul(train), compute_rul(calib), compute_rul(test)
        tr_lab = compute_labels(tr_rul)
        cal_lab, te_lab = compute_labels(cal_rul), compute_labels(te_rul)
        print(f"  engines train/calib/test = "
              f"{train['unit'].nunique()}/{calib['unit'].nunique()}/{test['unit'].nunique()}; "
              f"rows {len(train)}/{len(calib)}/{len(test)}; "
              f"test pos {int(te_lab.sum())}/{len(te_lab)}")

        p_anom = fit_r_anom(train)
        p_rul = fit_r_rul(train, tr_rul)
        p_nn = fit_r_nn(train, tr_lab)

        # Component predictions on calib (for weight fitting) and test (for eval).
        cal_stack = np.stack([p_anom(calib), p_rul(calib), p_nn(calib)], axis=1)
        te_stack = np.stack([p_anom(test), p_rul(test), p_nn(test)], axis=1)
        te_units = test["unit"].to_numpy()

        rows = [
            evaluate("R_anom only", te_stack, np.array([1.0, 0, 0]), te_lab, te_units),
            evaluate("R_RUL only", te_stack, np.array([0, 1.0, 0]), te_lab, te_units),
            evaluate("R_NN only", te_stack, np.array([0, 0, 1.0]), te_lab, te_units),
            evaluate("Equal weights", te_stack, np.array([1/3, 1/3, 1/3]), te_lab, te_units),
        ]
        agg = RiskAggregator()
        agg.calibrate_weights(cal_stack[:, 0], cal_stack[:, 1], cal_stack[:, 2], cal_lab.astype(float))
        rows.append(evaluate("Calibrated (SLSQP)", te_stack, agg.w, te_lab, te_units))

        for r in rows:
            r["seed"] = seed
            all_rows.append(r)
            print(f"  {r['config']:<20} PR-AUC={r['pr_auc']:.3f} ROC={r['roc_auc']:.3f} "
                  f"MCC={r['mcc']:.3f} Lead={r['lead_time_cycles']:.1f}  w={r['weights']}")

    out = RESULTS_DIR / "experiment_e2_cmapss_calib_split.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    df = pd.DataFrame(all_rows)
    agg_df = df.groupby("config")[["pr_auc", "roc_auc", "mcc", "lead_time_cycles"]].agg(
        ["mean", "std"]).round(3)
    print("\n=== FD001 fusion, held-out test engines (mean +/- std) ===")
    print(agg_df.to_string())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
