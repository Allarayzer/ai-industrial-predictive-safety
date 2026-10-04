"""Cost-ratio sensitivity for the FD001 fusion (revision experiment, GPT-P6).

The manuscript fixes the false-negative:false-positive cost ratio at 10:1 in both the
neural-risk loss and the SLSQP fusion objective. A reviewer will ask how sensitive the
conclusions are to that choice. This experiment reruns the paper's FD001 configuration
(R_anom + point-estimate R_RUL + R_NN, three-way engine split) with the ratio swept over
{2:1, 5:1, 10:1, 20:1}: the NN is retrained with class_weight {0:1, 1:ratio} and the
SLSQP objective uses RiskAggregator(cost_fn=ratio, cost_fp=1).

Reported per seed and ratio: R_NN only and Calibrated (SLSQP) — PR-AUC, ROC-AUC, MCC,
lead time, weights, plus FAR at the 0.3 warning threshold. Incremental append + resume
on (seed, ratio, config).

Output (analysis/results/): e2_cost_ratio_sensitivity.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np
import pandas as pd

import e2_cmapss_calib_split as base
from ai_cta.risk_model import RiskAggregator

OUT = base.RESULTS_DIR / "e2_cost_ratio_sensitivity.csv"
FIELDS = ["seed", "ratio", "config", "pr_auc", "roc_auc", "mcc",
          "lead_time_cycles", "far", "weights"]
RATIOS = (2, 5, 10, 20)


def fit_r_nn_ratio(train_df, train_labels, ratio):
    import tensorflow as tf
    tf.random.set_seed(1)
    np.random.seed(1)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_df[base.SELECTED_SENSORS])
    X_tr = scaler.transform(train_df[base.SELECTED_SENSORS]).astype(np.float32)
    inp = tf.keras.Input(shape=(len(base.SELECTED_SENSORS),))
    x = tf.keras.layers.Dense(64, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    model.fit(X_tr, train_labels.astype(np.float32), epochs=20, batch_size=64,
              validation_split=0.15, class_weight={0: 1.0, 1: float(ratio)},
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
              verbose=0)

    def predict(df):
        Xv = scaler.transform(df[base.SELECTED_SENSORS]).astype(np.float32)
        return model.predict(Xv, batch_size=512, verbose=0).flatten()
    return predict


def far_at_threshold(risk, labels, thr=base.WARNING_THRESHOLD):
    normal = labels == 0
    if normal.sum() == 0:
        return float("nan")
    return float((risk[normal] >= thr).mean())


def evaluate_with_far(name, stack, w, y, units):
    r = base.evaluate(name, stack, w, y, units)
    r["far"] = round(far_at_threshold(stack @ w, y), 4)
    return r


def done_keys():
    if not OUT.exists():
        return set()
    df = pd.read_csv(OUT)
    return {(int(r.seed), int(r.ratio), str(r.config)) for r in df.itertuples()}


def append_row(row):
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()
    base.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = done_keys()
    print(f"resume: {len(done)} rows present", flush=True)

    for seed in range(args.n_seeds):
        need = [(seed, r, c) for r in RATIOS
                for c in ("R_NN only", "Calibrated (SLSQP)")
                if (seed, r, c) not in done]
        if not need:
            print(f"seed {seed}: complete, skip", flush=True)
            continue
        print(f"\n=== seed {seed} ===", flush=True)
        train, calib, test = base.load_split(seed)
        tr_rul = base.compute_rul(train)
        cal_lab = base.compute_labels(base.compute_rul(calib))
        te_lab = base.compute_labels(base.compute_rul(test))
        te_units = test["unit"].to_numpy()

        p_anom = base.fit_r_anom(train)
        p_rul = base.fit_r_rul(train, tr_rul)
        anom_cal, anom_te = p_anom(calib), p_anom(test)
        rul_cal, rul_te = p_rul(calib), p_rul(test)

        for ratio in RATIOS:
            if all((seed, ratio, c) in done for c in ("R_NN only", "Calibrated (SLSQP)")):
                continue
            p_nn = fit_r_nn_ratio(train, base.compute_labels(tr_rul), ratio)
            nn_cal, nn_te = p_nn(calib), p_nn(test)
            cal_stack = np.stack([anom_cal, rul_cal, nn_cal], axis=1)
            te_stack = np.stack([anom_te, rul_te, nn_te], axis=1)

            rows = []
            if (seed, ratio, "R_NN only") not in done:
                rows.append(evaluate_with_far("R_NN only", te_stack,
                                              np.array([0, 0, 1.0]), te_lab, te_units))
            if (seed, ratio, "Calibrated (SLSQP)") not in done:
                agg = RiskAggregator(cost_fn=float(ratio), cost_fp=1.0)
                agg.calibrate_weights(cal_stack[:, 0], cal_stack[:, 1], cal_stack[:, 2],
                                      cal_lab.astype(float))
                rows.append(evaluate_with_far("Calibrated (SLSQP)", te_stack, agg.w,
                                              te_lab, te_units))
            for r in rows:
                r.update(seed=seed, ratio=ratio)
                append_row(r)
                print(f"  [{ratio:>2}:1] {r['config']:<20} MCC={r['mcc']:.3f} "
                      f"FAR={r['far']:.3f} w={r['weights']}", flush=True)

    df = pd.read_csv(OUT)
    print("\n=== mean ± std by ratio & config ===")
    print(df.groupby(["ratio", "config"])[["pr_auc", "mcc", "far", "lead_time_cycles"]]
            .agg(["mean", "std"]).round(3).to_string())
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
