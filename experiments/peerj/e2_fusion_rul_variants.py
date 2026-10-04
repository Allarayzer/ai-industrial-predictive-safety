"""FD001 3-component fusion with THREE R_RUL variants (revision experiment, GPT-P3).

Reviewer-anticipated question: the architecture advertises a quantile-LSTM horizon risk
as the R_RUL component, yet the headline FD001 fusion (e2_cmapss_calib_split.py) feeds
the simpler point-estimate risk 1 - RUL_hat/125 from an MSE LSTM. This experiment closes
that gap by running the SAME train/calib/test engine split with three R_RUL variants:

    mse-point    : clip(1 - RUL_hat/125, 0, 1) from the MSE LSTM     (paper's original)
    quantile     : risk_at_horizon = P(RUL <= 30) interpolated from a pinball-loss
                   quantile head (levels 0.1/0.5/0.9), same backbone as mse-point;
                   linear ramp below the lowest quantile, saturation at one above the
                   highest (matches the manuscript's Methods description)
    quantile-cqr : the same quantile predictor conformalized on the calibration engines
                   (CQR margin on the [0.1, 0.9] interval at alpha = 0.2), then the same
                   horizon interpolation on the adjusted quantiles

R_anom and R_NN are fit once per seed and shared across variants. Per variant we report
R_RUL only, Equal weights, and Calibrated (SLSQP); the shared R_anom-only / R_NN-only
rows are tagged variant='shared'. Incremental append + resume on (seed, variant, config).

Output (analysis/results/): e2_fusion_rul_variants.csv
    columns: seed, rul_variant, config, pr_auc, roc_auc, mcc, lead_time_cycles, weights
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib

import numpy as np
import pandas as pd

import e2_cmapss_calib_split as base
from ai_cta.risk_model import RiskAggregator

RESULTS_DIR = base.RESULTS_DIR
OUT = RESULTS_DIR / "e2_fusion_rul_variants.csv"
FIELDS = ["seed", "rul_variant", "config", "pr_auc", "roc_auc", "mcc",
          "lead_time_cycles", "weights"]
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
CQR_ALPHA = 0.2  # matches the [0.1, 0.9] central interval


def fit_quantile_rul(train_df, train_rul):
    """Same backbone as base.fit_r_rul (window-16 LSTM(32)) with a 3-quantile pinball head.

    Returns predict(df) -> (n, 3) quantile values in level order, clipped to [0, RUL_MAX].
    """
    import tensorflow as tf
    tf.random.set_seed(0)
    np.random.seed(0)
    from sklearn.preprocessing import StandardScaler
    q = tf.constant(np.array(QUANTILE_LEVELS, dtype=np.float32))

    def pinball(y_true, y_pred):
        e = tf.expand_dims(y_true, -1) - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))

    X = train_df[base.SELECTED_SENSORS].to_numpy(dtype=np.float32)
    y = np.minimum(train_rul, base.RUL_MAX).astype(np.float32)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    window = 16
    n_win = len(Xs) - window + 1
    Xw = np.stack([Xs[i:i + window] for i in range(n_win)], axis=0)
    yw = y[window - 1:]
    inp = tf.keras.Input(shape=(window, len(base.SELECTED_SENSORS)))
    x = tf.keras.layers.LSTM(32, dropout=0.1)(inp)
    out = tf.keras.layers.Dense(len(QUANTILE_LEVELS))(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=pinball)
    model.fit(Xw, yw, epochs=20, batch_size=64, validation_split=0.15,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
              verbose=0)

    def predict_quantiles(df):
        V = scaler.transform(df[base.SELECTED_SENSORS].to_numpy(dtype=np.float32)).astype(np.float32)
        wins = np.empty((len(V), window, len(base.SELECTED_SENSORS)), dtype=np.float32)
        for i in range(len(V)):
            start = max(0, i - window + 1)
            win = V[start:i + 1]
            if len(win) < window:
                win = np.concatenate([np.tile(win[0], (window - len(win), 1)), win], axis=0)
            wins[i] = win
        preds = model.predict(wins, batch_size=512, verbose=0)
        return np.clip(preds, 0.0, base.RUL_MAX)
    return predict_quantiles


def horizon_risk(V, h=float(base.HORIZON), levels=QUANTILE_LEVELS):
    """P(RUL <= h) by piecewise-linear interpolation of the quantile CDF in level order,
    with a linear ramp below the lowest quantile and saturation at one at/above the
    highest — mirroring the manuscript's risk_at_horizon description."""
    v1, v2, v3 = V[:, 0], V[:, 1], V[:, 2]
    q1, q2, q3 = levels
    seg_hi = q2 + (q3 - q2) * (h - v2) / np.maximum(v3 - v2, 1e-9)
    seg_lo = q1 + (q2 - q1) * (h - v1) / np.maximum(v2 - v1, 1e-9)
    ramp = q1 * np.clip(h / np.maximum(v1, 1e-9), 0.0, 1.0)
    p = np.where(h >= v3, 1.0, np.where(h >= v2, seg_hi, np.where(h >= v1, seg_lo, ramp)))
    return np.clip(p, 0.0, 1.0)


def cqr_margin(cal_quantiles, cal_rul, alpha=CQR_ALPHA):
    """Split-conformal (CQR) margin for the [q_lo, q_hi] interval on calibration engines."""
    y = np.minimum(cal_rul, base.RUL_MAX).astype(np.float64)
    scores = np.maximum(cal_quantiles[:, 0] - y, y - cal_quantiles[:, 2])
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))


def adjust_cqr(V, margin):
    W = V.copy()
    W[:, 0] = np.clip(V[:, 0] - margin, 0.0, base.RUL_MAX)
    W[:, 2] = np.clip(V[:, 2] + margin, 0.0, base.RUL_MAX)
    return W


def done_keys():
    if not OUT.exists():
        return set()
    df = pd.read_csv(OUT)
    return {(int(r.seed), str(r.rul_variant), str(r.config)) for r in df.itertuples()}


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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = done_keys()
    print(f"resume: {len(done)} rows already present", flush=True)

    for seed in range(args.n_seeds):
        variants_missing = any((seed, v, c) not in done
                               for v in ("shared", "mse-point", "quantile", "quantile-cqr")
                               for c in ("R_anom only", "R_NN only", "R_RUL only",
                                         "Equal weights", "Calibrated (SLSQP)"))
        if not variants_missing:
            print(f"seed {seed}: complete, skip", flush=True)
            continue
        print(f"\n=== seed {seed} ===", flush=True)
        train, calib, test = base.load_split(seed)
        tr_rul, cal_rul, te_rul = (base.compute_rul(train), base.compute_rul(calib),
                                   base.compute_rul(test))
        tr_lab = base.compute_labels(tr_rul)
        cal_lab, te_lab = base.compute_labels(cal_rul), base.compute_labels(te_rul)
        te_units = test["unit"].to_numpy()

        p_anom = base.fit_r_anom(train)
        p_nn = base.fit_r_nn(train, tr_lab)
        anom_cal, anom_te = p_anom(calib), p_anom(test)
        nn_cal, nn_te = p_nn(calib), p_nn(test)

        # Shared single-channel rows (variant-independent).
        for name, col in (("R_anom only", anom_te), ("R_NN only", nn_te)):
            if (seed, "shared", name) in done:
                continue
            stack = np.stack([col, np.zeros_like(col), np.zeros_like(col)], axis=1)
            r = base.evaluate(name, stack, np.array([1.0, 0, 0]), te_lab, te_units)
            r.update(seed=seed, rul_variant="shared")
            append_row(r)
            print(f"  {name:<22} MCC={r['mcc']:.3f}", flush=True)

        # RUL variants.
        p_rul_point = base.fit_r_rul(train, tr_rul)
        pq = fit_quantile_rul(train, tr_rul)
        q_cal, q_te = pq(calib), pq(test)
        margin = cqr_margin(q_cal, cal_rul)
        print(f"  CQR margin = {margin:.2f} cycles", flush=True)

        variant_channels = {
            "mse-point": (p_rul_point(calib), p_rul_point(test)),
            "quantile": (horizon_risk(q_cal), horizon_risk(q_te)),
            "quantile-cqr": (horizon_risk(adjust_cqr(q_cal, margin)),
                             horizon_risk(adjust_cqr(q_te, margin))),
        }
        for variant, (rul_cal, rul_te) in variant_channels.items():
            cal_stack = np.stack([anom_cal, rul_cal, nn_cal], axis=1)
            te_stack = np.stack([anom_te, rul_te, nn_te], axis=1)
            rows = []
            if (seed, variant, "R_RUL only") not in done:
                rows.append(base.evaluate("R_RUL only", te_stack,
                                          np.array([0, 1.0, 0]), te_lab, te_units))
            if (seed, variant, "Equal weights") not in done:
                rows.append(base.evaluate("Equal weights", te_stack,
                                          np.array([1/3, 1/3, 1/3]), te_lab, te_units))
            if (seed, variant, "Calibrated (SLSQP)") not in done:
                agg = RiskAggregator()
                agg.calibrate_weights(cal_stack[:, 0], cal_stack[:, 1], cal_stack[:, 2],
                                      cal_lab.astype(float))
                rows.append(base.evaluate("Calibrated (SLSQP)", te_stack, agg.w,
                                          te_lab, te_units))
            for r in rows:
                r.update(seed=seed, rul_variant=variant)
                append_row(r)
                print(f"  [{variant:<12}] {r['config']:<20} PR={r['pr_auc']:.3f} "
                      f"MCC={r['mcc']:.3f} w={r['weights']}", flush=True)

    df = pd.read_csv(OUT)
    print("\n=== mean +/- std by variant & config ===")
    print(df.groupby(["rul_variant", "config"])[["pr_auc", "roc_auc", "mcc"]]
            .agg(["mean", "std"]).round(3).to_string())
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
