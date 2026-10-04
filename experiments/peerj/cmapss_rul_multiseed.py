"""Multi-seed C-MAPSS RUL evaluation (council Tier-1 fix for the single-seed table).

Re-runs the same five RUL methods as run_cmapss_rul_normalized.py across N seeds, so
every reported number carries mean +/- SD, and the LSTM's RMSE advantage over the
classical baselines is tested with a paired Wilcoxon signed-rank test (paired by seed,
Cliff's delta effect size, Holm correction). All baselines are re-run here under the
identical split/normalization/protocol — these are paired observations, not numbers
copied from the literature.

Outputs (analysis/results/):
  cmapss_rul_multiseed.csv        per (seed, subset, method): rmse, mae, score, alpha_lambda
  cmapss_rul_multiseed_summary.md mean+-SD table + Wilcoxon LSTM-MSE/quantile vs baselines
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))
import run_cmapss_rul_normalized as R  # noqa: E402  (helpers + constants)

RES = pathlib.Path(__file__).resolve().parent / "results"
SENS = R.SELECTED_SENSORS
RUL_MAX = R.RUL_MAX


def lstm_mse_seeded(train, train_rul, test, epochs, window_size, seed):
    import tensorflow as tf
    tf.random.set_seed(seed); np.random.seed(seed)
    X = train[SENS].to_numpy(dtype=np.float32)
    y = np.minimum(train_rul, RUL_MAX).astype(np.float32)
    n_win = len(X) - window_size + 1
    Xs = np.stack([X[i:i + window_size] for i in range(n_win)], axis=0)
    ys = y[window_size - 1:]
    d = len(SENS)
    inp = tf.keras.Input(shape=(window_size, d))
    x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inp)
    x = tf.keras.layers.LSTM(32, dropout=0.2)(x)
    out = tf.keras.layers.Dense(1)(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    m.fit(Xs, ys, epochs=epochs, batch_size=64, validation_split=0.15,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
          verbose=0)
    units = sorted(test["unit"].unique())
    preds = np.zeros(len(units), dtype=np.float32)
    for i, u in enumerate(units):
        traj = test[test["unit"] == u][SENS].to_numpy(dtype=np.float32)
        if len(traj) < window_size:
            traj = np.concatenate([np.tile(traj[0], (window_size - len(traj), 1)), traj], axis=0)
        preds[i] = np.clip(float(m.predict(traj[-window_size:][None, ...], verbose=0).flatten()[0]),
                           0.0, RUL_MAX)
    return preds


def lstm_quantile_seeded(train, train_rul, test, epochs, window_size, seed):
    from ai_cta.rul_estimator import RULEstimator
    est = RULEstimator(window_size=window_size, quantiles=(0.1, 0.5, 0.9),
                       rul_max=RUL_MAX, epochs=epochs, random_state=seed)
    est.fit(train[SENS], train_rul)
    units = sorted(test["unit"].unique())
    preds = np.zeros(len(units), dtype=np.float32)
    for i, u in enumerate(units):
        traj = test[test["unit"] == u][SENS]
        if len(traj) < window_size:
            pad = pd.DataFrame(np.tile(traj.iloc[0].values, (window_size - len(traj), 1)), columns=SENS)
            traj = pd.concat([pad, traj], ignore_index=True)
        q = est.predict_quantiles(traj)
        preds[i] = float(np.clip(q[0.5][-1], 0.0, RUL_MAX))
    return preds


def methods_for_seed(seed):
    return [
        ("Linear Regression", lambda tr, y, te, ep, ws: R.fit_predict_pointwise(lambda: LinearRegression(), tr, y, te)),
        ("Random Forest", lambda tr, y, te, ep, ws: R.fit_predict_pointwise(
            lambda: RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1), tr, y, te)),
        ("Gradient Boosting", lambda tr, y, te, ep, ws: R.fit_predict_pointwise(
            lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=seed), tr, y, te)),
        ("LSTM (MSE)", lambda tr, y, te, ep, ws: lstm_mse_seeded(tr, y, te, ep, ws, seed)),
        ("Proposed (LSTM quantile)", lambda tr, y, te, ep, ws: lstm_quantile_seeded(tr, y, te, ep, ws, seed)),
    ]


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum(x > y for x in a for y in b); lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--window-size", type=int, default=32)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    subsets = [s.strip() for s in args.subsets.split(",")]

    out = RES / "cmapss_rul_multiseed.csv"
    fields = ["subset", "seed", "method", "rmse", "mae", "score", "alpha_lambda", "sec"]
    # Resume: skip (subset, seed, method) already present, write incrementally.
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = {(r.subset, int(r.seed), r.method) for r in prev.itertuples()}
        print(f"Resuming: {len(done)} runs already complete.")
    f = out.open("a", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    if not done:
        writer.writeheader(); f.flush()

    for subset in subsets:
        train, test, rul_true = R.load_train_test(subset)
        n_clusters = R.SUBSET_N_CLUSTERS.get(subset, 1)
        train_norm, test_norm = R.operational_normalize(train, test, n_clusters)
        train_rul = R.compute_rul(train_norm)
        for seed in range(args.seeds):
            for name, fn in methods_for_seed(seed):
                if (subset, seed, name) in done:
                    continue
                t0 = time.perf_counter()
                preds = fn(train_norm, train_rul, test_norm, args.epochs, args.window_size)
                d = preds - rul_true
                row = {
                    "subset": subset, "seed": seed, "method": name,
                    "rmse": round(float(np.sqrt((d**2).mean())), 3),
                    "mae": round(float(np.abs(d).mean()), 3),
                    "score": round(R.cmapss_score(d), 1),
                    "alpha_lambda": round(R.alpha_lambda_accuracy(preds, rul_true, 0.2), 3),
                    "sec": round(time.perf_counter() - t0, 1),
                }
                writer.writerow(row); f.flush()
                print(f"{subset} seed={seed} {name:<24} RMSE={row['rmse']:.2f} "
                      f"score={row['score']:.0f}", flush=True)
    f.close()

    df = pd.read_csv(out)
    lines = ["# Multi-seed C-MAPSS RUL (mean +/- SD over seeds)\n",
             f"Seeds: {args.seeds}; epochs: {args.epochs}. Baselines re-run under the identical "
             "split/normalization protocol; Wilcoxon is paired by seed (LSTM vs each classical), "
             "Cliff's delta effect size, Holm-corrected within each subset.\n"]
    for subset in subsets:
        sub = df[df["subset"] == subset]
        lines.append(f"\n## {subset}\n")
        lines.append("| Method | RMSE | MAE | Score | alpha-lambda |")
        lines.append("|---|---:|---:|---:|---:|")
        for m in sub["method"].unique():
            s = sub[sub["method"] == m]
            lines.append(f"| {m} | {s['rmse'].mean():.2f} ± {s['rmse'].std():.2f} "
                         f"| {s['mae'].mean():.2f} ± {s['mae'].std():.2f} "
                         f"| {s['score'].mean():.0f} ± {s['score'].std():.0f} "
                         f"| {s['alpha_lambda'].mean():.2f} ± {s['alpha_lambda'].std():.2f} |")
        # Wilcoxon: each LSTM variant vs the 3 classical baselines on RMSE (lower better)
        lines.append("\n_Paired Wilcoxon vs classical baselines (RMSE, paired by seed):_\n")
        lines.append("| Comparison | p (Wilcoxon) | Holm p | Cliff's d |")
        lines.append("|---|---:|---:|---:|")
        for ref in ["LSTM (MSE)", "Proposed (LSTM quantile)"]:
            refv = sub[sub["method"] == ref].sort_values("seed")["rmse"].values
            comps, ps = [], []
            for base in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
                bv = sub[sub["method"] == base].sort_values("seed")["rmse"].values
                try:
                    _, p = wilcoxon(refv, bv, alternative="less")  # ref RMSE < baseline
                except ValueError:
                    p = 1.0
                comps.append((ref, base, p, cliffs_delta(refv, bv))); ps.append(p)
            hp = holm(ps)
            for i, (r, b, p, d) in enumerate(comps):
                lines.append(f"| {r} < {b} | {p:.4g} | {hp[i]:.4g} | {d:+.2f} |")
    (RES / "cmapss_rul_multiseed_summary.md").write_text("\n".join(lines))
    print(f"\nWrote {out} and summary ({len(df)} rows).")


if __name__ == "__main__":
    main()
