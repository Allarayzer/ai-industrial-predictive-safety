"""W4 fix: randomized-shift robustness for the controlled location-shift benchmark.

The designed benchmark fixes onset (6,000), drift coefficients (0.85/0.40/0.50), ramp (700)
and recurrence period (800). Here 30 configurations are drawn at random from wide ranges that
were NOT used when designing the policies — onset U[4000, 8500], magnitude multiplier
U[0.5, 2.0], ramp U[200, 2000], period U[300, 1500], shape in {sudden, gradual, recurring,
mixed} — and the policy ranking is re-examined. Everything else (channels, cadence/missing/age,
calibration, delay 120, policy constants) matches the paper exactly, reusing the archived
generator/threshold code that e4_cmapss_rerun.py reproduced bit-identically.

Outputs analysis2/results/randomized_shift_results.csv (+ ranking summary; → Supplementary
Table S10).
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = pathlib.Path(__file__).resolve().parents[2]
RERUN = ROOT / "analysis2" / "results" / "rerun_cmapss"
OUT = ROOT / "analysis2" / "results"

spec = importlib.util.spec_from_file_location("p4bench", RERUN / "_patched_p4bench.py")
bench = importlib.util.module_from_spec(spec)
sys.modules["p4bench"] = bench
spec.loader.exec_module(bench)


def build_random(seed: int, cfg: dict, n: int = 12000):
    rng = np.random.default_rng(seed); cal_end = 2500
    onset = cfg["onset"]; mag = cfg["magnitude"]
    t = np.arange(n); regimes = ((t // 800) % 2).astype(int)
    base = np.arange(cal_end + 600, n - 200, 850)
    failures = base + rng.integers(-120, 121, size=len(base))
    dist = bench._nearest_failure(n, failures); horizon = 160
    labels = ((dist >= 0) & (dist <= horizon)).astype(int)
    prox = np.clip(1 - dist / horizon, 0, 1); prox[~np.isfinite(dist)] = 0
    drift = np.zeros(n)
    shape = cfg["shape"]
    if shape == "sudden":
        drift[onset:] = 1
    elif shape == "gradual":
        drift[onset:] = np.minimum((t[onset:] - onset) / cfg["ramp"], 1)
    elif shape == "recurring":
        drift[onset:] = (((t[onset:] - onset) // cfg["period"]) % 2 == 0).astype(float)
    elif shape == "mixed":  # half-step jump + slow ramp on top
        drift[onset:] = 0.5 + 0.5 * np.minimum((t[onset:] - onset) / cfg["ramp"], 1)
    comp = np.column_stack([
        expit(-3.0 + 2.7 * prox + .35 * regimes + .85 * mag * drift + rng.normal(0, .35, n)),
        expit(-3.3 + 4.3 * prox + .15 * regimes + .40 * mag * drift + rng.normal(0, .30, n)),
        expit(-3.1 + 3.8 * prox + .25 * regimes + .50 * mag * drift + rng.normal(0, .32, n)),
    ])
    w = np.array([.25, .35, .40])
    async_score, _ = bench.asynchronous_array(comp, np.zeros(n, int), w, seed + 10000,
                                              cadence=(1, 15, 5), miss=(.01, .08, .12),
                                              max_age=(2, 45, 12))
    return labels, comp, async_score, cal_end, onset


def sfogd(scores, y, ref_sorted, alpha=0.05, delay=120):
    n = len(scores); q = float(ref_sorted[min(len(ref_sorted) - 1,
        max(0, math.ceil((len(ref_sorted) + 1) * (1 - alpha)) - 1))])
    th = np.full(n, q); p = np.zeros(n, int); gsum = 0.; eta = 1 / math.sqrt(3)
    for i in range(n):
        th[i] = q; p[i] = int(scores[i] > q)
        r = i - delay
        if r >= 0 and y[r] == 0:
            err = int(scores[r] > th[r]); grad = alpha - err; gsum += grad * grad
            q = float(np.clip(q - eta * grad / max(math.sqrt(gsum), 1e-12), 0, 1))
    return p


CFG_RNG = np.random.default_rng(424242)
rows = []
for k in range(30):
    cfg = dict(shape=str(CFG_RNG.choice(["sudden", "gradual", "recurring", "mixed"])),
               onset=int(CFG_RNG.integers(4000, 8501)),
               magnitude=float(CFG_RNG.uniform(0.5, 2.0)),
               ramp=int(CFG_RNG.integers(200, 2001)),
               period=int(CFG_RNG.integers(300, 1501)))
    labels, comp, score, cal_end, onset = build_random(1000 + k, cfg)
    calidx = np.arange(cal_end); normal = labels[calidx] == 0
    ref = score[calidx][normal]; ref_comp = comp[calidx][normal]
    y = labels[cal_end:]; sc = score[cal_end:]; cm = comp[cal_end:]
    post_idx = max(0, onset - cal_end)
    for method in ["Static matched", "Rolling ACI", "Quantile tracking", "Guarded full"]:
        kwargs = {}
        if method == "Rolling ACI": kwargs["gamma"] = 0.001
        if method == "Quantile tracking": kwargs["pid_step"] = 0.0005
        pred, th, diag = bench.online_thresholds(method, sc, y, ref, delay=120, components=cm,
                                                 reference_components=ref_comp, interval=2200, **kwargs)
        met = bench.threshold_metrics(y, pred, sc)
        pn = y[post_idx:] == 0
        met["far_post"] = float(np.mean(pred[post_idx:][pn])) if pn.any() else np.nan
        rows.append(dict(config=k, **cfg, method=method, updates=diag["updates"], **met))
    pred = sfogd(sc, y, np.sort(ref))
    met = bench.threshold_metrics(y, pred, sc)
    pn = y[post_idx:] == 0
    met["far_post"] = float(np.mean(pred[post_idx:][pn])) if pn.any() else np.nan
    rows.append(dict(config=k, **cfg, method="SF-OGD", updates=int((y == 0).sum()), **met))
    print(f"config {k} done ({cfg['shape']}, onset {cfg['onset']}, mag {cfg['magnitude']:.2f})", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "randomized_shift_results.csv", index=False)

# ranking summary
piv = df.pivot(index="config", columns="method", values="mcc")
ranks = piv.rank(axis=1, ascending=False)
summ = pd.DataFrame({
    "mean_mcc": piv.mean(), "mean_rank_mcc": ranks.mean(),
    "times_best": (ranks == 1).sum(), "times_worst": (ranks == ranks.max(axis=1).max()).sum(),
})
farp = df.pivot(index="config", columns="method", values="far_post")
summ["far_post_mean"] = farp.mean()
summ["far_post_in_0.03_0.10"] = ((farp >= 0.03) & (farp <= 0.10)).sum()
summ = summ.round(4).sort_values("mean_rank_mcc")
summ.to_csv(OUT / "randomized_shift_summary.csv")
print(summ.to_string())
