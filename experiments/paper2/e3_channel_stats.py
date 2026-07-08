"""E3-B: statistics upgrade to the Paper-1 standard (audit action A8).

Recomputes the channel-reliability-shift paired comparisons (Governed escalation vs each
comparator, metrics post_mcc / expected_cost / mcc, n=10 seeds) from the archived per-seed file,
adding Holm correction across the 15-test family and Cliff's delta effect sizes.
Also emits Wilson 95% CIs for FAR proportions (pooled normals across seeds) for the channel
stress test, reading per-seed FAR and normal counts from the archived trace-free per-seed rows
(normals per seed are constant by design: 8,000 post-calibration events minus event windows).

Outputs:
  analysis2/results/channel_failure_paired_statistics_v2.csv
  analysis2/results/channel_failure_far_wilson.csv
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / 'paper2-incoming' / 'P4_Q1_literature_rebuilt_final'
OUT = ROOT / 'analysis2' / 'results'
OUT.mkdir(parents=True, exist_ok=True)

R = pd.read_csv(PKG / 'results' / 'channel_failure_results.csv')
FOCUS = 'Governed escalation'
COMPARATORS = ['Static', 'Scheduled', 'ACI', 'Quantile tracking', 'SF-OGD']
METRICS = ['post_mcc', 'expected_cost', 'mcc']


def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


rows = []
f = R[R.method == FOCUS].sort_values('seed')
for comp in COMPARATORS:
    c = R[R.method == comp].sort_values('seed')
    assert list(f.seed) == list(c.seed) == list(range(10))
    for met in METRICS:
        x = f[met].to_numpy(float); y = c[met].to_numpy(float); d = x - y
        stat, p = wilcoxon(x, y, alternative='two-sided', mode='exact')
        rows.append(dict(focus=FOCUS, comparator=comp, metric=met, n=len(d),
                         mean_difference=float(d.mean()), median_difference=float(np.median(d)),
                         wilcoxon_W=float(stat), p_value=float(p),
                         cliffs_delta=float(cliffs_delta(x, y))))

df = pd.DataFrame(rows)
# Holm correction across the full 15-test family
m = len(df)
order = np.argsort(df.p_value.to_numpy())
adj = np.empty(m)
running = 0.0
for rank, idx in enumerate(order):
    val = (m - rank) * df.p_value.iloc[idx]
    running = max(running, val)
    adj[idx] = min(1.0, running)
df['p_holm'] = adj
df['significant_holm_05'] = df.p_holm < 0.05
df.to_csv(OUT / 'channel_failure_paired_statistics_v2.csv', index=False)

# Wilson 95% CI for FAR per method: per-seed FAR is a proportion over the per-seed normal count.
# Reconstruct normal counts from per-seed far and the trace-equivalent event structure:
# normals per seed = post-cal events (8000) minus positives; recover from far & counts is not
# possible from far alone, so pool via the per-seed far values weighted equally (each seed has
# the same generator settings; normal counts vary only slightly with seeded failure jitter).
# We therefore report: mean FAR across seeds, sd, and Wilson CI treating the pooled normal count
# as 10 x median per-seed normals (documented approximation), plus exact per-seed range.
def wilson(phat, n, z=1.959963984540054):
    den = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / den
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


# Per-seed normal counts: derive exactly by regenerating the label stream with the archived
# generator logic (identical constants; no scores needed, labels only).
def normals_per_seed(seed):
    rng = np.random.default_rng(seed); n = 10000; cal = 2000
    base = np.arange(2600, n - 150, 750)
    failures = base + rng.integers(-80, 81, size=len(base))
    dist = np.full(n, np.inf); nxt = np.inf; fs = set(map(int, failures))
    for i in range(n - 1, -1, -1):
        if i in fs: nxt = i
        dist[i] = nxt - i
    y = ((dist >= 0) & (dist <= 140)).astype(int)
    return int((y[cal:] == 0).sum())


counts = {s: normals_per_seed(s) for s in range(10)}
wrows = []
for meth, g in R.groupby('method'):
    g = g.sort_values('seed')
    fars = g.far.to_numpy(float)
    ns = np.array([counts[s] for s in g.seed])
    pooled_fp = float((fars * ns).sum()); pooled_n = int(ns.sum())
    phat = pooled_fp / pooled_n
    lo, hi = wilson(phat, pooled_n)
    wrows.append(dict(method=meth, far_mean_seeds=float(fars.mean()), far_sd_seeds=float(fars.std(ddof=1)),
                      pooled_far=phat, pooled_normals=pooled_n, wilson_lo=float(lo), wilson_hi=float(hi)))
pd.DataFrame(wrows).to_csv(OUT / 'channel_failure_far_wilson.csv', index=False)

print(df.round(6).to_string(index=False))
print()
print(pd.DataFrame(wrows).round(5).to_string(index=False))
