"""W1 fix: one-at-a-time sensitivity of the governed-escalation controller constants.

Faithful parameterized copies of the two governed loops (channel-reliability stress test from
the archived run_q1_extension.py — bit-reproduced earlier by e3_channel_rerun.py — and the
leak-free battery replay from e3_battery_noleak.py). Every constant is varied one-at-a-time
around its operating value while all data generation, seeds, folds, and baseline calibration
stay fixed, so differences are attributable to the constant alone.

Outputs analysis2/results/governed_sensitivity.csv (→ Supplementary Table S9).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "paper2-incoming" / "P4_Q1_literature_rebuilt_final"
OUT = ROOT / "analysis2" / "results"
SCRATCH_OUT = OUT / "rerun_channel"  # module OUT redirect (unused writes)

spec = importlib.util.spec_from_file_location("q1ext", PKG / "code" / "run_q1_extension.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["q1ext"] = mod
spec.loader.exec_module(mod)
mod.OUT = SCRATCH_OUT

# battery pipeline (leak-free)
spec2 = importlib.util.spec_from_file_location("bat2", ROOT / "analysis2" / "scripts" / "e3_battery_noleak.py")
bat = importlib.util.module_from_spec(spec2)
sys.modules["bat2"] = bat
spec2.loader.exec_module(bat)


# ---------------- channel-reliability governed loop, parameterized ----------------
def channel_governed(seed: int, review_every=100, buffer_cap=1000, train_share=0.7,
                     auc_floor=0.58, proposal_weight=0.85, accept_cost=0.98,
                     accept_mcc=0.02, thr_jump=0.4, min_buffer=400):
    rng = np.random.default_rng(seed); n = 10000; cal = 2000; onset = 5000
    base = np.arange(2600, n - 150, 750)
    failures = base + rng.integers(-80, 81, size=len(base))
    dist = np.full(n, np.inf); nxt = np.inf; fs = set(map(int, failures))
    for i in range(n - 1, -1, -1):
        if i in fs: nxt = i
        dist[i] = nxt - i
    y = ((dist >= 0) & (dist <= 140)).astype(int)
    prox = np.clip(1 - dist / 140, 0, 1); prox[~np.isfinite(dist)] = 0
    from scipy.special import expit
    c1 = expit(-3 + 2.3 * prox + rng.normal(0, .38, n))
    c2 = expit(-3.2 + 4.0 * prox + rng.normal(0, .30, n))
    c3 = expit(-3 + 4.8 * prox + rng.normal(0, .28, n))
    c3[onset:] = expit(-1.8 + .5 * prox[onset:] + rng.normal(0, .65, n - onset))
    comp = np.c_[c1, c2, c3]
    w = mod.optimize_weights(comp[:cal], y[:cal])
    q0 = mod.conformal_q((comp[:cal] @ w)[y[:cal] == 0])
    yy = y[cal:]
    curw = w.copy(); q = q0; p = np.zeros(len(yy), int); s_dyn = np.zeros(len(yy)); buf = []
    wu = tu = esc = 0
    for i in range(len(yy)):
        s_dyn[i] = comp[cal + i] @ curw; p[i] = s_dyn[i] > q
        r = i - 80
        if r >= 0: buf.append((comp[cal + r].copy(), int(yy[r]))); buf = buf[-buffer_cap:]
        if i > 0 and i % review_every == 0 and len(buf) >= min_buffer:
            B = np.array([x for x, _ in buf]); Y = np.array([z for _, z in buf])
            sp = int(train_share * len(B)); Bt, Yt = B[:sp], Y[:sp]; Bv, Yv = B[sp:], Y[sp:]
            if len(np.unique(Yt)) == 2 and len(np.unique(Yv)) == 2:
                old = Bv @ curw; oldp = (old > q); om = mod.metrics(Yv, oldp, old)
                aucs = [roc_auc_score(Yv, Bv[:, j]) for j in range(3)]; fauc = roc_auc_score(Yv, old)
                bestj = int(np.argmax(aucs))
                if max(aucs) < auc_floor:
                    esc += 1
                else:
                    target = np.zeros(3); target[bestj] = 1.0
                    cand = (1 - proposal_weight) * curw + proposal_weight * target
                    cand = cand / cand.sum()
                    ns = Bv @ cand; norm = ns[Yv == 0]
                    if len(norm) >= 40:
                        nq = mod.conformal_q(norm); nm = mod.metrics(Yv, ns > nq, ns)
                        if (max(aucs) > fauc + .03 and
                                (nm['expected_cost'] < om['expected_cost'] * accept_cost
                                 or nm['mcc'] > om['mcc'] + accept_mcc)):
                            curw = cand; q = nq; wu += 1; tu += 1
            normal = np.array([x @ curw for x, z in buf if z == 0])
            if len(normal) > 200:
                nq = mod.conformal_q(normal)
                if abs(nq - q) < thr_jump: q = nq; tu += 1
    m = mod.metrics(yy, p, s_dyn)
    m['post_mcc'] = mod.metrics(yy[onset - cal:], p[onset - cal:], s_dyn[onset - cal:])['mcc']
    return dict(post_mcc=m['post_mcc'], mcc=m['mcc'], far=m['far'],
                expected_cost=m['expected_cost'], thr_updates=tu, weight_updates=wu, escalations=esc)


# ---------------- battery governed loop, parameterized (leak-free features) ----------------
def battery_governed(review_every=12, buffer_cap=70, train_share=0.7, auc_floor=0.56,
                     grid_step=0.1, shrink=0.08, accept_cost=0.95, l1_jump=1.2,
                     thr_jump_rel=0.6, min_buffer=35):
    d = bat.aggregate_battery(); rows = []
    for fi, (testb, calb, trainbs) in enumerate(bat.FOLDS):
        tr = d[d.Battery.isin(trainbs)]; calf = d[d.Battery == calb]; te = d[d.Battery == testb]
        m = bat.fit_models(tr); cc = bat.predict_models(m, calf); tc = bat.predict_models(m, te)
        w = bat.optimize_weights(cc, calf.label.to_numpy())
        cl, ca = bat.delivered_cache(cc, 100 + fi); tl, ta = bat.delivered_cache(tc, 200 + fi)
        cs = bat.fused_from_cache(cl, ca, w)
        q0 = bat.conformal_q(cs[calf.label.to_numpy() == 0])
        y = te.label.to_numpy(int)
        latest, avail = tl, ta; curw = w.copy(); q = q0
        p = np.zeros(len(y), int); w_hist = []; buf = []; wu = tu = esc = 0
        for i in range(len(y)):
            ww = curw * avail[i]; ww = ww / ww.sum() if ww.sum() > 0 else np.ones(3) / 3
            p[i] = float(latest[i] @ ww) > q; w_hist.append(curw.copy())
            r = i - 2
            if r >= 0: buf.append((tc[r].copy(), int(y[r]))); buf = buf[-buffer_cap:]
            if i > 0 and i % review_every == 0 and len(buf) >= min_buffer:
                B = np.array([x for x, _ in buf]); Y = np.array([yy_ for _, yy_ in buf])
                sp = max(24, int(train_share * len(B))); Bt, Yt = B[:sp], Y[:sp]; Bv, Yv = B[sp:], Y[sp:]
                if len(np.unique(Yt)) == 2 and len(np.unique(Yv)) == 2:
                    old_s = Bv @ curw; old_cost = bat.metrics(Yv, (old_s > q).astype(int), old_s)['expected_cost']
                    cand = bat.optimize_weights_grid(Bt, Yt, curw, step=grid_step, shrink=shrink)
                    new_s = Bv @ cand; normals = Bv[Yv == 0] @ cand
                    if len(normals) >= 5:
                        candq = bat.conformal_q(normals)
                        new_cost = bat.metrics(Yv, (new_s > candq).astype(int), new_s)['expected_cost']
                        aucs = []
                        for j in range(3):
                            try: aucs.append(roc_auc_score(Yv, Bv[:, j]))
                            except Exception: aucs.append(.5)
                        if max(aucs) < auc_floor: esc += 1
                        elif new_cost <= old_cost * accept_cost and np.sum(abs(cand - curw)) <= l1_jump:
                            curw = cand; q = candq; wu += 1; tu += 1
                normal_scores = np.array([x @ curw for x, yy_ in buf if yy_ == 0])
                if len(normal_scores) >= 25:
                    candq = bat.conformal_q(normal_scores)
                    if abs(candq - q) / max(abs(q), .05) < thr_jump_rel: q = candq; tu += 1
        score_dyn = np.array([latest[i] @ (w_hist[i] * avail[i] / max((w_hist[i] * avail[i]).sum(), 1e-12))
                              for i in range(len(y))])
        mm = bat.metrics(y, p, score_dyn)
        rows.append(dict(fold=fi, mcc=mm['mcc'], far=mm['far'], recall=mm['recall'],
                         expected_cost=mm['expected_cost'], thr_updates=tu,
                         weight_updates=wu, escalations=esc))
    R = pd.DataFrame(rows)
    return dict(mcc=R.mcc.mean(), far=R.far.mean(), recall=R.recall.mean(),
                expected_cost=R.expected_cost.mean(), thr_updates=R.thr_updates.mean(),
                weight_updates=R.weight_updates.mean(), escalations=R.escalations.mean())


CH_BASE = dict(review_every=100, buffer_cap=1000, train_share=0.7, auc_floor=0.58,
               proposal_weight=0.85, accept_cost=0.98, accept_mcc=0.02, thr_jump=0.4)
CH_GRID = dict(review_every=[50, 200], buffer_cap=[600, 1500], train_share=[0.6, 0.8],
               auc_floor=[0.54, 0.62], proposal_weight=[0.70, 0.95], accept_cost=[0.95, 1.0],
               accept_mcc=[0.01, 0.05], thr_jump=[0.2, 0.6])
BT_BASE = dict(review_every=12, buffer_cap=70, train_share=0.7, auc_floor=0.56,
               grid_step=0.1, shrink=0.08, accept_cost=0.95, l1_jump=1.2, thr_jump_rel=0.6)
BT_GRID = dict(review_every=[6, 24], buffer_cap=[50, 90], train_share=[0.6, 0.8],
               auc_floor=[0.52, 0.60], shrink=[0.04, 0.16], accept_cost=[0.90, 0.99],
               l1_jump=[0.8, 1.6], thr_jump_rel=[0.3, 0.9])

rows = []
print("channel: base", flush=True)
base_ch = [channel_governed(s, **CH_BASE) for s in range(10)]
b = pd.DataFrame(base_ch).mean()
rows.append(dict(experiment="channel", param="(base)", value="-", **b.to_dict()))
for pname, vals in CH_GRID.items():
    for v in vals:
        print(f"channel: {pname}={v}", flush=True)
        cfg = {**CH_BASE, pname: v}
        r = pd.DataFrame([channel_governed(s, **cfg) for s in range(10)]).mean()
        rows.append(dict(experiment="channel", param=pname, value=v, **r.to_dict()))

print("battery: base", flush=True)
rows.append(dict(experiment="battery", param="(base)", value="-", **battery_governed(**BT_BASE)))
for pname, vals in BT_GRID.items():
    for v in vals:
        print(f"battery: {pname}={v}", flush=True)
        cfg = {**BT_BASE, pname: v}
        rows.append(dict(experiment="battery", param=pname, value=v, **battery_governed(**cfg)))

df = pd.DataFrame(rows)
df.to_csv(OUT / "governed_sensitivity.csv", index=False)
print(df.round(4).to_string(index=False))

# consistency guard: base configs must reproduce the paper's numbers
ch_base = df[(df.experiment == "channel") & (df.param == "(base)")].iloc[0]
assert abs(ch_base.post_mcc - 0.6876) < 5e-3, ch_base.post_mcc
bt_base = df[(df.experiment == "battery") & (df.param == "(base)")].iloc[0]
assert abs(bt_base.expected_cost - 0.1272) < 5e-3, bt_base.expected_cost
print("BASE CONSISTENCY OK")
