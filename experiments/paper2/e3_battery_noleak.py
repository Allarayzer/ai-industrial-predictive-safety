"""E3-A: leak-free battery replay for Paper 2.

Faithful copy of the battery pipeline from the archived
paper2-incoming/P4_Q1_literature_rebuilt_final/code/run_q1_extension.py with EXACTLY ONE change:
'delta_capacity' is removed from FEATURES, because lag_capacity + delta_capacity reconstructs
current-cycle capacity exactly, contradicting the manuscript's exclusion claim
(Phase-0 audit action A3; author-approved 2026-07-07).

Everything else — aggregation, folds, model seeds, delivery cadence seeds (100+fi / 200+fi),
policies, governed controller constants — is unchanged, so any difference in results is
attributable solely to the feature removal. Outputs *_v2.csv into analysis2/results/.
"""
from __future__ import annotations
import math, json, pathlib, hashlib
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, roc_auc_score
from scipy.optimize import minimize

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]  # project root ("Scorpus научная статья")
PKG = ROOT / 'paper2-incoming' / 'P4_Q1_literature_rebuilt_final'
DATA = PKG / 'data' / 'nasa_battery_discharge.csv'
OUT = ROOT / 'analysis2' / 'results'
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = .05


def conformal_q(x, alpha=ALPHA):
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    if len(x) < 20: return float(np.quantile(x, 1 - alpha))
    rank = min(len(x) - 1, max(0, math.ceil((len(x) + 1) * (1 - alpha)) - 1))
    return float(x[rank])


def metrics(y, p, s):
    y = np.asarray(y, int); p = np.asarray(p, int); s = np.asarray(s, float); normal = y == 0
    try: auc = float(roc_auc_score(y, s))
    except Exception: auc = float('nan')
    return dict(mcc=float(matthews_corrcoef(y, p)), f1=float(f1_score(y, p, zero_division=0)),
                precision=float(precision_score(y, p, zero_division=0)), recall=float(recall_score(y, p, zero_division=0)),
                far=float(np.mean(p[normal])) if normal.any() else np.nan,
                expected_cost=float((10 * np.sum((y == 1) & (p == 0)) + np.sum((y == 0) & (p == 1))) / len(y)), roc_auc=auc)


def local_far_error(y, p, window=50, alpha=ALPHA):
    idx = np.flatnonzero(np.asarray(y) == 0); z = np.asarray(p)[idx]
    if len(z) < window: return np.nan
    vals = [abs(float(np.mean(z[i:i + window])) - alpha) for i in range(len(z) - window + 1)]
    return float(max(vals))


def optimize_weights(comp, y, old=None, shrink=.02):
    comp = np.asarray(comp, float); y = np.asarray(y, float)
    if len(np.unique(y)) < 2: return np.array([1 / 3] * 3) if old is None else old.copy()
    old = np.array([1 / 3] * 3) if old is None else np.asarray(old, float)
    def obj(w):
        r = np.clip(comp @ w, 1e-6, 1 - 1e-6)
        loss = -np.mean(10 * y * np.log(r) + (1 - y) * np.log(1 - r))
        return loss + shrink * np.sum((w - old) ** 2)
    res = minimize(obj, old, method='SLSQP', bounds=[(0, 1)] * 3,
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}, options={'maxiter': 300})
    w = res.x if res.success else old
    w = np.clip(w, 0, 1); return w / w.sum()


def optimize_weights_grid(comp, y, old=None, step=.1, shrink=.05):
    comp = np.asarray(comp, float); y = np.asarray(y, int); old = np.ones(3) / 3 if old is None else np.asarray(old, float)
    best = old.copy(); bestv = float('inf')
    vals = np.arange(0, 1 + 1e-9, step)
    for a in vals:
        for b in vals:
            c = 1 - a - b
            if c < -1e-9: continue
            w = np.array([a, b, max(0, c)])
            score = comp @ w; pred = score > 0.5
            v = (10 * np.sum((y == 1) & (~pred)) + np.sum((y == 0) & pred)) / len(y) + shrink * np.sum((w - old) ** 2)
            if v < bestv: bestv = v; best = w
    return best / best.sum()


def delivered_cache(comp, seed, cad=(1, 4, 2), miss=(.0, .05, .05), max_age=(2, 8, 4)):
    rng = np.random.default_rng(seed); n = len(comp); latest = np.zeros((n, 3)); avail = np.zeros((n, 3), bool)
    cur = comp[0].copy(); last = np.zeros(3, int)
    for i in range(n):
        for j in range(3):
            if i % cad[j] == 0 and rng.random() > miss[j]: cur[j] = comp[i, j]; last[j] = i
        latest[i] = cur; avail[i] = [(i - last[j]) <= max_age[j] for j in range(3)]
    return latest, avail


def fused_from_cache(latest, avail, w):
    ww = avail * w[None, :]; den = ww.sum(1); den[den == 0] = 1
    return (latest * ww).sum(1) / den


def run_threshold(method, scores, y, q0, delay=2, alpha=ALPHA, interval=25, buffer=60):
    n = len(scores); p = np.zeros(n, int); th = np.full(n, q0, float); normals = []; q = q0; updates = 0
    alpha_t = alpha; gsum = 0.; D = 1.; eta = D / math.sqrt(3)
    for i in range(n):
        th[i] = q; p[i] = int(scores[i] > q)
        r = i - delay
        if r >= 0 and y[r] == 0:
            x = float(scores[r]); err = int(x > th[r]); normals.append(x); normals = normals[-buffer:]
            if method == 'Scheduled' and i % interval == 0 and len(normals) >= 20:
                q = conformal_q(normals, alpha); updates += 1
            elif method == 'ACI':
                alpha_t = float(np.clip(alpha_t + .01 * (alpha - err), 1 / 61, .5))
                ref = np.sort(np.asarray(normals if len(normals) >= 20 else [q0]))
                q = float(np.quantile(ref, 1 - alpha_t)); updates += 1
            elif method == 'Quantile tracking':
                q = float(np.clip(q + .015 * (err - alpha), 0, 1)); updates += 1
            elif method == 'SF-OGD':
                grad = alpha - err; gsum += grad * grad
                q = float(np.clip(q - eta * grad / max(math.sqrt(gsum), 1e-12), 0, 1)); updates += 1
    return p, th, updates


def aggregate_battery():
    d = pd.read_csv(DATA)
    rows = []
    for (b, c), g in d.groupby(['Battery', 'id_cycle'], sort=True):
        t = g.Time.to_numpy(float); v = g.Voltage_measured.to_numpy(float); cur = g.Current_measured.to_numpy(float); temp = g.Temperature_measured.to_numpy(float)
        order = np.argsort(t); t = t[order]; v = v[order]; cur = cur[order]; temp = temp[order]
        dt = np.diff(t, prepend=t[0])
        rows.append(dict(Battery=b, cycle=int(c), capacity=float(g.Capacity.iloc[0]), duration=float(np.max(t)),
                         v_mean=float(v.mean()), v_min=float(v.min()), v_max=float(v.max()), v_std=float(v.std()),
                         v_early=float(v[:max(3, len(v) // 10)].mean()), v_late=float(v[-max(3, len(v) // 10):].mean()),
                         i_mean=float(cur.mean()), i_std=float(cur.std()), temp_mean=float(temp.mean()), temp_max=float(temp.max()),
                         temp_rise=float(temp.max() - temp[0]), energy_proxy=float(np.sum(np.abs(v * cur) * dt) / 3600)))
    z = pd.DataFrame(rows).sort_values(['Battery', 'cycle']).reset_index(drop=True)
    for col in ['capacity', 'duration', 'v_mean', 'v_min', 'v_std', 'v_late', 'temp_mean', 'temp_max', 'temp_rise', 'energy_proxy']:
        z['lag_' + col] = z.groupby('Battery')[col].shift(1)
        z['delta_' + col] = z.groupby('Battery')[col].diff()
    z['age'] = z.groupby('Battery').cumcount().astype(float)
    z['age_sqrt'] = np.sqrt(z.age)
    z = z.dropna().reset_index(drop=True)
    max_cycle = z.groupby('Battery').cycle.transform('max')
    z['rul'] = (max_cycle - z.cycle).clip(lower=0)
    z['label'] = (z.rul <= 20).astype(int)
    return z


# ONLY CHANGE vs archive: 'delta_capacity' removed (leak: lag_capacity + delta_capacity == capacity)
FEATURES = ['age', 'age_sqrt', 'lag_capacity', 'duration', 'lag_duration', 'delta_duration',
            'v_mean', 'v_min', 'v_std', 'v_late', 'lag_v_mean', 'delta_v_mean', 'temp_mean', 'temp_max',
            'temp_rise', 'lag_temp_mean', 'energy_proxy', 'lag_energy_proxy', 'delta_energy_proxy']


@dataclass
class BatteryModels:
    scaler: StandardScaler; iso: IsolationForest; rul: GradientBoostingRegressor; risk: LogisticRegression; healthy_ref: np.ndarray


def fit_models(train):
    X = train[FEATURES].to_numpy(float); sc = StandardScaler().fit(X); Z = sc.transform(X)
    healthy = train.rul.to_numpy() > 40
    iso = IsolationForest(n_estimators=300, contamination='auto', random_state=11).fit(Z[healthy])
    raw = -iso.score_samples(Z[healthy]); ref = np.sort(raw)
    rul = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=.035, loss='huber', random_state=12).fit(Z, train.rul)
    risk = LogisticRegression(max_iter=2000, class_weight={0: 1, 1: 10}, C=.5, random_state=13).fit(Z, train.label)
    return BatteryModels(sc, iso, rul, risk, ref)


def predict_models(m, d):
    Z = m.scaler.transform(d[FEATURES].to_numpy(float)); raw = -m.iso.score_samples(Z)
    anom = np.searchsorted(m.healthy_ref, raw, side='right') / len(m.healthy_ref)
    rh = np.clip(m.rul.predict(Z), 0, 200); rr = expit((20 - rh) / 5)
    risk = m.risk.predict_proba(Z)[:, 1]
    return np.column_stack([anom, rr, risk])


FOLDS = [('B0005', 'B0006', ['B0007', 'B0018']), ('B0006', 'B0007', ['B0005', 'B0018']),
         ('B0007', 'B0018', ['B0005', 'B0006']), ('B0018', 'B0005', ['B0006', 'B0007'])]


def run_battery():
    d = aggregate_battery(); rows = []; trace = []; diag = []
    for fi, (testb, calb, trainbs) in enumerate(FOLDS):
        tr = d[d.Battery.isin(trainbs)]; cal = d[d.Battery == calb]; te = d[d.Battery == testb]
        m = fit_models(tr); cc = predict_models(m, cal); tc = predict_models(m, te)
        w = optimize_weights(cc, cal.label.to_numpy())
        cl, ca = delivered_cache(cc, 100 + fi); tl, ta = delivered_cache(tc, 200 + fi)
        cs = fused_from_cache(cl, ca, w); ts = fused_from_cache(tl, ta, w)
        ref = cs[cal.label.to_numpy() == 0]; q0 = conformal_q(ref)
        y = te.label.to_numpy(int)
        for meth in ['Static', 'Scheduled', 'ACI', 'Quantile tracking', 'SF-OGD']:
            if meth == 'Static': p = (ts > q0).astype(int); th = np.full(len(ts), q0); upd = 0
            else: p, th, upd = run_threshold(meth, ts, y, q0)
            mm = metrics(y, p, ts); mm['lfe50'] = local_far_error(y, p, 50)
            rows.append(dict(fold=fi, test_battery=testb, calibration_battery=calb, train_batteries='+'.join(trainbs), method=meth, updates=upd, **mm))
            trace.append(pd.DataFrame(dict(fold=fi, Battery=testb, cycle=te.cycle.to_numpy(), label=y, score=ts, pred=p, threshold=th, method=meth)))
        # Governed escalation: threshold + supervised weight review with chronological holdout.
        latest, avail = tl, ta; curw = w.copy(); q = q0; p = np.zeros(len(y), int); th = np.zeros(len(y)); w_hist = []; buf = []; w_updates = 0; t_updates = 0; escalations = 0
        for i in range(len(y)):
            ww = curw * avail[i]; ww = ww / ww.sum() if ww.sum() > 0 else np.ones(3) / 3
            score = float(latest[i] @ ww); th[i] = q; p[i] = score > q; w_hist.append(curw.copy())
            r = i - 2
            if r >= 0:
                buf.append((tc[r].copy(), int(y[r]))); buf = buf[-70:]
            if i > 0 and i % 12 == 0 and len(buf) >= 35:
                B = np.array([x for x, _ in buf]); Y = np.array([yy for _, yy in buf])
                split = max(24, int(.7 * len(B))); Bt, Yt = B[:split], Y[:split]; Bv, Yv = B[split:], Y[split:]
                if len(np.unique(Yt)) == 2 and len(np.unique(Yv)) == 2:
                    old_s = Bv @ curw; old_p = (old_s > q).astype(int); old_cost = metrics(Yv, old_p, old_s)['expected_cost']
                    cand = optimize_weights_grid(Bt, Yt, curw, step=.1, shrink=.08); new_s = Bv @ cand
                    normals = Bv[Yv == 0] @ cand
                    if len(normals) >= 5:
                        candq = conformal_q(normals)
                        new_p = (new_s > candq).astype(int); new_cost = metrics(Yv, new_p, new_s)['expected_cost']
                        aucs = []
                        for j in range(3):
                            try: aucs.append(roc_auc_score(Yv, Bv[:, j]))
                            except Exception: aucs.append(.5)
                        try: fauc = roc_auc_score(Yv, old_s)
                        except Exception: fauc = .5
                        if max(aucs) < .56: escalations += 1
                        elif new_cost <= old_cost * .95 and np.sum(abs(cand - curw)) <= 1.2:
                            curw = cand; q = candq; w_updates += 1; t_updates += 1
                # threshold-only adjustment if FAR too high and ranking still useful
                normal_scores = np.array([x @ curw for x, yy in buf if yy == 0])
                if len(normal_scores) >= 25:
                    candq = conformal_q(normal_scores)
                    if abs(candq - q) / max(abs(q), .05) < .6: q = candq; t_updates += 1
        score_dyn = np.array([latest[i] @ (w_hist[i] * avail[i] / max((w_hist[i] * avail[i]).sum(), 1e-12)) for i in range(len(y))])
        mm = metrics(y, p, score_dyn); mm['lfe50'] = local_far_error(y, p, 50)
        rows.append(dict(fold=fi, test_battery=testb, calibration_battery=calb, train_batteries='+'.join(trainbs), method='Governed escalation', updates=t_updates, weight_updates=w_updates, model_escalations=escalations, **mm))
        trace.append(pd.DataFrame(dict(fold=fi, Battery=testb, cycle=te.cycle.to_numpy(), label=y, score=score_dyn, pred=p, threshold=th, method='Governed escalation')))
        diag.append(dict(fold=fi, test=testb, cal=calb, w_anomaly=w[0], w_rul=w[1], w_risk=w[2], n_train=len(tr), n_cal=len(cal), n_test=len(te)))
    R = pd.DataFrame(rows); T = pd.concat(trace, ignore_index=True); D = pd.DataFrame(diag)
    R.to_csv(OUT / 'battery_fold_results_v2.csv', index=False); T.to_csv(OUT / 'battery_trace_v2.csv', index=False); D.to_csv(OUT / 'battery_diagnostics_v2.csv', index=False)
    S = R.groupby('method').agg(mcc_mean=('mcc', 'mean'), mcc_sd=('mcc', 'std'), far_mean=('far', 'mean'),
                                recall_mean=('recall', 'mean'), cost_mean=('expected_cost', 'mean'),
                                auc_mean=('roc_auc', 'mean'), lfe50_mean=('lfe50', 'mean'), updates_mean=('updates', 'mean')).reset_index()
    S.to_csv(OUT / 'battery_summary_v2.csv', index=False)
    meta = {'script': 'analysis2/scripts/e3_battery_noleak.py',
            'change_vs_archive': "FEATURES minus 'delta_capacity' (leakage fix, audit action A3)",
            'battery_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
            'battery_primary_folds': FOLDS, 'event_horizon_cycles': 20, 'n_features': len(FEATURES)}
    (OUT / 'battery_metadata_v2.json').write_text(json.dumps(meta, indent=2))
    return S


if __name__ == '__main__':
    S = run_battery()
    print(S.round(4).to_string(index=False))
