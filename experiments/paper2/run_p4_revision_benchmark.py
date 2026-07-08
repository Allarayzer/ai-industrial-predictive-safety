from __future__ import annotations

import json
import math
import pathlib
import time
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

import sys
BASE = pathlib.Path('/mnt/data/ai-industrial-predictive-safety-p4')
sys.path.insert(0, str(BASE / 'src'))
from ai_cta.risk_model import RiskAggregator

OUT = pathlib.Path('/mnt/data/P4_7of10_revision/results')
OUT.mkdir(parents=True, exist_ok=True)
DATA = BASE / 'benchmarks' / 'data' / 'cmapss'
COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
OPS = ['op1', 'op2', 'op3']
SENSORS = [f's{i}' for i in range(1, 22)]
ALPHA = 0.05


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    x = np.asarray(scores, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max(20, int(math.ceil(1 / max(alpha, 1e-6)))):
        raise ValueError('insufficient calibration scores')
    level = min(1.0, math.ceil((len(x) + 1) * (1 - alpha)) / len(x))
    return float(np.quantile(x, level))


def empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ref, values, side='right') / max(len(ref), 1)


def threshold_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, int); pred = np.asarray(pred, int); score = np.asarray(score, float)
    normal = y == 0
    try:
        auc = float(roc_auc_score(y, score))
    except ValueError:
        auc = float('nan')
    return {
        'mcc': float(matthews_corrcoef(y, pred)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'far': float(np.mean(pred[normal])) if normal.any() else float('nan'),
        'expected_cost': float((10 * np.sum((y == 1) & (pred == 0)) + np.sum((y == 0) & (pred == 1))) / len(y)),
        'roc_auc': auc,
    }


def finite_quantile(reference: np.ndarray, alpha_t: float) -> float:
    x = np.asarray(reference, dtype=float)
    a = float(np.clip(alpha_t, 1 / (len(x) + 1), 0.5))
    level = min(1.0, math.ceil((len(x) + 1) * (1 - a)) / len(x))
    return float(np.quantile(x, level))


def online_thresholds(
    method: str,
    scores: np.ndarray,
    labels: np.ndarray,
    reference_scores: np.ndarray,
    alpha: float = ALPHA,
    delay: int = 40,
    components: np.ndarray | None = None,
    reference_components: np.ndarray | None = None,
    gamma: float = 0.01,
    pid_step: float = 0.015,
    interval: int = 600,
    buffer_size: int = 1600,
    guard_persistence: int = 2,
    guard_validation: bool = True,
    guard_coherence: bool = True,
    guard_jump_limit: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """One-sided false-alarm calibration with delayed normal feedback.

    ACI and AgACI are specialized here to an upper alarm threshold.  Updates
    occur only when a sample's normal status is revealed, matching the FAR
    target rather than overall classification error.
    """
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    n = len(scores)
    reference_scores = np.asarray(reference_scores, dtype=float)
    ref_sorted = np.sort(reference_scores)
    def q_fixed(a: float) -> float:
        aa = float(np.clip(a, 1 / (len(ref_sorted) + 1), 0.5))
        rank = int(min(len(ref_sorted) - 1, max(0, math.ceil((len(ref_sorted) + 1) * (1 - aa)) - 1)))
        return float(ref_sorted[rank])
    q0 = q_fixed(alpha)
    th = np.full(n, q0, float); pred = np.zeros(n, int)
    normal_buffer = list(map(float, reference_scores[-buffer_size:]))
    updates = rejected = drift_reports = 0

    if method == 'Static matched':
        pred = (scores > q0).astype(int)
        return pred, th, {'updates': 0, 'rejected': 0, 'drift_reports': 0}

    if method == 'Scheduled rolling':
        q = q0
        for i in range(n):
            reveal = i - delay
            if reveal >= 0 and labels[reveal] == 0:
                normal_buffer.append(float(scores[reveal])); normal_buffer = normal_buffer[-buffer_size:]
            if i > 0 and i % interval == 0 and len(normal_buffer) >= 200:
                q = conformal_quantile(np.asarray(normal_buffer), alpha); updates += 1
            th[i] = q; pred[i] = int(scores[i] > q)
        return pred, th, {'updates': updates, 'rejected': 0, 'drift_reports': 0}

    if method == 'ACI':
        a_t = alpha
        lower = 1 / (len(reference_scores) + 1)
        q = q0
        for i in range(n):
            th[i] = q; pred[i] = int(scores[i] > q)
            reveal = i - delay
            if reveal >= 0 and labels[reveal] == 0:
                err = int(scores[reveal] > th[reveal])
                a_t = float(np.clip(a_t + gamma * (alpha - err), lower, 0.5))
                q = q_fixed(a_t); updates += 1
        return pred, th, {'updates': updates, 'rejected': 0, 'drift_reports': 0}

    if method == 'Rolling ACI':
        a_t = alpha; q = q0
        for i in range(n):
            th[i] = q; pred[i] = int(scores[i] > q)
            reveal = i - delay
            if reveal >= 0 and labels[reveal] == 0:
                err = int(scores[reveal] > th[reveal])
                normal_buffer.append(float(scores[reveal])); normal_buffer = normal_buffer[-buffer_size:]
                lower = 1 / (len(normal_buffer) + 1)
                a_t = float(np.clip(a_t + gamma * (alpha - err), lower, 0.5))
                updates += 1
                if updates % 20 == 0:
                    q = finite_quantile(np.asarray(normal_buffer), a_t)
        return pred, th, {'updates': updates, 'rejected': 0, 'drift_reports': 0}

    if method == 'AgACI-style':
        gammas = np.asarray([0.001, 0.003, 0.01, 0.03, 0.08])
        a = np.full(len(gammas), alpha, float)
        w = np.ones(len(gammas), float) / len(gammas)
        q_exp = np.full(len(gammas), q0, float)
        eta = 3.0
        tau = 1 - alpha
        for i in range(n):
            q = float(np.dot(w, q_exp))
            th[i] = q; pred[i] = int(scores[i] > q)
            reveal = i - delay
            if reveal >= 0 and labels[reveal] == 0:
                x = float(scores[reveal])
                # Expert updates use their own alarm decisions.
                errs = (x > q_exp).astype(float)
                a = np.clip(a + gammas * (alpha - errs), 1 / (len(reference_scores) + 1), 0.5)
                q_exp = np.asarray([q_fixed(aj) for aj in a])
                # Bernstein/Hedge-style aggregation on pinball loss.
                losses = (tau - (x < q_exp).astype(float)) * (x - q_exp)
                scale = max(float(np.std(reference_scores)), 1e-3)
                w *= np.exp(-eta * losses / scale)
                w = np.clip(w, 1e-12, None); w /= w.sum(); updates += 1
        return pred, th, {'updates': updates, 'rejected': 0, 'drift_reports': 0}

    if method == 'Quantile tracking':
        # P-controller / online pinball-gradient baseline from Conformal PID.
        q = q0; step = pid_step
        for i in range(n):
            th[i] = q; pred[i] = int(scores[i] > q)
            reveal = i - delay
            if reveal >= 0 and labels[reveal] == 0:
                err = int(scores[reveal] > th[reveal])
                q = float(np.clip(q + step * (err - alpha), 0, 1)); updates += 1
        return pred, th, {'updates': updates, 'rejected': 0, 'drift_reports': 0}

    if method.startswith('Guarded'):
        if components is None or reference_components is None:
            raise ValueError('guarded method requires component scores')
        components = np.asarray(components, float); reference_components = np.asarray(reference_components, float)
        ref_median = np.median(reference_components, axis=0)
        ref_std = np.std(reference_components, axis=0) + 1e-6
        # Fixed quantile bins for PSI.
        bins = [np.unique(np.quantile(reference_components[:, j], np.linspace(0, 1, 11))) for j in range(3)]
        recent_components: list[np.ndarray] = []
        q = q0; streak = 0; cooldown = 0
        check_every = 60; window = 320
        max_jump = 1.5; far_tol = 1.4
        for i in range(n):
            th[i] = q; pred[i] = int(scores[i] > q)
            reveal = i - delay
            if reveal >= 0 and labels[reveal] == 0:
                normal_buffer.append(float(scores[reveal])); normal_buffer = normal_buffer[-buffer_size:]
                recent_components.append(components[reveal].copy()); recent_components = recent_components[-buffer_size:]
            cooldown += 1
            if i > 0 and i % check_every == 0 and len(recent_components) >= window:
                cur = np.asarray(recent_components[-window:])
                drift_ch = 0
                med_z = np.abs(np.median(cur, axis=0) - ref_median) / ref_std
                for j in range(3):
                    b = bins[j]
                    if len(b) < 4:
                        continue
                    b = b.copy(); b[0] = -np.inf; b[-1] = np.inf
                    rp = np.histogram(reference_components[:, j], bins=b)[0].astype(float) + 1
                    cp = np.histogram(cur[:, j], bins=b)[0].astype(float) + 1
                    rp /= rp.sum(); cp /= cp.sum()
                    psi = float(np.sum((cp - rp) * np.log(cp / rp)))
                    if psi > 0.25:
                        drift_ch += 1
                coherent = int(np.sum(med_z > 0.60)) >= 2
                drift = drift_ch >= 2 and (coherent or not guard_coherence)
                if drift:
                    drift_reports += 1; streak += 1
                else:
                    streak = 0
                if drift and streak >= guard_persistence and cooldown >= 300 and len(normal_buffer) >= 300:
                    x = np.asarray(normal_buffer)
                    split = int(len(x) * 0.75)
                    cal_x, val_x = x[:split], x[split:]
                    candidate = conformal_quantile(cal_x, alpha * 0.85)
                    val_far = float(np.mean(val_x > candidate))
                    jump_ok = (abs(candidate - q) / max(abs(q), 1e-6) <= max_jump) or not guard_jump_limit
                    val_ok = (val_far <= far_tol * alpha) or not guard_validation
                    if jump_ok and val_ok:
                        q = candidate; updates += 1; streak = 0
                        ref_median = np.median(cur, axis=0); ref_std = np.std(cur, axis=0) + 1e-6
                        reference_components = cur.copy()
                        bins = [np.unique(np.quantile(reference_components[:, j], np.linspace(0, 1, 11))) for j in range(3)]
                    else:
                        rejected += 1
                    cooldown = 0
        return pred, th, {'updates': updates, 'rejected': rejected, 'drift_reports': drift_reports}

    raise ValueError(method)


# --------------------------- Controlled replay ---------------------------
@dataclass
class ControlledStream:
    labels: np.ndarray
    regimes: np.ndarray
    components: np.ndarray
    async_score: np.ndarray
    sync_score: np.ndarray
    calibration_end: int
    drift_onset: int


def _nearest_failure(n: int, failures: np.ndarray) -> np.ndarray:
    d = np.full(n, np.inf); nxt = np.inf; fs = set(map(int, failures))
    for t in range(n - 1, -1, -1):
        if t in fs: nxt = t
        d[t] = nxt - t
    return d


def asynchronous_array(components: np.ndarray, engine_ids: np.ndarray, weights: np.ndarray, seed: int,
                       cadence=(1, 5, 3), miss=(0.01, 0.06, 0.08), max_age=(2, 12, 6)) -> tuple[np.ndarray, dict]:
    n = len(components); out = np.zeros(n); rng = np.random.default_rng(seed)
    latest = np.zeros(3); last = np.full(3, -10_000, int); available_sum = 0; fallback = 0
    prev_engine = None
    lat = []
    for i in range(n):
        eng = engine_ids[i]
        local = 0 if eng != prev_engine else local + 1
        if eng != prev_engine:
            latest[:] = components[i]; last[:] = i; local = 0; prev_engine = eng
        for j in range(3):
            if local % cadence[j] == 0 and rng.random() > miss[j]:
                latest[j] = components[i, j]; last[j] = i
        avail = np.asarray([(i - last[j]) <= max_age[j] for j in range(3)])
        w = weights * avail
        t0 = time.perf_counter_ns()
        if w.sum() <= 1e-12:
            w = avail.astype(float); fallback += 1
        w /= w.sum()
        out[i] = float(np.dot(w, latest))
        lat.append((time.perf_counter_ns() - t0) / 1000)
        available_sum += int(avail.sum())
    return out, {'mean_available': available_sum/n, 'fallback_fraction': fallback/n,
                 'latency_p50_us': float(np.percentile(lat,50)), 'latency_p95_us': float(np.percentile(lat,95))}


def build_controlled(seed: int, scenario: str, n: int = 12000) -> ControlledStream:
    rng = np.random.default_rng(seed); cal_end = 2500; onset = 6000
    t = np.arange(n); regimes = ((t // 800) % 2).astype(int)
    failures = np.arange(cal_end + 600, n - 200, 850) + rng.integers(-120, 121, size=len(np.arange(cal_end + 600, n - 200, 850)))
    dist = _nearest_failure(n, failures); horizon = 160
    labels = ((dist >= 0) & (dist <= horizon)).astype(int)
    prox = np.clip(1 - dist/horizon, 0, 1); prox[~np.isfinite(dist)] = 0
    drift = np.zeros(n)
    if scenario == 'sudden': drift[onset:] = 1
    elif scenario == 'gradual': drift[onset:] = np.minimum((t[onset:]-onset)/700,1)
    elif scenario == 'recurring': drift[onset:] = (((t[onset:]-onset)//800)%2==0).astype(float)
    elif scenario != 'none': raise ValueError(scenario)
    comp = np.column_stack([
        expit(-3.0 + 2.7*prox + .35*regimes + .85*drift + rng.normal(0,.35,n)),
        expit(-3.3 + 4.3*prox + .15*regimes + .40*drift + rng.normal(0,.30,n)),
        expit(-3.1 + 3.8*prox + .25*regimes + .50*drift + rng.normal(0,.32,n)),
    ])
    w = np.array([.25,.35,.40]); sync = comp @ w
    async_score,_ = asynchronous_array(comp, np.zeros(n,int), w, seed+10000, cadence=(1,15,5), miss=(.01,.08,.12), max_age=(2,45,12))
    return ControlledStream(labels, regimes, comp, async_score, sync, cal_end, onset)


def run_controlled() -> tuple[pd.DataFrame,pd.DataFrame]:
    methods = ['Static matched','Scheduled rolling','ACI','Rolling ACI','AgACI-style','Quantile tracking','Guarded full']
    rows=[]; sens=[]
    for scenario in ['none','sudden','gradual','recurring']:
        for seed in range(10):
            d=build_controlled(seed,scenario)
            cal=(np.arange(d.calibration_end)); normal=d.labels[cal]==0
            ref=d.async_score[cal][normal]; ref_comp=d.components[cal][normal]
            y=d.labels[d.calibration_end:]; sc=d.async_score[d.calibration_end:]; comp=d.components[d.calibration_end:]
            for method in methods:
                kwargs = {}
                if method in ('ACI','Rolling ACI'):
                    kwargs['gamma'] = 0.001
                if method == 'Quantile tracking':
                    kwargs['pid_step'] = 0.0005
                pred,th,diag=online_thresholds(method,sc,y,ref,delay=120,components=comp,reference_components=ref_comp,interval=2200,**kwargs)
                met=threshold_metrics(y,pred,sc)
                post_idx=max(0,d.drift_onset-d.calibration_end); post_norm=y[post_idx:]==0
                met['far_post']=float(np.mean(pred[post_idx:][post_norm])) if post_norm.any() else np.nan
                rows.append({'scenario':scenario,'seed':seed,'method':method,**met,**diag})
        # Sensitivity on the two one-way shift scenarios.
        if scenario in ('sudden','gradual'):
            for seed in range(5):
                d=build_controlled(seed,scenario); normal=d.labels[:d.calibration_end]==0
                ref=d.async_score[:d.calibration_end][normal]; ref_comp=d.components[:d.calibration_end][normal]
                y=d.labels[d.calibration_end:]; sc=d.async_score[d.calibration_end:]; comp=d.components[d.calibration_end:]
                for g in [0.001,0.003,0.01,0.03,0.08]:
                    p,t,di=online_thresholds('ACI',sc,y,ref,delay=120,gamma=g)
                    sens.append({'scenario':scenario,'seed':seed,'family':'ACI gamma','setting':str(g),**threshold_metrics(y,p,sc)})
                variants=[('Guarded full',2,1,1,1),('No persistence',1,1,1,1),('No validation',2,0,1,1),('No coherence',2,1,0,1),('No jump limit',2,1,1,0)]
                for name,per,val,coh,jump in variants:
                    p,t,di=online_thresholds('Guarded '+name,sc,y,ref,delay=120,components=comp,reference_components=ref_comp,
                                              guard_persistence=per,guard_validation=bool(val),guard_coherence=bool(coh),guard_jump_limit=bool(jump))
                    sens.append({'scenario':scenario,'seed':seed,'family':'Guard ablation','setting':name,**threshold_metrics(y,p,sc),**di})
    return pd.DataFrame(rows),pd.DataFrame(sens)


# --------------------------- C-MAPSS cross-fitting ---------------------------
@dataclass
class LinearChannels:
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    rul_beta: np.ndarray
    risk_beta: np.ndarray
    anomaly_reference: np.ndarray


def load_cmapss(name: str) -> pd.DataFrame:
    d=pd.read_csv(DATA/f'train_{name}.txt',sep=r'\s+',header=None,names=COLS)
    mc=d.groupby('unit')['cycle'].transform('max')
    d['rul']=(mc-d['cycle']).clip(upper=125).astype(float)
    d['label']=(d['rul']<=30).astype(int); d['cycle_scaled']=d['cycle']/300.0
    for s in SENSORS: d['d_'+s]=d.groupby('unit')[s].diff().fillna(0.0)
    return d


def engine_folds(d: pd.DataFrame,k=5) -> list[np.ndarray]:
    return [np.asarray(x,int) for x in np.array_split(np.array(sorted(d.unit.unique())),k)]


def fit_linear_channels(train: pd.DataFrame) -> LinearChannels:
    cand=OPS+['cycle_scaled']+SENSORS+['d_'+s for s in SENSORS]
    var=train[cand].var(); feats=list(var[var>1e-8].index)
    X=train[feats].to_numpy(float); mean=X.mean(0); std=X.std(0)+1e-6; Z=(X-mean)/std
    Xa=np.c_[np.ones(len(Z)),Z]
    ridge=5.0; A=Xa.T@Xa+ridge*np.eye(Xa.shape[1]); A[0,0]-=ridge
    rul_beta=np.linalg.solve(A,Xa.T@train.rul.to_numpy(float))
    y=train.label.to_numpy(float); w=np.where(y==1,10.,1.); sw=np.sqrt(w)
    Xw=Xa*sw[:,None]; yw=y*sw; ridge2=20.0
    B=Xw.T@Xw+ridge2*np.eye(Xa.shape[1]); B[0,0]-=ridge2
    risk_beta=np.linalg.solve(B,Xw.T@yw)
    healthy=Z[train.rul.to_numpy()>50]
    hmean=healthy.mean(0); hstd=healthy.std(0)+1e-6
    raw=np.mean(np.abs((healthy-hmean)/hstd),axis=1)
    # Store anomaly normalization parameters inside reference first two rows.
    anomaly_reference=np.concatenate([hmean,hstd,np.sort(raw)])
    return LinearChannels(feats,mean,std,rul_beta,risk_beta,anomaly_reference)


def predict_channels(model: LinearChannels, d: pd.DataFrame) -> np.ndarray:
    Z=(d[model.features].to_numpy(float)-model.mean)/model.std; Xa=np.c_[np.ones(len(Z)),Z]
    p=len(model.features); hmean=model.anomaly_reference[:p]; hstd=model.anomaly_reference[p:2*p]; ref=model.anomaly_reference[2*p:]
    raw=np.mean(np.abs((Z-hmean)/hstd),axis=1); anom=empirical_cdf(ref,raw)
    rulhat=np.clip(Xa@model.rul_beta,0,125); rul=expit((30-rulhat)/8)
    lin=Xa@model.risk_beta; risk=expit((lin-.35)*8)
    return np.column_stack([anom,rul,risk])


def fit_aggregator(comp: np.ndarray,y: np.ndarray) -> RiskAggregator:
    agg=RiskAggregator(cost_fn=10,cost_fp=1)
    agg.calibrate_weights(comp[:,0],comp[:,1],comp[:,2],y)
    return agg


def run_one_fold(subset: str,fold_idx: int) -> tuple[list[dict],pd.DataFrame,dict]:
    d=load_cmapss(subset); folds=engine_folds(d)
    test_u=folds[fold_idx]; cal_u=folds[(fold_idx+1)%5]
    train_u=np.concatenate([folds[j] for j in range(5) if j not in (fold_idx,(fold_idx+1)%5)])
    train=d[d.unit.isin(train_u)]; cal=d[d.unit.isin(cal_u)].sort_values(['unit','cycle']).reset_index(drop=True)
    test=d[d.unit.isin(test_u)].sort_values(['unit','cycle']).reset_index(drop=True)
    model=fit_linear_channels(train); c=predict_channels(model,cal); x=predict_channels(model,test)
    agg=fit_aggregator(c,cal.label.to_numpy())
    cal_sync=c@agg.w; test_sync=x@agg.w
    cal_async,diag_cal=asynchronous_array(c,cal.unit.to_numpy(),agg.w,1000+fold_idx)
    test_async,diag=asynchronous_array(x,test.unit.to_numpy(),agg.w,2000+fold_idx)
    ycal=cal.label.to_numpy(); y=test.label.to_numpy(); normal=ycal==0
    ref=cal_async[normal]; ref_comp=c[normal]
    methods=['Static matched','Scheduled rolling','ACI','Rolling ACI','AgACI-style','Quantile tracking','Guarded full']
    rows=[]; traces=[]
    # Mismatched synchronous threshold ablation.
    qsync=conformal_quantile(cal_sync[normal],ALPHA); pm=(test_async>qsync).astype(int)
    rows.append({'subset':subset,'fold':fold_idx,'method':'Static sync-calibrated (mismatch)',**threshold_metrics(y,pm,test_async),'updates':0,'rejected':0,'drift_reports':0})
    traces.append(pd.DataFrame({'subset':subset,'fold':fold_idx,'unit':test.unit,'cycle':test.cycle,'label':y,'score':test_async,'method':'Static sync-calibrated (mismatch)','threshold':qsync,'pred':pm}))
    for method in methods:
        kwargs = {}
        if method in ('ACI','Rolling ACI'):
            kwargs['gamma'] = 0.001
        if method == 'Quantile tracking':
            kwargs['pid_step'] = 0.0005
        p,t,di=online_thresholds(method,test_async,y,ref,delay=40,components=x,reference_components=ref_comp,interval=1200,**kwargs)
        rows.append({'subset':subset,'fold':fold_idx,'method':method,**threshold_metrics(y,p,test_async),**di})
        traces.append(pd.DataFrame({'subset':subset,'fold':fold_idx,'unit':test.unit,'cycle':test.cycle,'label':y,'score':test_async,'method':method,'threshold':t,'pred':p}))
    # Synchronous performance as an availability upper bound.
    q=conformal_quantile(cal_sync[normal],ALPHA); p=(test_sync>q).astype(int)
    rows.append({'subset':subset,'fold':fold_idx,'method':'Static synchronous upper bound',**threshold_metrics(y,p,test_sync),'updates':0,'rejected':0,'drift_reports':0})
    detail={'subset':subset,'fold':fold_idx,'n_train_engines':len(train_u),'n_cal_engines':len(cal_u),'n_test_engines':len(test_u),
            'weights_anomaly':float(agg.w[0]),'weights_rul':float(agg.w[1]),'weights_supervised':float(agg.w[2]),**diag,**{f'cal_{k}':v for k,v in diag_cal.items()}}
    return rows,pd.concat(traces,ignore_index=True),detail


def cluster_bootstrap_metrics(trace: pd.DataFrame,n_boot=500,seed=123) -> pd.DataFrame:
    rng=np.random.default_rng(seed); out=[]
    for (subset,method),g in trace.groupby(['subset','method'],sort=False):
        units=np.array(sorted(g.unit.unique()))
        arrays=[]
        for u in units:
            z=g[g.unit==u]
            arrays.append((z.label.to_numpy(int),z.pred.to_numpy(int),z.score.to_numpy(float)))
        point=threshold_metrics(g.label.to_numpy(),g.pred.to_numpy(),g.score.to_numpy())
        boot={k:[] for k in ['mcc','f1','recall','far','expected_cost']}
        for _ in range(n_boot):
            idx=rng.integers(0,len(units),len(units))
            yy=np.concatenate([arrays[j][0] for j in idx]); pp=np.concatenate([arrays[j][1] for j in idx]); ss=np.concatenate([arrays[j][2] for j in idx])
            m=threshold_metrics(yy,pp,ss)
            for k in boot: boot[k].append(m[k])
        row={'subset':subset,'method':method}
        for k in boot:
            row[k]=point[k]; row[k+'_lo']=float(np.quantile(boot[k],.025)); row[k+'_hi']=float(np.quantile(boot[k],.975))
        out.append(row)
    return pd.DataFrame(out)

def paired_bootstrap_differences(trace: pd.DataFrame,focus='Guarded full',n_boot=500,seed=456) -> pd.DataFrame:
    rng=np.random.default_rng(seed); rows=[]
    comparators=['Static matched','Scheduled rolling','ACI','Rolling ACI','AgACI-style','Quantile tracking']
    for subset,gsub in trace.groupby('subset',sort=False):
        units=np.array(sorted(gsub.unit.unique()))
        gf=gsub[gsub.method==focus]
        mapf=[]
        for u in units:
            z=gf[gf.unit==u]; mapf.append((z.label.to_numpy(int),z.pred.to_numpy(int),z.score.to_numpy(float)))
        for comp in comparators:
            gc=gsub[gsub.method==comp]
            if gc.empty: continue
            mapc=[]
            for u in units:
                z=gc[gc.unit==u]; mapc.append((z.label.to_numpy(int),z.pred.to_numpy(int),z.score.to_numpy(float)))
            diffs={k:[] for k in ['mcc','far','expected_cost']}
            for _ in range(n_boot):
                idx=rng.integers(0,len(units),len(units))
                yf=np.concatenate([mapf[j][0] for j in idx]);pf=np.concatenate([mapf[j][1] for j in idx]);sf=np.concatenate([mapf[j][2] for j in idx])
                yc=np.concatenate([mapc[j][0] for j in idx]);pc=np.concatenate([mapc[j][1] for j in idx]);sc=np.concatenate([mapc[j][2] for j in idx])
                mf=threshold_metrics(yf,pf,sf);mc=threshold_metrics(yc,pc,sc)
                for k in diffs: diffs[k].append(mf[k]-mc[k])
            row={'subset':subset,'focus':focus,'comparator':comp}
            for k,v in diffs.items():
                row['delta_'+k]=float(np.mean(v));row['delta_'+k+'_lo']=float(np.quantile(v,.025));row['delta_'+k+'_hi']=float(np.quantile(v,.975))
            rows.append(row)
    return pd.DataFrame(rows)

def run_cmapss() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    all_rows=[];traces=[];details=[]
    for subset in ['FD001','FD002','FD003','FD004']:
        for fold in range(5):
            print('C-MAPSS',subset,fold,flush=True)
            rows,tr,det=run_one_fold(subset,fold);all_rows.extend(rows);traces.append(tr);details.append(det)
    return pd.DataFrame(all_rows),pd.concat(traces,ignore_index=True),pd.DataFrame(details)


def summarize_and_plot(controlled: pd.DataFrame,sensitivity: pd.DataFrame,cmapss_rows: pd.DataFrame,cmapss_trace: pd.DataFrame):
    controlled.to_csv(OUT/'controlled_results.csv',index=False)
    sensitivity.to_csv(OUT/'controlled_sensitivity.csv',index=False)
    csum=cmapss_rows.groupby(['subset','method']).agg(mcc_mean=('mcc','mean'),mcc_sd=('mcc','std'),far_mean=('far','mean'),recall_mean=('recall','mean'),cost_mean=('expected_cost','mean'),updates_mean=('updates','mean')).reset_index()
    csum.to_csv(OUT/'cmapss_fold_summary.csv',index=False)
    cboot=cluster_bootstrap_metrics(cmapss_trace,500);cboot.to_csv(OUT/'cmapss_cluster_bootstrap.csv',index=False)
    pdiff=paired_bootstrap_differences(cmapss_trace,500);pdiff.to_csv(OUT/'cmapss_paired_bootstrap.csv',index=False)
    ctrl=controlled.groupby(['scenario','method']).agg(mcc_mean=('mcc','mean'),mcc_sd=('mcc','std'),far_mean=('far','mean'),far_post_mean=('far_post','mean'),cost_mean=('expected_cost','mean'),updates_mean=('updates','mean')).reset_index()
    ctrl.to_csv(OUT/'controlled_summary.csv',index=False)

    keep=['Static matched','Scheduled rolling','ACI','AgACI-style','Quantile tracking','Guarded full']
    for metric in ['mcc_mean','far_post_mean','cost_mean']:
        p=ctrl[ctrl.method.isin(keep)].pivot(index='scenario',columns='method',values=metric)
        ax=p.plot(kind='bar',figsize=(11,5));ax.set_ylabel(metric.replace('_',' ').title());ax.set_title('Controlled replay: '+metric.replace('_',' '));plt.xticks(rotation=0);plt.tight_layout();plt.savefig(OUT/f'controlled_{metric}.png',dpi=220);plt.close()
    # C-MAPSS plot from cluster-bootstrap point estimates.
    p=cboot[cboot.method.isin(keep)].pivot(index='subset',columns='method',values='mcc')
    ax=p.plot(kind='bar',figsize=(11,5));ax.set_ylabel('MCC');ax.set_title('Five-fold engine-disjoint C-MAPSS replay');plt.xticks(rotation=0);plt.tight_layout();plt.savefig(OUT/'cmapss_mcc_methods.png',dpi=220);plt.close()
    p=cboot[cboot.method.isin(keep)].pivot(index='subset',columns='method',values='far')
    ax=p.plot(kind='bar',figsize=(11,5));ax.axhline(ALPHA,linestyle='--',linewidth=1);ax.set_ylabel('False-alarm rate');ax.set_title('C-MAPSS replay false-alarm rate');plt.xticks(rotation=0);plt.tight_layout();plt.savefig(OUT/'cmapss_far_methods.png',dpi=220);plt.close()
    return ctrl,csum,cboot,pdiff


def main():
    t0=time.time();controlled,sens=run_controlled();print('controlled done',time.time()-t0,flush=True)
    cmapss_rows,cmapss_trace,details=run_cmapss();print('cmapss done',time.time()-t0,flush=True)
    cmapss_rows.to_csv(OUT/'cmapss_fold_results.csv',index=False);cmapss_trace.to_csv(OUT/'cmapss_crossfit_trace.csv',index=False);details.to_csv(OUT/'cmapss_diagnostics.csv',index=False)
    ctrl,csum,cboot,pdiff=summarize_and_plot(controlled,sens,cmapss_rows,cmapss_trace)
    meta={'created':'2026-07-08','alpha':ALPHA,'controlled_seeds':10,'cmapss_folds':5,'cmapss_subsets':['FD001','FD002','FD003','FD004'],
          'notes':'C-MAPSS is simulated run-to-failure data. Calibration and deployment share the same asynchronous cadence/missingness mechanism.'}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(ctrl.round(4).to_string(index=False));print(cboot.round(4).to_string(index=False));print('total',time.time()-t0)

if __name__=='__main__': main()
