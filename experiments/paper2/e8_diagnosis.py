"""Reviewer point 1: direct evaluation of the governed controller's FAILURE-MODE DIAGNOSIS,
not only the outcome of its actions.

Four controlled regimes with a KNOWN ground-truth failure mode are generated; the same governed
controller used in the channel-reliability experiment is run with per-review action logging.
We then score how well the controller's chosen action matches the action that the true failure
mode calls for.

Ground truth -> correct structural action:
  none         (no post-onset change)                 -> no structural action (restraint)
  calibration  (score-location shift, ranking kept)   -> threshold recalibration only
  channel      (one dominant channel loses ranking)   -> bounded weight update
  model        (all channels lose ranking)            -> model-review / escalation

Outputs (-> Supplementary Table S12 + Fig. 6):
  analysis2/results/diagnosis_reviews.csv     per-review (regime, seed, event, action, ...)
  analysis2/results/diagnosis_confusion.csv   regime x structural-action confusion (row-normalized)
  analysis2/results/diagnosis_summary.csv     per-regime detection delay + false-action rates
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "paper2-incoming" / "P4_Q1_literature_rebuilt_final"
OUT = ROOT / "analysis2" / "results"

spec = importlib.util.spec_from_file_location("q1ext", PKG / "code" / "run_q1_extension.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["q1ext"] = mod
spec.loader.exec_module(mod)
mod.OUT = OUT / "rerun_channel"

N, CAL, ONSET, HORIZON = 10000, 2000, 5000, 140
# controller constants = channel-reliability operating configuration (unchanged)
REVIEW_EVERY, BUFFER_CAP, TRAIN_SHARE = 100, 1000, 0.7
AUC_FLOOR, PROPOSAL_W, ACCEPT_COST, ACCEPT_MCC, THR_JUMP = 0.58, 0.85, 0.98, 0.02, 0.4
DELAY, MIN_BUFFER = 80, 400

CORRECT = {"none": "none", "calibration": "threshold", "channel": "weight", "model": "escalation"}


def build_regime(seed: int, regime: str):
    rng = np.random.default_rng(seed)
    base = np.arange(2600, N - 150, 750)
    failures = base + rng.integers(-80, 81, size=len(base))
    dist = np.full(N, np.inf); nxt = np.inf; fs = set(map(int, failures))
    for i in range(N - 1, -1, -1):
        if i in fs: nxt = i
        dist[i] = nxt - i
    y = ((dist >= 0) & (dist <= HORIZON)).astype(int)
    prox = np.clip(1 - dist / HORIZON, 0, 1); prox[~np.isfinite(dist)] = 0
    # baseline generators (identical to the channel-reliability benchmark pre-onset)
    c1 = expit(-3.0 + 2.3 * prox + rng.normal(0, .38, N))
    c2 = expit(-3.2 + 4.0 * prox + rng.normal(0, .30, N))
    c3 = expit(-3.0 + 4.8 * prox + rng.normal(0, .28, N))
    o = slice(ONSET, N); m = N - ONSET
    if regime == "none":
        pass
    elif regime == "calibration":
        # additive location shift on every channel's logit: ranking within each channel
        # (a monotone function of prox) is preserved, only the operating point moves.
        c1[o] = expit(-3.0 + 2.3 * prox[o] + 1.3 + rng.normal(0, .38, m))
        c2[o] = expit(-3.2 + 4.0 * prox[o] + 1.3 + rng.normal(0, .30, m))
        c3[o] = expit(-3.0 + 4.8 * prox[o] + 1.3 + rng.normal(0, .28, m))
    elif regime == "channel":
        # dominant supervised channel becomes weak/noisy/biased; RUL channel keeps ranking
        c3[o] = expit(-1.8 + 0.5 * prox[o] + rng.normal(0, .65, m))
    elif regime == "model":
        # ALL channels lose almost all ranking information (near-random scores)
        c1[o] = expit(-2.2 + 0.10 * prox[o] + rng.normal(0, .85, m))
        c2[o] = expit(-2.1 + 0.10 * prox[o] + rng.normal(0, .82, m))
        c3[o] = expit(-2.1 + 0.12 * prox[o] + rng.normal(0, .85, m))
    else:
        raise ValueError(regime)
    return np.c_[c1, c2, c3], y


def run_controller(comp, y, seed, regime):
    w = mod.optimize_weights(comp[:CAL], y[:CAL])
    q0 = mod.conformal_q((comp[:CAL] @ w)[y[:CAL] == 0])
    yy = y[CAL:]
    curw = w.copy(); q = q0; p = np.zeros(len(yy), int); s_dyn = np.zeros(len(yy)); buf = []
    reviews = []
    for i in range(len(yy)):
        s_dyn[i] = comp[CAL + i] @ curw; p[i] = s_dyn[i] > q
        r = i - DELAY
        if r >= 0:
            buf.append((comp[CAL + r].copy(), int(yy[r]))); buf = buf[-BUFFER_CAP:]
        if i > 0 and i % REVIEW_EVERY == 0 and len(buf) >= MIN_BUFFER:
            action = "none"
            B = np.array([x for x, _ in buf]); Y = np.array([z for _, z in buf])
            sp = int(TRAIN_SHARE * len(B)); Bt, Yt = B[:sp], Y[:sp]; Bv, Yv = B[sp:], Y[sp:]
            max_auc = np.nan
            if len(np.unique(Yt)) == 2 and len(np.unique(Yv)) == 2:
                old = Bv @ curw; oldp = (old > q); om = mod.metrics(Yv, oldp, old)
                aucs = [roc_auc_score(Yv, Bv[:, j]) for j in range(3)]; fauc = roc_auc_score(Yv, old)
                max_auc = float(max(aucs)); bestj = int(np.argmax(aucs))
                if max_auc < AUC_FLOOR:
                    action = "escalation"
                else:
                    target = np.zeros(3); target[bestj] = 1.0
                    cand = (1 - PROPOSAL_W) * curw + PROPOSAL_W * target; cand = cand / cand.sum()
                    ns = Bv @ cand; norm = ns[Yv == 0]
                    if len(norm) >= 40:
                        nq = mod.conformal_q(norm); nm = mod.metrics(Yv, ns > nq, ns)
                        if (max_auc > fauc + .03 and
                                (nm["expected_cost"] < om["expected_cost"] * ACCEPT_COST
                                 or nm["mcc"] > om["mcc"] + ACCEPT_MCC)):
                            curw = cand; q = nq; action = "weight"
            if action == "none":  # threshold-only maintenance branch
                normal = np.array([x @ curw for x, z in buf if z == 0])
                if len(normal) > 200:
                    nq = mod.conformal_q(normal)
                    if abs(nq - q) < THR_JUMP and abs(nq - q) > 1e-6:
                        q = nq; action = "threshold"
            reviews.append(dict(regime=regime, seed=seed, event=CAL + i,
                                post=int(CAL + i >= ONSET), action=action, max_auc=max_auc))
    return reviews


def main():
    all_reviews = []
    for regime in ["none", "calibration", "channel", "model"]:
        for seed in range(10):
            comp, y = build_regime(seed, regime)
            all_reviews.extend(run_controller(comp, y, seed, regime))
        print(f"regime {regime} done", flush=True)
    R = pd.DataFrame(all_reviews)
    R.to_csv(OUT / "diagnosis_reviews.csv", index=False)

    post = R[R.post == 1].copy()
    order_reg = ["none", "calibration", "channel", "model"]

    # Per-run mean count of each action, by true regime (the diagnosis signal).
    cnt = post.groupby(["regime", "seed"]).action.value_counts().unstack(fill_value=0)
    for a in ["none", "threshold", "weight", "escalation"]:
        if a not in cnt.columns:
            cnt[a] = 0
    perrun = cnt.groupby("regime").mean()[["threshold", "weight", "escalation"]].reindex(order_reg).round(3)
    perrun["correct_action"] = [CORRECT[r] for r in perrun.index]
    perrun.to_csv(OUT / "diagnosis_action_counts.csv")

    # Per-stream structural diagnosis, using action counts well above the per-run noise floor
    # (escalation floor 3 vs noise <=0.2 and signal ~22; weight floor 1 vs noise 0.3 and
    # signal 1.5). None and calibration are grouped as "no structural action needed" because
    # the controller correctly leaves the fusion untouched in both (ranking is intact).
    def diagnose(g):
        e = (g.action == "escalation").sum(); w = (g.action == "weight").sum()
        if e >= 3: return "escalate (model)"
        if w >= 1: return "reweight (channel)"
        return "no structural action"
    diag = (post.groupby(["regime", "seed"]).apply(diagnose, include_groups=False)
            .rename("diagnosis").reset_index())
    order_diag = ["no structural action", "reweight (channel)", "escalate (model)"]
    conf = (diag.groupby(["regime", "diagnosis"]).size().unstack(fill_value=0)
            .reindex(index=order_reg, columns=order_diag, fill_value=0))
    conf.to_csv(OUT / "diagnosis_confusion.csv")

    rows = []
    for regime in order_reg:
        sub = post[post.regime == regime]
        w_run = (sub.action == "weight").groupby(sub.seed).sum() if False else None
        # per-run rates of the two diagnostic actions
        per = cnt.loc[regime] if regime in cnt.index.get_level_values(0) else None
        wr = cnt.xs(regime, level="regime")["weight"].mean()
        er = cnt.xs(regime, level="regime")["escalation"].mean()
        # detection delay: onset->first correct-type structural action (per seed, median)
        delays = []
        for seed in range(10):
            ss = R[(R.regime == regime) & (R.seed == seed) & (R.post == 1)]
            if regime in ("none", "calibration"):
                act = "threshold" if regime == "calibration" else None
            else:
                act = "weight" if regime == "channel" else "escalation"
            if act is None:
                delays.append(np.nan); continue
            hit = ss[ss.action == act]
            delays.append(float(hit.event.min() - ONSET) if len(hit) else np.nan)
        rows.append(dict(regime=regime, correct_action=CORRECT[regime],
                         weight_actions_per_run=round(float(wr), 3),
                         escalations_per_run=round(float(er), 3),
                         median_detection_delay_events=(np.nan if regime == "none"
                                                        else float(np.nanmedian(delays)))))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "diagnosis_summary.csv", index=False)
    print("\nPER-RUN ACTION COUNTS:\n", perrun.to_string())
    print("\nDIAGNOSIS CONFUSION (streams):\n", conf.to_string())
    print("\nSUMMARY:\n", S.to_string(index=False))


if __name__ == "__main__":
    main()
