"""FD001 fusion vs single channels on the SLSQP objective itself (revision experiment, GPT-P5).

Reviewer-anticipated question: Table 5 shows the quantile-derived R_RUL alone beating the
calibrated fusion on threshold-level MCC, and the manuscript explains that SLSQP minimizes
a cost-asymmetric cross-entropy, not MCC at the warning threshold. This experiment reports
the metrics that the optimizer actually targets, on the held-out TEST engines:

    cost_bce_10to1      : cost-weighted binary cross-entropy, FN:FP = 10:1 (the family of
                          the RiskAggregator calibration objective), mean per sample
    expected_cost_10to1 : (10*FN + FP) / n at the 0.3 warning threshold
    brier               : mean squared error of the score against the binary label
    mcc_at_thr, far_at_thr : threshold-level cross-checks against Table 5

Same seeds, same engine splits, same three R_RUL variants as e2_fusion_rul_variants.py;
R_anom and R_NN fit once per seed and shared. Incremental append + resume on
(seed, rul_variant, config).

Output (analysis/results/): e2_fusion_cost_objective.csv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

import e2_cmapss_calib_split as base
import e2_fusion_rul_variants as vr
from ai_cta.risk_model import RiskAggregator

RESULTS_DIR = base.RESULTS_DIR
OUT = RESULTS_DIR / "e2_fusion_cost_objective.csv"
FIELDS = ["seed", "rul_variant", "config", "cost_bce_10to1", "expected_cost_10to1",
          "brier", "mcc_at_thr", "far_at_thr", "weights"]
THRESHOLD = 0.3
COST_FN, COST_FP = 10.0, 1.0


def cost_metrics(scores, labels):
    y = labels.astype(float)
    p = np.clip(scores, 1e-7, 1.0 - 1e-7)
    cost_bce = float(np.mean(COST_FN * y * (-np.log(p))
                             + COST_FP * (1.0 - y) * (-np.log(1.0 - p))))
    pred = scores >= THRESHOLD
    fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    n_neg = max(int(np.sum(y == 0)), 1)
    return {
        "cost_bce_10to1": round(cost_bce, 4),
        "expected_cost_10to1": round((COST_FN * fn + COST_FP * fp) / len(y), 4),
        "brier": round(float(np.mean((scores - y) ** 2)), 4),
        "mcc_at_thr": round(float(matthews_corrcoef(y, pred)), 4) if pred.any() and not pred.all() else 0.0,
        "far_at_thr": round(fp / n_neg, 4),
    }


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


def emit(seed, variant, config, scores, labels, weights):
    r = cost_metrics(scores, labels)
    r.update(seed=seed, rul_variant=variant, config=config,
             weights=np.array2string(np.asarray(weights), precision=3))
    append_row(r)
    print(f"  [{variant:<12}] {config:<20} costBCE={r['cost_bce_10to1']:.3f} "
          f"expCost={r['expected_cost_10to1']:.3f} brier={r['brier']:.3f} "
          f"MCC={r['mcc_at_thr']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = done_keys()
    print(f"resume: {len(done)} rows already present", flush=True)

    all_keys = [("shared", c) for c in ("R_anom only", "R_NN only")] + [
        (v, c) for v in ("mse-point", "quantile", "quantile-cqr")
        for c in ("R_RUL only", "Equal weights", "Calibrated (SLSQP)")]

    for seed in range(args.n_seeds):
        if all((seed, v, c) in done for v, c in all_keys):
            print(f"seed {seed}: complete, skip", flush=True)
            continue
        print(f"\n=== seed {seed} ===", flush=True)
        train, calib, test = base.load_split(seed)
        tr_rul, cal_rul = base.compute_rul(train), base.compute_rul(calib)
        te_rul = base.compute_rul(test)
        tr_lab = base.compute_labels(tr_rul)
        cal_lab, te_lab = base.compute_labels(cal_rul), base.compute_labels(te_rul)

        p_anom = base.fit_r_anom(train)
        p_nn = base.fit_r_nn(train, tr_lab)
        anom_cal, anom_te = p_anom(calib), p_anom(test)
        nn_cal, nn_te = p_nn(calib), p_nn(test)

        if (seed, "shared", "R_anom only") not in done:
            emit(seed, "shared", "R_anom only", anom_te, te_lab, [1.0, 0, 0])
        if (seed, "shared", "R_NN only") not in done:
            emit(seed, "shared", "R_NN only", nn_te, te_lab, [0, 0, 1.0])

        p_rul_point = base.fit_r_rul(train, tr_rul)
        pq = vr.fit_quantile_rul(train, tr_rul)
        q_cal, q_te = pq(calib), pq(test)
        margin = vr.cqr_margin(q_cal, cal_rul)
        print(f"  CQR margin = {margin:.2f} cycles", flush=True)

        variant_channels = {
            "mse-point": (p_rul_point(calib), p_rul_point(test)),
            "quantile": (vr.horizon_risk(q_cal), vr.horizon_risk(q_te)),
            "quantile-cqr": (vr.horizon_risk(vr.adjust_cqr(q_cal, margin)),
                             vr.horizon_risk(vr.adjust_cqr(q_te, margin))),
        }
        for variant, (rul_cal, rul_te) in variant_channels.items():
            cal_stack = np.stack([anom_cal, rul_cal, nn_cal], axis=1)
            te_stack = np.stack([anom_te, rul_te, nn_te], axis=1)
            if (seed, variant, "R_RUL only") not in done:
                emit(seed, variant, "R_RUL only", rul_te, te_lab, [0, 1.0, 0])
            if (seed, variant, "Equal weights") not in done:
                w = np.array([1 / 3, 1 / 3, 1 / 3])
                emit(seed, variant, "Equal weights", te_stack @ w, te_lab, w)
            if (seed, variant, "Calibrated (SLSQP)") not in done:
                agg = RiskAggregator()
                agg.calibrate_weights(cal_stack[:, 0], cal_stack[:, 1], cal_stack[:, 2],
                                      cal_lab.astype(float))
                emit(seed, variant, "Calibrated (SLSQP)", te_stack @ agg.w, te_lab, agg.w)

    df = pd.read_csv(OUT)
    print("\n=== mean +/- std by variant & config ===")
    print(df.groupby(["rul_variant", "config"])[
        ["cost_bce_10to1", "expected_cost_10to1", "brier", "mcc_at_thr"]]
        .agg(["mean", "std"]).round(4).to_string())
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
