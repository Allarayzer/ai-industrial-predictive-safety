"""Maintained C-MAPSS RUL runner: grouped windows and disjoint early stopping.

Protocol changed on 2026-10-12; not a replay of frozen PeerJ reference numbers.

Re-runs the same five RUL model families as run_cmapss_rul_normalized.py across N seeds, so
every reported number carries mean +/- SD, and the LSTM's RMSE difference from the
classical baselines is tested with a paired Wilcoxon signed-rank test (paired by seed,
Cliff's delta effect size, Holm correction). All baselines are re-run here under the
shared engine split/normalization; classical baselines remain pointwise — these are paired observations, not numbers
copied from the literature.

Outputs (experiments/maintained/results/):
  cmapss_rul_multiseed.csv        per (seed, subset, method): rmse, mae, score, endpoint_accuracy_20pct
  cmapss_rul_multiseed_summary.md mean+-SD table + Wilcoxon LSTM-MSE/quantile vs baselines
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import provenance
import run_cmapss_rul_normalized as rul_helpers  # noqa: E402  (helpers + constants)

RES = pathlib.Path(__file__).resolve().parent / "results"
SENS = rul_helpers.SELECTED_SENSORS
RUL_MAX = rul_helpers.RUL_MAX


def methods_for_seed(seed):
    """Same pointwise baseline budget; grouped neural fitting with explicit val."""
    return [
        (
            "Linear Regression",
            lambda tr, y, te, ep, ws, **kw: rul_helpers.fit_predict_pointwise(
                lambda: LinearRegression(), tr, y, te
            ),
        ),
        (
            "Random Forest",
            lambda tr, y, te, ep, ws, **kw: rul_helpers.fit_predict_pointwise(
                lambda: RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
                tr,
                y,
                te,
            ),
        ),
        (
            "Gradient Boosting",
            lambda tr, y, te, ep, ws, **kw: rul_helpers.fit_predict_pointwise(
                lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=seed),
                tr,
                y,
                te,
            ),
        ),
        (
            "LSTM (MSE)",
            lambda tr, y, te, ep, ws, **kw: rul_helpers.fit_predict_lstm_mse(
                tr, y, te, ep, ws, seed=seed, **kw
            ),
        ),
        (
            "Proposed (LSTM quantile)",
            lambda tr, y, te, ep, ws, **kw: rul_helpers.fit_predict_lstm_ensemble(
                tr, y, te, ep, ws, seed=seed, **kw
            ),
        ),
    ]


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


REFS = ["LSTM (MSE)", "Proposed (LSTM quantile)"]
BASELINES = ["Linear Regression", "Random Forest", "Gradient Boosting"]


def paired_comparison(frame, reference, baseline):
    selected = frame[frame["method"].isin([reference, baseline])]
    if selected.duplicated(["method", "seed"]).any():
        raise ValueError("Duplicate method/seed rows in paired comparison.")
    left = selected[selected.method == reference][["seed", "rmse"]]
    right = selected[selected.method == baseline][["seed", "rmse"]]
    paired = left.merge(
        right, on="seed", validate="one_to_one", suffixes=("_ref", "_base")
    ).sort_values("seed")
    a, b = paired.rmse_ref.to_numpy(), paired.rmse_base.to_numpy()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired RMSE values must be finite.")
    n = len(paired)
    result = {
        "reference": reference,
        "baseline": baseline,
        "seeds": paired.seed.tolist(),
        "n_pairs": n,
        "unpaired_rows": len(selected) - 2 * n,
        "mean_difference": float(np.mean(a - b)) if n else None,
        "cliffs_delta": cliffs_delta(a, b) if n else None,
        "p_value": None,
        "status": "insufficient_pairs",
    }
    if n >= 2:
        if np.all(a == b):
            result.update(p_value=1.0, status="all_ties")
        else:
            p_value = float(wilcoxon(a, b, alternative="less").pvalue)
            if not np.isfinite(p_value):
                raise ValueError("Wilcoxon returned a non-finite p-value.")
            result.update(p_value=p_value, status="tested")
    return result


def comparison_table(frame):
    comparisons = [paired_comparison(frame, ref, base) for ref in REFS for base in BASELINES]
    # One predeclared family of SIX hypotheses per subset. Unavailable tests
    # occupy their family position (p=1), but are displayed as untested.
    adjusted = holm([row["p_value"] if row["p_value"] is not None else 1.0 for row in comparisons])
    for row, value in zip(comparisons, adjusted, strict=True):
        row["holm_p"] = float(value) if row["p_value"] is not None else None
    return comparisons


def summary_text(frame, subsets, args):
    lines = [
        "# Multi-seed C-MAPSS RUL",
        "",
        f"Requested seeds: {args.seeds}; epochs: {args.epochs}.",
        "Neural models use windows; classical baselines remain pointwise. SD across seeds is descriptive.",
        "Wilcoxon pairs matching seed IDs only; fewer than two pairs are not tested. All ties give p=1.",
        "Holm correction uses one six-comparison family per subset (two neural models × three baselines), not across subsets.",
        "Cliff's delta describes the marginal seed distributions, not a paired effect size.",
    ]

    def fmt(value):
        return "n/a" if value is None else f"{value:.4g}"

    for subset in subsets:
        sub = frame[frame.subset == subset]
        lines += [
            f"\n## {subset}\n",
            "| Method | n | RMSE mean ± SD | MAE | Score | endpoint ±20% |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for method in sub.method.unique():
            values = sub[sub.method == method]
            sd = f"{values.rmse.std():.3f}" if len(values) >= 2 else "n/a"
            lines.append(
                f"| {method} | {len(values)} | {values.rmse.mean():.3f} ± {sd} | "
                f"{values.mae.mean():.3f} | {values.score.mean():.1f} | {values.endpoint_accuracy_20pct.mean():.3f} |"
            )
        lines += [
            "\n| Comparison (RMSE, less) | pairs | unmatched rows | p | Holm p | mean paired difference | Cliff's d | status |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in comparison_table(sub):
            lines.append(
                f"| {row['reference']} < {row['baseline']} | {row['n_pairs']} | {row['unpaired_rows']} | "
                f"{fmt(row['p_value'])} | {fmt(row['holm_p'])} | {fmt(row['mean_difference'])} | "
                f"{fmt(row['cliffs_delta'])} | {row['status']} |"
            )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--window-size", type=int, default=32)
    ap.add_argument("--output-dir", type=pathlib.Path, default=RES)
    ap.add_argument("--data-dir", type=pathlib.Path, default=None)
    args = ap.parse_args()
    subsets = [s.strip() for s in args.subsets.split(",")]
    if (
        min(args.seeds, args.epochs, args.window_size) < 1
        or len(set(subsets)) != len(subsets)
        or not set(subsets).issubset(rul_helpers.SUBSET_N_CLUSTERS)
    ):
        ap.error("Require positive seeds/epochs/window-size and distinct supported subsets.")
    if args.data_dir is not None:
        rul_helpers.DATA_DIR = args.data_dir.resolve()
    data_dir = rul_helpers.DATA_DIR.resolve()
    output_dir = args.output_dir.resolve()
    seeds = list(range(args.seeds))
    loaded = {subset: rul_helpers.load_train_test(subset) for subset in subsets}
    roles = {}
    for subset, (train, test, _) in loaded.items():
        for seed in seeds:
            fit, validation = rul_helpers.split_engine_validation(train, seed=seed)
            roles[f"{subset}/{seed}"] = {
                "fit": sorted(map(int, fit.unit.unique())),
                "validation": sorted(map(int, validation.unit.unique())),
                "official_test": sorted(map(int, test.unit.unique())),
            }
    provenance.prepare_run(
        output_dir,
        "cmapss_rul_multiseed",
        {**vars(args), "output_dir": str(output_dir), "data_dir": str(data_dir)},
        {
            f"{prefix}_{subset}": data_dir / f"{prefix}_{subset}.txt"
            for subset in subsets
            for prefix in ("train", "test", "RUL")
        },
        provenance.source_files(REPO, __file__),
        seeds,
        roles,
    )
    out = output_dir / "cmapss_rul_multiseed.csv"
    fields = ["subset", "seed", "method", "rmse", "mae", "score", "endpoint_accuracy_20pct", "sec"]
    allowed = {
        (subset, seed, name) for subset in subsets for seed in seeds for name in BASELINES + REFS
    }
    rows = provenance.read_checkpoint(
        out, fields, ["subset", "seed", "method"], allowed, fields[3:]
    )
    done = {(row["subset"], row["seed"], row["method"]) for row in rows}
    for subset, (train, test, rul_true) in loaded.items():
        for seed in seeds:
            if all((subset, seed, name) in done for name in BASELINES + REFS):
                continue
            train_norm, val_norm, test_norm = rul_helpers.normalize_roles(
                train, test, rul_helpers.SUBSET_N_CLUSTERS[subset], seed
            )
            train_rul, val_rul = (
                rul_helpers.compute_rul(train_norm),
                rul_helpers.compute_rul(val_norm),
            )
            for name, fn in methods_for_seed(seed):
                if (subset, seed, name) in done:
                    continue
                t0 = time.perf_counter()
                preds = provenance.predictions_vector(
                    fn(
                        train_norm,
                        train_rul,
                        test_norm,
                        args.epochs,
                        args.window_size,
                        validation_data=(val_norm, val_rul),
                    ),
                    rul_true,
                )
                d = preds - rul_true
                row = {
                    "subset": subset,
                    "seed": seed,
                    "method": name,
                    "rmse": float(np.sqrt((d**2).mean())),
                    "mae": float(np.abs(d).mean()),
                    "score": rul_helpers.cmapss_score(d),
                    "endpoint_accuracy_20pct": rul_helpers.alpha_lambda_accuracy(
                        preds, rul_true, 0.2
                    ),
                    "sec": time.perf_counter() - t0,
                }
                if not np.isfinite([row[field] for field in fields[3:]]).all():
                    raise ValueError("Computed metrics must be finite before checkpointing.")
                rows.append(row)
                provenance.write_checkpoint(out, fields, rows)
                done.add((subset, seed, name))
                print(f"{subset} seed={seed} {name}: RMSE={row['rmse']:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    (output_dir / "cmapss_rul_multiseed_summary.md").write_text(summary_text(frame, subsets, args))
    print(f"Wrote {out} and summary ({len(rows)} rows).")


if __name__ == "__main__":
    main()
