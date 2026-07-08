"""Full re-execution of the archived controlled + C-MAPSS benchmark for reproducibility.

Loads the archived revision/code/run_p4_revision_benchmark.py and analyze_crossfit_fast.py
with their hardcoded /mnt/data paths redirected to the local package and to
analysis2/results/rerun_cmapss/ (the archive itself is never written to), runs the complete
pipeline, then diffs the regenerated CSVs against the archived ones that back the manuscript.

Deterministic seeds throughout (controlled: seeds 0-9; folds: sorted engine ids;
bootstrap rng: default_rng(20260708)), so results should match up to floating-point/library
differences (archive: python 3.13.5/numpy 2.3.5; local: python 3.11/numpy per venv).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "paper2-incoming" / "P4_Q1_literature_rebuilt_final"
RERUN = ROOT / "analysis2" / "results" / "rerun_cmapss"
RERUN.mkdir(parents=True, exist_ok=True)


def load_patched(src: pathlib.Path, name: str):
    code = src.read_text()
    code = code.replace("/mnt/data/ai-industrial-predictive-safety-p4", str(PKG / "software"))
    code = code.replace("/mnt/data/P4_7of10_revision/results", str(RERUN))
    patched = RERUN / f"_patched_{name}.py"
    patched.write_text(code)
    spec = importlib.util.spec_from_file_location(name, patched)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # analyze script runs at import; benchmark defines main()
    return mod


t0 = time.time()
print("=== stage 1: benchmark (controlled + C-MAPSS crossfit) ===", flush=True)
bench = load_patched(PKG / "revision" / "code" / "run_p4_revision_benchmark.py", "p4bench")
bench.main()
print(f"benchmark finished in {time.time() - t0:.0f}s", flush=True)

print("=== stage 2: engine-macro bootstrap analyzer ===", flush=True)
load_patched(PKG / "revision" / "code" / "analyze_crossfit_fast.py", "p4analyze")
print(f"analyzer finished at {time.time() - t0:.0f}s", flush=True)

print("=== stage 3: diff vs archived results ===", flush=True)
import pandas as pd

CHECKS = [
    ("controlled_summary.csv", ["scenario", "method"],
     ["mcc_mean", "far_post_mean", "cost_mean", "updates_mean"]),
    ("cmapss_fold_summary.csv", ["subset", "method"], ["mcc_mean", "far_mean", "cost_mean"]),
    ("cmapss_engine_bootstrap.csv", ["subset", "method"],
     ["mcc_macro", "mcc_lo", "mcc_hi", "far_macro", "expected_cost_macro"]),
]
worst_overall = 0.0
for fname, key, cols in CHECKS:
    new = pd.read_csv(RERUN / fname)
    old = pd.read_csv(PKG / "revision" / "results" / fname)
    m = new.merge(old, on=key, suffixes=("_new", "_old"))
    print(f"{fname}: rows new={len(new)} old={len(old)} matched={len(m)}")
    for c in cols:
        if f"{c}_new" not in m.columns:
            print(f"  {c}: MISSING")
            continue
        d = (m[f"{c}_new"] - m[f"{c}_old"]).abs().max()
        worst_overall = max(worst_overall, d)
        print(f"  {c:>22}: max |diff| = {d:.3e}")
print(f"WORST deviation across all checks: {worst_overall:.3e}")
print("Manuscript-facing values live in cmapss_engine_bootstrap.csv and controlled_summary.csv; "
      "a deviation <= 5e-4 keeps every 3-decimal number in the paper unchanged.")
print(f"total {time.time() - t0:.0f}s")
