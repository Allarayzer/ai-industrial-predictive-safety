"""Re-execute the channel-reliability stress test (E2) from the archived generator code and
compare bit-level/tolerance against the archived CSVs — reproducibility confirmation for the
paper's central experiment (claim C2).

Imports the archived module unmodified and redirects its OUT directory so the archive is never
written to.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "paper2-incoming" / "P4_Q1_literature_rebuilt_final"
OUT = ROOT / "analysis2" / "results" / "rerun_channel"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("q1ext", PKG / "code" / "run_q1_extension.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["q1ext"] = mod
spec.loader.exec_module(mod)
mod.OUT = OUT  # redirect all writes away from the archive

R, S = mod.run_channel_failure_controlled()

arch_R = pd.read_csv(PKG / "results" / "channel_failure_results.csv")
arch_S = pd.read_csv(PKG / "results" / "channel_failure_summary.csv")

key = ["seed", "method"]
merged = R.merge(arch_R, on=key, suffixes=("_new", "_old"))
num_cols = [c for c in R.columns if c not in key and R[c].dtype.kind in "fi"]
print(f"rows: rerun {len(R)} vs archive {len(arch_R)}; matched {len(merged)}")
worst = 0.0
for c in num_cols:
    a, b = merged[f"{c}_new"], merged[f"{c}_old"]
    d = (a - b).abs().max()
    worst = max(worst, d)
    print(f"  {c:>18}: max |diff| = {d:.3e}")
print(f"WORST deviation: {worst:.3e}")

ms = S.merge(arch_S, on="method", suffixes=("_new", "_old"))
for c in ["post_mcc_mean", "cost_mean", "far_mean", "updates_mean", "weight_updates_mean"]:
    d = (ms[f"{c}_new"] - ms[f"{c}_old"]).abs().max()
    print(f"  summary {c:>20}: max |diff| = {d:.3e}")
print("Headline check — governed post_mcc_mean:",
      float(S.set_index('method').loc['Governed escalation', 'post_mcc_mean']),
      "(paper: 0.688)")
