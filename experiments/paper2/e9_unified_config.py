"""Reviewer point 3: run the governed controller with a SINGLE shared dimensionless
configuration across the channel-reliability and battery experiments, to show the conclusions
do not depend on per-experiment constant choices.

Scale-dependent constants (review cadence, buffer size) necessarily track stream length and
label delay and are kept at their per-experiment values; the dimensionless acceptance thresholds
are unified:
    validation share       = 0.70   (channel 0.70 / battery 0.70 -> unchanged)
    escalation AUC floor    = 0.57   (channel 0.58 / battery 0.56 -> unified)
    cost-improvement margin = 0.97   (channel 0.98 / battery 0.95 -> unified)

Reuses the parameterized loops validated in e5_governed_sensitivity.py (base configs reproduce
the paper numbers). Outputs analysis2/results/unified_config.csv (-> Supplementary Table S9b).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis2" / "results"

spec = importlib.util.spec_from_file_location("e5", ROOT / "analysis2" / "scripts" / "e5_governed_sensitivity.py")
# e5 runs its grid on import; avoid that by loading the module object without executing __main__.
src = (ROOT / "analysis2" / "scripts" / "e5_governed_sensitivity.py").read_text()
src = src.split("CH_BASE = dict(")[0]  # keep only the function definitions
mod = importlib.util.module_from_spec(spec)
sys.modules["e5"] = mod
exec(compile(src, str(spec.origin), "exec"), mod.__dict__)

UNIFIED = dict(train_share=0.70, auc_floor=0.57, accept_cost=0.97)

# channel: base for reference, then unified
ch_base = pd.DataFrame([mod.channel_governed(s) for s in range(10)]).mean()
ch_uni = pd.DataFrame([mod.channel_governed(s, train_share=0.70, auc_floor=0.57, accept_cost=0.97)
                       for s in range(10)]).mean()
bt_base = pd.Series(mod.battery_governed())
bt_uni = pd.Series(mod.battery_governed(train_share=0.70, auc_floor=0.57, accept_cost=0.97))

rows = [
    dict(experiment="channel", config="per-experiment (paper)", post_mcc=round(ch_base.post_mcc, 3),
         expected_cost=round(ch_base.expected_cost, 3), weight_updates=round(ch_base.weight_updates, 2),
         escalations=round(ch_base.escalations, 2)),
    dict(experiment="channel", config="unified dimensionless", post_mcc=round(ch_uni.post_mcc, 3),
         expected_cost=round(ch_uni.expected_cost, 3), weight_updates=round(ch_uni.weight_updates, 2),
         escalations=round(ch_uni.escalations, 2)),
    dict(experiment="battery", config="per-experiment (paper)", mcc=round(bt_base.mcc, 3),
         expected_cost=round(bt_base.expected_cost, 3), recall=round(bt_base.recall, 3),
         weight_updates=round(bt_base.weight_updates, 2)),
    dict(experiment="battery", config="unified dimensionless", mcc=round(bt_uni.mcc, 3),
         expected_cost=round(bt_uni.expected_cost, 3), recall=round(bt_uni.recall, 3),
         weight_updates=round(bt_uni.weight_updates, 2)),
]
df = pd.DataFrame(rows)
df.to_csv(OUT / "unified_config.csv", index=False)
print(df.to_string(index=False))
print()
print(f"channel unified post_mcc {ch_uni.post_mcc:.3f} vs ACI 0.589 -> "
      f"{'beats' if ch_uni.post_mcc > 0.589 else 'FAILS'}; "
      f"cost {ch_uni.expected_cost:.3f} vs ACI 0.458 -> "
      f"{'beats' if ch_uni.expected_cost < 0.458 else 'FAILS'}")
print(f"battery unified cost {bt_uni.expected_cost:.3f} vs static 0.174 -> "
      f"{'beats' if bt_uni.expected_cost < 0.174 else 'FAILS'}")
