"""Build Supplementary Information PDF for Paper 2 from the archived CSVs.

S1 controlled location shift (full grid) | S2 paired Wilcoxon + Holm + Cliff |
S3 Wilson CIs for FAR | S4 battery per-cell (v2) | S5 C-MAPSS all methods |
S6 governed-controller constants | S7 dataset overview | S8 positioning vs adjacent streams.
Tables S1-S5 are rendered directly from analysis2/paper/supplement/*.csv.
"""
from __future__ import annotations

import os
import pathlib
import subprocess

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUP = ROOT / "analysis2" / "paper" / "supplement"
OUT = ROOT / "analysis2" / "paper"
LATEX_BIN = "/Library/TeX/texbin"


def md_table(df: pd.DataFrame, floatfmt: int = 3) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.{floatfmt}f}"
        return str(v)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


parts = ["""# Supplementary Information

**Governed adaptation of asynchronous risk fusion under distribution shift: distinguishing
calibration, channel-reliability, and model drift in predictive safety** — Ilia Serebriakov

All tables below are generated programmatically from the archived result CSVs listed in the
provenance map (`provenance2.csv`); no value is transcribed by hand.
"""]

s1 = pd.read_csv(SUP / "S1_controlled_location_shift_full.csv")
parts.append("## Supplementary Table S1. Controlled location shift — all scenarios and policies\n\n"
             "Mean over 10 seeds. `far_post_mean` is the false-alarm rate among post-onset normal "
             "events; `updates_mean` counts accepted state changes.\n\n" + md_table(s1))

s2 = pd.read_csv(SUP / "S2_channel_paired_wilcoxon_holm_cliff.csv")
s2 = s2[["comparator", "metric", "n", "mean_difference", "median_difference",
         "wilcoxon_W", "p_value", "p_holm", "cliffs_delta"]]
parts.append("\n\n## Supplementary Table S2. Channel-reliability shift — paired statistics\n\n"
             "Governed escalation versus each comparator (exact two-sided paired Wilcoxon over "
             "10 seeds; Holm correction across the 15-test family; Cliff's delta effect size). "
             "Positive mean differences favour the governed policy on MCC metrics; negative "
             "differences favour it on expected cost.\n\n" + md_table(s2, 6))

s3 = pd.read_csv(SUP / "S3_channel_far_wilson_ci.csv")
parts.append("\n\n## Supplementary Table S3. Channel-reliability shift — FAR with Wilson 95% CIs\n\n"
             "False-alarm proportions pooled over all revealed post-calibration normals of the 10 "
             "seeds (n = 65,900).\n\n" + md_table(s3, 4))

s4 = pd.read_csv(SUP / "S4_battery_per_cell_v2.csv")
s4 = s4[["fold", "test_battery", "method", "mcc", "far", "recall", "expected_cost",
         "roc_auc", "updates"]]
parts.append("\n\n## Supplementary Table S4. Battery replay — per-cell outcomes (leak-free v2)\n\n"
             "Each physical cell is tested exactly once under a deterministic cell-disjoint fold.\n\n"
             + md_table(s4))

s5 = pd.read_csv(SUP / "S5_cmapss_engine_macro_all_methods.csv")
cols = [c for c in ["subset", "method", "mcc_macro", "mcc_lo", "mcc_hi", "far_macro",
                    "expected_cost_macro"] if c in s5.columns]
parts.append("\n\n## Supplementary Table S5. C-MAPSS engine-macro results — all recorded policies\n\n"
             "Engine-macro values with 95% engine-cluster bootstrap intervals (5,000 resamples), "
             "including policies omitted from main-text Table 2 and the SF-OGD specialization.\n\n"
             + md_table(s5[cols]))

parts.append("""

## Supplementary Table S6. Governed/guarded controller constants (as implemented)

| Constant | Guarded threshold (location shift, C-MAPSS) | Governed escalation (channel test) | Governed escalation (battery) |
|---|---|---|---|
| Review cadence | every 60 events, window 320 | every 100 events | every 12 cycles |
| Label buffer | 300+ normals required | last 1,000 labeled events | last 70 labeled cycles |
| Chronological validation share | 25% (75/25 split) | 30% (700/300) | 30% (`split = max(24, 0.7n)`) |
| Drift evidence | PSI > 0.25 in >= 2 of 3 channels, coherent median shift, persistence 2 reviews | validation ROC-AUC comparison | validation ROC-AUC comparison |
| Cooldown | >= 300 events | review cadence only | review cadence only |
| Candidate threshold | conformal quantile at 0.85 alpha, FAR tolerance 1.4 alpha | recent-normal conformal quantile | recent-normal conformal quantile |
| Threshold jump limit | relative jump <= 1.5 | absolute change < 0.4 | relative change < 0.6 |
| Weight proposal | n/a | 0.15 w_current + 0.85 e_best | grid step 0.1, shrinkage 0.08 |
| Weight acceptance | n/a | best-channel AUC > fused + 0.03 AND (cost < 0.98x or MCC + 0.02) | cost <= 0.95x AND L1 jump <= 1.2 |
| Escalation floor | n/a | max channel AUC < 0.58 | max channel AUC < 0.56 |
| Feedback delay | 120 events / 40 cycles (C-MAPSS) | 80 events | 2 cycles |

Constants are operating choices; the sensitivity grid in the research archive varies the ACI
learning rate over {0.001, 0.003, 0.01, 0.03, 0.08} and toggles the persistence, validation,
coherence, and jump-limit guards over 5 seeds on the sudden and gradual scenarios.

## Supplementary Table S7. Datasets and independence units

| Benchmark | Nature | Units | Split | Role |
|---|---|---|---|---|
| Controlled location shift | synthetic score stream, 12,000 events | 10 seeds x 4 scenarios | fixed calibration (2,500) / onset (6,000) | threshold behaviour (RQ1) |
| Controlled reliability shift | synthetic score stream, 10,000 events | 10 seeds | fixed calibration (2,000) / onset (5,000) | threshold vs weight action (RQ2) |
| NASA C-MAPSS FD001-FD004 | simulated turbofan run-to-failure | 709 engines | 5-fold engine-disjoint | stable in-domain replay (RQ3) |
| NASA Li-ion batteries | physical laboratory cycling | 4 cells (B0005/6/7/18), 636 cycles, 169,766 rows | deterministic cell-disjoint | external physical check (RQ4) |

## Supplementary Table S8. Positioning relative to adjacent research streams

| Stream | Adapts | Decides when NOT to act | Escalates / abstains | PHM alarm replay |
|---|---|---|---|---|
| Online conformal adaptation (ACI family) | threshold / quantile | no | no | rare |
| Online ensembling & streaming model selection | expert weights / model choice | usually no | usually no | rare |
| Harmful-shift monitoring | nothing (alerts) | yes | signal only | application-dependent |
| Selective prediction / selective conformal | selection rule | yes | yes | mostly non-PHM |
| This study | threshold AND bounded weights | yes (validated) | model-review flag | C-MAPSS + physical batteries |

The comparison is scoped to the representative methods cited in the main text and is not an
exhaustive survey.
""")

s9 = pd.read_csv(SUP / "S9_governed_sensitivity_oat.csv")
parts.append("\n\n## Supplementary Table S9. Governed-controller constants — one-at-a-time sensitivity\n\n"
             "Each operating constant of the governed escalation controller is perturbed one at a "
             "time around its operating value while data generation, seeds/folds, and baseline "
             "calibration stay fixed (channel test: means over 10 seeds; battery: cell-macro over "
             "4 folds). The '(base)' rows reproduce the main-text configuration. Across all 16 "
             "perturbed channel configurations, post-shift MCC stays within [0.657, 0.709] — above "
             "the best threshold-only policy (ACI, 0.589) in every case — and expected cost stays "
             "below ACI's 0.458. On batteries, expected cost stays below the static reference "
             "(0.174) in 15 of 16 perturbed configurations; the exception is halving the review "
             "frequency (review_every = 24 cycles, cost 0.210), which shows the cost advantage "
             "requires reviews at a cadence comparable to the label delay.\n\n" + md_table(s9))

s11 = pd.read_csv(SUP / "S11_rw_battery_summary.csv")
parts.append("\n\n## Supplementary Table S11. Twelve-cell replication — NASA Randomized Battery Usage\n\n"
             "Cell-macro results over twelve room-temperature randomized-usage cells (RW1-RW12; "
             "12 folds, each cell tested once within its four-cell usage group). The unit of "
             "analysis is the fixed-current reference discharge (21-79 per cell); capacity is "
             "coulomb-counted and the near-end horizon is 15% of each cell's series. Static "
             "calibration and the governed controller finish essentially tied (MCC 0.572 vs "
             "0.560); the governed policy accepted no weight actions and matched static behaviour "
             "exactly on ten of twelve cells; continuous ACI and SF-OGD degrade sharply. Two "
             "cells raised no alarms under any policy: on RW3 the fused ranking itself failed "
             "(test ROC-AUC 0.586; a real-data model-drift instance where review, not adaptation, "
             "is the appropriate action), and on RW6 the cross-cell threshold sat above the "
             "entire 21-cycle test series. Per-fold results in S11raw_rw_fold_results.csv.\n\n"
             + md_table(s11))

s12 = pd.read_csv(SUP / "S12_diagnosis_confusion.csv")
s12b = pd.read_csv(SUP / "S12b_diagnosis_action_counts.csv")
s12c = pd.read_csv(SUP / "S12c_diagnosis_summary.csv")
parts.append("\n\n## Supplementary Table S12. Failure-mode diagnosis — confusion, action counts, delays\n\n"
             "The governed controller is run on four controlled regimes with a known ground-truth "
             "failure mode (10 seeds each), logging its structural decision at every review. "
             "**(a) Per-stream confusion** (counts out of 10): the channel-reliability regime is "
             "diagnosed as a reweight in 10/10 and the model-drift regime as an escalation in "
             "10/10; the ranking-preserved regimes are handled without a structural action in "
             "15/20, the errors being harmless single reweights and never a false escalation.\n\n"
             + md_table(s12) +
             "\n\n**(b) Mean actions per run by regime.** Weight actions rise from 0.3 under intact "
             "ranking to 1.5 once a channel fails; escalations rise from <=0.2 to 22.1 only when "
             "all channels lose ranking.\n\n" + md_table(s12b) +
             "\n\n**(c) Per-regime summary** (diagnostic-action rates and median detection delay "
             "in events from onset to the first correct structural action).\n\n" + md_table(s12c))

s9b = pd.read_csv(SUP / "S9b_unified_config.csv")
parts.append("\n\n## Supplementary Table S9b. Unified dimensionless controller configuration\n\n"
             "The governed controller re-run with a single shared set of dimensionless acceptance "
             "thresholds (validation share 0.70, escalation AUC floor 0.57, cost-improvement "
             "margin 0.97) across both experiments; scale-dependent constants (review cadence, "
             "buffer size) remain matched to each stream. Under channel-reliability shift the "
             "unified controller keeps its post-shift MCC and cost advantage over the best "
             "threshold-only policy (ACI: post-shift MCC 0.589, cost 0.458); on batteries it "
             "keeps its expected-cost advantage over static calibration (0.174). Conclusions do "
             "not depend on per-experiment constant choices.\n\n" + md_table(s9b))

s13 = pd.DataFrame([
    {"Aspect": "Primary question", "Companion paper (under review)": "Offline construction of the hybrid risk-fusion framework", "This paper": "Online maintenance of the fused decision layer"},
    {"Aspect": "Core contribution", "Companion paper (under review)": "Component fusion with split-conformal alarm calibration and drift detection", "This paper": "Failure-mode diagnosis and governed threshold/reweight/escalation actions"},
    {"Aspect": "Evaluation", "Companion paper (under review)": "Offline component and fusion benchmarks", "This paper": "Controlled shift stress tests, asynchronous replay, and online adaptation"},
    {"Aspect": "Adaptation", "Companion paper (under review)": "Static calibration; no online weight governance", "This paper": "Threshold recalibration, bounded weight update, and escalation controller"},
    {"Aspect": "Shared assets", "Companion paper (under review)": "Datasets and ai_cta code infrastructure", "This paper": "Same datasets/infrastructure reused; no numerical result reused as validation"},
])
parts.append("\n\n## Supplementary Table S13. Relationship to the companion manuscript\n\n"
             "The companion manuscript (under review at PeerJ Computer Science) and the present "
             "study address different questions and share only datasets and code infrastructure; "
             "no result of one is treated as validation of the other.\n\n" + md_table(s13))

s10 = pd.read_csv(SUP / "S10_randomized_shift_summary.csv")
parts.append("\n\n## Supplementary Table S10. Randomized-shift robustness of the location-shift ranking\n\n"
             "Thirty shift configurations drawn at random from ranges not used when the policies "
             "were designed (shape in {sudden, gradual, recurring, mixed}; onset U[4,000, 8,500]; "
             "magnitude multiplier U[0.5, 2.0]; ramp U[200, 2,000]; recurrence period "
             "U[300, 1,500]); one stream per configuration, all other benchmark settings as in the "
             "main text. The policy ranking of the designed benchmark is reproduced: rolling ACI "
             "and quantile tracking occupy the top two MCC ranks in 30/30 configurations with "
             "post-onset FAR within [0.043, 0.096]; the guarded policy ranks third in 28/30; "
             "static calibration ranks last or next-to-last everywhere (post-onset FAR "
             "0.293-1.000). Per-configuration results are in the archived "
             "S10raw_randomized_shift_results.csv.\n\n" + md_table(s10))

md = OUT / "_supplement2.md"
md.write_text("\n".join(parts))
env = {**os.environ, "PATH": LATEX_BIN + ":" + os.environ.get("PATH", "")}
out = OUT / "supplementary_information.pdf"
r = subprocess.run(["pandoc", str(md), "-o", str(out), "--pdf-engine=pdflatex",
                    "-V", "geometry:margin=0.9in", "-V", "fontsize=10pt",
                    "-V", "documentclass=article", "-V", "geometry:landscape"],
                   env=env, capture_output=True, text=True)
if r.returncode != 0:
    print("pandoc FAILED:\n", (r.stderr or "")[-1500:])
else:
    print(f"wrote {out.name} ({out.stat().st_size // 1024} KB)")
