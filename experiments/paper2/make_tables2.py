"""Paper-2 tables generated directly from analysis2/results CSVs (no manual transcription).

Main-text tables (display budget: 5 figures + 3 tables = 8):
  Table 1 - channel-reliability shift summary
  Table 2 - C-MAPSS engine-macro with bootstrap CIs
  Table 3 - battery replay v2 (leak-free)
Supplementary tables (S1-S3) as CSVs ready for the supplement:
  S1 - controlled location shift, all scenarios/policies incl. 'none' and SF-OGD
  S2 - paired Wilcoxon with Holm correction and Cliff's delta (channel test)
  S3 - Wilson 95% CIs for FAR (channel test)

Writes LaTeX fragments to analysis2/paper/tables/table{1,2,3}.tex and caption .txt files,
plus supplementary CSVs to analysis2/paper/supplement/.
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "analysis2" / "results"
TBL = ROOT / "analysis2" / "paper" / "tables"
SUP = ROOT / "analysis2" / "paper" / "supplement"
TBL.mkdir(parents=True, exist_ok=True)
SUP.mkdir(parents=True, exist_ok=True)


def latex_table(caption, header, rows, colspec=None, label=None):
    ncol = len(header)
    colspec = colspec or ("l" + "r" * (ncol - 1))
    out = ["\\begin{table}[htbp]\\centering\\footnotesize",
           f"\\caption{{{caption}}}"]
    if label:
        out.append(f"\\label{{{label}}}")
    out.append(f"\\begin{{tabular}}{{{colspec}}}")
    out.append("\\toprule")
    out.append(" & ".join(header) + r" \\")
    out.append("\\midrule")
    for r in rows:
        out.append(" & ".join(str(c) for c in r) + r" \\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


def f3(x):
    return f"{float(x):.3f}"


def table1_channel():
    s = pd.read_csv(RES / "channel_failure_summary.csv")
    order = ["Static", "Scheduled", "ACI", "Quantile tracking", "SF-OGD", "Governed escalation"]
    s = s.set_index("method").loc[order].reset_index()
    rows = [[m.method, f3(m.post_mcc_mean), f3(m.far_mean), f3(m.cost_mean),
             f"{m.updates_mean:g}", f"{m.weight_updates_mean:g}"] for m in s.itertuples()]
    cap = ("Channel-reliability-shift stress test: post-shift MCC, false-alarm rate among normal "
           "events, expected cost at the 10:1 ratio, and mean accepted threshold and weight state "
           "changes (mean over 10 seeds). All governed-versus-comparator differences on post-shift "
           "MCC, expected cost, and full-stream MCC are significant after Holm correction "
           "(exact paired Wilcoxon, adjusted $p=0.029$, $|\\delta_{\\mathrm{Cliff}}|=1$; "
           "Supplementary Table S2).")
    (TBL / "table1.tex").write_text(latex_table(
        cap, ["Policy", "Post-shift MCC", "FAR", "Cost", "Thr.\\ updates", "Wt.\\ updates"], rows,
        label="tab:channel"))
    print("  table1.tex")


def cmapss_supplement_table():
    # C-MAPSS three-method engine-macro view — moved to supplement (S5b) to stay within the
    # Scientific Reports 8-display-item budget; the all-methods S5 already exists.
    b = pd.read_csv(RES / "cmapss_engine_bootstrap.csv")
    methods = ["Static matched", "Quantile tracking", "Guarded full"]
    rows = []
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        for m in methods:
            r = b[(b.subset == sub) & (b.method == m)].iloc[0]
            rows.append(dict(subset=sub, method=m,
                             mcc_macro=round(float(r.mcc_macro), 3),
                             mcc_lo=round(float(r.mcc_lo), 3), mcc_hi=round(float(r.mcc_hi), 3),
                             far_macro=round(float(r.far_macro), 3),
                             expected_cost_macro=round(float(r.expected_cost_macro), 3)))
    pd.DataFrame(rows).to_csv(SUP / "S5b_cmapss_main_three_methods.csv", index=False)
    print("  S5b_cmapss_main_three_methods.csv")


def table3_battery():
    s = pd.read_csv(RES / "battery_summary_v2.csv")
    order = ["Static", "Scheduled", "ACI", "Quantile tracking", "SF-OGD", "Governed escalation"]
    s = s.set_index("method").loc[order].reset_index()
    rows = [[m.method, f3(m.mcc_mean), f3(m.far_mean), f3(m.recall_mean), f3(m.cost_mean),
             f"{m.updates_mean:g}"] for m in s.itertuples()]
    cap = ("Physical NASA lithium-ion battery replay, macro-averaged over four held-out cells "
           "(cell-disjoint splits; leak-free feature set). MCC, false-alarm rate, recall, "
           "expected cost at the 10:1 ratio, and mean accepted policy updates. With four cells "
           "the summaries are descriptive; per-cell outcomes are in Supplementary Table S4.")
    # Battery is main-text Table 2 (C-MAPSS detail moved to Supplementary Table S5 to keep the
    # 8-item Scientific Reports display budget: 6 figures + 2 tables).
    (TBL / "table2.tex").write_text(latex_table(
        cap, ["Policy", "MCC", "FAR", "Recall", "Cost", "Updates"], rows, label="tab:battery"))
    print("  table2.tex (battery)")


def supplement():
    # S1: controlled location shift, full grid incl. none + SF-OGD merged, with per-seed SDs
    # recomputed from the raw per-seed files so every scenario-policy pair carries mcc/far_post/
    # cost standard deviations (the archived summaries store only mcc_sd).
    raw = pd.concat([
        pd.read_csv(RES / "controlled_results.csv"),
        pd.read_csv(RES / "controlled_sfogd_results.csv").assign(method="SF-OGD"),
    ], ignore_index=True)
    s1 = (raw.groupby(["scenario", "method"])
             .agg(mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
                  far_post_mean=("far_post", "mean"), far_post_sd=("far_post", "std"),
                  cost_mean=("expected_cost", "mean"), cost_sd=("expected_cost", "std"),
                  updates_mean=("updates", "mean"))
             .reset_index().sort_values(["scenario", "method"]))
    # consistency guard: seed-mean recomputation must reproduce the archived summary
    c = pd.read_csv(RES / "controlled_summary.csv")
    chk = s1.merge(c, on=["scenario", "method"], suffixes=("", "_arch"))
    assert (chk.mcc_mean - chk.mcc_mean_arch).abs().max() < 1e-9
    s1.to_csv(SUP / "S1_controlled_location_shift_full.csv", index=False)
    # S2 / S3: stats
    pd.read_csv(RES / "channel_failure_paired_statistics_v2.csv").to_csv(
        SUP / "S2_channel_paired_wilcoxon_holm_cliff.csv", index=False)
    pd.read_csv(RES / "channel_failure_far_wilson.csv").to_csv(
        SUP / "S3_channel_far_wilson_ci.csv", index=False)
    # S4: battery per-cell
    pd.read_csv(RES / "battery_fold_results_v2.csv").to_csv(
        SUP / "S4_battery_per_cell_v2.csv", index=False)
    # S5: C-MAPSS all methods engine-macro + SF-OGD engine-macro (SF-OGD file names its cost
    # column cost_macro; coalesce into expected_cost_macro so the rendered table has no gaps)
    b = pd.read_csv(RES / "cmapss_engine_bootstrap.csv")
    bs = pd.read_csv(RES / "cmapss_sfogd_engine_bootstrap.csv")
    s5 = pd.concat([b, bs], ignore_index=True)
    if "cost_macro" in s5.columns:
        s5["expected_cost_macro"] = s5["expected_cost_macro"].fillna(s5["cost_macro"])
    s5.to_csv(SUP / "S5_cmapss_engine_macro_all_methods.csv", index=False)
    # S9: governed-controller constants sensitivity (OAT); S10: randomized-shift robustness
    pd.read_csv(RES / "governed_sensitivity.csv").to_csv(
        SUP / "S9_governed_sensitivity_oat.csv", index=False)
    pd.read_csv(RES / "randomized_shift_summary.csv").to_csv(
        SUP / "S10_randomized_shift_summary.csv", index=False)
    pd.read_csv(RES / "randomized_shift_results.csv").to_csv(
        SUP / "S10raw_randomized_shift_results.csv", index=False)
    # S11: NASA Randomized Battery Usage replication
    pd.read_csv(RES / "rw_summary.csv").to_csv(SUP / "S11_rw_battery_summary.csv", index=False)
    pd.read_csv(RES / "rw_fold_results.csv").to_csv(SUP / "S11raw_rw_fold_results.csv", index=False)
    # S12: failure-mode diagnosis (confusion + per-run action counts + delays)
    pd.read_csv(RES / "diagnosis_confusion.csv").to_csv(SUP / "S12_diagnosis_confusion.csv", index=False)
    pd.read_csv(RES / "diagnosis_action_counts.csv").to_csv(SUP / "S12b_diagnosis_action_counts.csv", index=False)
    pd.read_csv(RES / "diagnosis_summary.csv").to_csv(SUP / "S12c_diagnosis_summary.csv", index=False)
    pd.read_csv(RES / "diagnosis_threshold_sensitivity.csv").to_csv(SUP / "S12d_diagnosis_threshold_sensitivity.csv", index=False)
    # S9b: unified dimensionless controller configuration
    pd.read_csv(RES / "unified_config.csv").to_csv(SUP / "S9b_unified_config.csv", index=False)
    print("  supplement CSVs")


if __name__ == "__main__":
    print("Generating Paper-2 tables ->", TBL)
    table1_channel()
    table3_battery()       # writes table2.tex (battery = main Table 2)
    cmapss_supplement_table()
    supplement()
    print("done")
