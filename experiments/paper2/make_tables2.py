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


def table2_cmapss():
    b = pd.read_csv(RES / "cmapss_engine_bootstrap.csv")
    methods = ["Static matched", "Quantile tracking", "Guarded full"]
    rows = []
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        for m in methods:
            r = b[(b.subset == sub) & (b.method == m)].iloc[0]
            rows.append([sub, m, f"{f3(r.mcc_macro)} [{f3(r.mcc_lo)}, {f3(r.mcc_hi)}]",
                         f3(r.far_macro), f3(r.expected_cost_macro)])
    cap = ("Five-fold engine-disjoint C-MAPSS replay: engine-macro MCC with 95\\% engine-cluster "
           "bootstrap intervals (5{,}000 resamples), false-alarm rate, and expected cost at the "
           "10:1 ratio; 709 test engines in total (100/260/100/249 in FD001--FD004). C-MAPSS is "
           "a simulation. Additional policies and the SF-OGD specialization are reported in "
           "Supplementary Table S5.")
    (TBL / "table2.tex").write_text(latex_table(
        cap, ["Subset", "Policy", "MCC [95\\% CI]", "FAR", "Cost"], rows,
        colspec="llrrr", label="tab:cmapss"))
    print("  table2.tex")


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
    (TBL / "table3.tex").write_text(latex_table(
        cap, ["Policy", "MCC", "FAR", "Recall", "Cost", "Updates"], rows, label="tab:battery"))
    print("  table3.tex")


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
    print("  supplement S1-S5 CSVs")


if __name__ == "__main__":
    print("Generating Paper-2 tables ->", TBL)
    table1_channel()
    table2_cmapss()
    table3_battery()
    supplement()
    print("done")
