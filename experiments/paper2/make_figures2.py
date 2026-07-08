"""Paper-2 figures, drawn programmatically from analysis2/results CSVs (no manual numbers).

Style matches Paper 1 (analysis/scripts/make_diagrams.py): matplotlib, DejaVu Sans,
Okabe-Ito colorblind-safe palette, FancyBboxPatch boxes; Figure 1 legend: solid = online
scoring path, dashed = offline / delayed-label governance path.

Outputs Figure_1..Figure_5 (.pdf + .png 400 dpi) and caption .txt files to analysis2/paper/figures/.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "analysis2" / "results"
FIGDIR = ROOT / "analysis2" / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "yellow": "#F0E442", "grey": "#E8E8E8", "dark": "#222222"}
plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans"})

POLICY_COLOR = {"Static": OK["dark"], "Static matched": OK["dark"],
                "Scheduled": OK["yellow"], "Scheduled rolling": OK["yellow"],
                "ACI": OK["sky"], "Rolling ACI": OK["blue"],
                "Quantile tracking": OK["green"], "SF-OGD": OK["orange"],
                "Guarded full": OK["purple"], "Governed escalation": OK["purple"]}


def box(ax, x, y, w, h, text, fc, fontsize=10.5, bold=False, ec="#333333", tc="#111111", ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.2, edgecolor=ec, facecolor=fc, alpha=0.92, linestyle=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            color=tc, wrap=True, zorder=5)


def arrow(ax, x1, y1, x2, y2, style="-|>", color="#333333", lw=1.6, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=14, lw=lw, color=color, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}",
                 shrinkA=2, shrinkB=2, zorder=3))


def save(fig, n, caption):
    fig.savefig(FIGDIR / f"Figure_{n}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"Figure_{n}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    (FIGDIR / f"Figure_{n}_caption.txt").write_text(caption + "\n")
    print(f"  wrote Figure_{n}.pdf/.png")


def figure1_architecture():
    fig, ax = plt.subplots(figsize=(10.8, 7.4))
    ax.set_xlim(0, 103)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Channel services (top)
    box(ax, 3, 86, 28, 11, "Anomaly channel\n(service, own cadence)\n→ $R_{anom}$", OK["blue"], fontsize=9.5, tc="white")
    box(ax, 36, 86, 28, 11, "RUL channel\n(service, own cadence)\n→ $R_{RUL}$", OK["green"], fontsize=9.5, tc="white")
    box(ax, 69, 86, 28, 11, "Supervised risk channel\n(service, own cadence)\n→ $R_{sup}$", OK["orange"], fontsize=9.5)

    # Asynchronous delivery layer
    box(ax, 14, 71, 72, 9,
        "Asynchronous delivery: last value per channel, missing updates, maximum valid age $a_j$",
        OK["grey"], fontsize=9.5)

    # Fusion + threshold (online path)
    box(ax, 8, 55, 46, 10,
        "Freshness-aware fusion\n$R_t=\\sum_j w_j^{*}(t)\\, s_j(t_j)$,  renormalized weights",
        OK["purple"], fontsize=9.5, bold=True, tc="white")
    box(ax, 62, 55, 33, 10, "Alarm decision\n$R_t > q_t$ ?", OK["sky"], fontsize=10, bold=True)
    box(ax, 62, 40, 33, 8, "Alarm / no alarm\n→ operator", OK["dark"], fontsize=9.5, bold=True, tc="white")

    # Governance monitor (offline, delayed labels)
    box(ax, 3, 8, 39, 27,
        "Governance monitor\n(delayed trusted labels)\n\nFAR vs target •\nfused ranking •\nper-channel ranking",
        OK["grey"], fontsize=9.5, ls="--")
    box(ax, 50, 27.5, 47, 7.5, "Action 1: validated threshold recalibration\n(jump-limited)", OK["sky"], fontsize=8.4)
    box(ax, 50, 17.75, 47, 7.5, "Action 2: bounded simplex weight update\n(validated, shrunk toward previous, logged)", OK["green"], fontsize=8.4, tc="white")
    box(ax, 50, 8, 47, 7.5, "Action 3: model-review / abstention flag\n(no automated repair)", OK["red"], fontsize=8.4, tc="white")

    # Online (solid) arrows
    for xc in (17, 50, 83):
        arrow(ax, xc, 86, xc, 80)
    arrow(ax, 50, 71, 31, 65)
    arrow(ax, 54, 60, 62, 60)
    arrow(ax, 78, 55, 78, 48)

    # Offline / delayed (dashed) arrows
    arrow(ax, 63, 40, 28, 35, ls="--", color="#777777")     # delayed outcomes -> monitor
    arrow(ax, 42, 31.25, 50, 31.25, ls="--", color="#777777")
    arrow(ax, 42, 21.5, 50, 21.5, ls="--", color="#777777")
    arrow(ax, 42, 11.75, 50, 11.75, ls="--", color="#777777")
    # Action 1 updates q_t: curve right around the operator box up to the alarm decision
    arrow(ax, 96, 35, 94, 55, ls="--", color="#777777", rad=-0.45)
    # Action 2 updates the fusion weights: pass between monitor (x<=42) and actions (x>=50)
    arrow(ax, 49, 25.25, 31, 55, ls="--", color="#777777")

    ax.legend(handles=[Line2D([0], [0], color="#333333", lw=1.8, label="online scoring path (solid)"),
                       Line2D([0], [0], color="#777777", lw=1.8, ls="--",
                              label="offline / delayed-label governance path (dashed)")],
              loc="lower left", fontsize=8.8, frameon=False, bbox_to_anchor=(0.0, -0.02))

    save(fig, 1,
         "Figure 1. Governed online decision layer for asynchronous risk fusion. Three "
         "independently deployed channel services emit anomaly, remaining-useful-life, and "
         "supervised failure-risk scores at their own cadences; an asynchronous delivery layer "
         "retains the last value of each channel subject to missing updates and a maximum valid "
         "age. The weights of fresh channels are renormalized on the simplex and the fresh scores "
         "are fused into a single "
         "value that is compared with the current alarm threshold (solid arrows: online scoring "
         "path). Delayed trusted outcomes feed a governance monitor that tracks the empirical "
         "false-alarm rate together with fused and per-channel ranking quality (dashed arrows: "
         "offline path). The monitor selects among three bounded actions: a validated, "
         "jump-limited threshold recalibration when only the operating point has drifted; a "
         "validated, shrinkage-limited simplex weight update when relative channel reliability "
         "has changed; or a model-review/abstention flag when no channel retains adequate "
         "ranking, in which case no automated repair is attempted.")


def figure2_location_shift():
    c = pd.read_csv(RES / "controlled_summary.csv")
    sf = pd.read_csv(RES / "controlled_sfogd_summary.csv")
    scen = ["sudden", "gradual", "recurring"]
    pols = ["Static matched", "Rolling ACI", "Quantile tracking", "Guarded full", "SF-OGD"]
    labels = ["Static", "Rolling ACI", "Quantile tracking", "Guarded", "SF-OGD"]

    def val(s, p, col):
        if p == "SF-OGD":
            r = sf[sf.scenario == s]
            return float(r[col.replace("far_post_mean", "far_post_mean")].iloc[0])
        r = c[(c.scenario == s) & (c.method == p)]
        return float(r[col].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(scen))
    wbar = 0.16
    for ax, col, title, hline in ((axes[0], "mcc_mean", "MCC (whole stream)", None),
                                  (axes[1], "far_post_mean", "Post-onset FAR", 0.05)):
        for k, (p, lab) in enumerate(zip(pols, labels)):
            vals = [val(s, p, col) for s in scen]
            ax.bar(x + (k - 2) * wbar, vals, wbar * 0.92, label=lab,
                   color=POLICY_COLOR[p], edgecolor="#333333", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(scen)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        if hline is not None:
            ax.axhline(hline, color="#777777", ls="--", lw=1.0)
            ax.text(2.42, hline + 0.012, "target 0.05", fontsize=8, color="#666666")
    axes[0].set_ylim(0, 0.9)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, 2,
         "Figure 2. Controlled location-shift benchmark (mean over 10 seeds). Left: MCC over the "
         "full evaluation stream for the sudden, gradual, and recurring shift scenarios. Right: "
         "false-alarm rate among post-onset normal events against the 0.05 target (dashed line). "
         "Continuous rolling adaptive conformal inference and proportional quantile tracking "
         "restore false-alarm control most completely but change state after every trusted-normal "
         "feedback (7,609 updates); the guarded policy changes state only 2.1-5.7 times on "
         "average yet recovers only part of the discrimination, and the bounded one-sided SF-OGD "
         "specialization controls false alarms near 0.10 while remaining weak on MCC and cost. "
         "Static deployment-matched calibration fails under every persistent shift. Bars are "
         "seed means; standard deviations for every scenario-policy pair appear in Supplementary "
         "Table S1.")


def figure3_channel():
    s = pd.read_csv(RES / "channel_failure_summary.csv")
    order = ["Static", "Scheduled", "ACI", "Quantile tracking", "SF-OGD", "Governed escalation"]
    s = s.set_index("method").loc[order].reset_index()
    labels = ["Static", "Scheduled", "ACI", "Quantile\ntracking", "SF-OGD", "Governed\nescalation"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(order))
    colors = [POLICY_COLOR[m] for m in order]
    axes[0].bar(x, s.post_mcc_mean, 0.62, color=colors, edgecolor="#333333", linewidth=0.5)
    axes[0].set_title("Post-shift MCC", fontsize=11)
    axes[1].bar(x, s.cost_mean, 0.62, color=colors, edgecolor="#333333", linewidth=0.5)
    axes[1].set_title("Expected cost (10:1)", fontsize=11)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 0.78)
    for i, (t, w) in enumerate(zip(s.updates_mean, s.weight_updates_mean)):
        axes[0].text(i, float(s.post_mcc_mean.iloc[i]) + 0.015,
                     f"{t:g}t/{w:g}w", ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout()
    save(fig, 3,
         "Figure 3. Controlled channel-reliability shift (mean over 10 seeds). After the change "
         "point the supervised channel loses ranking information while the RUL channel remains "
         "informative. Left: MCC restricted to the post-shift window, annotated with the mean "
         "number of accepted threshold (t) and weight (w) state changes per run. Right: expected "
         "cost at the 10:1 cost ratio. Threshold-only policies "
         "recover the operating point at best partially, whereas the governed controller "
         "validates a sparse, bounded weight update on a chronological holdout (1.5 accepted "
         "weight changes on average) and attains the highest post-shift MCC and the lowest "
         "cost. All 15 paired Wilcoxon comparisons of the governed policy against the five "
         "alternatives on post-shift MCC, expected cost, and full-stream MCC remain significant "
         "after Holm correction (adjusted p = 0.029) with maximal Cliff's delta.")


def figure4_cmapss():
    b = pd.read_csv(RES / "cmapss_engine_bootstrap.csv")
    methods = ["Static matched", "Quantile tracking", "Guarded full"]
    labels = ["Static matched", "Quantile tracking", "Guarded"]
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    x = np.arange(len(subsets))
    wbar = 0.24
    for k, (m, lab) in enumerate(zip(methods, labels)):
        rows = b[b.method == m].set_index("subset").loc[subsets]
        y = rows.mcc_macro.to_numpy()
        lo = y - rows.mcc_lo.to_numpy()
        hi = rows.mcc_hi.to_numpy() - y
        ax.bar(x + (k - 1) * wbar, y, wbar * 0.9, label=lab, color=POLICY_COLOR[m],
               edgecolor="#333333", linewidth=0.5, yerr=[lo, hi], capsize=3,
               error_kw=dict(lw=1.0, ecolor="#333333"))
    ax.set_ylim(0.70, 0.87)
    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.set_ylabel("Engine-macro MCC")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, 4,
         "Figure 4. Five-fold engine-disjoint C-MAPSS replay (709 test engines across FD001-"
         "FD004; simulated data). Bars show engine-macro MCC with 95% engine-cluster bootstrap "
         "intervals (5,000 resamples). Without an injected deployment shift, deployment-matched "
         "static calibration is strongest or statistically indistinguishable everywhere; "
         "quantile tracking yields at most a small FD003 improvement, and the guarded policy "
         "accepts no changes at all in FD002 and FD004, matching static behavior there. The "
         "bounded one-sided SF-OGD specialization (not shown; supplementary tables) is "
         "substantially worse on every subset.")


def figure5_battery():
    s = pd.read_csv(RES / "battery_summary_v2.csv")
    order = ["Static", "Scheduled", "ACI", "Quantile tracking", "SF-OGD", "Governed escalation"]
    s = s.set_index("method").loc[order].reset_index()
    labels = ["Static", "Sched.", "ACI", "Quant.\ntrack.", "SF-OGD", "Governed\nescal."]
    colors = [POLICY_COLOR[m] for m in order]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))
    panels = [("mcc_mean", "MCC"), ("cost_mean", "Expected cost (10:1)"), ("recall_mean", "Recall")]
    x = np.arange(len(order))
    for ax, (col, title) in zip(axes, panels):
        ax.bar(x, s[col], 0.62, color=colors, edgecolor="#333333", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    for i, u in enumerate(s.updates_mean):
        axes[0].text(i, float(s.mcc_mean.iloc[i]) + 0.015, f"{u:g}", ha="center",
                     fontsize=7.5, color="#555555")
    fig.tight_layout()
    save(fig, 5,
         "Figure 5. Physical NASA lithium-ion battery replay, macro-averaged over four held-out "
         "cells (cell-disjoint train/calibration/test split; leak-free feature set in which "
         "current-cycle capacity is not reconstructible from the predictors). Panels show MCC "
         "(annotated with mean accepted policy updates), expected cost at the 10:1 ratio, and "
         "recall. Static deployment-matched calibration attains the highest cell-macro MCC, "
         "while governed escalation selects a different cost-sensitive operating point: the "
         "lowest expected cost among all policies and near-maximal recall (only scheduled "
         "recalibration reaches recall 1.000, at a false-alarm rate of 0.565), using 4.25 "
         "accepted updates per cell, all of which were threshold actions (no weight updates or "
         "escalations were triggered). With only four cells these are descriptive summaries; "
         "per-cell results are given in the supplementary tables.")


def figure6_diagnosis():
    conf = pd.read_csv(RES / "diagnosis_confusion.csv", index_col=0)
    counts = pd.read_csv(RES / "diagnosis_action_counts.csv", index_col=0)
    reg_order = ["none", "calibration", "channel", "model"]
    reg_labels = ["No shift", "Calibration\ndrift", "Channel-\nreliability", "Model\ndrift"]
    conf = conf.reindex(reg_order)
    counts = counts.reindex(reg_order)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    # Panel A: confusion matrix heatmap (counts out of 10 streams)
    ax = axes[0]
    M = conf.to_numpy(float)
    col_labels = ["No structural\naction", "Reweight\n(channel)", "Escalate\n(model)"]
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=10, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(reg_labels, fontsize=9)
    ax.set_xlabel("Controller's structural verdict", fontsize=9.5)
    ax.set_ylabel("True failure mode", fontsize=9.5)
    ax.set_title("Failure-mode diagnosis (streams, n=10 each)", fontsize=10.5)
    for i in range(4):
        for j in range(3):
            v = int(M[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                    color="white" if v >= 6 else "#222222", fontweight="bold")

    # Panel B: per-run diagnostic-action counts, log scale (weight 0.3-1.5, escalation 0.1-22)
    ax = axes[1]
    x = np.arange(4); wbar = 0.36
    ax.bar(x - wbar / 2, counts["weight"].to_numpy() + 1e-2, wbar, label="Weight actions",
           color=OK["green"], edgecolor="#333333", linewidth=0.5)
    ax.bar(x + wbar / 2, counts["escalation"].to_numpy() + 1e-2, wbar, label="Escalations",
           color=OK["red"], edgecolor="#333333", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 40)
    ax.set_xticks(x); ax.set_xticklabels(reg_labels, fontsize=9)
    ax.set_ylabel("Actions per run (mean, log scale)", fontsize=9.5)
    ax.set_title("Diagnostic actions by failure mode", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False)
    for i in range(4):
        ax.text(i - wbar / 2, counts["weight"].iloc[i] + 0.03, f"{counts['weight'].iloc[i]:.1f}",
                ha="center", fontsize=7.5, color="#555555")
        ax.text(i + wbar / 2, counts["escalation"].iloc[i] + 0.03, f"{counts['escalation'].iloc[i]:.1f}",
                ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout()
    save(fig, 6,
         "Figure 6. Direct evaluation of failure-mode diagnosis. The governed controller is run on "
         "four controlled regimes with a known ground-truth failure mode (10 seeds each), logging "
         "its structural decision at every review. Left: confusion matrix of the per-stream "
         "structural verdict against the true failure mode (counts out of 10). The channel-"
         "reliability regime is diagnosed as a reweight in 10/10 streams and the model-drift "
         "regime as an escalation in 10/10; the ranking-preserved regimes (no shift, calibration "
         "drift) are correctly handled without a structural action in 15/20 streams, the five "
         "errors being unnecessary single reweights (never a false escalation) that do not "
         "materially change stream-level metrics. Right: mean "
         "number of weight actions and escalations per run by regime (log scale). Weight actions "
         "rise from 0.3 under intact ranking to 1.5 once a channel fails, and escalations rise "
         "from at most 0.2 to 22.1 only when all channels lose ranking, so the escalation signal "
         "cleanly separates model drift from channel-reliability drift.")


if __name__ == "__main__":
    print("Generating Paper-2 figures ->", FIGDIR)
    figure1_architecture()
    figure2_location_shift()
    figure3_channel()
    figure4_cmapss()
    figure5_battery()
    figure6_diagnosis()
    print("done")
