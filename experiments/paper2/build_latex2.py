"""Build the Paper-2 manuscript PDF/TEX (Scientific Reports structure) with the Paper-1
typography: pandoc -> pdflatex, Computer Modern, 1-inch margins, 11 pt.

Sections: title block + abstract -> Introduction -> Results -> Discussion -> Methods ->
statements -> numbered references (Nature style, square brackets).
Citations are written in the drafts as [Rxx] or [Rxx, Ryy] tokens and are renumbered here
by order of first appearance. Figures/tables are inserted at {{FIGn}} / {{TABn}} markers;
figure captions come from analysis2/paper/figures/Figure_n_caption.txt, tables from
analysis2/paper/tables/tablen.tex (raw LaTeX generated from the results CSVs).
"""
from __future__ import annotations

import csv
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
DRAFT = ROOT / "analysis2" / "draft"
FIGDIR = ROOT / "analysis2" / "paper" / "figures"
TBLDIR = ROOT / "analysis2" / "paper" / "tables"
OUTDIR = ROOT / "analysis2" / "paper"
REFS = ROOT / "analysis2" / "references_final.csv"

TITLE = ("Governed adaptation of asynchronous risk fusion under distribution shift: "
         "distinguishing calibration, channel-reliability, and model drift in predictive safety")
PANDOC = "pandoc"
LATEX_BIN = "/Library/TeX/texbin"

PREAMBLE = r"""
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\setlength{\emergencystretch}{3em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
"""

SECTIONS = ["introduction", "results", "discussion", "methods"]


def fig_block(n: int) -> str:
    png = FIGDIR / f"Figure_{n}.png"
    cap = (FIGDIR / f"Figure_{n}_caption.txt").read_text().strip()
    cap = re.sub(r"^Figure\s*\d+\.\s*", "", cap).replace("\n", " ")
    cap = cap.replace("%", r"\%").replace("&", r"\&")
    return f"\n\n![{cap}]({png}){{width=92%}}\n\n"


def tab_block(n: int) -> str:
    tex = (TBLDIR / f"table{n}.tex").read_text()
    return f"\n\n```{{=latex}}\n{tex}\n```\n\n"


def load_refs() -> dict[str, str]:
    return {r["key"]: r["nature_citation"] for r in csv.DictReader(open(REFS))}


def main() -> None:
    refs = load_refs()
    order: list[str] = []

    def renumber(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        nums = []
        for k in keys:
            if k not in refs:
                raise SystemExit(f"unknown reference key {k}")
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        return "[" + ",".join(str(n) for n in sorted(nums)) + "]"

    bodies = []
    for name in SECTIONS:
        text = (DRAFT / f"{name}.md").read_text()
        text = re.sub(r"\[((?:R\d{2})(?:\s*,\s*R\d{2})*)\]", renumber, text)
        text = re.sub(r"\{\{FIG(\d)\}\}", lambda m: fig_block(int(m.group(1))), text)
        text = re.sub(r"\{\{TAB(\d)\}\}", lambda m: tab_block(int(m.group(1))), text)
        bodies.append(text)

    abstract = (DRAFT / "abstract.md").read_text().strip()
    abstract = re.sub(r"(?<=\d)-(?=\d)", "--", abstract)  # en dashes; ORCID stays outside
    title_block = (
        "```{=latex}\n"
        "\\begin{center}\n"
        f"{{\\LARGE\\bfseries {TITLE}}}\\\\[1.2em]\n"
        "{\\large Ilia Serebriakov}\\\\[0.3em]\n"
        "{\\small Engineering Science, The City University of New York, New York, NY, USA}\\\\\n"
        "{\\small ORCID 0009-0009-1548-390X}\\\\[0.3em]\n"
        "{\\small Correspondence: ilia.serebriakov98@login.cuny.edu}\n"
        "\\end{center}\n"
        "\\vspace{0.5em}\\hrule\\vspace{1em}\n"
        "\\noindent\\textbf{Abstract.}\n"
        "```\n\n"
        + abstract
        + "\n\n```{=latex}\n\\vspace{0.5em}\\hrule\\vspace{1em}\n```\n"
    )

    data_availability = (
        "\n# Data availability {-}\n\n"
        "The NASA C-MAPSS turbofan simulation, the NASA Ames "
        "lithium-ion battery aging experiments, and the NASA Randomized Battery Usage "
        "experiments (Prognostics Data Repository; Zenodo mirror "
        "https://doi.org/10.5281/zenodo.15277374) are public benchmark datasets; the exact "
        "transported battery CSV is archived in the supplementary research archive with its "
        "SHA-256 checksum. All raw and aggregate result files behind every table and figure, "
        "per-event traces, and a claim-to-file provenance map are included in the supplementary "
        "archive. Code availability is stated in Methods.\n")
    back_matter = "\n".join([
        "\n# Author contributions {-}\n",
        "I.S. conceived the study, implemented the software and "
        "experiments, analyzed the results, prepared the figures, and wrote and reviewed the "
        "manuscript.\n",
        "\n# Competing interests {-}\n",
        "The author declares no competing interests.\n",
        "\n# Funding {-}\n",
        "This research received no external funding.\n",
    ])

    ref_lines = ["\n# References {-}\n",
                 "```{=latex}\n\\begingroup\\small\n```\n"]
    for i, k in enumerate(order):
        ref_lines.append(f"{i + 1}. {refs[k]}\n")
    ref_lines.append("\n```{=latex}\n\\endgroup\n```\n")

    body_md = "\n".join(bodies + [data_availability] + ref_lines + [back_matter])
    # Nature style: en dashes for numeric and FDxxx ranges. Applied to the body only (the
    # title block holds the ORCID, which must keep plain hyphens); digit-hyphen-digit leaves
    # compound words, minus signs, and identifiers like SHA-256 untouched. The 10:1 ratio has
    # no hyphen and is unaffected.
    body_md = re.sub(r"(?<=\d)-(?=\d)", "--", body_md)
    body_md = re.sub(r"(FD\d{3})-(?=FD\d{3})", r"\1--", body_md)
    md = OUTDIR / "_manuscript2.md"
    md.write_text(title_block + "\n" + body_md)
    header = OUTDIR / "_preamble2.tex"
    header.write_text(PREAMBLE)

    env = {**os.environ, "PATH": LATEX_BIN + ":" + os.environ.get("PATH", "")}
    for ext, args in [("tex", []), ("pdf", ["--pdf-engine=pdflatex"])]:
        out = OUTDIR / f"manuscript2_latex.{ext}"
        cmd = [PANDOC, str(md), "-o", str(out), "-H", str(header),
               "--number-sections", "-V", "geometry:margin=1in", "-V", "fontsize=11pt",
               "-V", "colorlinks=true", "-V", "urlcolor=blue", "-V", "linkcolor=blue",
               "-V", "documentclass=article"] + args
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[{ext}] pandoc FAILED:\n", (r.stderr or "")[-1800:])
        else:
            print(f"  wrote {out.name} ({out.stat().st_size // 1024} KB)")
    print(f"  references used (in order): {len(order)} of {len(refs)}")
    unused = sorted(set(refs) - set(order))
    if unused:
        print("  UNUSED reference keys:", ", ".join(unused))


if __name__ == "__main__":
    main()
