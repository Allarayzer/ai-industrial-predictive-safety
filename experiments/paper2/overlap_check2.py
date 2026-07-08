"""Text-overlap check for Paper 2 (mission rule: no >20 consecutive words reused from the
monograph, Paper 1, the previous P4 draft, or the author's published articles).

Reports every common word run of >= REPORT_AT words between the new draft sections and each
source; exits nonzero if any run exceeds HARD_LIMIT words.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DRAFT = ROOT / "analysis2" / "draft"
SCRATCH = pathlib.Path("/private/tmp/claude-501/-Users-ilyaserebryakov-Desktop-Scorpus"
                       "---------------/f062bc07-863f-44af-9a00-f41942acddba/scratchpad")
REPORT_AT = 10
HARD_LIMIT = 20


def norm_words(text: str) -> list[str]:
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text.lower())
    return text.split()


def longest_runs(a: list[str], b: list[str], min_len: int):
    """All maximal common runs >= min_len via shingle index (memory-light)."""
    if not a or not b:
        return []
    k = min_len
    index = {}
    for i in range(len(b) - k + 1):
        index.setdefault(tuple(b[i:i + k]), []).append(i)
    runs = []
    i = 0
    while i < len(a) - k + 1:
        key = tuple(a[i:i + k])
        best = 0
        for j in index.get(key, []):
            L = k
            while i + L < len(a) and j + L < len(b) and a[i + L] == b[j + L]:
                L += 1
            best = max(best, L)
        if best:
            runs.append((i, best, " ".join(a[i:i + best])))
            i += best
        else:
            i += 1
    return runs


def pdf_text(path: pathlib.Path) -> str:
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)


draft_text = "\n".join((DRAFT / f"{n}.md").read_text()
                       for n in ("abstract", "introduction", "results", "discussion", "methods"))
draft_words = norm_words(draft_text)

sources = {
    "P4-previous-draft": (SCRATCH / "draft_p4_text.txt").read_text(),
    "paper1-draft": "\n".join(p.read_text() for p in sorted((ROOT / "analysis" / "draft").glob("*.md"))),
    "monograph": (ROOT / "analysis" / "monograph" / "monograph.txt").read_text(errors="ignore"),
    "ULOA-published": pdf_text(ROOT / "paper2-incoming" / "2. Опубликованная статья.pdf"),
}

fail = False
for name, text in sources.items():
    runs = longest_runs(draft_words, norm_words(text), REPORT_AT)
    print(f"== vs {name}: {len(runs)} run(s) >= {REPORT_AT} words")
    for _, L, s in sorted(runs, key=lambda r: -r[1])[:12]:
        flag = " <-- EXCEEDS HARD LIMIT" if L > HARD_LIMIT else ""
        print(f"   [{L}w]{flag} {s[:180]}")
        if L > HARD_LIMIT:
            fail = True

sys.exit(1 if fail else 0)
