# Paper 2 experiments — governed adaptation under distribution shift

Replication package for *"Governed adaptation of asynchronous risk fusion under distribution
shift: distinguishing calibration, channel-reliability, and model drift in predictive safety"*
(submitted to Scientific Reports). The library components used by these experiments live in
`src/ai_cta/` (`online_fusion.py`, `adaptive_calibration.py`, `adaptive_pipeline.py`, added in
v1.2.0); unit tests in `tests/test_online_fusion.py`, `tests/test_adaptive_calibration.py`.

## Experiment generators (archived versions that produced the paper's numbers)

| Script | Produces |
|---|---|
| `run_p4_revision_benchmark.py` | controlled location-shift benchmark (Fig. 2, Table S1) + five-fold engine-disjoint C-MAPSS crossfit replay (Table 2, Fig. 4) |
| `analyze_crossfit_fast.py` | engine-macro metrics + 5,000-resample engine-cluster bootstrap CIs from the crossfit trace |
| `run_q1_extension.py` | channel-reliability stress test (Table 1, Fig. 3) + original battery replay + SF-OGD C-MAPSS add-on |
| `add_sfogd_controlled.py` | SF-OGD rows of the controlled benchmark |
| `e3_battery_noleak.py` | **battery replay used in the paper** (Table 3, Fig. 5): identical to `run_q1_extension.run_battery` except `delta_capacity` is removed from FEATURES (leakage fix — with `lag_capacity` it reconstructs current-cycle capacity exactly) |
| `e3_channel_stats.py` | exact paired Wilcoxon + Holm + Cliff's delta + Wilson CIs (Tables S2, S3) |

Note: `run_p4_revision_benchmark.py` / `analyze_crossfit_fast.py` carry historical hardcoded
`/mnt/data/...` paths from the machine that produced the archived results; run them through the
re-execution drivers below, which redirect paths without modifying the originals.

## Reproducibility re-execution drivers

- `e3_channel_rerun.py` — re-runs the channel-reliability test and diffs against archived CSVs.
  Verified 2026-07-08: all 60 rows match to machine epsilon (max |diff| = 1.1e-16).
- `e4_cmapss_rerun.py` — re-runs the full controlled + C-MAPSS pipeline and bootstrap analyzer,
  then diffs the manuscript-facing CSVs against the archived ones. Verified 2026-07-08:
  controlled per-seed results bit-identical (280/280 rows, diff 0.0); engine-macro bootstrap
  table incl. all CIs bit-identical (32/32 rows); worst deviation anywhere 1.1e-16.
  Note: `summarize_and_plot` in the archived benchmark crashes at the
  `paired_bootstrap_differences(trace, 500)` call (500 lands in the `focus` parameter — a
  known latent bug); the archive lacks the two files written after that point, confirming the
  original run crashed identically. Nothing the paper uses is affected; engine bootstraps come
  from `analyze_crossfit_fast.py`.

## Manuscript build (fully data-driven; no hand-typed numbers)

`make_figures2.py` → figures; `make_tables2.py` → main tables + supplementary CSVs;
`build_supplement2.py` / `build_latex2.py` → supplement and manuscript PDFs (pandoc + pdflatex);
`overlap_check2.py` → 20-word text-reuse guard. `provenance2.csv` maps every result file to its
origin with SHA-256.

Data: NASA C-MAPSS (via `benchmarks/download_cmapss.py`) and the NASA Ames battery aging
dataset (checksummed transport CSV; see the paper's Data availability statement).
