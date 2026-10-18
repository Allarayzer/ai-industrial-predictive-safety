# Versions and reproduction

## Choose the study and preserve its provenance

| Material | Location | Status |
|---|---|---|
| Maintained C-MAPSS runners | [experiments/maintained](../experiments/maintained/README.md) | New runs with engine-grouped windows and explicit validation; no replacement reference results |
| PeerJ corrected-protocol snapshot and reference CSVs | [experiments/peerj](../experiments/peerj/README.md) | Frozen imported code and ten reference CSVs; known limitations remain |
| Governed adaptation study | [experiments/paper2](../experiments/paper2/README.md) | Separate generators, rerun drivers, provenance map, and assumptions |
| Historical demonstrations | [benchmarks](../benchmarks) | Not all use independent evaluation splits; see warnings below |
| Archived software v1.1.0 | [DOI 10.5281/zenodo.21145078](https://doi.org/10.5281/zenodo.21145078) | Historical source archive; not the current development code |

The PeerJ source record identifies library commit `fb9a0b9`. The new snapshot
under `experiments/peerj` imports corrected runners from the article workspace,
with only filesystem-path changes. `manifest.json` records each original source
hash, imported hash, and unchanged reference-result hash. It is not a claim that
the v1.1.0 Zenodo archive already contained these corrected runners. "Corrected"
describes the imported study revision; it is not a claim that every remaining
protocol issue was resolved. Keep these files and their manifest unchanged.

For new C-MAPSS runs, use `experiments/maintained/`. Its E2 and multi-seed RUL
copies split engines before scaling/windowing and use explicit early-stopping
validation. E2 lead time is measured to actual failure. These changes define new
runs; they do not retroactively correct the frozen CSVs. Pointwise classical
RUL baselines still have a different input context from windowed models.
The separate bounded `fd001_validation.py` comparison provides the same
32-cycle context to all four models and saves per-engine predictions; it is a
one-seed diagnostic, not a replacement multi-subset study.

Current development code is **1.2.2.dev0**. It includes REST and loss-shape fixes
that postdate the experiments. A new run against this library is a new validation
run, not a replacement for archived results. For historical replication, use the
recorded library revision and experiment environment alongside the corrected
runners. In particular, target-array shape and TensorFlow/Keras versions can
change loss behavior; do not assume numerical identity after the maintenance fix.

## Environment and outputs

Install dependencies from `pyproject.toml` as compatibility ranges. These are not
a lockfile. The historical PeerJ environment record lists Python 3.11.2 on Apple
M4/macOS, NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.9.0, SciPy 1.17.1 and TensorFlow
2.21.0. This describes the recorded experiment environment, not a clean-install
compatibility guarantee for every platform.

For new runs, save `git rev-parse HEAD`, `python --version`, `pip freeze`, CLI
arguments, dataset hashes, and hardware/backend details alongside outputs.
Seeded neural-network training need not be bitwise reproducible across platforms.
Maintained runners write to `experiments/maintained/results/`; frozen PeerJ
runners write to `experiments/peerj/results/`. The checked-in `reference_results/`
directory is never a rerun destination. Some historical
runners overwrite their output while others resume it: use an empty output
folder for an independent replication.
The maintained multi-seed and E2 runners now reject resume when their recorded
arguments, data/source hashes, environment or engine roles differ. They also
reject an old CSV without a matching provenance manifest. Use explicit
`--data-dir` and `--output-dir`; file hashes identify the local bytes and do not
independently establish dataset authenticity.

## Historical protocol limitations

- `benchmarks/run_cwru_benchmark.py`: training-normal observations reappear in
  evaluation; the frozen PeerJ CWRU protocol uses disjoint recordings but still
  has only one normal test recording and its own sampling-rate/data conventions.
- `benchmarks/run_experiment_e2_cmapss.py`: SLSQP weights are fit and evaluated on
  the same validation engines; use the maintained E2 runner for new evaluations.
- `benchmarks/run_ablation.py`: weights and evaluation share data. The frozen
  PeerJ `ablation_calib_split.py` records a separate study protocol.
- `experiments/peerj/cmapss_rul_multiseed.py` and
  `e2_cmapss_calib_split.py` retain concatenated-engine windowing; the latter's
  lead-time helper measures from entry into the positive label horizon rather
  than actual failure. Use their maintained copies for new runs.
- Legacy E1 selects thresholds using test labels. `run_experiment_e4.py` is a
  synthetic HybridDetector demonstration with score-to-label padding;
  `run_cwru_robustness.py` is a distinct Isolation Forest demonstration, not the
  proposed IF-to-autoencoder cascade.
- Historical E5 assigned alarms to window starts and used an ADWIN-like KS
  approximation. Historical Bosch results used path order rather than calendar
  splits and reconstructed pooled counts from rounded metrics. Current changes
  to those runners require new evaluation; saved historical CSVs are unchanged.

These files are retained to identify historical runs, with warnings. Do not
interpret in-sample metrics as held-out generalization evidence. See the full
[protocol status map](benchmarks.md#protocol-status), including RUL and E6 limits.

The six historical E1, E2-CMAPSS, E4, CWRU benchmark/robustness and E6 CLI entry
points refuse execution unless `--allow-legacy` and a new `--output-dir` are
provided. The opt-in acknowledges known invalid assumptions; it does not repair
their scientific protocols. E1 also has score-time alignment problems, CWRU E4
has overlapping raw windows/causal imputation limits, and E6 omits failed
requests from its historical metrics. Frozen PeerJ code stays byte-for-byte
unchanged; its limitations remain listed above.

## Scope of this maintenance validation

The maintenance checks exercise HTTP requests, streaming threshold decisions,
adaptive-controller acceptance/rejection, and numerical losses. They do not
rerun full datasets or update submitted manuscript numbers. The governed-adaptation
package and previously uploaded journal/Zenodo artifacts remain versioned study
materials. A future release needs its own verified archival record; the old DOI
must not be reassigned to these changes.

## Local maintenance check — 2026-10-04

On macOS / Python 3.11.2:

- Full installed-environment suite: **83 passed**. Combined library/API statement
  coverage: **72%**; API **95%**; adaptive pipeline **96%**.
- With TensorFlow imports disabled: **78 passed**, deep-test module skipped.
- Ruff passes for the library, tests, API, quick start, and manifest verifier.
- Quick start completes with separate training/calibration/test streams.
- All ten imported experiment entry points accept `--help`; all 20 source/result
  SHA-256 checks pass. Imported code differs from source only in filesystem paths.
- Wheel build and editable install succeed; runtime, installed distribution, and
  citation version agree on `1.2.2.dev0`.

A dependency emits a Starlette TestClient/httpx deprecation warning; tests pass.
The full dataset experiments, Linux CI matrix, Docker stack, and field behavior
were not rerun in this maintenance pass. In particular, RUL ensemble coverage
remains absent from this suite. These checks are bounded regression evidence,
not an assertion that every module or scientific result has been revalidated.
