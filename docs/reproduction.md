# Versions and reproduction

## Choose the study and preserve its provenance

| Material | Location | Status |
|---|---|---|
| PeerJ corrected protocols and reference CSVs | [experiments/peerj](../experiments/peerj/README.md) | Working-source snapshot; ten reference CSVs preserved byte-for-byte |
| Governed adaptation study | [experiments/paper2](../experiments/paper2/README.md) | Separate generators, rerun drivers, provenance map, and assumptions |
| Historical demonstrations | [benchmarks](../benchmarks) | Not all use independent evaluation splits; see warnings below |
| Archived software v1.1.0 | [DOI 10.5281/zenodo.21145078](https://doi.org/10.5281/zenodo.21145078) | Historical source archive; not the current development code |

The PeerJ source record identifies library commit `fb9a0b9`. The new snapshot
under `experiments/peerj` imports corrected runners from the article workspace,
with only filesystem-path changes. `manifest.json` records each original source
hash, imported hash, and unchanged reference-result hash. It is not a claim that
the v1.1.0 Zenodo archive already contained these corrected runners.

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
The corrected runners write to `experiments/peerj/results/`; the checked-in
`reference_results/` directory is never a rerun destination. Some historical
runners overwrite their output while others resume it: use an empty output
folder for an independent replication.

## Historical scripts with overlapping evaluation

- `benchmarks/run_cwru_benchmark.py`: training-normal observations reappear in
  evaluation; use `experiments/peerj/cwru_leakage_safe.py` for the study protocol.
- `benchmarks/run_experiment_e2_cmapss.py`: SLSQP weights are fit and evaluated on
  the same validation engines; use `e2_cmapss_calib_split.py` instead.
- `benchmarks/run_ablation.py`: weights and evaluation share data; use
  `ablation_calib_split.py` instead.

These files are retained to identify historical runs, with warnings. Do not
interpret their in-sample metrics as held-out generalization evidence.

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
