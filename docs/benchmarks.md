# Benchmarks

This page describes the public-dataset benchmarks shipped with the
repository, the ablation and baseline-comparison studies, and the
expected reproduction protocol.

## Select the correct protocol first

Use [experiments/maintained](../experiments/maintained/README.md) for new C-MAPSS
validation runs. [experiments/peerj](../experiments/peerj/README.md) is a frozen
study snapshot, including known limitations; [experiments/paper2](../experiments/paper2/README.md)
is the separate governed-adaptation package.
See [version/provenance notes](reproduction.md) before comparing outputs.

## Protocol status

| Entry point | Status and interpretation |
|---|---|
| `experiments/maintained/cmapss_rul_multiseed.py` | New grouped-engine RUL runs with fit-only normalization and explicit validation. Classical baselines remain pointwise while neural models use windows; no equal-information-budget or superiority claim. |
| `experiments/maintained/e2_cmapss_calib_split.py` | New grouped-engine E2 runs; lead time uses actual failure and is conditional on a warning occurring (missed warnings are excluded). |
| Frozen PeerJ C-MAPSS runners | Preserve the imported scripts/results, including concatenated-engine windows and the old E2 label-horizon lead definition. Use maintained copies for new work. |
| `run_experiment_e1.py` | Historical demonstration; thresholds use test labels and the "3sigma" row is not a fixed 3-sigma threshold. Do not use for held-out detector comparison. |
| `run_experiment_e4.py` | Historical synthetic HybridDetector robustness demonstration; padding scores to row labels does not establish timestamp alignment. Distinct from the CWRU IF demonstration below. |
| `run_cwru_benchmark.py`, `run_experiment_e2_cmapss.py`, `run_ablation.py` | Historical evaluation overlaps model-fitting or weight-selection data. |
| `run_cwru_robustness.py` (E4) | Historical IF robustness test with 11 features and a percentile threshold; not a trained IF-to-LSTM-autoencoder cascade. |
| Frozen PeerJ CWRU | Disjoint recording roles, but one normal test recording, cross-load shift, and study-specific sampling-rate/data conventions. |
| `run_experiment_e5.py` | Current runner timestamps checks at the last available sample and reports observed pre-drift exposure. Its "ADWIN" comparator remains an ADWIN-like KS approximation, not River ADWIN. Historical CSVs predate the timing correction. |
| `run_bosch_benchmark.py` | Current runner splits by calendar month, retains exact confusion counts and labels mean machine AUC/AP as macro averages. Historical CSVs predate these changes. |
| `run_bosch_path_order_corrected.py`, `run_experiment_e5_causal.py` | Separate reviewed study protocols: lexicographic Bosch roles and causal E5 censoring/traces. See [protocol contracts and source identity](bosch-e5-protocols.md); these do not replace the calendar/null-control entrypoints. |
| `run_cmapss_rul_normalized.py` | Current helpers use engine-grouped fit/validation windows; historical CSVs predate that correction. Pointwise baselines still differ in context from neural models, and the full ensemble uses equal weights. The alpha-lambda column is endpoint tolerance accuracy, without a trajectory lambda criterion. |
| `run_experiment_e6.py` | Historical API timing demo; non-200 requests are omitted from latency samples and pipeline startup is included in throughput timing. This is not a deployment capacity test. |

Unqualified filenames in this table are under `benchmarks/`. Historical results
have not been replaced. Corrected runner behavior needs a new full run before
new empirical claims; unit tests alone do not establish benchmark performance.

The six legacy CLIs `run_experiment_e1.py`, `run_experiment_e2_cmapss.py`,
`run_experiment_e4.py`, `run_cwru_benchmark.py`, `run_cwru_robustness.py` and
`run_experiment_e6.py` refuse to run by default. An intentional historical run
requires both `--allow-legacy` and `--output-dir /path/to/new-run`. The directory
must not exist; the runner creates it and refuses existing paths, including empty
directories. `--help` and refusal paths need only the Python standard library.
This guard preserves the algorithms and their limitations; it does not validate
their results. Other historical runners remain subject to the warnings above.

## Datasets

### NASA C-MAPSS Turbofan Engine Degradation

- **Source:** NASA Prognostics Center of Excellence.
- **Direct download (verified working as of 2026-04):**
  - PHM Society S3 archive:
    <https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip>
    (12 MB outer zip; contains `CMAPSSData.zip` with 14 files).
  - NASA's own page:
    <https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data>
  - Kaggle mirror (no NASA account required):
    <https://www.kaggle.com/datasets/behrad3d/nasa-cmaps>
- **Contents:** four sub-datasets (FD001–FD004). All four are used by
  the reference benchmark; `FD001` is the primary headline result.
  Training trajectories run to failure; operating conditions and fault modes
  vary by subset. There are 21 sensor channels plus three operational settings.
- **License:** public domain (17 U.S.C. § 105, NASA-produced data).
- **Task:** flag runs in the last `k` cycles of each engine as
  "degrading" vs. normal; report RMSE of remaining useful life
  prediction for the primary quantile.
- **Expected file layout:**
  ```
  benchmarks/data/cmapss/
      train_FD001.txt
      test_FD001.txt
      RUL_FD001.txt
      train_FD002.txt
      test_FD002.txt
      RUL_FD002.txt
      train_FD003.txt
      test_FD003.txt
      RUL_FD003.txt
      train_FD004.txt
      test_FD004.txt
      RUL_FD004.txt
      Damage Propagation Modeling.pdf
      readme.txt
  ```
- **Citation:** Saxena, Goebel, Simon, Eklund (2008), *Damage
  Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*,
  PHM 2008.

### Case Western Reserve University (CWRU) Bearing Dataset

- **Source:** CWRU Bearing Data Center.
- **Direct download (verified working as of 2026-04):** every
  `.mat` file can be fetched directly at
  `https://engineering.case.edu/sites/default/files/<N>.mat`
  where `<N>` is the record ID listed on the index pages:
  - Normal baseline: `97, 98, 99, 100.mat` (~30 MB)
  - 12 kHz Drive End: 60 files, IDs `105–234` (~150 MB)
  - 12 kHz Fan End: 45 files (~125 MB)
  - 48 kHz Drive End: 52 files (~340 MB)
  Full dataset is ~670 MB.
- **Contents:** accelerometer traces from a drive-end bearing under
  normal operation and under three seeded fault classes (inner-race,
  outer-race, rolling-element) at several severity levels (0.007",
  0.014", 0.021", 0.028" defect diameters). Sampling at 12 or 48 kHz.
- **License:** free for academic use with citation of the Bearing
  Data Center.
- **Task:** binary classification of a bearing recording as normal vs.
  faulty.
- **Expected file layout:**
  ```
  benchmarks/data/cwru/
      normal/    # 97.mat, 98.mat, 99.mat, 100.mat
      12k_DE/    # 105.mat ... 234.mat
      12k_FE/    # fan-end fault files
      48k_DE/    # high-sampling-rate fault files
  ```

### Bosch CNC Machining Dataset (cross-domain real-world vibration)

- **Source:** Robert Bosch GmbH / Technical University of Munich.
  Publication: Tnani, Feil, Diepold (2022), *Procedia CIRP* 107, 131–136.
  DOI: `10.1016/j.procir.2022.04.022`.
- **Repository:** <https://github.com/boschresearch/CNC_Machining>
- **Dataset reference commit:**
  `d60581d6a3ab6015dcc5488c3d76112bb8e1bcb1`. Record the checkout and file
  manifest for each run; a newer checkout is a different input collection.
- **Licences:** CC-BY-4.0 (data) + BSD-3-Clause (code).
- **Contents:** tri-axial accelerometer (Bosch CISS) readings from three
  brownfield CNC milling machines (M01, M02, M03) executing 15 process
  types (OP00–OP14) over 2018-10..2021-08. Each recording is roughly
  268k samples × 3 axes at 2 kHz (~134 s — one process cycle). Labels:
  `good` (normal) or `bad` (process anomaly / tool wear). 1 632 normal
  + 70 anomalous recordings (imbalance ≈ 1:23 pooled, up to 1:59 on M03).
- **Task:** per-machine anomaly detection (normal vs faulted process).
- **Download:** clone the repo — data lives in `data/M0N/OPNN/{good,bad}/*.h5`.
  ~911 MB total.
- **Expected layout after cloning/symlinking:**
  ```
  benchmarks/data/bosch/data/M01/OP00/good/*.h5
  benchmarks/data/bosch/data/M01/OP00/bad/*.h5
  ... (M01–M03, OP00–OP14, good/bad)
  ```
- **Current benchmark script:** `benchmarks/run_bosch_benchmark.py` extracts
  27 vibration features (9 per axis × 3: mean, std, RMS, crest factor,
  peak-to-peak, dominant FFT bin, spectral entropy, skewness, kurtosis).
  It selects a calendar-month boundary near the 70% normal-recording target
  without splitting a month across roles. IF fits earlier normal recordings;
  test normal/fault recordings are at or after the boundary. Earlier faults are
  excluded from this prospective test. The threshold still uses fit-normal
  scores, not an independent calibration split.
- **Historical results:** `benchmarks/results/bosch_results.csv` describes the
  earlier path-sorted split, not this calendar protocol. Its reconstructed
  confusion counts and "pooled" AUC/AP labels are not authoritative for new
  evaluation. Current code sums exact counts and distinguishes macro AUC/AP.

### Optional: FEMTO PHM 2012 Bearings

- **Source:** FEMTO-ST Institute.
- **Download:** <https://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM/IEEE-PHM-2012-Data-challenge.php>
  (requires filling in a short request form).
- **Task:** bearing RUL prediction on accelerated-life tests.
- Not required by the default benchmarks, but a drop-in `run_femto_benchmark.py`
  can be added if needed for a specific paper.
## Download scripts
For convenience, `benchmarks/download_cmapss.py` and
`benchmarks/download_cwru.py` print the expected layout and a checklist;
they do not download automatically because both datasets require the
user to accept licensing terms at the source.
```bash
python benchmarks/download_cmapss.py
python benchmarks/download_cwru.py
```
## Running historical demonstrations

These commands retain historical entry points. Read the status map above before
using them; method coverage and output columns differ by script. For current
C-MAPSS work, use the maintained README instead. Keep new outputs separate from
historical reference files.

```bash
# C-MAPSS turbofan benchmark
python benchmarks/run_cmapss_benchmark.py

# Intentional historical CWRU run; ./legacy-cwru-run must not yet exist
python benchmarks/run_cwru_benchmark.py --allow-legacy --output-dir ./legacy-cwru-run

# Ablation of 3-component hybrid risk (monograph § 8.4)
python benchmarks/run_ablation.py  --n-seeds 10 --n-samples 5000

# Baselines: IsolationForest vs LOF vs OC-SVM vs z-score
python benchmarks/run_baselines.py --n-seeds 10 --n-samples 5000
```

The ablation script measures the contribution of each component in
`R_final = w_1 R_anom + w_2 R_RUL + w_3 R_NN`. Results aggregated
across `--n-seeds` random seeds are reported as mean ± std, which is
descriptive variation across the specified runs. It is not a confidence interval
or a count of independent industrial trials.

### Historical 3-model RUL ensemble demonstration

The monograph's §9.6 specifies a three-model ensemble for RUL
estimation:

- **Model A** — XGBoost multi-quantile regression over tabular
  statistical features (implemented in
  `ai_cta.rul_ensemble.XGBoostQuantileRegressor`).
- **Model B** — LSTM multi-quantile regression over sequential raw
  signals (implemented in `ai_cta.rul_estimator.RULEstimator`).
- **Model C** — Weibull physics-guided fallback for engineering
  sanity-check and extreme-extrapolation guard (implemented in
  `ai_cta.rul_ensemble.PhysicsGuidedRUL`).

The stacked ensemble is available via
`ai_cta.rul_ensemble.RULEnsemble`.

To run the full ensemble on C-MAPSS:

```bash
python benchmarks/run_cmapss_rul_normalized.py --subsets FD001 --full-ensemble
```

**Historical FD001 results (seed=42, epochs=30; not a corrected rerun):**

| Method | RMSE | MAE | Score | α-λ |
|---|---:|---:|---:|---:|
| Linear Regression | 20.59 | 16.37 | 1224 | 0.52 |
| Random Forest | 17.19 | 12.17 | 916 | 0.65 |
| Gradient Boosting | 17.42 | 12.03 | 1108 | 0.70 |
| **LSTM (MSE)** | **12.85** | **10.27** | **268** | 0.65 |
| Proposed Ensemble (LSTM quantile, Model B only) | 16.23 | 12.43 | 598 | 0.55 |
| Full Ensemble (A+B+C, equal weights) | 17.80 | 13.40 | 584 | 0.57 |

These rows belong to the historical protocol and original book context. The
full ensemble uses equal weights in this runner; `RULEnsemble.fit()` supporting
validation-based tuning does not mean this run used it. The rows do not explain
the cause of the observed ranking or establish tuned-ensemble performance.

The baseline script runs four classical anomaly detectors (IsolationForest,
LOF, One-Class SVM, z-score) on a synthetic stream. A shared stream alone does
not establish fairness: information budgets, calibration, tuning and evaluation
roles must also match the comparison being claimed.
Results on the reference hardware (commodity laptop, Python 3.11, no
GPU) are **indicative only**; they are not a substitute for publication-
grade evaluation and will vary with hyperparameters, seeds, and the
exact subset used.
## Baseline context
The results should be compared to the following kinds of baselines from
the literature:
- One-Class SVM (Schölkopf et al., 1999).
- LSTM Encoder-Decoder (Malhotra et al., 2016).
- Deep SVDD (Ruff et al., 2018).
- TranAD (Tuli et al., 2022) — transformer-based anomaly detection.
- USAD (Audibert et al., 2020) — autoencoder-based.
Any improvement claim in a published paper should include **at least**
OC-SVM and one deep baseline on the same split, with reported variance
across multiple random seeds.
## Reproducibility checklist
Before reporting results publicly, verify:
- Random seeds fixed at every stage (`random_state=42` is the default).
- Data splits described with enough detail to reproduce
  (train/calibration/test proportions, time-aware vs. random).
- Hyperparameters logged (print `detector.get_params()` in a final
  cell).
- Exact package versions recorded in a per-run lock/environment file;
  repository compatibility ranges are not pins.
- The benchmark script output CSV committed alongside the paper or
  deposited on Zenodo.
