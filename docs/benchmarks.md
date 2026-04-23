# Benchmarks

This page describes the public-dataset benchmarks shipped with the
repository, the ablation and baseline-comparison studies, and the
expected reproduction protocol.

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
  Each engine unit is run under stable operating conditions until
  failure. 21 sensor channels plus three operational settings.
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
- **Version pinned in the monograph:** commit `d60581d` (2024-06-06).
  Later releases may have ±5–10 extra recordings.
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
- **Benchmark script:** `benchmarks/run_bosch_benchmark.py` — extracts
  27 vibration features (9 per axis × 3: mean, std, RMS, crest factor,
  peak-to-peak, dominant FFT bin, spectral entropy, skewness, kurtosis),
  fits IsolationForest on the first 70 % normal recordings (time-based
  split, no data leakage), thresholds at the 95th percentile of train
  scores, and evaluates on the remaining 30 % normal + all anomalous.
- **Reference results (commit `d60581d`, seed 42):** M01 F1=0.77,
  ROC-AUC=0.92; pooled ROC-AUC=0.85 across 3 machines. See monograph
  Appendix 13.A.2 (Experiment E7) and `benchmarks/results/bosch_results.csv`.

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
## Running benchmarks

Each benchmark script evaluates at least three configurations:

1. A single-track Isolation Forest over engineered features.
2. A single-track LSTM forecaster (requires `tensorflow`).
3. The hybrid ensemble.

A CSV of per-method metrics (accuracy, precision, recall, F1, false-alarm
rate, runtime) is written to `benchmarks/results/` along with a
LaTeX-ready summary table.

```bash
# C-MAPSS turbofan benchmark
python benchmarks/run_cmapss_benchmark.py

# CWRU bearing benchmark
python benchmarks/run_cwru_benchmark.py

# Ablation of 3-component hybrid risk (monograph § 8.4)
python benchmarks/run_ablation.py  --n-seeds 10 --n-samples 5000

# Baselines: IsolationForest vs LOF vs OC-SVM vs z-score
python benchmarks/run_baselines.py --n-seeds 10 --n-samples 5000
```

The ablation script measures the contribution of each component in
`R_final = w_1 R_anom + w_2 R_RUL + w_3 R_NN`. Results aggregated
across `--n-seeds` random seeds are reported as mean ± std, which is
the standard format for Scopus / IEEE / NeurIPS papers.

The baseline script runs four classical anomaly detectors (IsolationForest,
LOF, One-Class SVM, z-score) on the same synthetic stream so the paper
can quote a fair head-to-head.
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
- Package versions pinned in `requirements.txt` or `pyproject.toml`.
- The benchmark script output CSV committed alongside the paper or
  deposited on Zenodo.
