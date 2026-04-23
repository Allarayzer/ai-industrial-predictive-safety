# Benchmarks
This page describes the two public-dataset benchmarks shipped with the
repository and the expected reproduction protocol.
## Datasets
### NASA C-MAPSS Turbofan Engine Degradation
- **Source:** NASA Prognostics Center of Excellence.
- **Download:** <https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6>
  or the Kaggle mirror at
  <https://www.kaggle.com/datasets/behrad3d/nasa-cmaps>.
- **Contents:** four sub-datasets (FD001–FD004); `FD001` is used as the
  primary benchmark here. Each engine unit is run under stable operating
  conditions until failure. 21 sensor channels plus three operational
  settings.
- **Task:** flag runs in the last `k` cycles of each engine as
  "degrading" vs. normal.
- **Expected file layout:**
  ```
  benchmarks/data/cmapss/
      train_FD001.txt
      test_FD001.txt
      RUL_FD001.txt
  ```
### Case Western Reserve University (CWRU) Bearing Dataset
- **Source:** CWRU Bearing Data Center.
- **Download:** <https://engineering.case.edu/bearingdatacenter/download-data-file>
  (free registration required).
- **Contents:** accelerometer traces from a drive-end bearing under
  normal operation and under three seeded fault classes (inner-race,
  outer-race, rolling-element) at several severity levels (0.007",
  0.014", 0.021" mm defect diameters). Sampling at 12 or 48 kHz.
- **Task:** binary classification of a bearing recording as normal vs.
  faulty.
- **Expected file layout:**
  ```
  benchmarks/data/cwru/
      normal/
          *.mat
      fault/
          *.mat
  ```
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
rate, runtime) is written to `benchmarks/results/`.
```bash
# C-MAPSS turbofan benchmark
python benchmarks/run_cmapss_benchmark.py
# CWRU bearing benchmark
python benchmarks/run_cwru_benchmark.py
```
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
