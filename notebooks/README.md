# Jupyter notebooks

Exploratory, experimental, and reproducibility notebooks that accompany
the monograph and Scopus publications.

## Conventions

- **Do not commit notebook outputs.** Strip outputs before committing:
  ```
  jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
  ```
  A `nb-clean` pre-commit hook is provided in `.github/` for CI
  enforcement.
- **Seeds must be set.** Every notebook that trains a model sets seeds
  for `numpy`, `random`, `tensorflow`, and `torch` (where used).
- **Data paths are relative to repo root.** Access benchmark data via
  `benchmarks/data/<dataset>/` after downloading per the instructions
  in `docs/benchmarks.md`.

## Suggested notebook structure

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb`      | Inspect C-MAPSS / CWRU signal distributions, missingness, and regime shifts. |
| `02_preprocess_tuning.ipynb`     | Sweep `Preprocessor` window size / smoothing parameters and measure downstream detection F1. |
| `03_anomaly_detection_eval.ipynb`| Compare IsolationForest / LSTM-AE / HybridDetector on held-out C-MAPSS folds. |
| `04_rul_calibration.ipynb`       | Quantile-calibration diagnostics for RULEstimator (coverage plots, pinball loss). |
| `05_hybrid_risk_ablation.ipynb`  | Ablation of the three-component hybrid risk model (monograph § 8.4). |
| `06_drift_case_study.ipynb`      | Reproduce the drift-detection case study from monograph § 13.9. |
| `07_paper_tables.ipynb`          | Generate LaTeX tables for the companion Scopus paper. |

## Running

```
pip install -e ".[test,deep]"
jupyter lab
```

Each notebook begins with `%run` lines that import from
`ai_cta` so that the source of truth remains the Python package.
Notebook cells should be thin wrappers over `ai_cta` APIs, not
re-implementations of the algorithms.
