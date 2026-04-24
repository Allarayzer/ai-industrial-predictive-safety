# Book ↔ Code mapping

The monograph "Artificial Intelligence for Preventing Accidents at High-Risk
Industrial Facilities" (Serebriakov, 2026) uses some simplified illustrative
class names in its code listings (Chapter 10) and architecture sections
(Chapter 7). The reference implementation in `src/ai_cta/` is more modular
and uses different names for better reusability.

This document maps book names to repo names so reviewers can navigate
between the two without confusion.

## Core detection classes

| Book name (§10.5, §7.5)        | Repo class                                            | Module                          |
|--------------------------------|-------------------------------------------------------|---------------------------------|
| `AnomalyDetector` (unified)    | `IsolationForestDetector`                             | `ai_cta.anomaly_detector`       |
|                                | `LSTMDetector` (§9.4 "LSTM Autoencoder")              | `ai_cta.anomaly_detector`       |
|                                | `HybridDetector` (§9.5 "Cascade IF → AE")             | `ai_cta.anomaly_detector`       |
| `Preprocessor` (unified class) | `StatisticalFeatureExtractor`                         | `ai_cta.preprocess`             |
|                                | `RollingWindowFeatureExtractor`                       | `ai_cta.preprocess`             |
|                                | `FrequencyDomainFeatureExtractor`                     | `ai_cta.preprocess`             |

Rationale: the book presents a single `AnomalyDetector` class to keep the
pedagogical text concise. The actual package splits this into three
independently testable classes (one per algorithm) to follow the
single-responsibility principle and allow each to be used standalone.
`HybridDetector` composes `IsolationForestDetector` + `LSTMDetector`
exactly as §9.5 describes.

## RUL + risk model classes

| Book name (§10.6, §10.7)                | Repo class                        | Module                  |
|-----------------------------------------|-----------------------------------|-------------------------|
| `RULEstimator` (or `RULRegressor`)      | `RULEstimator` (Model B)          | `ai_cta.rul_estimator`  |
| §9.6 Model A (XGBoost quantile)         | `XGBoostQuantileRegressor`        | `ai_cta.rul_ensemble`   |
| §9.6 Model C (physics-guided fallback)  | `PhysicsGuidedRUL`, `WeibullParams` | `ai_cta.rul_ensemble` |
| §9.6 stacked ensemble                   | `RULEnsemble`                     | `ai_cta.rul_ensemble`   |
| `RiskAggregator`                        | `RiskAggregator`                  | `ai_cta.risk_model`     |
| (not in book) composite rule scorer     | `RiskScorer`, `ChannelLimits`     | `ai_cta.risk_model`     |
| (not in book) conformal calibration     | `ConformalThresholdCalibrator`    | `ai_cta.risk_model`     |
| `NeuralRiskEstimator` (§9.7, §10.7)     | `NeuralRiskEstimator`             | `ai_cta.risk_model`     |

## Drift + calibration classes

| Book name (§9.9, §9.10)       | Repo class           | Module                   |
|-------------------------------|----------------------|--------------------------|
| `DriftDetector`               | `DriftDetector`      | `ai_cta.drift_detector`  |
| (not in book) drift report    | `DriftReport`        | `ai_cta.drift_detector`  |
| `OnlineCalibrator`            | `OnlineCalibrator`   | `ai_cta.calibration`     |
| (not in book) diff record     | `CalibrationUpdate`  | `ai_cta.calibration`     |

## API Pydantic schemas (§10.7, §10.8)

| Book name in code listing (§10.7)  | Repo class            | Module             |
|------------------------------------|-----------------------|--------------------|
| `TelemetryPayload`                 | `SensorReading` (single-sample)<br>`BatchRequest` (window) | `api.main` |
| `RiskResponse`                     | `ScoreResponse`       | `api.main`         |
| (not in book) health probe schema  | `HealthResponse`      | `api.main`         |
| (not in book) version probe schema | `VersionResponse`     | `api.main`         |

Rationale: book shows abstract `TelemetryPayload` → `RiskResponse` flow.
The repo splits telemetry payload into per-sample (`SensorReading`) and
per-window (`BatchRequest`) schemas so single-sample `/predict` and
windowed `/score-batch` endpoints can have distinct validated types.
`ScoreResponse` is the output shape for both — equivalent to the book's
`RiskResponse` but named to match the endpoint verb.

OpenAPI 3.1 schema including all types: `api/openapi.yaml` (auto-exported
from the running FastAPI app).

## Pipeline, data utilities

| Book name (architecture only)      | Repo class / function                      | Module            |
|------------------------------------|--------------------------------------------|-------------------|
| "SafetyPipeline" (§9, §10)         | `SafetyPipeline`                           | `ai_cta.pipeline` |
| (not in book) pipeline event type  | `PipelineEvent`                            | `ai_cta.pipeline` |
| Industrial simulator (§12.3.1)     | `IndustrialSimulator`, `SimulationEvent`   | `ai_cta.simulator`|
| Synthetic stream helpers           | `generate_synthetic_stream`, `inject_anomalies` | `ai_cta.data` |
| Evaluation metrics                 | `evaluate_binary_detector`, `DetectorMetrics` | `ai_cta.evaluation` |

## File path alias notes

The book's code listings include filename comments like `# models/anomaly.py`
or `# features/vibration.py`. These are **illustrative pedagogy**, not
claims that the repo uses that directory layout. The repo follows
Chapter 10.2's canonical flat layout `src/ai_cta/*.py` — see §10.2 for the
authoritative structure.

| Book code comment   | Actual repo path                  |
|---------------------|-----------------------------------|
| `# models/anomaly.py`    | `src/ai_cta/anomaly_detector.py`   |
| `# models/rul.py`        | `src/ai_cta/rul_estimator.py`      |
| `# models/health_index.py` | (no direct equivalent; composed via `RiskAggregator`) |
| `# features/vibration.py`| spectral features live inside `src/ai_cta/preprocess.py` |
| `# evaluation/metrics.py`| `src/ai_cta/evaluation.py`         |

## Algorithm ↔ module

See §9.2–9.10 of the monograph. Direct module mapping:

| Algorithm (book §9.N)                        | Module                      |
|---------------------------------------------|-----------------------------|
| 9.2 Алгоритм 1: Preprocessor                | `ai_cta.preprocess`         |
| 9.3 Алгоритм 2: Isolation Forest Anomaly    | `ai_cta.anomaly_detector`   |
| 9.4 Алгоритм 3: LSTM Autoencoder            | `ai_cta.anomaly_detector`   |
| 9.5 Алгоритм 4: Cascade IF → AE             | `ai_cta.anomaly_detector`   |
| 9.6 Алгоритм 5: RUL Estimator (ensemble)    | `ai_cta.rul_estimator`      |
| 9.7 Алгоритм 6: Neural Risk Estimator       | `ai_cta.risk_model`         |
| 9.8 Алгоритм 7: Risk Aggregator (SLSQP)     | `ai_cta.risk_model`         |
| 9.9 Алгоритм 8: Drift Detector (PSI + KS)   | `ai_cta.drift_detector`     |
| 9.10 Алгоритм 9: Online Calibration         | `ai_cta.calibration`        |

## Benchmark ↔ book experiment

| Book experiment          | Benchmark script                         |
|--------------------------|------------------------------------------|
| §13.5 E1 (S1 anomaly)    | `benchmarks/run_experiment_e1.py`        |
| §13.6 E2 (C-MAPSS hybrid risk) | `benchmarks/run_experiment_e2_cmapss.py` |
| §13.7 E3 (C-MAPSS RUL)   | `benchmarks/run_cmapss_rul_normalized.py` |
| §13.8 E4 (CWRU robustness) | `benchmarks/run_cwru_robustness.py`    |
| §13.9 E5 (drift)         | `benchmarks/run_experiment_e5.py`        |
| §13.10 E6 (API perf)     | `benchmarks/run_experiment_e6.py`        |
| §13.11 Ablation          | `benchmarks/run_ablation.py`             |
| §13.A.1 CWRU binary      | `benchmarks/run_cwru_benchmark.py`       |
| §13.A.2 Bosch E7         | `benchmarks/run_bosch_benchmark.py`      |
