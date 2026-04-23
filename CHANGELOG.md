# Changelog
All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).
## [1.1.0] — 2026-04-22
### Added
- **Prognostics module** (`ai_cta.prognostics`):
  - `RULEstimator`: LSTM quantile regression for Remaining Useful Life
    prediction with multi-quantile pinball loss; `risk_at_horizon()`
    method recovers the R_RUL component defined in monograph § 8.3.2.
  - `NeuralRiskEstimator`: feedforward NN with cost-asymmetric BCE for
    the contextualized R_NN risk component (§ 8.3.3).
  - `DriftDetector`: distribution drift monitoring via Population
    Stability Index and Kolmogorov-Smirnov test (§ 9.9).
  - `OnlineCalibrator`: scheduled recalibration of conformal thresholds
    on a sliding buffer of normal-data scores (§ 9.10).
- **`RiskAggregator`** (`ai_cta.risk.aggregator`): three-component
  weighted risk fusion with SLSQP weight calibration on the
  probability simplex (monograph § 8.4 and § 10.7).
- **`IndustrialSimulator`** (`ai_cta.utils.simulator`): full
  multi-channel telemetry generator implementing the mathematical
  model from § 12.3.1 — periodic seasonal components, multi-mode
  degradation, Poisson anomaly events with stochastic failure delays.
- **REST API** (`api/main.py`): FastAPI service exposing the pipeline
  via `/score`, `/score-batch`, `/health`, and `/version` endpoints
  (§ 10.8). Auto-generated Swagger UI at `/docs`.
- **Docker reference deployment** (`docker/`): Dockerfile for the API
  service plus a docker-compose stack with API, n8n, Postgres, and
  Redis (§ 12.5).
### Changed
- **Package renamed** from `ai_industrial_safety` to `ai_cta` to align
  with the structure given in monograph § 10.2. Update imports
  accordingly.
- **Risk levels** changed from `LOW` / `MEDIUM` / `HIGH` / `CRITICAL`
  to `OK` / `Warning` / `Critical` / `Emergency` to match the
  convention in monograph § 8.4.3.
- Default risk-level boundaries adjusted: OK below 0.3, Warning
  through 0.6, Critical through 0.85, Emergency above.
- `OnlineCalibrator.buffer_size()` renamed to `current_buffer_size()`
  to avoid collision with the `buffer_size` constructor parameter.
### Tests
- Test suite expanded from 34 to 62 tests covering the new prognostics,
  aggregator, and simulator modules.
## [1.0.0] — 2026-04-22
### Added
- Structured Python package with submodules:
  - `features`: three families of feature extractors (statistical,
    rolling-window, frequency-domain).
  - `detection`: unsupervised detectors — IsolationForestDetector,
    LSTMDetector, HybridDetector.
  - `risk`: composite scoring with channel limits and split conformal
    threshold calibration.
  - `streaming`: SafetyPipeline with optional webhook alerting.
  - `utils`: synthetic data generation, evaluation helpers.
- pytest suite (34 tests).
- GitHub Actions CI for Python 3.10/3.11/3.12 with `ruff` linting.
- Benchmark scripts for NASA C-MAPSS and CWRU datasets.
- Quick-start example, professional README with badges and Mermaid
  architecture diagram, BibTeX citations.
- `CITATION.cff` with ORCID 0009-0009-1548-390X.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- `docs/architecture.md`, `docs/api.md`, `docs/benchmarks.md`.
## [0.1.0] — 2025-12-16
### Added
- Initial demonstration prototype with basic Isolation Forest wrapper,
  MinMax preprocessing, heuristic risk scoring, and n8n webhook notifier.
