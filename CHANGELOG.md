# Changelog
All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).
## [Unreleased] — 1.2.2.dev0
### Fixed
- Conformal calibration uses the exact finite-sample order statistic instead of
  conservative interpolation; online batches reject invalid scores atomically.
  Regime decisions reject nonfinite scores and incompatible shapes.
- Fusion now requires aligned lengths, and rule scoring requires configured
  channels. Historical behavior requires explicit `legacy_resample` or
  `legacy_ignore` policies; these defaults are intentional compatibility changes.
- Normalized RUL aggregation preserves untouched subsets during partial reruns,
  validates schemas and values, and replaces each output CSV atomically.
- Maintained runners reject incompatible resume using data/source/environment
  provenance; multi-seed inference pairs by seed and handles ties/missing pairs.
  E2 forwards seeds to every model and reports missing warnings explicitly.
- E5 reports censoring rather than turning non-detection into an event, uses
  minute timestamps, and distinguishes periodic scheduling from drift detection.
- Binary metrics and weight calibration normalize vector/column labels and reject
  invalid inputs. Weight-calibration costs must be finite and positive. Failed
  optimization now raises `RuntimeError` while retaining the previous weights;
  invalid optimizer outputs are rejected before any weights are replaced.
- RUL ensemble tuning preserves the active-model simplex and requires elapsed
  cycles when the physics component is used. RUL horizon scores preserve the
  upper quantile level instead of jumping to one; invalid quantile grids fail
  explicitly.
- Constant-channel PSI remains sensitive to shifts; nonfinite calibration and
  future-dated fusion observations are rejected instead of yielding silent decisions.
- REST input validation returns 422 for invalid timestamps and nonfinite values;
  webhook HTTP errors are logged. The demo n8n service uses regular execution
  mode, and the unavailable API metrics target is no longer advertised.
- Maintained C-MAPSS paths use engine-grouped windows and held-out validation;
  E5 timestamps reflect window availability, and Bosch uses calendar splits and
  exact pooled outcomes. Historical reference results are unchanged.
- Single-reading REST requests now reach the detector's final window; omitted
  timestamps receive a UTC value instead of the string `None`.
- Quantile and asymmetric-BCE losses preserve sample pairing for both vector
  and column-shaped targets; incompatible broadcasting is regression-tested.
- Package version is read from `_version.py`; citation metadata distinguishes
  development code from the archived v1.1.0 DOI.
### Added
- A bounded FD001 diagnostic runner with equal window context across four model
  families, acquisition-manifest checks, engine-level predictions, conditional
  bootstrap intervals, and saved source/environment provenance.
- Explicit opt-in and fresh output directories for six historical benchmark
  CLIs with known scientific limitations; archived source snapshots stay unchanged.
- Separate maintained experiment runners and regression checks for numerical,
  temporal, and input-validation failures. These define new runs, not replacement
  manuscript metrics or a replay of the frozen PeerJ snapshot.
- Optional held-out composite-risk calibration in `SafetyPipeline`, with a
  three-stream example and strict conformal threshold decisions for callbacks.
- HTTP, adaptive-pipeline, and TensorFlow numerical/integration regression tests.
- A PeerJ protocol snapshot, unchanged reference results, and SHA-256 manifest.
### Documentation
- Separate sensor/REST and adaptive-fusion execution paths; label historical
  overlapping evaluation protocols and remove unverified environment-lock claims.

## [1.2.1] — 2026-07-08
- Historical runtime version bump (commit `b361060`). Packaging/citation metadata
  remained at 1.1.0; the unreleased maintenance changes above reconcile it.

## [1.2.0]
- Asynchronous fusion, per-regime calibration, guarded threshold updates, and
  the adaptive pipeline; governed-adaptation experiments under `experiments/paper2`.

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
