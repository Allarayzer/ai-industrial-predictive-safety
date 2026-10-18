# API Reference
Public API of `ai_cta`. Import directly from the top-level package:
```python
from ai_cta import (
    # Features
    StatisticalFeatureExtractor, RollingWindowFeatureExtractor,
    FrequencyDomainFeatureExtractor,
    # Detection
    IsolationForestDetector, LSTMDetector, HybridDetector,
    # Risk
    RiskScorer, ChannelLimits, RiskLevel,
    ConformalThresholdCalibrator, RiskAggregator,
    # Prognostics
    RULEstimator, NeuralRiskEstimator,
    DriftDetector, DriftReport,
    OnlineCalibrator, CalibrationUpdate,
    # Streaming
    SafetyPipeline, PipelineEvent,
)
```
This page describes the current library contract. Historical experiment snapshots retain their own code and results; see [reproduction notes](reproduction.md).
## Feature extractors
All three follow the scikit-learn transformer protocol.
### `StatisticalFeatureExtractor(channels=None)`
9 descriptors per channel: mean, std, min, max, RMS, peak-to-peak,
skewness, kurtosis, crest factor.
### `RollingWindowFeatureExtractor(window_size=32, step=1, channels=None, aggregations=("mean","std","min","max"))`
Sliding-window aggregations per channel. The first `window_size − 1`
rows of each channel contain `NaN` until the window is filled.
### `FrequencyDomainFeatureExtractor(sampling_rate=1.0, channels=None)`
FFT-based spectral descriptors per channel: spectral centroid,
bandwidth, energy, dominant frequency. The DC component is removed
before the FFT so centroid/bandwidth are not dominated by the signal
mean.
## Detectors
All detectors expose `fit(X)`, `decision_function(X) -> scores in [0, 1]`,
and `predict(X, threshold=0.5) -> 0/1 labels`.
### `IsolationForestDetector(window_size=64, stride=32, sampling_rate=1.0, use_spectral=True, contamination=0.05, n_estimators=200, random_state=42)`
Unsupervised detector over engineered windows. After fitting:
`feature_names_`, `calibration_mean_`, `calibration_std_`,
`n_windows_fit_`.
### `LSTMDetector(window_size=32, lstm_units=(64, 32), dropout=0.1, learning_rate=1e-3, epochs=20, batch_size=64, validation_split=0.1, patience=5, random_state=42)`
Requires `tensorflow`. Anomaly scores via the modified z-score
(median / MAD) of L2 prediction residuals. After fitting: `model_`,
`history_`, `residual_median_`, `residual_mad_`.
### `HybridDetector(if_weight=0.5, if_params=None, lstm_params=None)`
Weighted ensemble of `IsolationForestDetector` and `LSTMDetector`.
`tune_weights(X_val, y_val, n_grid=21) -> best_weight` finds the
F1-optimal weight on a labeled validation set.
## Prognostics
### `RULEstimator(window_size=32, quantiles=(0.1, 0.5, 0.9), rul_max=125.0, lstm_units=(64, 32), dropout=0.2, learning_rate=1e-3, epochs=30, batch_size=64, validation_split=0.15, patience=8, random_state=42)`
Multi-quantile LSTM regressor for Remaining Useful Life prediction.
Requires `tensorflow`.
- `fit(X: DataFrame, y: array, *, groups=None, validation_data=None)` — `y` is
  per-timestep RUL. Pass engine IDs in `groups` for multiple trajectories;
  windows stay within a group. With no explicit validation data, grouped input
  is split by engine before scaling/windowing using `validation_split`.
  Explicit validation is `(X_val, y_val)` for one stream or
  `(X_val, y_val, groups_val)` for grouped data; grouped training requires
  disjoint validation engine IDs. With groups or explicit validation, the scaler
  fits training data only.
  Plain ungrouped input remains a single-stream interface; engine identities
  are not inferred from sensor values or DataFrame indices. Its implicit Keras
  window split does not establish independent validation; provide explicit
  validation data when that distinction matters.
- `predict_quantiles(X) -> dict[float, ndarray]` — per-quantile
  predictions for one trajectory. Call separately per engine at inference;
  this method does not accept `groups`.
- `predict(X) -> ndarray` — median (or mean of quantiles if 0.5 not
  configured).
- `risk_at_horizon(X, horizon: float) -> ndarray` — a CDF-like score per window,
  using piecewise-linear interpolation of predicted quantiles. Below the first
  positive quantile it interpolates from `(0, 0)`; at/above the largest value it
  plateaus at the largest configured quantile level. The sparse quantiles do
  not identify an upper-tail distribution. Duplicate quantile values use their
  rightmost level. Crossed, negative or non-finite values and a negative or
  non-finite horizon raise `ValueError`. This score needs independent
  calibration before interpretation as a failure probability.

The q10–q90 range has **nominal 80%** coverage. Its empirical coverage must be
measured on independent engines; a quantile label is not a coverage guarantee.

### `RULEnsemble(quantiles=(0.1, 0.5, 0.9), rul_max=125.0, use_physics=True, weight_grid=11)`
Imported from `ai_cta.rul_ensemble`; combines already fitted quantile models.
Weights are nonnegative and sum to one over active models. Without validation
inputs the weights are equal. Tuning requires `X_tabular_val`,
`X_sequential_val`, and `y_val`; an active physical component additionally
requires `elapsed_val`. Prediction with a positive physical-component weight
requires `elapsed` and raises `ValueError` if it is missing. These API guards
do not establish the physical fallback's calibration or ensemble superiority.
### `NeuralRiskEstimator(hidden_units=(128, 64, 32), dropout=0.3, learning_rate=1e-3, epochs=40, batch_size=128, validation_split=0.15, patience=10, cost_fn=10.0, cost_fp=1.0, random_state=42)`
Feedforward classifier with cost-asymmetric BCE loss. Requires
`tensorflow`.
- `fit(X: DataFrame, y: array)` — `y` is the binary failure label.
- `predict_proba(X) -> ndarray` — cost-sensitive score in `[0, 1]` per sample.
  Despite the method name, weighted BCE does not by itself produce calibrated
  failure probabilities. Use separate calibration data before probabilistic
  interpretation.
- `predict(X, threshold=0.5) -> ndarray` — binary prediction.
### `DriftDetector(method="psi" | "ks", threshold=None, n_bins=10, bonferroni=True)`
Distribution drift monitor. Default thresholds: PSI 0.25, KS p-value
0.01 (Bonferroni-corrected when many channels are tested).
- `fit(reference: DataFrame) -> self`
- `detect(current: DataFrame) -> DriftReport`
`DriftReport` exposes `drift_detected`, `method`, `per_channel_score`,
`per_channel_drift`, `threshold`, and `channels_with_drift()`.
### `OnlineCalibrator(alpha=0.05, buffer_size=2000, min_recalibrate=200, schedule_seconds=3600, schedule_samples=500)`
Periodic recalibration of an anomaly threshold on a sliding buffer
of normal-data scores.
- `add(score: float)` / `add_batch(scores: ndarray)` — ingest finite scores;
  batches must be one-dimensional and are validated before changing the buffer.
- `maybe_recalibrate() -> CalibrationUpdate | None` — returns a
  fresh threshold if the schedule allows; otherwise `None`.
- `current_buffer_size() -> int` — current size of the sliding buffer.
- `total_observations() -> int` — total scores ever added.
`CalibrationUpdate` has `threshold`, `n_samples_used`, `timestamp`.
## Risk
### `ChannelLimits(warn_low, warn_high, alarm_low, alarm_high)`
Operational limits for one channel. Method `exceedance(value) ->
float in [0, 1]`.
### `RiskScorer(ml_weight=0.6, limits=None, levels=DEFAULT_LEVELS, missing_channels="raise")`
Composite scoring. Default levels: `OK` (< 0.3), `Warning` (< 0.6),
`Critical` (< 0.85), `Emergency` (≥ 0.85).
With no configured channel limits, the score is the ML score. These fixed bands
are engineering thresholds, not calibrated probabilities or certification levels.
All configured rule channels must be present and finite. The explicit
`missing_channels="legacy_ignore"` policy reproduces historical omission of
missing channels; changing the score composition requires fresh calibration.
Nonfinite anomaly or configured channel scores are rejected in either mode.
- `score(anomaly_score, channel_values=None) -> float in [0, 1]`
- `level(score) -> str`
### `ConformalThresholdCalibrator(alpha=0.05, random_state=42)`
Split conformal prediction with finite-sample correction.
- `calibrate(calibration_scores) -> self` — selects the exact observed order
  statistic with rank `k = ceil((n + 1) * (1 - alpha))`. Finite calibration
  computes this rank from the decimal representation `str(alpha)` to avoid
  floating-point roundoff adding an extra rank at integer boundaries. It
  requires `k <= n` (for example, n=19 at alpha=.05). Smaller samples are
  rejected without replacing a previous threshold. Use finite, held-out
  known-normal scores; marginal control assumes exchangeability. There is no
  automatic conditional or adaptive time-series guarantee.
- `apply(scores) -> ndarray of 0/1`
After calibration: `threshold_`, `calibration_size_`. Alerts use strict `>`;
ties are normal decisions. No interpolation between order statistics is used.
### `RiskAggregator(weights=(1/3, 1/3, 1/3), thresholds=(0.3, 0.6, 0.85), cost_fn=10.0, cost_fp=1.0, alignment_policy="strict")`
Three-component fusion (monograph § 8.4):
    R_final = w₁ R_anom + w₂ R_RUL + w₃ R_NN
Cost weights must be finite and strictly positive; invalid costs raise `ValueError`.
- `aggregate(R_anom, R_RUL, R_NN) -> (R_final, alert_levels)` — requires equal
  lengths by default. Explicit `alignment_policy="legacy_resample"` retains
  historical index resampling, which can discard events; use timestamp-aware
  fusion for asynchronous channels. Equal lengths alone do not establish alignment.
- `calibrate_weights(R_anom_val, R_RUL_val, R_NN_val, y_val) -> ndarray`
  — SLSQP minimizes cost-asymmetric BCE on a separate labeled calibration split.
  Score arrays must align observation by observation; labels may have shape
  `(n,)` or `(n, 1)`. Keep weight selection and final evaluation separate.
  Failed optimization raises `RuntimeError`, preserving the prior weights and
  setting `last_optimization_warning_`; it no longer returns prior weights as
  a calibration result. Reported success must also have a finite objective and
  finite nonnegative weights summing to one within absolute tolerance `1e-6`.
  Only then are weights normalized and assigned, and the warning cleared.
Levels: `OK` / `Warning` / `Critical` / `Emergency`.
## Streaming
### `SafetyPipeline(detector, risk_scorer, window_size=32, alert_threshold=0.6, alert_callback=None, risk_calibrator=None)`
Consumes an iterable of dict readings and yields `PipelineEvent` per
step. `PipelineEvent` has `timestamp`, `anomaly_score`, `risk_score`,
`risk_level`, `channel_values`, and `to_dict()`.
An optional fitted `risk_calibrator` must use held-out **composite** risk scores.
It controls callbacks with a strict `score > threshold` decision, replacing the
fixed callback threshold. Event risk levels retain their separate fixed bands.

Static helper: `SafetyPipeline.make_webhook_callback(url, timeout=5.0)`
builds a callback that POSTs events to an HTTP endpoint in
n8n-compatible JSON.
## Utilities (`ai_cta.utils`)
### `generate_synthetic_stream(n_samples=1000, sampling_rate=1.0, random_state=42) -> DataFrame`
Simple multi-channel generator. Columns: `timestamp`, `temperature`,
`vibration`, `pressure`. For richer simulations use
`IndustrialSimulator` below.
### `inject_anomalies(df, n_anomalies=20, anomaly_types=("spike","drift","oscillation"), channels=("temperature","vibration","pressure"), random_state=42) -> (df, labels)`
Injects three kinds of anomalies into a sensor DataFrame; returns the
contaminated frame plus per-sample binary labels.
### `evaluate_binary_detector(y_true, y_pred) -> DetectorMetrics`
Standard classification metrics: accuracy, precision, recall, F1,
confusion-matrix counts, false-alarm rate.
### `IndustrialSimulator(n_sensors=10, duration_days=30, frequency_minutes=1, anomaly_intensity=0.005, degradation_rate=0.02, failure_probability=0.6, failure_delay_mean=720, seed=42)`
Full multi-channel telemetry generator implementing the model from
monograph § 12.3.1: μ + seasonal sinusoid + multi-mode degradation +
Poisson anomaly events with stochastic failure delays + Gaussian
noise.
- `generate() -> (DataFrame, list[SimulationEvent])` — seeded generation;
  record runtime versions when comparing runs across environments.
`SimulationEvent` has `timestep`, `channel`, `kind`
(`spike` | `level_shift` | `drift`), `magnitude`, `failure_at`.
`make_failure_labels(n_steps, events, horizon=720) -> ndarray` —
converts simulator events into a binary failure label per timestep.
