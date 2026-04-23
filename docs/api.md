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
- `fit(X: DataFrame, y: array)` — `y` is the per-timestep RUL value.
- `predict_quantiles(X) -> dict[float, ndarray]` — per-quantile
  predictions.
- `predict(X) -> ndarray` — median (or mean of quantiles if 0.5 not
  configured).
- `risk_at_horizon(X, horizon: float) -> ndarray` — probability per
  window that RUL ≤ horizon. Implements R_RUL (monograph § 8.3.2)
  via piecewise-linear empirical CDF interpolation.
### `NeuralRiskEstimator(hidden_units=(128, 64, 32), dropout=0.3, learning_rate=1e-3, epochs=40, batch_size=128, validation_split=0.15, patience=10, cost_fn=10.0, cost_fp=1.0, random_state=42)`
Feedforward classifier with cost-asymmetric BCE loss. Requires
`tensorflow`.
- `fit(X: DataFrame, y: array)` — `y` is the binary failure label.
- `predict_proba(X) -> ndarray` — failure probability per sample.
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
- `add(score: float)` / `add_batch(scores: ndarray)` — ingest scores.
- `maybe_recalibrate() -> CalibrationUpdate | None` — returns a
  fresh threshold if the schedule allows; otherwise `None`.
- `current_buffer_size() -> int` — current size of the sliding buffer.
- `total_observations() -> int` — total scores ever added.
`CalibrationUpdate` has `threshold`, `n_samples_used`, `timestamp`.
## Risk
### `ChannelLimits(warn_low, warn_high, alarm_low, alarm_high)`
Operational limits for one channel. Method `exceedance(value) ->
float in [0, 1]`.
### `RiskScorer(ml_weight=0.6, limits=None, levels=DEFAULT_LEVELS)`
Composite scoring. Default levels: `OK` (< 0.3), `Warning` (< 0.6),
`Critical` (< 0.85), `Emergency` (≥ 0.85).
- `score(anomaly_score, channel_values=None) -> float in [0, 1]`
- `level(score) -> str`
### `ConformalThresholdCalibrator(alpha=0.05, random_state=42)`
Split conformal prediction with finite-sample correction.
- `calibrate(calibration_scores) -> self` — requires
  `len(scores) >= ceil(1 / alpha)`.
- `apply(scores) -> ndarray of 0/1`
After calibration: `threshold_`, `calibration_size_`.
### `RiskAggregator(weights=(1/3, 1/3, 1/3), thresholds=(0.3, 0.6, 0.85), cost_fn=10.0, cost_fp=1.0)`
Three-component fusion (monograph § 8.4):
    R_final = w₁ R_anom + w₂ R_RUL + w₃ R_NN
- `aggregate(R_anom, R_RUL, R_NN) -> (R_final, alert_levels)`
- `calibrate_weights(R_anom_val, R_RUL_val, R_NN_val, y_val) -> ndarray`
  — SLSQP optimization on the simplex.
Levels: `OK` / `Warning` / `Critical` / `Emergency`.
## Streaming
### `SafetyPipeline(detector, risk_scorer, window_size=32, alert_threshold=0.6, alert_callback=None)`
Consumes an iterable of dict readings and yields `PipelineEvent` per
step. `PipelineEvent` has `timestamp`, `anomaly_score`, `risk_score`,
`risk_level`, `channel_values`, and `to_dict()`.
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
- `generate() -> (DataFrame, list[SimulationEvent])` — bit-identical
  reproducibility from the seed.
`SimulationEvent` has `timestep`, `channel`, `kind`
(`spike` | `level_shift` | `drift`), `magnitude`, `failure_at`.
`make_failure_labels(n_steps, events, horizon=720) -> ndarray` —
converts simulator events into a binary failure label per timestep.
