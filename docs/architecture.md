# Architecture

AI-CTA exposes two separate streaming paths and a set of offline model-building
components. The monograph's broader vision (computer vision, digital twins,
SCADA) should not be read as a list of implemented services.

## Sensor path and demo REST

`SafetyPipeline` buffers sensor readings, asks a fitted detector for a window
score, and combines that score with physical limits through `RiskScorer`.
Callbacks use a fixed threshold by default. A fitted `risk_calibrator` can instead
threshold the final composite score. The caller must fit it on separate,
known-normal windows using the same detector, rules, and score definition.
`risk_level` still denotes the scorer's fixed bands; it is not the calibrated
alarm decision. The existing callback behavior is unchanged when no calibrator
is supplied.

The FastAPI demo creates a synthetic-trained Isolation Forest and a fixed-threshold
sensor pipeline at startup. `/predict` and `/score` prepend 63 synthetic readings
to the request, forming one 64-reading window that includes the new observation.
These stateless endpoints are smoke-test demonstrations, not real sensor history.
`/score-batch` accepts actual history and emits one event per complete sliding
window. Every request is independent; callers must supply the required history.
Neither endpoint runs the three-channel fusion or adaptive controller.

## Adaptive score path

`AdaptiveSafetyPipeline` accepts timestamps, drift-monitoring features, and
precomputed updates for anomaly, RUL, and neural scores. Models are run by the
caller, not by this class.

1. `AsynchronousRiskFusion` keeps the latest observation per channel, rejects
   out-of-order updates, excludes stale channels, and renormalizes available
   weights (or requires all three in strict mode). A zero-weight fallback is
   explicitly recorded in the returned snapshot.
2. `DriftDetector` compares a moving feature window with a fitted reference,
   using PSI or KS. A drift signal is not itself a failure label.
3. The caller marks externally verified normal observations with `trusted_normal`.
   `GuardedRecalibrationController` requires persistent drift, discards the old
   calibration buffer, and splits new trusted scores chronologically into
   calibration and validation portions.
4. A candidate threshold is accepted only if validation FAR and maximum threshold
   change checks pass. A rejection keeps the previous threshold. Decisions are
   recorded; this is a threshold update, not model retraining or weight optimization.

Adaptive `alarm` uses the controller's threshold. The snapshot's `risk_level`
continues to describe the aggregator's fixed bands, so the two are distinct.
The adaptive interface is not connected to the demo REST service.

## Offline components and score meaning

- `preprocess.py`: statistical, rolling, and spectral feature transformers.
- `anomaly_detector.py`: Isolation Forest, prediction-residual LSTM, and a hybrid
  detector. Logistic score squashing does not by itself calibrate probabilities.
- `rul_estimator.py`: LSTM quantile regression with pinball loss and an interpolated
  horizon-risk estimate. Quantile coverage and probability calibration need
  separate assessment; quantile crossing remains a limitation.
- `risk_model.py`: detector-plus-rules `RiskScorer`, split-conformal thresholds,
  the neural score, and `RiskAggregator`. SLSQP minimizes cost-asymmetric binary
  cross-entropy on a separate labeled calibration set, not F1 or MCC. The
  offline aggregator's positional resampling is not timestamp synchronization.
- `calibration.py`: scheduled threshold proposals from caller-supplied normal
  scores; the consumer decides whether/how to adopt them.
- `adaptive_calibration.py`: per-regime thresholds with a pooled fallback, plus
  the guarded controller described above.

A split-conformal marginal FAR bound requires exchangeable normal calibration
and test scores with a fixed scoring function. Temporal dependence, drift,
changing channel availability/weights, or using the wrong score for calibration
can invalidate that premise. Guard checks and regime stratification do not
establish a general guarantee under arbitrary shift or repeated adaptation.

## Deployment scope

The HTTP webhook callback is implemented. Kafka, MQTT, and OPC-UA connections
require application-specific integration. Docker Compose provides reference
services, not evidence of their end-to-end integration or operational readiness.
This Python research implementation makes no hard-real-time, safety-certification,
or accident-prevention guarantee. Study-specific evidence and limitations are
mapped in [reproduction notes](reproduction.md).
