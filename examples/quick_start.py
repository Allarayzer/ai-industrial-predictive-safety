"""End-to-end demonstration on synthetic sensor data.
Run with:
    python examples/quick_start.py
Shows how to:
    1. Generate synthetic sensor data.
    2. Inject anomalies for evaluation.
    3. Fit an Isolation Forest detector.
    4. Calibrate a conformal threshold.
    5. Combine it with rule-based channel limits.
    6. Run the streaming pipeline with an alert callback.
"""
from __future__ import annotations
import numpy as np
from ai_cta import (
    ConformalThresholdCalibrator,
    IsolationForestDetector,
    RiskScorer,
    SafetyPipeline,
)
from ai_cta.risk_model import ChannelLimits
from ai_cta.pipeline import PipelineEvent
from ai_cta.data import generate_synthetic_stream, inject_anomalies
from ai_cta.evaluation import evaluate_binary_detector

def main() -> None:
    # ---------- 1-2. Data ------------------------------------------------
    print("Generating synthetic training and test streams ...")
    train = generate_synthetic_stream(n_samples=2000, random_state=0)
    test, labels = inject_anomalies(
        generate_synthetic_stream(n_samples=1000, random_state=1),
        n_anomalies=15,
        random_state=1,
    )
    print(f"  Train: {len(train)} samples")
    print(f"  Test : {len(test)} samples ({int(labels.sum())} anomalous)")
    # ---------- 3. Detector ---------------------------------------------
    print("\nFitting IsolationForestDetector ...")
    detector = IsolationForestDetector(
        window_size=64,
        stride=32,
        use_spectral=False,
    ).fit(train.drop(columns=["timestamp"]))
    print(f"  Engineered feature count: {len(detector.feature_names_)}")
    print(f"  Training windows:         {detector.n_windows_fit_}")
    # ---------- 4. Conformal calibration --------------------------------
    print("\nCalibrating threshold at alpha=0.05 ...")
    calib_scores = detector.decision_function(train.drop(columns=["timestamp"]))
    calibrator = ConformalThresholdCalibrator(alpha=0.05).calibrate(calib_scores)
    print(f"  Calibrated threshold: {calibrator.threshold_:.3f}")
    # Evaluate the conformal-thresholded detector on the test stream.
    test_scores = detector.decision_function(test.drop(columns=["timestamp"]))
    test_preds = calibrator.apply(test_scores)
    label_idx = np.linspace(0, len(labels) - 1, len(test_scores)).astype(int)
    aligned_labels = labels[label_idx]
    metrics = evaluate_binary_detector(aligned_labels, test_preds)
    print(
        f"  F1={metrics.f1:.3f}  "
        f"precision={metrics.precision:.3f}  "
        f"recall={metrics.recall:.3f}  "
        f"FAR={metrics.false_alarm_rate:.3f}"
    )
    # ---------- 5. Composite risk scorer --------------------------------
    scorer = RiskScorer(
        ml_weight=0.6,
        limits={
            "temperature": ChannelLimits(warn_low=45, warn_high=55, alarm_low=30, alarm_high=70),
            "vibration": ChannelLimits(warn_low=0.1, warn_high=0.5, alarm_low=0.0, alarm_high=0.8),
            "pressure": ChannelLimits(warn_low=0.9, warn_high=1.1, alarm_low=0.7, alarm_high=1.3),
        },
    )
    # ---------- 6. Streaming pipeline -----------------------------------
    print("\nRunning SafetyPipeline on the test stream ...")
    alerts: list[PipelineEvent] = []
    pipeline = SafetyPipeline(
        detector=detector,
        risk_scorer=scorer,
        window_size=64,
        alert_threshold=0.7,
        alert_callback=alerts.append,
    )
    event_count = 0
    for _event in pipeline.run(test.to_dict(orient="records")):
        event_count += 1
    print(f"  Events emitted: {event_count}")
    print(f"  Alerts raised:  {len(alerts)}")
    if alerts:
        first, last = alerts[0], alerts[-1]
        print(
            f"  First alert: t={first.timestamp} "
            f"risk={first.risk_score:.2f} level={first.risk_level}"
        )
        print(
            f"  Last alert:  t={last.timestamp} "
            f"risk={last.risk_score:.2f} level={last.risk_level}"
        )

if __name__ == "__main__":
    main()
