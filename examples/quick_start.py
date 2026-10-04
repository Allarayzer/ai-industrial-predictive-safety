"""Demonstrate held-out calibration of the *composite* streaming risk.

Run from the repository root: python examples/quick_start.py
This synthetic time-series example illustrates wiring, not a distribution-shift
or conditional-coverage guarantee. See experiments/peerj for study protocols.
"""
from __future__ import annotations

import numpy as np

from ai_cta import ConformalThresholdCalibrator, IsolationForestDetector, RiskScorer, SafetyPipeline
from ai_cta.data import generate_synthetic_stream, inject_anomalies
from ai_cta.pipeline import PipelineEvent
from ai_cta.risk_model import ChannelLimits


def main() -> None:
    window = 64
    # Three separate streams: neither calibration nor test data fit the detector.
    train = generate_synthetic_stream(n_samples=2048, random_state=0)
    calibration = generate_synthetic_stream(n_samples=2048, random_state=1)
    test, _ = inject_anomalies(
        generate_synthetic_stream(n_samples=256, random_state=2),
        n_anomalies=5, random_state=2,
    )
    detector = IsolationForestDetector(
        window_size=window, stride=window, use_spectral=False,
    ).fit(train.drop(columns=["timestamp"]))
    scorer = RiskScorer(ml_weight=0.6, limits={
        "temperature": ChannelLimits(45, 55, 30, 70),
        "vibration": ChannelLimits(0.1, 0.5, 0.0, 0.8),
        "pressure": ChannelLimits(0.9, 1.1, 0.7, 1.3),
    })
    # Calibrate the SAME final score that the pipeline thresholds. Each score
    # uses a disjoint 64-sample window and its final sensor reading for rules.
    scores = detector.decision_function(calibration.drop(columns=["timestamp"]))
    window_ends = calibration.iloc[window - 1::window]
    risks = np.asarray([
        scorer.score(float(score), row.drop(labels="timestamp").to_dict())
        for score, (_, row) in zip(scores, window_ends.iterrows(), strict=True)
    ])
    calibrator = ConformalThresholdCalibrator(alpha=0.05).calibrate(risks)
    alerts: list[PipelineEvent] = []
    pipeline = SafetyPipeline(
        detector, scorer, window_size=window, risk_calibrator=calibrator,
        alert_callback=alerts.append,
    )
    events = list(pipeline.run(test.to_dict(orient="records")))
    print(f"Calibration windows: {len(risks)}; composite risk threshold: {calibrator.threshold_:.4f}")
    print(f"Streaming events: {len(events)}; calibrated alerts: {len(alerts)}")
    print("Risk-level labels retain the fixed scorer bands; callbacks use the calibrated threshold.")
    print("Marginal FAR control requires exchangeable normal calibration/test scores; temporal dependence or shift can invalidate it.")


if __name__ == "__main__":
    main()
