"""Integration checks for drift, trusted scores, recalibration, and alarms."""
import numpy as np
import pandas as pd
import pytest

from ai_cta import (
    AdaptiveSafetyPipeline,
    AsynchronousRiskFusion,
    DriftDetector,
    GuardedRecalibrationController,
    RiskAggregator,
)


def make_pipeline(initial_threshold, max_change=1.0):
    controller = GuardedRecalibrationController(
        initial_threshold=initial_threshold, alpha=0.1, min_samples=30,
        drift_persistence=2, cooldown_samples=0, max_relative_change=max_change,
    )
    return AdaptiveSafetyPipeline(
        fusion=AsynchronousRiskFusion(RiskAggregator(), max_age_seconds=10),
        drift_detector=DriftDetector(method="ks", threshold=0.01),
        recalibration=controller,
        reference_features=pd.DataFrame({"sensor": np.linspace(0, 1, 100)}),
        drift_window_size=8, drift_check_interval=1,
    )


@pytest.mark.parametrize("trusted", [True, False])
def test_only_trusted_post_drift_scores_can_change_threshold(trusted):
    pipeline = make_pipeline(initial_threshold=0.5)
    start = pd.Timestamp("2026-01-01")
    events = [pipeline.process(
        start + pd.Timedelta(seconds=i), {"sensor": 5.0},
        {"anomaly": 0.3, "rul": 0.3, "neural": 0.3}, trusted_normal=trusted,
    ) for i in range(45)]
    assert any(e.drift_report and e.drift_report.drift_detected for e in events)
    accepted = [e for e in events if e.recalibration and e.recalibration.accepted]
    if trusted:
        assert accepted
        assert pipeline.recalibration.threshold == pytest.approx(0.3)
    else:
        assert not accepted
        assert pipeline.recalibration.threshold == 0.5
    alarm = pipeline.process(
        start + pd.Timedelta(seconds=45), {"sensor": 5.0},
        {"anomaly": 0.9, "rul": 0.9, "neural": 0.9}, trusted_normal=False,
    )
    assert alarm.alarm


def test_rejected_recalibration_preserves_threshold_and_alarm():
    pipeline = make_pipeline(initial_threshold=0.2, max_change=0.1)
    events = [pipeline.process(
        pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=i), {"sensor": 5.0},
        {"anomaly": 0.8, "rul": 0.8, "neural": 0.8}, trusted_normal=True,
    ) for i in range(45)]
    attempts = [e.recalibration for e in events if e.recalibration and e.recalibration.attempted]
    assert attempts
    assert all(not d.accepted and d.reason == "threshold_jump_too_large" for d in attempts)
    assert pipeline.recalibration.threshold == 0.2
    assert all(e.alarm for e in events)
