import numpy as np

from ai_cta.adaptive_calibration import (
    GuardedRecalibrationController,
    RegimeConformalCalibrator,
)


def test_regime_calibrator_uses_distinct_thresholds():
    rng = np.random.default_rng(0)
    a = rng.normal(0.2, 0.02, 200)
    b = rng.normal(0.5, 0.02, 200)
    scores = np.r_[a, b]
    regimes = np.array(["A"] * len(a) + ["B"] * len(b))
    cal = RegimeConformalCalibrator(alpha=0.05).calibrate(scores, regimes)
    assert cal.threshold_for("B") > cal.threshold_for("A")
    assert cal.threshold_for("unseen") == cal.pooled_


def test_guarded_recalibration_accepts_valid_candidate():
    rng = np.random.default_rng(1)
    ctrl = GuardedRecalibrationController(
        initial_threshold=0.35,
        alpha=0.05,
        min_samples=200,
        drift_persistence=2,
        cooldown_samples=0,
        max_relative_change=1.0,
        far_tolerance=2.0,
    )
    ctrl.update_drift(True)
    ctrl.update_drift(True)
    for score in rng.normal(0.25, 0.03, 400):
        ctrl.observe_normal(score)
    decision = ctrl.maybe_recalibrate()
    assert decision.attempted
    assert decision.accepted
    assert 0.2 < ctrl.threshold < 0.4


def test_guarded_recalibration_rejects_large_jump():
    rng = np.random.default_rng(2)
    ctrl = GuardedRecalibrationController(
        initial_threshold=0.2,
        alpha=0.05,
        min_samples=200,
        drift_persistence=1,
        cooldown_samples=0,
        max_relative_change=0.1,
        far_tolerance=2.0,
    )
    ctrl.update_drift(True)
    for score in rng.normal(0.6, 0.02, 400):
        ctrl.observe_normal(score)
    decision = ctrl.maybe_recalibrate()
    assert decision.attempted
    assert not decision.accepted
    assert decision.reason == "threshold_jump_too_large"
    assert ctrl.threshold == 0.2
