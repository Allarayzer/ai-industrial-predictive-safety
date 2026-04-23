"""Tests for risk scoring and conformal calibration."""
import numpy as np
import pytest
from ai_cta.risk_model import ChannelLimits, RiskScorer
from ai_cta.risk_model import ConformalThresholdCalibrator

def test_risk_scorer_pure_ml():
    scorer = RiskScorer(ml_weight=1.0, limits={})
    assert scorer.score(0.3) == pytest.approx(0.3)
    assert scorer.score(0.9) == pytest.approx(0.9)
    assert scorer.score(0.0) == pytest.approx(0.0)

def test_risk_scorer_with_limits():
    limits = {"temp": ChannelLimits(warn_low=40, warn_high=80, alarm_low=20, alarm_high=100)}
    scorer = RiskScorer(ml_weight=0.5, limits=limits)
    # Temperature well within warning band: only ML contributes.
    score = scorer.score(0.4, {"temp": 60.0})
    assert score == pytest.approx(0.2)  # 0.5 * 0.4 + 0.5 * 0.0
    # Temperature at alarm high: rule contributes max.
    score_high = scorer.score(0.4, {"temp": 100.0})
    assert score_high > score

def test_risk_scorer_invalid_weight():
    with pytest.raises(ValueError):
        RiskScorer(ml_weight=1.5)

def test_risk_scorer_levels():
    scorer = RiskScorer(ml_weight=1.0, limits={})
    assert scorer.level(0.1) == "OK"
    assert scorer.level(0.4) == "Warning"
    assert scorer.level(0.7) == "Critical"
    assert scorer.level(0.9) == "Emergency"

def test_channel_limits_exceedance():
    limits = ChannelLimits(warn_low=40, warn_high=80, alarm_low=20, alarm_high=100)
    # Well inside warning band.
    assert limits.exceedance(60.0) == 0.0
    # At warn high.
    assert limits.exceedance(80.0) == pytest.approx(0.0)
    # Halfway between warn_high and alarm_high.
    assert 0.0 < limits.exceedance(90.0) < 1.0
    # Beyond alarm_high.
    assert limits.exceedance(120.0) == 1.0

def test_conformal_calibrator_coverage():
    rng = np.random.default_rng(0)
    # Draw 1000 normal calibration scores uniformly from a plausible distribution.
    scores = rng.uniform(0.0, 0.5, 1000)
    cal = ConformalThresholdCalibrator(alpha=0.1).calibrate(scores)
    # On test scores from the same distribution, the false-alarm rate should
    # be approximately alpha.
    test_scores = rng.uniform(0.0, 0.5, 10000)
    preds = cal.apply(test_scores)
    far = preds.mean()
    assert 0.05 < far < 0.15

def test_conformal_calibrator_invalid_alpha():
    with pytest.raises(ValueError):
        ConformalThresholdCalibrator(alpha=0.0)
    with pytest.raises(ValueError):
        ConformalThresholdCalibrator(alpha=1.0)

def test_conformal_calibrator_insufficient_data():
    cal = ConformalThresholdCalibrator(alpha=0.05)
    with pytest.raises(ValueError):
        cal.calibrate(np.array([0.1, 0.2, 0.3]))

def test_conformal_calibrator_requires_calibration():
    cal = ConformalThresholdCalibrator(alpha=0.1)
    with pytest.raises(RuntimeError):
        cal.apply(np.array([0.1, 0.2]))
