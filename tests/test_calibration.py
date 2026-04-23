"""Tests for the OnlineCalibrator."""
import numpy as np
import pytest

from ai_cta.calibration import CalibrationUpdate, OnlineCalibrator


def test_calibrator_initial_state():
    cal = OnlineCalibrator(alpha=0.05, min_recalibrate=100)
    assert cal.current_buffer_size() == 0
    assert cal.total_observations() == 0
    # Not enough data → no update
    assert cal.maybe_recalibrate() is None

def test_calibrator_first_calibration():
    cal = OnlineCalibrator(
        alpha=0.05,
        buffer_size=500,
        min_recalibrate=200,
        schedule_seconds=0.0,  # immediate
        schedule_samples=0,
    )
    rng = np.random.default_rng(0)
    cal.add_batch(rng.uniform(0.0, 0.5, 250))
    assert cal.current_buffer_size() == 250
    update = cal.maybe_recalibrate()
    assert isinstance(update, CalibrationUpdate)
    assert 0.0 <= update.threshold <= 1.0
    assert update.n_samples_used == 250

def test_calibrator_throttles_by_sample_count():
    cal = OnlineCalibrator(
        alpha=0.1,
        buffer_size=500,
        min_recalibrate=100,
        schedule_seconds=0.0,
        schedule_samples=200,
    )
    rng = np.random.default_rng(0)
    cal.add_batch(rng.uniform(0, 0.5, 150))
    first = cal.maybe_recalibrate()
    assert first is not None  # first call always allowed once min met
    # Add only a few more — should not retrigger
    cal.add_batch(rng.uniform(0, 0.5, 50))
    second = cal.maybe_recalibrate()
    assert second is None
    # Add enough to meet schedule_samples
    cal.add_batch(rng.uniform(0, 0.5, 200))
    third = cal.maybe_recalibrate()
    assert third is not None

def test_calibrator_validates_alpha():
    with pytest.raises(ValueError):
        OnlineCalibrator(alpha=0.0)
    with pytest.raises(ValueError):
        OnlineCalibrator(alpha=1.0)

def test_calibrator_validates_buffer():
    with pytest.raises(ValueError):
        OnlineCalibrator(buffer_size=50, min_recalibrate=100)

def test_calibrator_buffer_bounded():
    cal = OnlineCalibrator(
        alpha=0.05, buffer_size=100, min_recalibrate=50
    )
    rng = np.random.default_rng(0)
    cal.add_batch(rng.uniform(0, 1, 250))
    # Buffer should not exceed buffer_size even though 250 added
    assert cal.current_buffer_size() == 100
    assert cal.total_observations() == 250
