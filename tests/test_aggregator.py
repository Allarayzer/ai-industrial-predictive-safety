"""Tests for the RiskAggregator."""
import numpy as np
import pytest
from ai_cta.risk import RiskAggregator

def test_aggregator_default_weights():
    agg = RiskAggregator()
    assert agg.w.shape == (3,)
    np.testing.assert_allclose(agg.w.sum(), 1.0, atol=1e-9)

def test_aggregator_aggregates_correctly():
    agg = RiskAggregator(weights=(1.0, 0.0, 0.0))
    R_anom = np.array([0.1, 0.5, 0.9])
    R_RUL = np.array([0.0, 0.0, 0.0])
    R_NN = np.array([0.0, 0.0, 0.0])
    R_final, _ = agg.aggregate(R_anom, R_RUL, R_NN)
    np.testing.assert_allclose(R_final, R_anom)

def test_aggregator_alert_levels():
    agg = RiskAggregator(weights=(1.0, 0.0, 0.0))
    R_anom = np.array([0.1, 0.4, 0.7, 0.9])
    R_zero = np.zeros(4)
    _, alerts = agg.aggregate(R_anom, R_zero, R_zero)
    assert list(alerts) == ["OK", "Warning", "Critical", "Emergency"]

def test_aggregator_validates_weights():
    with pytest.raises(ValueError):
        RiskAggregator(weights=(0.5, 0.5))  # wrong length
    with pytest.raises(ValueError):
        RiskAggregator(weights=(-0.1, 0.5, 0.6))  # negative
    with pytest.raises(ValueError):
        RiskAggregator(weights=(0.0, 0.0, 0.0))  # all zero

def test_aggregator_validates_thresholds():
    with pytest.raises(ValueError):
        RiskAggregator(thresholds=(0.5, 0.3, 0.8))  # not increasing
    with pytest.raises(ValueError):
        RiskAggregator(thresholds=(0.0, 0.5, 0.8))  # boundary at 0
    with pytest.raises(ValueError):
        RiskAggregator(thresholds=(0.3, 0.6, 1.0))  # boundary at 1

def test_aggregator_validates_costs():
    with pytest.raises(ValueError):
        RiskAggregator(cost_fn=0)
    with pytest.raises(ValueError):
        RiskAggregator(cost_fp=-1)

def test_aggregator_calibrate_weights_runs():
    """Verify SLSQP optimization completes without error."""
    rng = np.random.default_rng(0)
    n = 200
    # Make R_anom strongly informative; others noise.
    y = rng.integers(0, 2, n).astype(float)
    R_anom = 0.7 * y + rng.uniform(0, 0.3, n)
    R_RUL = rng.uniform(0, 1, n)
    R_NN = rng.uniform(0, 1, n)
    agg = RiskAggregator()
    weights = agg.calibrate_weights(R_anom, R_RUL, R_NN, y)
    assert weights.shape == (3,)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
    # R_anom should win highest weight in this construction
    assert weights[0] > weights[1]
    assert weights[0] > weights[2]

def test_aggregator_alignment_handles_unequal_lengths():
    agg = RiskAggregator(weights=(1.0, 0.0, 0.0))
    a = np.linspace(0.1, 0.9, 100)
    b = np.linspace(0.0, 1.0, 50)  # shorter
    c = np.linspace(0.0, 1.0, 200)  # longer
    R_final, alerts = agg.aggregate(a, b, c)
    # Output length matches the shortest
    assert len(R_final) == 50
    assert len(alerts) == 50
