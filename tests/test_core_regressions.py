"""Small audit regressions for numerical contracts; no model training."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai_cta.evaluation import evaluate_binary_detector
from ai_cta.risk_model import ConformalThresholdCalibrator, RiskAggregator, RiskScorer
from ai_cta.rul_ensemble import RULEnsemble
from ai_cta.rul_estimator import RULEstimator


@pytest.mark.parametrize("true_column,pred_column", [(False, True), (True, False), (True, True)])
def test_binary_metrics_preserve_sample_pairing(true_column, pred_column):
    labels = np.array([0, 0, 1])
    actual = evaluate_binary_detector(
        labels[:, None] if true_column else labels,
        labels[:, None] if pred_column else labels,
    )
    assert actual.accuracy == 1
    assert actual.f1 == 1
    assert actual.true_positive + actual.true_negative == len(labels)


@pytest.mark.parametrize("invalid", [[0, .5, 1], [0, np.nan, 1], [0, np.inf, 1], [[0, 1]] * 3])
def test_binary_metrics_reject_invalid_labels(invalid):
    with pytest.raises(ValueError):
        evaluate_binary_detector(np.array([0, 0, 1]), np.array(invalid))


def test_aggregator_calibration_accepts_column_labels_without_broadcasting():
    labels = np.array([0., 0., 1., 1.])
    informative = .01 + .98 * labels
    neutral = np.full(4, .5)
    vector = RiskAggregator(cost_fn=1, cost_fp=1).calibrate_weights(
        informative, neutral, neutral, labels,
    )
    column = RiskAggregator(cost_fn=1, cost_fp=1).calibrate_weights(
        informative[:, None], neutral, neutral, labels[:, None],
    )
    np.testing.assert_allclose(column, vector, atol=1e-6)
    assert column[0] > .99


@pytest.mark.parametrize("component,labels", [
    ([.1, .9], [0, 0, 1, 1]),
    ([.1, .2, .8, .9], [0, 1]),
    ([.1, .2, .8, .9], [0, 0, .5, 1]),
    ([.1, np.nan, .8, .9], [0, 0, 1, 1]),
])
def test_aggregator_calibration_rejects_unaligned_or_invalid_inputs(component, labels):
    with pytest.raises(ValueError):
        RiskAggregator().calibrate_weights(np.array(component), np.full(4, .5),
                                           np.full(4, .5), np.array(labels))


def test_aggregator_failed_optimization_preserves_prior(monkeypatch):
    aggregator = RiskAggregator(weights=(.2, .3, .5))
    prior = aggregator.w.copy()
    monkeypatch.setattr("ai_cta.risk_model.minimize", lambda *args, **kwargs: SimpleNamespace(
        success=False, message="forced numerical failure", x=np.array([1., 0., 0.]),
    ))
    with pytest.raises(RuntimeError, match="forced numerical failure"):
        aggregator.calibrate_weights(np.array([.1, .9]), np.full(2, .5),
                                     np.full(2, .5), np.array([0, 1]))
    np.testing.assert_array_equal(aggregator.w, prior)
    assert aggregator.last_optimization_warning_ == "forced numerical failure"


@pytest.mark.parametrize("candidate,objective", [
    ([.5, .5], .1),
    ([np.nan, .3, .7], .1),
    ([np.inf, .3, .7], .1),
    ([-.1, .5, .6], .1),
    ([0., 0., 0.], .1),
    ([.2, .3, .4], .1),
    ([.2, .3, .5], np.nan),
    ([.2, .3, .5], np.inf),
])
def test_aggregator_rejects_invalid_optimizer_success_atomically(monkeypatch, candidate, objective):
    aggregator = RiskAggregator(weights=(.1, .2, .7))
    prior = aggregator.w.copy()
    monkeypatch.setattr("ai_cta.risk_model.minimize", lambda *args, **kwargs: SimpleNamespace(
        success=True, message="reported success", x=np.array(candidate), fun=objective,
    ))
    with pytest.raises(RuntimeError, match="calibration"):
        aggregator.calibrate_weights(np.array([.1, .9]), np.full(2, .5),
                                     np.full(2, .5), np.array([0, 1]))
    np.testing.assert_array_equal(aggregator.w, prior)
    assert aggregator.last_optimization_warning_


def test_aggregator_success_clears_warning_and_accepts_simplex_roundoff(monkeypatch):
    aggregator = RiskAggregator(weights=(.1, .2, .7))
    aggregator.last_optimization_warning_ = "previous failure"
    candidate = np.array([.2, .3, .5000005])
    monkeypatch.setattr("ai_cta.risk_model.minimize", lambda *args, **kwargs: SimpleNamespace(
        success=True, message="success", x=candidate, fun=.1,
    ))
    result = aggregator.calibrate_weights(np.array([.1, .9]), np.full(2, .5),
                                          np.full(2, .5), np.array([0, 1]))
    np.testing.assert_allclose(result, candidate / candidate.sum())
    np.testing.assert_array_equal(aggregator.w, result)
    assert aggregator.last_optimization_warning_ is None


class _ConstantQuantiles:
    def __init__(self, value):
        self.value = value

    def predict_quantiles(self, X):
        return {q: np.full(len(X), self.value) for q in (.1, .5, .9)}


class _ConstantPhysics:
    params_ = SimpleNamespace(scale=100.)

    def predict(self, elapsed):
        return np.full(len(elapsed), 100.)


def test_two_model_ensemble_stays_on_simplex():
    X = np.zeros((4, 1))
    model = RULEnsemble(use_physics=False).fit(
        _ConstantQuantiles(100), _ConstantQuantiles(100), None,
        X, pd.DataFrame(X), np.zeros(4), np.full(4, 50.),
    )
    assert model.weights_.sum() == pytest.approx(1)
    np.testing.assert_allclose(model.predict(X, pd.DataFrame(X)), 100)


def test_two_model_tuning_does_not_require_elapsed():
    X = np.zeros((4, 1))
    model = RULEnsemble(use_physics=False).fit(
        _ConstantQuantiles(20), _ConstantQuantiles(100), None,
        X_tabular_val=X, X_sequential_val=pd.DataFrame(X), y_val=np.full((4, 1), 20.),
    )
    np.testing.assert_allclose(model.weights_, [1, 0, 0])


def test_active_physics_requires_elapsed_at_prediction():
    model = RULEnsemble().fit(_ConstantQuantiles(100), _ConstantQuantiles(100), _ConstantPhysics())
    with pytest.raises(ValueError, match="elapsed"):
        model.predict(np.zeros((4, 1)), pd.DataFrame(np.zeros((4, 1))))


def test_ensemble_rejects_mismatched_prediction_counts():
    model = RULEnsemble(use_physics=False).fit(_ConstantQuantiles(20), _ConstantQuantiles(100), None)
    with pytest.raises(ValueError):
        model.predict(np.zeros((4, 1)), pd.DataFrame(np.zeros((1, 1))))


def test_ensemble_pinball_preserves_column_label_pairing():
    model = RULEnsemble()
    labels = np.array([20., 80.])
    predictions = {q: labels.copy() for q in (.1, .5, .9)}
    assert model._pinball_loss(labels[:, None], predictions) == pytest.approx(0)


def _rul_with_quantiles(monkeypatch, values):
    estimator = RULEstimator()
    monkeypatch.setattr(estimator, "predict_quantiles", lambda _: {
        q: np.array([value]) for q, value in zip((.1, .5, .9), values, strict=True)
    })
    return estimator


def test_rul_cdf_matches_quantiles_and_uses_explicit_upper_plateau(monkeypatch):
    estimator = _rul_with_quantiles(monkeypatch, [10., 50., 90.])
    for horizon, expected in [(0, 0), (5, .05), (10, .1), (50, .5), (90, .9), (100, .9)]:
        assert estimator.risk_at_horizon(pd.DataFrame(), horizon)[0] == pytest.approx(expected)


@pytest.mark.parametrize("values", [[10., 9., 90.], [10., np.nan, 90.], [-1., 50., 90.]])
def test_rul_cdf_rejects_invalid_inverse_cdf(monkeypatch, values):
    estimator = _rul_with_quantiles(monkeypatch, values)
    with pytest.raises(ValueError):
        estimator.risk_at_horizon(pd.DataFrame(), 30)


def test_rul_cdf_repeated_quantiles_use_rightmost_probability(monkeypatch):
    estimator = _rul_with_quantiles(monkeypatch, [0., 0., 90.])
    assert estimator.risk_at_horizon(pd.DataFrame(), 0)[0] == pytest.approx(.5)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_conformal_rejects_nonfinite_calibration_preserving_previous_threshold(invalid):
    calibrator = ConformalThresholdCalibrator().calibrate(np.linspace(.1, .5, 100))
    previous = calibrator.threshold_
    with pytest.raises(ValueError):
        calibrator.calibrate(np.r_[np.linspace(.1, .5, 99), invalid])
    assert calibrator.threshold_ == previous
    assert calibrator.apply(np.array([1.]))[0] == 1


def test_conformal_rejects_nonfinite_prediction():
    calibrator = ConformalThresholdCalibrator().calibrate(np.linspace(.1, .5, 100))
    with pytest.raises(ValueError):
        calibrator.apply(np.array([np.nan]))


def test_default_scorer_without_rules_preserves_ml_score():
    scorer = RiskScorer()
    assert scorer.score(.99) == pytest.approx(.99)
    assert scorer.level(scorer.score(.99)) == "Emergency"
