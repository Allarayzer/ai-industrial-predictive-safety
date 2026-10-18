"""Independent rank and input-contract checks; no training or downloaded data."""
import math
from fractions import Fraction

import numpy as np
import pytest

from ai_cta.adaptive_calibration import RegimeConformalCalibrator
from ai_cta.calibration import OnlineCalibrator
from ai_cta.risk_model import (
    ChannelLimits,
    ConformalThresholdCalibrator,
    RiskAggregator,
    RiskScorer,
)


def _reference_order_statistic(scores, alpha):
    """Definition using exact rational arithmetic and sorting, not np.quantile."""
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError("A finite conformal threshold is unavailable at this sample size.")
    return sorted(scores)[rank - 1]


@pytest.mark.parametrize("scores,alpha", [
    ([0., 1., 2., 3., 100.], Fraction(2, 5)),
    (list(range(100)), Fraction(1, 20)),
    ([0., 0., 0., 3., 100.], Fraction(2, 5)),
])
def test_conformal_uses_exact_observed_order_statistic(scores, alpha):
    expected = _reference_order_statistic(scores, alpha)
    calibrator = ConformalThresholdCalibrator(alpha=float(alpha)).calibrate(np.array(scores))
    assert calibrator.threshold_ == expected
    assert calibrator.threshold_ in scores
    # Equality is covered, including ties, so the decision is strictly >.
    assert calibrator.apply([expected])[0] == 0


@pytest.mark.parametrize("alpha,n", [(Fraction(1, 10), 9), (Fraction(1, 20), 19)])
def test_conformal_accepts_smallest_sample_size_with_finite_threshold(alpha, n):
    scores = list(range(n))
    calibrator = ConformalThresholdCalibrator(alpha=float(alpha)).calibrate(np.array(scores))
    assert calibrator.threshold_ == _reference_order_statistic(scores, alpha)


@pytest.mark.parametrize("alpha,n,expected_threshold", [
    (.7, 9, 2),
    (.95, 19, 0),
    (np.nextafter(.7, 0), 9, 3),
    (1 / 3, 5, 4),
])
def test_conformal_rank_uses_decimal_alpha_without_rounding_tolerance(alpha, n, expected_threshold):
    # .7 and .95 are exact user-facing decimal levels. Values just below
    # a boundary must not be rounded up to it: float(1/3) is the decimal
    # .3333333333333333, strictly below the exact rational 1/3.
    scores = np.arange(n)
    calibrator = ConformalThresholdCalibrator(alpha=alpha).calibrate(scores)
    assert calibrator.threshold_ == expected_threshold
    assert calibrator.threshold_ == _reference_order_statistic(scores, Fraction(str(alpha)))


def test_conformal_rejects_insufficient_sample_size_without_replacing_fit():
    calibrator = ConformalThresholdCalibrator(alpha=.1).calibrate(np.arange(20))
    previous = (calibrator.threshold_, calibrator.calibration_size_)
    with pytest.raises(ValueError):
        calibrator.calibrate(np.arange(8))
    assert (calibrator.threshold_, calibrator.calibration_size_) == previous


@pytest.mark.parametrize("population", [np.array([0., 1., 2., 3., 4., 100.]),
                                       np.array([0., 0., 0., 1., 1., 1.])])
def test_conformal_exchangeable_rank_bound_by_leave_one_out_enumeration(population):
    # Each held-out rank is equally likely under exchangeability. This checks
    # the finite-sample bound exactly, without relying on stochastic coverage.
    alpha = Fraction(1, 2)
    decisions = []
    for index, test_score in enumerate(population):
        calibration = np.delete(population, index)
        expected = _reference_order_statistic(calibration.tolist(), alpha)
        fitted = ConformalThresholdCalibrator(alpha=float(alpha)).calibrate(calibration)
        assert fitted.threshold_ == expected
        decisions.append(int(fitted.apply([test_score])[0]))
    assert Fraction(sum(decisions), len(decisions)) <= alpha


def _scorer(**kwargs):
    return RiskScorer(ml_weight=.5, limits={
        "temperature": ChannelLimits(40, 80, 20, 100),
        "pressure": ChannelLimits(2, 4, 1, 5),
    }, **kwargs)


@pytest.mark.parametrize("channels", [None, {}, {"temperature": 90}])
def test_scorer_rejects_missing_configured_rule_channels(channels):
    with pytest.raises(ValueError, match="[Mm]issing"):
        _scorer().score(.2, channels)


def test_scorer_legacy_missing_behavior_requires_explicit_opt_in():
    assert _scorer(missing_channels="legacy_ignore").score(.2, {"temperature": 90}) == .35
    assert _scorer(missing_channels="legacy_ignore").score(.2) == .1


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_scorer_rejects_nonfinite_anomaly_scores(invalid):
    with pytest.raises(ValueError, match="finite"):
        RiskScorer().score(invalid)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_scorer_rejects_nonfinite_configured_channel_values(invalid):
    with pytest.raises(ValueError, match="finite"):
        _scorer().score(.2, {"temperature": invalid, "pressure": 3})


def test_aggregator_rejects_implicit_length_resampling():
    # Middle anomaly=.9 disappears under the previous linspace resampling.
    with pytest.raises(ValueError, match="equal length"):
        RiskAggregator(weights=(1, 0, 0)).aggregate(np.array([.1, .9, .1]),
                                                   np.array([.2, .2]), np.array([.2, .2]))


def test_aggregator_legacy_resampling_requires_explicit_opt_in():
    aggregator = RiskAggregator(weights=(1, 0, 0), alignment_policy="legacy_resample")
    scores, _ = aggregator.aggregate(np.array([.1, .9, .1]), np.array([.2, .2]), np.array([.2, .2]))
    np.testing.assert_array_equal(scores, [.1, .1])


@pytest.mark.parametrize("parameter", ["cost_fn", "cost_fp"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 0., -1.])
def test_aggregator_rejects_nonfinite_or_nonpositive_costs(parameter, invalid):
    with pytest.raises(ValueError, match="positive|finite"):
        RiskAggregator(**{parameter: invalid})


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_online_calibrator_rejects_invalid_observation_without_poisoning_buffer(invalid):
    online = OnlineCalibrator(alpha=.1, min_recalibrate=20, buffer_size=20)
    online.add_batch(np.linspace(0, .5, 20))
    before = (online.current_buffer_size(), online.total_observations())
    with pytest.raises(ValueError, match="finite"):
        online.add(invalid)
    assert (online.current_buffer_size(), online.total_observations()) == before
    assert online.maybe_recalibrate() is not None


def test_online_calibrator_invalid_batch_is_atomic():
    online = OnlineCalibrator()
    with pytest.raises(ValueError, match="finite"):
        online.add_batch(np.array([.1, .2, np.nan]))
    assert online.current_buffer_size() == online.total_observations() == 0


@pytest.mark.parametrize("invalid", [np.array([np.nan, .3]), np.array([[.2], [.3]])])
def test_regime_calibrator_rejects_invalid_score_arrays(invalid):
    calibrator = RegimeConformalCalibrator(alpha=.1).calibrate(np.linspace(0, .5, 40),
                                                            np.array(["a"] * 40))
    with pytest.raises(ValueError):
        calibrator.apply(invalid, np.array(["a", "a"]))
