"""Small numerical checks; no expensive model fitting or dataset downloads."""
import numpy as np
import pytest

from ai_cta.risk_model import NeuralRiskEstimator
from ai_cta.rul_estimator import RULEstimator

tf = pytest.importorskip("tensorflow")


@pytest.mark.parametrize("column_labels", [False, True])
def test_pinball_loss_pairs_targets_with_their_own_prediction(column_labels):
    y = np.array([2.0, 9.0], dtype=np.float32)
    predictions = np.array([[1, 2, 3], [7, 8, 10]], dtype=np.float32)
    quantiles = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    error = y[:, None] - predictions
    expected = np.maximum(quantiles * error, (quantiles - 1) * error).mean()
    labels = y[:, None] if column_labels else y
    actual = RULEstimator()._pinball_loss()(tf.constant(labels), tf.constant(predictions))
    assert float(actual) == pytest.approx(float(expected))


@pytest.mark.parametrize("column_labels", [False, True])
def test_asymmetric_bce_pairs_targets_with_their_own_prediction(column_labels):
    y = np.array([0.0, 1.0], dtype=np.float32)
    predictions = np.array([[0.1], [0.8]], dtype=np.float32)
    expected = -(np.log(0.9) + 10 * np.log(0.8)) / 2
    labels = y[:, None] if column_labels else y
    actual = NeuralRiskEstimator()._asymmetric_bce()(
        tf.constant(labels), tf.constant(predictions),
    )
    assert float(actual) == pytest.approx(expected)


def test_quantile_model_can_train_on_a_small_batch():
    estimator = RULEstimator(window_size=4, lstm_units=(2,), dropout=0)
    model = estimator._build_model(n_channels=2)
    x = np.random.default_rng(0).normal(size=(4, 4, 2)).astype(np.float32)
    loss = model.train_on_batch(x, np.array([4, 3, 2, 1], dtype=np.float32))
    assert np.isfinite(loss)
    assert model(x, training=False).shape == (4, 3)
