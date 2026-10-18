"""Non-finite model outputs must be rejected before RUL clipping; no training."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def models(monkeypatch):
    helpers = load_module("benchmarks/run_cmapss_rul_normalized.py")
    frame = pd.DataFrame({"unit": np.repeat([1, 2, 3], 18), "cycle": np.tile(np.arange(1, 19), 3)})
    for sensor in helpers.SELECTED_SENSORS:
        frame[sensor] = frame.cycle.astype(float)
    train = frame[frame.unit.isin([1, 2])].copy()
    validation = frame[frame.unit == 3].copy()
    state = SimpleNamespace(prediction=1.0)

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def compile(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return self

        def predict(self, inputs, **kwargs):
            return np.full((len(inputs), 1), state.prediction)

        def predict_quantiles(self, *args, **kwargs):
            return {0.5: np.array([state.prediction])}

    def layer(*args, **kwargs):
        return lambda inputs: inputs
    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        SimpleNamespace(
            keras=SimpleNamespace(
                utils=SimpleNamespace(set_random_seed=lambda seed: None),
                Input=lambda **kwargs: None,
                layers=SimpleNamespace(LSTM=layer, Dense=layer),
                Model=FakeModel,
                optimizers=SimpleNamespace(Adam=lambda *args: None),
                callbacks=SimpleNamespace(EarlyStopping=lambda **kwargs: None),
            )
        ),
    )
    import ai_cta.rul_ensemble as ensemble
    import ai_cta.rul_estimator as estimator

    monkeypatch.setattr(estimator, "RULEstimator", FakeModel)
    for name in ("XGBoostQuantileRegressor", "PhysicsGuidedRUL", "RULEnsemble"):
        monkeypatch.setattr(ensemble, name, FakeModel)
    return helpers, train, validation, state, FakeModel


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("family", ["pointwise", "mse", "quantile", "full", "e2"])
def test_predictors_reject_nonfinite_before_clipping(models, value, family):
    helpers, train, validation, state, fake_model = models
    state.prediction = value
    train_rul, val_rul = helpers.compute_rul(train), helpers.compute_rul(validation)
    with pytest.raises(ValueError, match="finite"):
        if family == "pointwise":
            helpers.fit_predict_pointwise(fake_model, train, train_rul, validation)
        elif family == "e2":
            runner = load_module("experiments/maintained/e2_cmapss_calib_split.py")
            predict = runner.fit_r_rul(train, train_rul, (validation, val_rul))
            predict(validation)
        else:
            function = {
                "mse": helpers.fit_predict_lstm_mse,
                "quantile": helpers.fit_predict_lstm_ensemble,
                "full": helpers.fit_predict_full_ensemble,
            }[family]
            function(train, train_rul, validation, 1, 2, validation_data=(validation, val_rul))


@pytest.mark.parametrize("values", [10.0, np.array([[-1.0, 50.0, 130.0]], dtype=np.float32)])
def test_finite_clipping_preserves_values_and_shape(values):
    helpers = load_module("benchmarks/run_cmapss_rul_normalized.py")
    expected = np.clip(values, 0.0, helpers.RUL_MAX)
    actual = helpers.finite_clip_rul(values)
    assert np.shape(actual) == np.shape(values)
    np.testing.assert_array_equal(actual, expected)
