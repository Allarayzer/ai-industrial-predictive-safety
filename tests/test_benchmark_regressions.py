"""Scientific protocol regressions using tiny arrays; no model training/downloads."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai_cta.rul_estimator import RULEstimator

ROOT = Path(__file__).resolve().parents[1]


def load_runner(relative_path):
    spec = importlib.util.spec_from_file_location(Path(relative_path).stem, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rul_windows_never_join_engines():
    estimator = RULEstimator(window_size=3)
    X = np.array([10, 11, 12, 13, 20, 21, 22, 23], dtype=np.float32)[:, None]
    windows, targets = estimator._make_windows(X, np.arange(8), groups=np.repeat([1, 2], 4))
    np.testing.assert_array_equal(
        windows[:, :, 0], [[10, 11, 12], [11, 12, 13], [20, 21, 22], [21, 22, 23]]
    )
    np.testing.assert_array_equal(targets, [2, 3, 6, 7])


def test_rul_explicit_validation_fits_scaler_only_on_training(monkeypatch):
    pytest.importorskip("tensorflow")
    captured = {}

    class Recorder:
        def fit(self, x, y, **kwargs):
            captured.update(x=x, y=y, **kwargs)
            return SimpleNamespace(history={})

    estimator = RULEstimator(window_size=2, epochs=1)
    monkeypatch.setattr(estimator, "_build_model", lambda n_channels: Recorder())
    train = pd.DataFrame({"sensor": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    validation = pd.DataFrame({"sensor": [100.0, 101.0, 102.0]})
    estimator.fit(
        train,
        np.arange(6),
        groups=np.repeat([1, 2], 3),
        validation_data=(validation, np.arange(3), np.repeat(3, 3)),
    )
    np.testing.assert_allclose(estimator._scaler.mean_, [2.5])
    assert captured.get("validation_split", 0) == 0
    assert captured["x"].shape[0] == 4
    assert captured["validation_data"][0].shape[0] == 2
    assert captured["validation_data"][0].min() > captured["x"].max()


def test_rul_rejects_overlapping_validation_engines():
    pytest.importorskip("tensorflow")
    estimator = RULEstimator(window_size=2)
    frame = pd.DataFrame({"sensor": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="disjoint"):
        estimator.fit(
            frame, np.arange(3), groups=[1, 1, 1], validation_data=(frame, np.arange(3), [1, 1, 1])
        )


def test_cmapss_split_is_disjoint_by_engine_and_windows_stay_inside():
    runner = load_runner("benchmarks/run_cmapss_rul_normalized.py")
    data = pd.DataFrame({"unit": np.repeat(np.arange(1, 5), 4), "cycle": np.tile([1, 2, 3, 4], 4)})
    for col in runner.SELECTED_SENSORS:
        data[col] = data["unit"] * 10 + data["cycle"]
    train, validation = runner.split_engine_validation(data, validation_fraction=0.25, seed=0)
    assert set(train.unit).isdisjoint(validation.unit)
    assert set(train.unit) | set(validation.unit) == {1, 2, 3, 4}
    windows, targets = runner.make_engine_windows(train, runner.compute_rul(train), 3)
    assert windows.shape == (6, 3, len(runner.SELECTED_SENSORS))
    assert np.all(np.ptp(windows[:, :, 0], axis=1) == 2)
    np.testing.assert_array_equal(targets, np.tile([1, 0], 3))


@pytest.mark.parametrize("method", ["run_ks_bonferroni", "run_adwin_approx"])
def test_drift_alert_time_is_when_the_window_is_available(method):
    runner = load_runner("benchmarks/run_experiment_e5.py")
    onset = 2 * 1440
    values = np.zeros((onset + 1440, 3))
    values[onset:] = 10
    stream = pd.DataFrame(values, columns=["temperature", "vibration", "pressure"])
    delay, false_alarms = getattr(runner, method)(stream, onset, reference_window_days=1)
    assert false_alarms == 0, "post-onset samples cannot create pre-onset false alarms"
    assert delay > 0, "the detector cannot observe the future 12-hour window at onset"
    assert delay < 12


def test_bosch_split_uses_calendar_months_and_excludes_old_faults():
    runner = load_runner("benchmarks/run_bosch_benchmark.py")
    files = [
        Path(f"OP{i}/good/M01_{month}_{year}_OP{i:02}_000.h5")
        for i, (month, year) in enumerate(
            [("Aug", 2021), ("Feb", 2019), ("Feb", 2021), ("Aug", 2019)]
        )
    ]
    old_fault = Path("OP00/bad/M01_Feb_2019_OP00_000.h5")
    new_fault = Path("OP00/bad/M01_Aug_2021_OP00_000.h5")
    train, normal_test, fault_test = runner.chronological_split(files, [old_fault, new_fault], 0.5)
    assert {p.name for p in train} == {files[1].name, files[3].name}
    assert {p.name for p in normal_test} == {files[0].name, files[2].name}
    assert fault_test == [new_fault]
    with pytest.raises(ValueError, match="date|month"):
        runner.chronological_split(files + [Path("unknown.h5")], [], 0.5)


def test_bosch_pooled_counts_survive_metric_rounding(tmp_path, monkeypatch):
    runner = load_runner("benchmarks/run_bosch_benchmark.py")
    row = {
        "machine": "M01",
        "n_train_normal": 20,
        "n_test_normal": 3,
        "n_test_anom": 3,
        "precision": 0.5,
        "recall": 0.333,
        "f1": 0.4,
        "far": 0.333,
        "roc_auc": 0.75,
        "pr_auc": 0.6,
        "tp": 1,
        "fn": 2,
        "fp": 1,
        "tn": 2,
        "threshold": 0.1,
        "fit_time_sec": 0,
        "macro_roc_auc": float("nan"),
        "macro_pr_auc": float("nan"),
    }
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "evaluate_machine", lambda machine, seed: dict(row))
    monkeypatch.setattr(sys, "argv", ["benchmark", "--machines", "M01"])
    runner.main()
    pooled = pd.read_csv(tmp_path / "bosch_results.csv").iloc[-1]
    assert pooled["recall"] == pytest.approx(0.333)
    assert pooled["far"] == pytest.approx(0.333)
    assert pooled["f1"] == pytest.approx(0.4)
    assert pooled["tp"] == 1
    assert np.isnan(pooled["roc_auc"]), "machine AUC mean is not pooled AUC"
    assert pooled["macro_roc_auc"] == pytest.approx(0.75)


def cmapss_fixture(runner, units=4, cycles=4):
    frame = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(1, units + 1), cycles),
            "cycle": np.tile(np.arange(1, cycles + 1), units),
        }
    )
    for col in runner.SELECTED_SENSORS:
        frame[col] = frame.unit * 10.0 + frame.cycle
    for col in runner.OP_SETTINGS:
        frame[col] = 0.0
    return frame


def test_cmapss_preprocessing_never_fits_on_early_stopping_engines():
    runner = load_runner("benchmarks/run_cmapss_rul_normalized.py")
    train = cmapss_fixture(runner)
    test = train.iloc[:4].copy()
    _, validation = runner.split_engine_validation(train, seed=0)
    first_fit, _, _ = runner.normalize_roles(train, test, 1, seed=0)
    altered = train.copy()
    altered.loc[validation.index, runner.SELECTED_SENSORS] += 1e6
    second_fit, _, _ = runner.normalize_roles(altered, test, 1, seed=0)
    np.testing.assert_allclose(
        first_fit[runner.SELECTED_SENSORS], second_fit[runner.SELECTED_SENSORS]
    )


def test_maintained_multiseed_main_smoke_without_training(tmp_path, monkeypatch):
    runner = load_runner("experiments/maintained/cmapss_rul_multiseed.py")
    helpers = runner.rul_helpers
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for filename in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"):
        (data_dir / filename).write_text("synthetic fixture identity\n")
    monkeypatch.setattr(helpers, "DATA_DIR", data_dir)
    train = cmapss_fixture(helpers)
    test = train[train.unit.isin([1, 2])].copy()
    monkeypatch.setattr(
        helpers, "load_train_test", lambda subset: (train, test, np.array([3.0, 4.0]))
    )
    names = [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "LSTM (MSE)",
        "Proposed (LSTM quantile)",
    ]

    def predict(fit, targets, test, epochs, window_size, validation_data):
        validation, _ = validation_data
        assert set(fit.unit).isdisjoint(validation.unit)
        np.testing.assert_allclose(fit[helpers.SELECTED_SENSORS].mean(), 0, atol=1e-7)
        return np.array([3.0, 4.0])

    monkeypatch.setattr(
        runner, "methods_for_seed", lambda seed: [(name, predict) for name in names]
    )
    monkeypatch.setattr(runner, "RES", tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner", "--subsets", "FD001", "--seeds", "1"])
    runner.main()
    result = pd.read_csv(tmp_path / "cmapss_rul_multiseed.csv")
    assert len(result) == 5
    assert (result.rmse == 0).all()
    assert (result.endpoint_accuracy_20pct == 1).all()


def test_maintained_e2_lead_uses_failure_endpoint_including_late_warnings():
    runner = load_runner("experiments/maintained/e2_cmapss_calib_split.py")
    labels = (np.arange(101) >= 70).astype(int)
    units = np.ones(101)
    for warning, expected in [(60, 40), (90, 10)]:
        risk = (np.arange(101) >= warning).astype(float)
        assert runner.median_lead_time(risk, labels, units) == expected


def test_maintained_e2_windows_omit_each_engine_warmup():
    runner = load_runner("experiments/maintained/e2_cmapss_calib_split.py")
    frame = cmapss_fixture(runner.rul_helpers, units=2)
    windows, endpoints = runner.window_inputs(frame, window=3)
    assert windows.shape == (4, 3, 14)
    np.testing.assert_array_equal(endpoints, [2, 3, 6, 7])
    assert np.all(np.ptp(windows[:, :, 0], axis=1) == 2)


def test_rul_grouped_fit_accepts_real_keras_validation_for_one_tiny_epoch():
    tf = pytest.importorskip("tensorflow")
    tf.keras.backend.clear_session()
    estimator = RULEstimator(window_size=2, lstm_units=(2,), dropout=0, epochs=1, batch_size=4)
    train = pd.DataFrame({"sensor": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]})
    validation = pd.DataFrame({"sensor": [20.0, 21.0, 22.0, 23.0]})
    estimator.fit(
        train,
        np.arange(8)[:, None],
        groups=np.repeat([1, 2], 4),
        validation_data=(validation, np.arange(4)[:, None], np.repeat(3, 4)),
    )
    assert np.isfinite(estimator.history_["val_loss"]).all()
    assert len(estimator.predict(validation)) == 3
