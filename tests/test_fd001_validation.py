"""Independent arithmetic and dataset contracts for the bounded FD001 runner."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def runner():
    file = Path(__file__).resolve().parents[1] / "experiments/maintained/fd001_validation.py"
    spec = importlib.util.spec_from_file_location("fd001_validation", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_error_metrics_match_independent_closed_form(runner):
    result = runner.metrics([20, 20], [7, 30])
    assert result["rmse"] == pytest.approx(np.sqrt(269 / 2))
    assert result["mae"] == 11.5
    assert result["nasa_score_sum"] == pytest.approx(2 * (np.e - 1))
    assert result["endpoint_accuracy_20pct"] == 0


@pytest.mark.parametrize("prediction", [np.array([[1], [2]]), [1], [np.nan, 2]])
def test_metric_rejects_broadcasting_and_nonfinite(runner, prediction):
    with pytest.raises(ValueError):
        runner.metrics([1, 2], prediction)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_clipping_cannot_hide_nonfinite_model_outputs(runner, value):
    with pytest.raises(ValueError, match="before clipping"):
        runner.clip_finite_predictions([0, value, 100])


def test_endpoint_windows_preserve_engine_order_and_padding(runner):
    frame = pd.DataFrame({"unit": [2, 2, 1, 1, 1], "cycle": [2, 1, 3, 1, 2]})
    for sensor in runner.helpers.SELECTED_SENSORS:
        frame[sensor] = frame.unit * 10 + frame.cycle
    windows, padded = runner.endpoint_windows(frame, 3)
    np.testing.assert_array_equal(windows[:, :, 0], [[11, 12, 13], [21, 21, 22]])
    assert padded == [2]


def test_dataset_rejects_duplicate_cycle_and_bad_label_alignment(runner):
    frame = pd.DataFrame(np.ones((3, 26)), columns=runner.helpers.COL_NAMES)
    frame["cycle"] = [1, 2, 3]
    runner.validate_frames(frame, frame, [10])
    bad = frame.copy()
    bad["cycle"] = [1, 2, 2]
    with pytest.raises(ValueError, match="cycles"):
        runner.validate_frames(bad, frame, [10])
    with pytest.raises(ValueError, match="align"):
        runner.validate_frames(frame, frame, [10, 20])


def test_paired_bootstrap_identical_predictions_has_zero_difference(runner):
    result = runner.paired_engine_bootstrap([10, 20, 30], [15, 18, 32], [15, 18, 32])
    assert result["delta_rmse"] == 0
    assert result["percentile_95"] == [0, 0]


def test_runner_never_overwrites_populated_output(runner, tmp_path, monkeypatch):
    marker = tmp_path / "existing.csv"
    marker.write_text("preserve")
    monkeypatch.setattr(sys, "argv", ["run", "--data-dir", "absent", "--data-manifest", "absent",
                                      "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert marker.read_text() == "preserve"
