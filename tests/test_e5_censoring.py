"""E5 reports observable events and right censoring on its minute time axis."""
import math
import sys

import numpy as np
import pandas as pd
import pytest

from benchmarks import run_experiment_e5 as runner


def _constant_stream(pre_days=2, post_days=1):
    return pd.DataFrame(np.zeros(((pre_days + post_days) * 1440, 3)),
                        columns=["temperature", "vibration", "pressure"])


@pytest.mark.parametrize("method", [runner.run_ks_bonferroni, runner.run_adwin_approx])
def test_no_alert_is_censored_not_an_event_at_end_of_followup(method):
    delay, false_alarms = method(_constant_stream(), 2 * 1440, reference_window_days=1)
    assert math.isinf(delay)
    assert false_alarms == 0


def test_periodic_retraining_outside_followup_is_not_observed():
    delay, _ = runner.run_periodic(_constant_stream(), 2 * 1440)
    assert math.isinf(delay), "the scheduled 120-hour event is beyond the observed 24 hours"


def test_periodic_retraining_at_onset_is_observed_with_zero_delay():
    delay, _ = runner.run_periodic(_constant_stream(pre_days=7), 7 * 1440)
    assert delay == 0


@pytest.mark.parametrize("method", [runner.run_ks_bonferroni, runner.run_adwin_approx])
def test_reference_cannot_include_post_onset_data(method):
    with pytest.raises(ValueError, match="reference|Reference"):
        method(_constant_stream(), 2 * 1440, reference_window_days=3)


def test_generated_stream_timestamps_match_minute_assumption():
    stream, onset = runner.build_dataset_s2(pre_drift_days=1, post_drift_days=1)
    assert stream.timestamp.diff().dropna().eq(pd.Timedelta(minutes=1)).all()
    assert stream.timestamp.iloc[onset] - stream.timestamp.iloc[0] == pd.Timedelta(days=1)


def test_zero_drift_multiplier_provides_unshifted_null_control():
    stream, onset = runner.build_dataset_s2(1, 1, seed=3, drift_multiplier=0)
    original = runner.generate_synthetic_stream(len(stream), random_state=3)
    np.testing.assert_array_equal(stream.drop(columns="timestamp"), original.drop(columns="timestamp"))
    assert onset == 1440


def test_output_distinguishes_detection_from_censoring(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "METHODS", {
        "observed": lambda stream, onset: (2., 0),
        "missing": lambda stream, onset: (float("inf"), 0),
    })
    monkeypatch.setattr(sys, "argv", ["runner", "--n-seeds", "1", "--pre-drift-days", "8",
                                      "--post-drift-days", "1", "--output-dir", str(tmp_path)])
    assert runner.main() == 0
    frame = pd.read_csv(tmp_path / "experiment_e5_drift.csv").set_index("method")
    assert bool(frame.loc["observed", "detected"])
    assert not bool(frame.loc["missing", "detected"])
    assert bool(frame.loc["missing", "censored"])
    assert pd.isna(frame.loc["missing", "time_to_detect_hours"])
    assert frame.loc["missing", "censor_horizon_hours"] == pytest.approx(24 - 1/60)
    summary = pd.read_csv(tmp_path / "experiment_e5_drift_summary.csv").set_index("method")
    assert summary.loc["missing", "detection_fraction"] == 0
    assert pd.isna(summary.loc["missing", "conditional_delay_mean_hours"])


def test_null_control_alert_is_an_event_but_not_a_true_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "METHODS", {"null alarm": lambda stream, onset: (2., 0)})
    monkeypatch.setattr(sys, "argv", ["runner", "--n-seeds", "1", "--pre-drift-days", "8",
                                      "--post-drift-days", "1", "--drift-multiplier", "0",
                                      "--output-dir", str(tmp_path)])
    assert runner.main() == 0
    row = pd.read_csv(tmp_path / "experiment_e5_drift.csv").iloc[0]
    assert row.event_observed and not row.detected and not row.censored
    assert pd.isna(row.time_to_detect_hours)
    summary = pd.read_csv(tmp_path / "experiment_e5_drift_summary.csv").iloc[0]
    assert summary.event_fraction == 1
    assert pd.isna(summary.detection_fraction)
