"""Causal E5 event times and censoring, using only tiny deterministic fixtures.

Set E5_RUNNER_PATH to the immutable October runner for the red reproduction.
No model training or scientific S2 benchmark is run by this module.
"""
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def runner():
    path = Path(os.environ.get(
        "E5_RUNNER_PATH", str(Path(__file__).resolve().parents[1]
                              / "benchmarks/run_experiment_e5_causal.py")))
    spec = importlib.util.spec_from_file_location("e5_correctness_target", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stream(n):
    return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
                         "temperature": np.arange(n, dtype=float)})


def force_alerts(runner, monkeypatch, predicate):
    class Detector:
        def __init__(self, **kwargs):
            pass

        def fit(self, reference):
            return self

        def detect(self, window):
            return SimpleNamespace(drift_detected=predicate(window.temperature.to_numpy()))

    monkeypatch.setattr(runner, "DriftDetector", Detector)
    monkeypatch.setattr(runner.stats, "ks_2samp", lambda ref, window:
                        (1.0, 0.0) if predicate(window) else (0.0, 1.0))


def delay_and_false_alarms(result):
    # Both interfaces express the scientific quantities tested in the red run.
    if hasattr(result, "event_delay_hours"):
        delay = result.event_delay_hours
        return float("inf") if delay is None else delay, result.false_alarms
    return result


@pytest.mark.parametrize("name", ["run_ks_bonferroni", "run_adwin_approx"])
def test_cross_onset_window_alert_is_available_at_last_sample(runner, monkeypatch, name):
    force_alerts(runner, monkeypatch, lambda values: np.any(values >= 1500))
    result = getattr(runner, name)(stream(1680), 1500, reference_window_days=1,
                                  detection_window_hours=2, step_hours=1)
    delay, false_alarms = delay_and_false_alarms(result)
    assert delay == pytest.approx(59 / 60), "window [1440,1559] is observable at 1559"
    assert false_alarms == 0, "a window ending after onset is not a pre-onset alarm"


@pytest.mark.parametrize("name", ["run_ks_bonferroni", "run_adwin_approx"])
def test_pre_onset_alert_does_not_hide_first_observed_post_onset_alert(runner, monkeypatch, name):
    force_alerts(runner, monkeypatch, lambda values: int(values[-1]) in {1499, 1619})
    result = getattr(runner, name)(stream(1720), 1600, reference_window_days=1,
                                  detection_window_hours=1, step_hours=1)
    delay, false_alarms = delay_and_false_alarms(result)
    assert delay == pytest.approx(19 / 60)
    assert false_alarms == 1


@pytest.mark.parametrize("name", ["run_ks_bonferroni", "run_adwin_approx"])
def test_last_complete_window_is_evaluated(runner, monkeypatch, name):
    force_alerts(runner, monkeypatch, lambda values: int(values[-1]) == 1559)
    result = getattr(runner, name)(stream(1560), 1500, reference_window_days=1,
                                  detection_window_hours=1, step_hours=1)
    delay, _ = delay_and_false_alarms(result)
    assert delay == pytest.approx(59 / 60)


@pytest.mark.parametrize("name", ["run_ks_bonferroni", "run_adwin_approx"])
def test_no_alert_is_censored_not_an_event_at_stream_end(runner, monkeypatch, name):
    force_alerts(runner, monkeypatch, lambda values: False)
    result = getattr(runner, name)(stream(1680), 1500, reference_window_days=1,
                                  detection_window_hours=1, step_hours=1)
    delay, false_alarms = delay_and_false_alarms(result)
    assert math.isinf(delay)
    assert false_alarms == 0


def test_periodic_event_outside_observation_is_censored(runner):
    delay, _ = delay_and_false_alarms(runner.run_periodic(stream(2880), 1440, period_days=7))
    assert math.isinf(delay)


def test_periodic_boundary_at_onset_has_zero_delay(runner):
    delay, _ = delay_and_false_alarms(runner.run_periodic(stream(1500), 1440, period_days=1))
    assert delay == 0


@pytest.mark.parametrize("name", ["run_ks_bonferroni", "run_adwin_approx"])
def test_reference_cannot_use_post_onset_observations(runner, name):
    with pytest.raises(ValueError, match="reference|Reference"):
        getattr(runner, name)(stream(3000), 1500, reference_window_days=2)


@pytest.mark.parametrize("name", ["run_ks_bonferroni", "run_adwin_approx", "run_periodic"])
def test_nonminute_timestamps_are_rejected(runner, name):
    frame = stream(3000)
    frame.timestamp = pd.date_range("2026-01-01", periods=len(frame), freq="s")
    with pytest.raises(ValueError, match="minute"):
        kwargs = {} if name == "run_periodic" else {"reference_window_days": 1}
        getattr(runner, name)(frame, 2000, **kwargs)


def test_generator_sets_minute_timestamps_without_changing_fixture_signals(runner, monkeypatch):
    def fake_generator(n_samples, random_state):
        frame = stream(n_samples)
        frame.timestamp = pd.date_range("2026-01-01", periods=n_samples, freq="s")
        frame["vibration"] = frame.temperature
        return frame

    monkeypatch.setattr(runner, "generate_synthetic_stream", fake_generator)
    frame, onset = runner.build_dataset_s2(1, 1, seed=0)
    assert frame.timestamp.diff().dropna().eq(pd.Timedelta(minutes=1)).all()
    assert frame.timestamp.iloc[onset] - frame.timestamp.iloc[0] == pd.Timedelta(days=1)
    np.testing.assert_array_equal(frame.temperature.iloc[:onset], np.arange(onset))
    expected = np.arange(len(frame), dtype=float)
    expected[onset:] += np.arange(len(frame) - onset) / 2880 * 2 * pd.Series(np.arange(onset)).std()
    np.testing.assert_allclose(frame.temperature, expected)


@pytest.mark.parametrize("name", ["run_ks_bonferroni", "run_adwin_approx", "run_periodic"])
@pytest.mark.parametrize("defect", ["nat", "all_nat", "duplicate", "reverse", "gap", "invalid"])
def test_incomplete_or_irregular_timestamps_are_rejected(runner, name, defect):
    frame = stream(1600)
    if defect == "nat":
        frame.loc[1500, "timestamp"] = pd.NaT
    elif defect == "all_nat":
        frame["timestamp"] = pd.NaT
    elif defect == "duplicate":
        frame.loc[1500, "timestamp"] = frame.timestamp.iloc[1499]
    elif defect == "reverse":
        frame["timestamp"] = frame.timestamp.iloc[::-1].to_numpy()
    elif defect == "gap":
        frame.loc[1500:, "timestamp"] += pd.Timedelta(minutes=1)
    else:
        frame["timestamp"] = frame.timestamp.astype(str)
        frame.loc[1500, "timestamp"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="minute"):
        kwargs = {} if name == "run_periodic" else {"reference_window_days": 1}
        getattr(runner, name)(frame, 1500, **kwargs)


@pytest.mark.parametrize("name", ["run_ks_bonferroni", "run_adwin_approx"])
def test_trace_preserves_first_pre_alert_and_post_event(runner, monkeypatch, name):
    force_alerts(runner, monkeypatch, lambda values: int(values[-1]) in {1499, 1619})
    result = getattr(runner, name)(stream(1720), 1600, reference_window_days=1,
                                  detection_window_hours=1)
    fields = result.fields
    assert fields["first_alert_index"] == 1499
    assert fields["event_index"] == fields["event_window_end_index"] == 1619
    assert fields["event_window_start_index"] == 1560
    assert fields["event_timestamp"] == "2026-01-02T02:59:00"
    assert fields["event_observed"] and fields["detected"] and not fields["censored"]
    assert fields["pre_onset_alert_opportunities"] == 2
    assert fields["false_alarms_per_opportunity"] == 0.5
    assert fields["pre_onset_decision_span_hours"] == 1
    assert [r["decision_index"] for r in result.trace] == [1499, 1559, 1619]
    assert all(r["window_end_index"] == r["decision_index"] for r in result.trace)
    assert all(r["window_start_index"] >= r["reference_end_exclusive"] for r in result.trace)


@pytest.mark.parametrize("name", ["run_ks_bonferroni", "run_adwin_approx"])
def test_censor_at_last_decision_distinguishes_unchecked_tail(runner, monkeypatch, name):
    # The untested partial tail would alert, but it is not a scheduled full window.
    force_alerts(runner, monkeypatch, lambda values: np.any(values >= 1680))
    result = getattr(runner, name)(stream(1700), 1500, reference_window_days=1,
                                  detection_window_hours=1)
    fields = result.fields
    assert fields["censored"] and not fields["event_observed"]
    assert fields["event_index"] is None and fields["time_to_detect_hours"] is None
    assert fields["observation_end_index"] == 1699
    assert fields["last_decision_index"] == fields["censor_index"] == 1679
    assert fields["observed_followup_hours"] == pytest.approx(179 / 60)
    assert fields["observation_followup_hours"] == pytest.approx(199 / 60)
    assert fields["first_post_onset_decision_index"] == 1559
    assert fields["post_onset_evaluable"]


@pytest.mark.parametrize("name", ["run_ks_bonferroni", "run_adwin_approx"])
def test_insufficient_window_does_not_invent_followup(runner, monkeypatch, name):
    force_alerts(runner, monkeypatch, lambda values: True)
    result = getattr(runner, name)(stream(1499), 1440, reference_window_days=1,
                                  detection_window_hours=1)
    assert result.trace == []
    assert result.fields["censored"]
    assert not result.fields["post_onset_evaluable"]
    assert result.fields["observed_followup_hours"] is None
    assert result.fields["censor_index"] is None
    assert result.fields["pre_onset_alert_opportunities"] == 0
    assert result.fields["false_alarms_per_opportunity"] is None


def test_heuristic_reference_updates_depend_on_alert_not_known_onset(runner, monkeypatch):
    force_alerts(runner, monkeypatch, lambda values: True)
    first = runner.run_adwin_approx(stream(6060), 5900, reference_window_days=4,
                                    detection_window_hours=1)
    second = runner.run_adwin_approx(stream(6060), 5950, reference_window_days=4,
                                     detection_window_hours=1)
    assert first.trace[0]["reference_start_index"] == 0
    assert first.trace[0]["reference_start_after_alert"] == 1440
    assert first.trace[1]["reference_start_index"] == 1440
    for a, b in zip(first.trace, second.trace, strict=False):
        for key in ("decision_index", "window_start_index", "window_end_index", "alert",
                    "reference_start_index", "reference_start_after_alert"):
            assert a[key] == b[key]


def test_periodic_is_calendar_event_and_last_observed_boundary_counts(runner):
    observed = runner.run_periodic(stream(1441), 1400, period_days=1)
    censored = runner.run_periodic(stream(1440), 1400, period_days=1)
    assert observed.fields["event_kind"] == "scheduled_retraining"
    assert observed.fields["event_index"] == 1440
    assert observed.fields["event_observed"]
    assert observed.fields["detected"] is None
    assert observed.fields["time_to_detect_hours"] is None
    assert observed.fields["false_alarms"] is None
    assert observed.event_delay_hours == pytest.approx(40 / 60)
    assert censored.fields["censored"]
    assert censored.fields["censor_index"] == 1439
    assert censored.fields["last_decision_index"] == 0
    assert censored.fields["censor_basis"] == "known_calendar_observation_end"


@pytest.mark.parametrize("name", ["run_ks_bonferroni", "run_adwin_approx"])
def test_actual_ks_on_small_fixed_step_fixture(runner, name):
    frame = stream(1500)
    frame["temperature"] = 0.0
    frame.loc[1440:, "temperature"] = 1.0
    result = getattr(runner, name)(frame, 1440, reference_window_days=1,
                                  detection_window_hours=1)
    assert result.fields["event_index"] == 1499
    assert result.event_delay_hours == pytest.approx(59 / 60)


def test_main_saves_raw_trace_hashes_and_explicit_censor_fields(runner, monkeypatch, tmp_path):
    # Exercise serialization only; no synthetic generator or full benchmark runs.
    frame = stream(1700)
    monkeypatch.setattr(runner, "build_dataset_s2", lambda *args: (frame, 1500))
    force_alerts(runner, monkeypatch, lambda values: False)
    monkeypatch.setattr(runner, "METHODS", {
        "KS fixture": lambda s, o: runner.run_ks_bonferroni(
            s, o, reference_window_days=1, detection_window_hours=1),
        "Periodic fixture": lambda s, o: runner.run_periodic(s, o, period_days=7),
    })
    out = tmp_path / "new-causal-series"
    monkeypatch.setattr(sys, "argv", ["e5", "--n-seeds", "1", "--output-dir", str(out)])
    assert runner.main() == 0
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert "not established" in metadata["historical_stream_identity"]
    for name, digest in metadata["files_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert set(metadata["files_sha256"]) == {
        "s2_seed_0.csv", "decisions_seed_0.csv", "experiment_e5_causal.csv"}
    with (out / "experiment_e5_causal.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["event_observed"] == "False" and rows[0]["censored"] == "True"
    assert rows[0]["time_to_detect_hours"] == "" and rows[0]["censor_index"] == "1679"
    assert rows[1]["event_kind"] == "scheduled_retraining"
    assert rows[1]["detected"] == ""
    with (out / "decisions_seed_0.csv").open() as handle:
        trace = list(csv.DictReader(handle))
    assert trace[0]["decision_index"] == trace[0]["window_end_index"] == "1499"
    assert trace[0]["window_start_index"] == "1440"
    with pytest.raises(FileExistsError):
        runner.main()


def test_main_requires_explicit_new_output_directory(runner, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["e5"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


@pytest.mark.parametrize("name", ["run_ks_bonferroni", "run_adwin_approx"])
@pytest.mark.parametrize("defect", ["numeric_strings", "complex", "bool", "nan"])
def test_detectors_reject_channels_they_would_handle_differently(runner, name, defect):
    frame = stream(1500)
    if defect == "numeric_strings":
        frame["temperature"] = frame.temperature.astype(str)
    elif defect == "complex":
        frame["temperature"] = frame.temperature.astype(complex) + 1j
    elif defect == "bool":
        frame["temperature"] = True
    else:
        frame.loc[1499, "temperature"] = np.nan
    with pytest.raises(ValueError, match="numeric"):
        getattr(runner, name)(frame, 1440, reference_window_days=1, detection_window_hours=1)


@pytest.mark.parametrize("name", ["run_ks_bonferroni", "run_adwin_approx"])
def test_pre_onset_checks_without_post_check_do_not_create_negative_followup(runner, monkeypatch, name):
    force_alerts(runner, monkeypatch, lambda values: False)
    result = getattr(runner, name)(stream(1540), 1520, reference_window_days=1,
                                  detection_window_hours=1)
    assert result.fields["last_decision_index"] == 1499
    assert result.fields["pre_onset_alert_opportunities"] == 1
    assert not result.fields["post_onset_evaluable"]
    assert result.fields["censor_index"] is None
    assert result.fields["observed_followup_hours"] is None
    assert result.fields["observation_followup_hours"] == pytest.approx(19 / 60)


def test_failed_method_preserves_completed_method_and_raw_hash(runner, monkeypatch, tmp_path):
    frame = stream(1700)
    monkeypatch.setattr(runner, "build_dataset_s2", lambda *args: (frame, 1500))
    force_alerts(runner, monkeypatch, lambda values: False)

    def failure(*args):
        raise RuntimeError("injected fixture failure")

    monkeypatch.setattr(runner, "METHODS", {
        "completed": lambda s, o: runner.run_ks_bonferroni(
            s, o, reference_window_days=1, detection_window_hours=1),
        "failed": failure,
    })
    out = tmp_path / "failed-series"
    monkeypatch.setattr(sys, "argv", ["e5", "--n-seeds", "1", "--output-dir", str(out)])
    with pytest.raises(RuntimeError, match="injected fixture failure"):
        runner.main()
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["active_seed"] == 0 and metadata["active_method"] == "failed"
    assert metadata["error_type"] == "RuntimeError"
    assert [row["method"] for row in metadata["completed_results"]] == ["completed"]
    for name, digest in metadata["files_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    with (out / "decisions_seed_0.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 4
    assert not (out / "experiment_e5_causal.csv").exists()
