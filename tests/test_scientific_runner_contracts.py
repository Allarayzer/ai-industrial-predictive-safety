"""Runner provenance/statistical contracts; deterministic fixtures, no training."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    path = ROOT / "experiments" / "maintained" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_multiseed(tmp_path, monkeypatch):
    runner = load_runner("cmapss_rul_multiseed")
    helpers = runner.rul_helpers
    frame = pd.DataFrame(
        {"unit": np.repeat(np.arange(1, 5), 4), "cycle": np.tile(np.arange(1, 5), 4)}
    )
    for name in helpers.SELECTED_SENSORS:
        frame[name] = frame.unit * 10.0 + frame.cycle
    for name in helpers.OP_SETTINGS:
        frame[name] = 0.0
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"):
        (data / name).write_text("fixture bytes\n")
    monkeypatch.setattr(helpers, "DATA_DIR", data)
    monkeypatch.setattr(
        helpers, "load_train_test", lambda subset: (frame, frame.iloc[:4], np.array([3.0]))
    )
    names = [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "LSTM (MSE)",
        "Proposed (LSTM quantile)",
    ]
    calls = []

    def predict(*args, **kwargs):
        calls.append(1)
        return np.array([3.0])

    monkeypatch.setattr(
        runner, "methods_for_seed", lambda seed: [(name, predict) for name in names]
    )
    monkeypatch.setattr(runner, "RES", tmp_path / "output")

    def run(epochs=1):
        monkeypatch.setattr(
            sys, "argv", ["runner", "--subsets", "FD001", "--seeds", "2", "--epochs", str(epochs)]
        )
        runner.main()

    return SimpleNamespace(runner=runner, run=run, calls=calls, data=data)


def test_resume_rejects_changed_config_before_reusing_rows(fake_multiseed):
    fake_multiseed.run()
    assert len(fake_multiseed.calls) == 10
    with pytest.raises(ValueError, match="provenance|contract|config"):
        fake_multiseed.run(epochs=2)
    assert len(fake_multiseed.calls) == 10


def test_resume_rejects_changed_dataset_bytes(fake_multiseed):
    fake_multiseed.run()
    (fake_multiseed.data / "train_FD001.txt").write_text("changed bytes\n")
    with pytest.raises(ValueError, match="provenance|contract|dataset"):
        fake_multiseed.run()


def test_resume_same_contract_skips_completed_rows_and_records_identity(fake_multiseed):
    fake_multiseed.run()
    fake_multiseed.run()
    assert len(fake_multiseed.calls) == 10
    manifest = fake_multiseed.runner.RES / "cmapss_rul_multiseed.provenance.json"
    saved = json.loads(manifest.read_text())
    contract = saved["contract"]
    assert contract["seeds"] == [0, 1]
    assert contract["arguments"]["epochs"] == 1
    assert contract["environment"]["python"]
    assert len(contract["datasets"]) == 3
    assert all(len(item["sha256"]) == 64 for item in contract["datasets"].values())
    assert set(contract["engine_roles"]["FD001/0"]["fit"]).isdisjoint(
        contract["engine_roles"]["FD001/0"]["validation"]
    )


def test_pairing_joins_seed_identity_instead_of_sorted_positions():
    runner = load_runner("cmapss_rul_multiseed")
    frame = pd.DataFrame(
        {"method": ["A", "A", "B", "B"], "seed": [0, 1, 1, 2], "rmse": [1.0, 5.0, 7.0, 20.0]}
    )
    result = runner.paired_comparison(frame, "A", "B")
    assert result["seeds"] == [1]
    assert result["n_pairs"] == 1
    assert result["mean_difference"] == -2
    assert result["p_value"] is None
    assert result["status"] == "insufficient_pairs"


def test_wilcoxon_all_ties_is_explicit_one_not_nan():
    runner = load_runner("cmapss_rul_multiseed")
    frame = pd.DataFrame(
        {"method": ["A", "A", "B", "B"], "seed": [0, 1, 0, 1], "rmse": [3.0, 4.0, 3.0, 4.0]}
    )
    result = runner.paired_comparison(frame, "A", "B")
    assert result["p_value"] == 1.0
    assert result["status"] == "all_ties"


def test_holm_family_contains_all_six_comparisons(monkeypatch):
    runner = load_runner("cmapss_rul_multiseed")
    names = [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "LSTM (MSE)",
        "Proposed (LSTM quantile)",
    ]
    frame = pd.DataFrame(
        [
            {"method": name, "seed": seed, "rmse": float(index + seed)}
            for index, name in enumerate(names)
            for seed in range(2)
        ]
    )
    monkeypatch.setattr(runner, "wilcoxon", lambda *a, **kw: SimpleNamespace(pvalue=0.01))
    comparisons = runner.comparison_table(frame)
    assert len(comparisons) == 6
    assert all(row["holm_p"] == pytest.approx(0.06) for row in comparisons)


def test_duplicate_seed_method_pairs_rejected():
    runner = load_runner("cmapss_rul_multiseed")
    frame = pd.DataFrame({"method": ["A", "A", "B"], "seed": [0, 0, 0], "rmse": [3.0, 4.0, 3.0]})
    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        runner.paired_comparison(frame, "A", "B")


def test_e2_reports_warning_denominator_and_misses():
    runner = load_runner("e2_cmapss_calib_split")
    stack = np.array([[0.0, 0.0, 0.0], [0.6, 0.6, 0.6], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = runner.evaluate(
        "fixture",
        stack,
        np.array([1.0, 0.0, 0.0]),
        np.ones(4),
        np.array([1, 1, 2, 2]),
        np.array([1, 2, 1, 2]),
    )
    assert result["n_engines"] == 2
    assert result["n_warned_engines"] == 1
    assert result["n_missed_engines"] == 1
    assert result["warning_coverage"] == 0.5


def test_e2_optimizer_failure_does_not_claim_calibrated_output(monkeypatch):
    runner = load_runner("e2_cmapss_calib_split")

    class FailedAggregator:
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        last_optimization_warning_ = "failed test optimization"

        def calibrate_weights(self, *args):
            return self.w

    monkeypatch.setattr(runner, "RiskAggregator", FailedAggregator)
    with pytest.raises(RuntimeError, match="optimization|calibration"):
        runner.fit_fusion_weights(np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]), np.array([0, 1]))


def test_e2_if_seed_is_forwarded_without_training(monkeypatch):
    runner = load_runner("e2_cmapss_calib_split")
    captured = []

    class FakeIF:
        def __init__(self, **kwargs):
            captured.append(kwargs["random_state"])

        def fit(self, X):
            return self

        def score_samples(self, X):
            return np.linspace(0.1, 0.2, len(X))

    monkeypatch.setattr(runner, "IsolationForest", FakeIF)
    frame = pd.DataFrame({"unit": np.repeat([1, 2], 18), "cycle": np.tile(np.arange(18), 2)})
    for col in runner.SELECTED_SENSORS:
        frame[col] = np.arange(36, dtype=float)
    runner.fit_r_anom(frame, seed=7)
    assert captured == [7]


@pytest.mark.parametrize("changed", ["source", "environment", "engine_roles"])
def test_resume_locks_source_environment_and_roles(tmp_path, monkeypatch, changed):
    provenance = load_runner("provenance")
    source, dataset = tmp_path / "runner.py", tmp_path / "data.txt"
    source.write_text("version 1")
    dataset.write_text("dataset")
    environment = {"python": "fixture", "packages": {"numpy": "fixture"}}
    monkeypatch.setattr(provenance, "environment", lambda: environment)
    roles = {"0": {"fit": [1, 2], "validation": [3]}}
    kwargs = {
        "output_dir": tmp_path / "output",
        "name": "fixture",
        "arguments": {"epochs": 1},
        "datasets": {"train": dataset},
        "sources": [source],
        "seeds": [0],
        "engine_roles": roles,
    }
    provenance.prepare_run(**kwargs)
    if changed == "source":
        source.write_text("version 2")
    elif changed == "environment":
        environment["packages"]["numpy"] = "changed"
    else:
        roles["0"] = {"fit": [1, 3], "validation": [2]}
    with pytest.raises(ValueError, match="contract changed"):
        provenance.prepare_run(**kwargs)


def test_resume_rejects_unidentified_or_duplicate_checkpoint(tmp_path):
    provenance = load_runner("provenance")
    output = tmp_path / "fixture.csv"
    output.write_text("seed,metric\n0,1.0\n0,2.0\n")
    with pytest.raises(ValueError, match="no provenance"):
        provenance.prepare_run(tmp_path, "fixture", {}, {}, [], [0], {})
    with pytest.raises(ValueError, match="Duplicate"):
        provenance.read_checkpoint(output, ["seed", "metric"], ["seed"], {(0,)}, ["metric"])


def test_e2_main_seed_provenance_and_resume_without_training(tmp_path, monkeypatch):
    runner = load_runner("e2_cmapss_calib_split")
    data = tmp_path / "data"
    data.mkdir()
    (data / "train_FD001.txt").write_text("synthetic fixture identity\n")
    output = tmp_path / "output"
    frame = pd.DataFrame(
        {"unit": np.repeat(np.arange(1, 9), 64), "cycle": np.tile(np.arange(1, 65), 8)}
    )
    monkeypatch.setattr(
        runner,
        "load_split",
        lambda seed: tuple(
            frame[frame.unit.isin(ids)].reset_index(drop=True)
            for ids in ([1, 2, 3, 4], [5, 6], [7, 8])
        ),
    )
    fitted = []

    def fake_fit(train, *args, seed):
        fitted.append(seed)
        return lambda df: np.where(runner.compute_labels(runner.compute_rul(df)), 0.9, 0.1)

    monkeypatch.setattr(runner, "fit_r_anom", fake_fit)
    monkeypatch.setattr(runner, "fit_r_rul", fake_fit)
    monkeypatch.setattr(runner, "fit_r_nn", fake_fit)
    monkeypatch.setattr(runner, "fit_fusion_weights", lambda stack, labels: np.ones(3) / 3)
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--n-seeds", "2", "--data-dir", str(data), "--output-dir", str(output)],
    )
    runner.main()
    runner.main()
    assert fitted == [0, 0, 0, 1, 1, 1]
    rows = pd.read_csv(output / "experiment_e2_cmapss_calib_split.csv")
    assert len(rows) == 10
    assert (rows.n_engines == 2).all()
    assert (rows.warning_coverage == 1.0).all()
    settings = json.loads(
        (output / "experiment_e2_cmapss_calib_split.seed1.settings.json").read_text()
    )
    assert set(settings["component_seeds"].values()) == {1}
    assert settings["optimizer_status"] == "success"
    (data / "train_FD001.txt").write_text("changed identity")
    with pytest.raises(ValueError, match="contract changed"):
        runner.main()


@pytest.mark.parametrize(
    "stack", [np.array([[np.nan, 0.2, 0.3]]), np.array([[2.0, 0.2, 0.3]]), np.empty((0, 3))]
)
def test_e2_rejects_invalid_available_scores(stack):
    runner = load_runner("e2_cmapss_calib_split")
    with pytest.raises(ValueError):
        runner.evaluate("fixture", stack, np.ones(3) / 3, np.zeros(len(stack)), np.ones(len(stack)))
