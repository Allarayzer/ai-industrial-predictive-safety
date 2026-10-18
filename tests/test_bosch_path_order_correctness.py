"""Synthetic/mocked Bosch accounting regressions; no real model is fitted."""
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score


@pytest.fixture
def runner():
    path = Path(os.environ.get("BOSCH_RUNNER_PATH", str(Path(__file__).parents[1] /
                                                         "benchmarks/run_bosch_path_order_corrected.py")))
    spec = importlib.util.spec_from_file_location("bosch_under_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hand_labeled_row(machine, offset):
    labels = [0, 0, 0, 1, 1, 1]
    scores = [offset + x for x in (0, .1, .2, .3, .4, .5)]
    threshold = offset + .45
    records = [{"recording_id": f"{machine}/record-{i}.h5", "machine": machine,
                "seed": 42, "role": "test", "label": y, "score": s,
                "threshold": threshold, "prediction": int(s >= threshold)}
               for i, (y, s) in enumerate(zip(labels, scores, strict=False))]
    return {"machine": machine, "n_train_normal": 14, "n_test_normal": 3,
            "n_test_anom": 3, "tp": 1, "fp": 0, "tn": 3, "fn": 2,
            "precision": 1., "recall": .333, "f1": .5, "far": 0.,
            "roc_auc": 1., "pr_auc": 1., "threshold": threshold, "fit_time_sec": 0.,
            "_prediction_records": records,
            "_split_metadata": {"ordering_policy": "lexicographic", "temporal_order_validated": False}}


@pytest.fixture
def mocked_main(runner, monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(runner, "DATA_DIR", data)
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "results").mkdir()
    historical = tmp_path / "results/bosch_seed0.csv"
    historical.write_text("historical aggregate must remain unchanged\n")
    rows = {"M01": hand_labeled_row("M01", 0), "M02": hand_labeled_row("M02", 100)}
    monkeypatch.setattr(runner, "evaluate_machine", lambda m, **kw: rows[m])
    monkeypatch.setattr(sys, "argv", ["run_bosch_benchmark.py", "--machines", "M01,M02"])
    assert runner.main() == 0
    result = next((tmp_path / "results").rglob("bosch_results.csv"))
    assert result.parent.name == "bosch-corrected"
    assert historical.read_text() == "historical aggregate must remain unchanged\n"
    with result.open() as stream:
        output = list(csv.DictReader(stream))
    return result.parent, output, rows


def test_pooled_counts_do_not_invert_rounded_recall(mocked_main):
    _, output, _ = mocked_main
    pooled = output[-1]
    # Each real confusion table has TP=1 of 3 positives. int(.333 * 3) is 0.
    assert float(pooled["recall"]) == pytest.approx(2 / 6, abs=1e-14)
    assert [int(pooled[x]) for x in ("tp", "fp", "tn", "fn")] == [2, 0, 6, 4]


def test_macro_auc_is_not_labeled_pooled_auc(mocked_main):
    _, output, rows = mocked_main
    raw = [record for row in rows.values() for record in row["_prediction_records"]]
    assert roc_auc_score([r["label"] for r in raw], [r["score"] for r in raw]) == .75
    pooled = output[-1]
    assert pooled["roc_auc"] == ""  # Separate detector scales: no pooled ranking claim.
    assert pooled["pr_auc"] == ""
    assert float(pooled["macro_roc_auc"]) == 1.0
    assert float(pooled["macro_pr_auc"]) == 1.0


def test_mocked_run_preserves_raw_records_and_split_metadata(mocked_main):
    folder, _, _ = mocked_main
    with (folder / "bosch_predictions.csv").open() as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == len({r["recording_id"] for r in records}) == 12
    assert sum(int(r["prediction"]) and int(r["label"]) for r in records) == 2
    assert {float(r["threshold"]) for r in records} == {.45, 100.45}
    manifest = json.loads((folder / "bosch_split_metadata.json").read_text())
    assert manifest["ordering_policy"] == "lexicographic"
    assert manifest["temporal_order_validated"] is False
    assert "separately fitted" in manifest["pooled_ranking_note"]


@pytest.fixture
def synthetic_machine(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    # OP path dominates the known month/year: late OP00 sorts before early OP01.
    normal = []
    for op, period in (("OP00", "Aug_2021"), ("OP01", "Feb_2019")):
        for i in range(10):
            p = tmp_path / "M01" / op / "good" / f"M01_{period}_{op}_{i:03d}.h5"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            normal.append(p)
    anomaly = tmp_path / "M01/OP00/bad/M01_Feb_2019_OP00_000.h5"
    anomaly.parent.mkdir(parents=True)
    anomaly.touch()
    ordered, faults = runner.collect_files("M01")
    assert "Aug_2021" in ordered[0].name and "Feb_2019" in ordered[-1].name
    values = {p: float(i) for i, p in enumerate(ordered + faults)}
    monkeypatch.setattr(runner, "build_feature_matrix",
                        lambda files: np.array([[values[p]] + [0.] * 26 for p in files]).reshape(-1, 27))

    class MockScaler:
        def fit(self, x): return self
        def transform(self, x): return x

    class MockForest:
        def __init__(self, **kwargs): pass
        def fit(self, x): return self
        def decision_function(self, x): return -x[:, 0]

    import sklearn.ensemble
    import sklearn.preprocessing
    monkeypatch.setattr(sklearn.ensemble, "IsolationForest", MockForest)
    monkeypatch.setattr(sklearn.preprocessing, "RobustScaler", MockScaler)
    return runner, ordered, faults, values


def test_default_path_split_is_not_claimed_chronological(synthetic_machine):
    runner, normal, _, _ = synthetic_machine
    row = runner.evaluate_machine("M01")
    assert row["ordering_policy"] == "lexicographic"
    metadata = row["_split_metadata"]
    assert metadata["temporal_order_validated"] is False
    assert metadata["train_normal_ids"] == [p.relative_to(runner.DATA_DIR).as_posix() for p in normal[:14]]
    assert metadata["test_normal_ids"] == [p.relative_to(runner.DATA_DIR).as_posix() for p in normal[14:]]


def test_record_ids_stay_aligned_with_scores(synthetic_machine):
    runner, normal, faults, values = synthetic_machine
    row = runner.evaluate_machine("M01", seed=9)
    records = row["_prediction_records"]
    assert len(records) == 21
    for record in records:
        path = runner.DATA_DIR / record["recording_id"]
        assert record["score"] == values[path]
        assert record["seed"] == 9
        assert record["prediction"] == int(record["score"] >= record["threshold"])
    test = [r for r in records if r["role"] == "test"]
    assert {r["recording_id"] for r in test} == {p.relative_to(runner.DATA_DIR).as_posix() for p in normal[14:] + faults}
    assert row["tp"] == sum(r["label"] == 1 and r["prediction"] == 1 for r in test)
    assert row["fp"] == sum(r["label"] == 0 and r["prediction"] == 1 for r in test)


def test_unreadable_recording_fails_instead_of_shifting_ids(runner, monkeypatch, tmp_path):
    bad, good = tmp_path / "bad.h5", tmp_path / "good.h5"
    def load(path):
        if path == bad:
            raise OSError("synthetic unreadable file")
        return np.ones((8, 3))
    monkeypatch.setattr(runner, "load_h5", load)
    monkeypatch.setattr(runner, "extract_features", lambda x: np.ones(27))
    with pytest.raises(ValueError, match="bad.h5"):
        runner.build_feature_matrix([bad, good])


def test_nonfinite_features_rejected(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_h5", lambda p: np.ones((8, 3)))
    monkeypatch.setattr(runner, "extract_features", lambda x: np.full(27, np.nan))
    with pytest.raises(ValueError, match="nonfinite"):
        runner.build_feature_matrix([tmp_path / "bad-feature.h5"])


def test_nonempty_output_refused_before_evaluation(runner, monkeypatch, tmp_path):
    historical = tmp_path / "bosch_results.csv"
    historical.write_text("preserved historical result\n")
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_bosch_benchmark.py", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(runner, "evaluate_machine",
                        lambda *a, **kw: pytest.fail("No fit may start for a nonempty output directory"))
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
    assert historical.read_text() == "preserved historical result\n"


def test_existing_empty_output_refused_before_evaluation(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_bosch_benchmark.py", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(runner, "evaluate_machine",
                        lambda *a, **kw: pytest.fail("Existing output must fail before fitting"))
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("machines", ["M01,M01", "M01,", " ,M01"])
def test_duplicate_or_empty_machine_names_rejected(runner, monkeypatch, tmp_path, machines):
    output = tmp_path / "new-output"
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_bosch_benchmark.py", "--machines", machines,
                                     "--output-dir", str(output)])
    monkeypatch.setattr(runner, "evaluate_machine",
                        lambda *a, **kw: pytest.fail("Invalid machine selection must fail before fitting"))
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
    assert not output.exists()
