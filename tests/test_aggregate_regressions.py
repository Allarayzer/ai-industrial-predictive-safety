"""Preserve normalized benchmark results when rebuilding combined tables."""
import csv
import sys

import pytest

from benchmarks import run_cmapss_rul_normalized as runner

FIELDS = ["subset", "method", "rmse", "mae", "score", "alpha_lambda",
          "fit_predict_time_sec", "n_test_engines", "n_op_clusters"]


def _row(subset="FD001", method="Linear Regression"):
    return dict(zip(FIELDS, [subset, method, "12.00", "8.50", "100.0", "0.500",
                             "1.0", "100", "1" if subset in ("FD001", "FD003") else "6"], strict=True))


def _write(path, rows, fields=FIELDS):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_single_subset_run_preserves_other_subsets_in_aggregate(tmp_path, monkeypatch):
    _write(tmp_path / "cmapss_rul_normalized_fd002.csv", [_row("FD002")])
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "evaluate_subset", lambda subset, args: [_row(subset)])
    monkeypatch.setattr(sys, "argv", ["runner", "--subsets", "FD001"])
    assert runner.main() == 0
    rows = _read(tmp_path / "cmapss_rul_normalized_all_subsets.csv")
    assert [row["subset"] for row in rows] == ["FD001", "FD002"]


def test_aggregate_only_uses_selected_output_without_model_evaluation(tmp_path, monkeypatch):
    _write(tmp_path / "cmapss_rul_normalized_fd002.csv", [_row("FD002")])

    def forbidden(*args, **kwargs):
        pytest.fail("aggregate-only must not load data or evaluate models")

    monkeypatch.setattr(runner, "evaluate_subset", forbidden)
    monkeypatch.setattr(runner, "load_train_test", forbidden)
    monkeypatch.setattr(sys, "argv", ["runner", "--aggregate-only", "--output-dir", str(tmp_path)])
    assert runner.main() == 0
    assert _read(tmp_path / "cmapss_rul_normalized_all_subsets.csv") == [_row("FD002")]


def test_cli_routes_data_and_output_paths(tmp_path, monkeypatch):
    data_dir, output_dir = tmp_path / "data", tmp_path / "output"

    def evaluate(subset, args):
        assert args.data_dir == data_dir
        return [_row(subset)]

    monkeypatch.setattr(runner, "evaluate_subset", evaluate)
    monkeypatch.setattr(sys, "argv", ["runner", "--subsets", "FD001", "--data-dir", str(data_dir),
                                      "--output-dir", str(output_dir)])
    assert runner.main() == 0
    assert _read(output_dir / "cmapss_rul_normalized_fd001.csv") == [_row()]


def test_loader_reads_explicit_data_directory(tmp_path):
    lines = [" ".join(map(str, [unit, 1] + [0] * 24)) for unit in (1, 2)]
    for name in ("train_FD001.txt", "test_FD001.txt"):
        (tmp_path / name).write_text("\n".join(lines) + "\n")
    (tmp_path / "RUL_FD001.txt").write_text("3\n4\n")
    train, test, targets = runner.load_train_test("FD001", tmp_path)
    assert train.unit.tolist() == test.unit.tolist() == [1, 2]
    assert targets.tolist() == [3.0, 4.0]


def test_rebuild_sorts_subsets_and_preserves_written_values(tmp_path):
    _write(tmp_path / "cmapss_rul_normalized_fd002.csv", [_row("FD002")])
    _write(tmp_path / "cmapss_rul_normalized_fd001.csv", [_row()])
    output = runner.rebuild_aggregate(tmp_path)
    assert _read(output) == [_row(), _row("FD002")]
    assert _read(output)[0]["rmse"] == "12.00"


@pytest.mark.parametrize("defect", [
    "empty", "header_only", "schema", "duplicate", "wrong_subset", "nonfinite", "extra_cell",
])
def test_invalid_subset_file_leaves_existing_aggregate_intact(tmp_path, defect):
    aggregate = tmp_path / "cmapss_rul_normalized_all_subsets.csv"
    aggregate.write_text("previous aggregate\n")
    source = tmp_path / "cmapss_rul_normalized_fd001.csv"
    row = _row()
    if defect == "empty":
        source.write_text("")
    elif defect == "header_only":
        _write(source, [])
    elif defect == "schema":
        _write(source, [{k: v for k, v in row.items() if k != "method"}], FIELDS[:1] + FIELDS[2:])
    elif defect == "duplicate":
        _write(source, [row, row])
    elif defect == "wrong_subset":
        _write(source, [_row("FD002")])
    elif defect == "nonfinite":
        _write(source, [{**row, "rmse": "nan"}])
    else:
        _write(source, [row])
        with source.open("a") as handle:
            handle.write(",".join(row.values()) + ",unexpected\n")
    with pytest.raises(ValueError):
        runner.rebuild_aggregate(tmp_path)
    assert aggregate.read_text() == "previous aggregate\n"


def test_no_subset_files_does_not_truncate_existing_aggregate(tmp_path):
    aggregate = tmp_path / "cmapss_rul_normalized_all_subsets.csv"
    aggregate.write_text("previous aggregate\n")
    assert runner.rebuild_aggregate(tmp_path) is None
    assert aggregate.read_text() == "previous aggregate\n"


def test_failed_replacement_keeps_aggregate_and_cleans_temporary_file(tmp_path, monkeypatch):
    _write(tmp_path / "cmapss_rul_normalized_fd001.csv", [_row()])
    aggregate = tmp_path / "cmapss_rul_normalized_all_subsets.csv"
    aggregate.write_text("previous aggregate\n")

    def fail_replace(*args, **kwargs):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(runner.pathlib.Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        runner.rebuild_aggregate(tmp_path)
    assert aggregate.read_text() == "previous aggregate\n"
    assert not list(tmp_path.glob("*.tmp"))
