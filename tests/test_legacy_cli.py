"""Legacy entry points must fail safely without numerical/API dependencies."""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPTS = (
    "run_experiment_e1",
    "run_experiment_e2_cmapss",
    "run_experiment_e4",
    "run_cwru_benchmark",
    "run_cwru_robustness",
    "run_experiment_e6",
)


def cli(name, *args, cwd=ROOT, module=False):
    target = ["-m", f"benchmarks.{name}"] if module else [str(ROOT / "benchmarks" / f"{name}.py")]
    env = {**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"}
    # -S disables site initialization: argparse/guard must precede all optional
    # dependencies. A regression importing numpy/ai_cta first fails this check.
    return subprocess.run(
        [sys.executable, "-S", *target, *args], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=10, check=False,
    )


@pytest.mark.parametrize("name", LEGACY_SCRIPTS)
@pytest.mark.parametrize("module", [False, True], ids=["script", "module"])
def test_help_and_default_do_not_load_dependencies_or_run(name, module):
    help_result = cli(name, "--help", module=module)
    assert help_result.returncode == 0, help_result.stderr
    assert "--allow-legacy" in help_result.stdout
    assert "--output-dir" in help_result.stdout
    default_result = cli(name, module=module)
    assert default_result.returncode == 2
    assert "legacy benchmark disabled by default" in default_result.stderr
    assert "Traceback" not in default_result.stderr


@pytest.mark.parametrize("name", LEGACY_SCRIPTS)
def test_acknowledgement_and_output_are_both_required(name, tmp_path):
    output = tmp_path / "should-not-exist"
    no_ack = cli(name, "--output-dir", str(output), cwd=tmp_path)
    assert no_ack.returncode == 2
    assert "legacy benchmark disabled by default" in no_ack.stderr
    assert not output.exists()
    no_output = cli(name, "--allow-legacy", cwd=tmp_path)
    assert no_output.returncode == 2
    assert "--output-dir is required" in no_output.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", LEGACY_SCRIPTS)
@pytest.mark.parametrize("occupied", [False, True], ids=["empty-dir", "saved-result"])
def test_existing_output_rejected_before_dependencies(name, occupied, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    if occupied:
        (output / "saved.csv").write_text("immutable reference\n")
    result = cli(name, "--allow-legacy", "--output-dir", str(output), cwd=tmp_path)
    assert result.returncode == 2
    assert "refusing an existing path" in result.stderr
    assert "Traceback" not in result.stderr
    if occupied:
        assert (output / "saved.csv").read_text() == "immutable reference\n"
        assert len(list(output.iterdir())) == 1
    else:
        assert list(output.iterdir()) == []


def load_guard():
    spec = importlib.util.spec_from_file_location("legacy_guard", ROOT / "benchmarks" / "_legacy_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acknowledged_guard_reserves_only_new_empty_output(tmp_path, monkeypatch, capsys):
    output = tmp_path / "new-output"
    monkeypatch.setattr(sys, "argv", ["runner", "--allow-legacy", "--output-dir", str(output)])
    args = load_guard().parse_legacy_args(argparse.ArgumentParser())
    assert args.output_dir == output
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert "known evaluation limitations" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        load_guard().parse_legacy_args(argparse.ArgumentParser())
    assert error.value.code == 2


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_guard_refuses_files_and_symlinks(tmp_path, monkeypatch, kind):
    output = tmp_path / "existing"
    if kind == "file":
        output.write_text("reference")
    else:
        target = tmp_path / "target"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(sys, "argv", ["runner", "--allow-legacy", "--output-dir", str(output)])
    with pytest.raises(SystemExit) as error:
        load_guard().parse_legacy_args(argparse.ArgumentParser())
    assert error.value.code == 2
    if kind == "file":
        assert output.read_text() == "reference"
    else:
        assert output.is_symlink()
        assert list(output.iterdir()) == []
