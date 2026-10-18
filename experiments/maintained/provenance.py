"""Strict local run identity and atomic checkpoints for maintained experiments."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def environment():
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "tensorflow",
        "keras",
        "xgboost",
        "ai-cta",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "numerical_environment": {
            name: os.environ.get(name)
            for name in (
                "TF_DETERMINISTIC_OPS",
                "TF_ENABLE_ONEDNN_OPTS",
                "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_run(output_dir, name, arguments, datasets, sources, seeds, engine_roles):
    """Refuse changed inputs/settings/software, or existing results without identity."""
    output_dir = Path(output_dir).resolve()
    contract = {
        "schema_version": 1,
        "runner": name,
        "arguments": arguments,
        "seeds": list(seeds),
        "engine_roles": engine_roles,
        "datasets": {key: fingerprint(path) for key, path in sorted(datasets.items())},
        "sources": {str(Path(path).resolve()): fingerprint(path) for path in sorted(sources)},
        "environment": environment(),
    }
    path = output_dir / f"{name}.provenance.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("contract") != contract:
            changed = [
                key for key in contract if previous.get("contract", {}).get(key) != contract[key]
            ]
            raise ValueError(
                f"Run provenance contract changed ({', '.join(changed)}); use a new --output-dir."
            )
    else:
        if output_dir.exists() and any(output_dir.glob(f"{name}*")):
            raise ValueError(
                "Existing results have no provenance contract; use a new --output-dir."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            path,
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "argv": sys.argv,
                "contract": contract,
            },
        )
    return contract


def source_files(repo, runner):
    """Hash runner, shared helper and all source modules that can affect outputs."""
    repo = Path(repo)
    return sorted(
        {
            Path(runner),
            Path(__file__),
            repo / "benchmarks/run_cmapss_rul_normalized.py",
            *repo.glob("src/ai_cta/*.py"),
        }
    )


def read_checkpoint(path, fields, key_columns, allowed_keys, finite_columns):
    path = Path(path)
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if list(frame.columns) != list(fields):
        raise ValueError("Checkpoint schema differs from the runner contract.")
    if frame.duplicated(list(key_columns)).any():
        raise ValueError("Duplicate keys in checkpoint; refusing ambiguous resume.")
    keys = set(frame[list(key_columns)].itertuples(index=False, name=None))
    if not keys.issubset(allowed_keys):
        raise ValueError("Checkpoint contains keys outside the requested run contract.")
    if not np.isfinite(frame[list(finite_columns)].to_numpy(dtype=float)).all():
        raise ValueError("Checkpoint contains non-finite metrics.")
    return frame.to_dict("records")


def write_checkpoint(path, fields, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def predictions_vector(predictions, truth):
    values = np.asarray(predictions, dtype=float)
    target = np.asarray(truth, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if target.ndim != 1 or values.shape != target.shape or not len(target):
        raise ValueError("Predictions and truth must be aligned, non-empty vectors.")
    if not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("Predictions and truth must be finite.")
    return values
