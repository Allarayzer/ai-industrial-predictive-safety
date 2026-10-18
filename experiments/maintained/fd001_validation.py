"""Bounded one-seed FD001 comparison with equal window context and saved predictions.

This is a diagnostic run, not a multi-seed reproduction or a tuned leaderboard.
All methods see the same fitting engines and 32-cycle sensor windows. Official
test endpoints are evaluated once after training, without parameter selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))
import run_cmapss_rul_normalized as helpers


def finite_vector(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty finite vector.")
    return values


def clip_finite_predictions(values):
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2) or not values.size or not np.isfinite(values).all():
        raise ValueError("Model outputs must be finite before clipping.")
    return np.clip(values, 0, 125)


def metrics(truth, prediction):
    truth, prediction = finite_vector(truth, "truth"), finite_vector(prediction, "prediction")
    if truth.shape != prediction.shape:
        raise ValueError("Targets and predictions must have matching shape.")
    error = prediction - truth
    score = np.expm1(np.where(error < 0, -error / 13.0, error / 10.0)).sum()
    result = {"rmse": float(np.sqrt(np.mean(error**2))),
              "mae": float(np.abs(error).mean()), "nasa_score_sum": float(score),
              "endpoint_accuracy_20pct": float((np.abs(error) <= 0.2 * truth).mean())}
    # A separate library calculation checks the two point-error formulas.
    np.testing.assert_allclose(result["rmse"] ** 2, mean_squared_error(truth, prediction))
    np.testing.assert_allclose(result["mae"], mean_absolute_error(truth, prediction))
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Metric overflow/non-finite result.")
    return result


def validate_frames(train, test, uncapped):
    for name, frame in [("train", train), ("test", test)]:
        if list(frame.columns) != helpers.COL_NAMES or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"Invalid {name} schema or nonfinite values.")
        ids = sorted(frame.unit.unique())
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError(f"{name} engine IDs must be contiguous starting at one.")
        for _, group in frame.groupby("unit"):
            if group.cycle.tolist() != list(range(1, len(group) + 1)):
                raise ValueError(f"{name} cycles must be chronological, unique, and consecutive.")
    uncapped = finite_vector(uncapped, "RUL")
    if len(uncapped) != test.unit.nunique() or np.any(uncapped < 0):
        raise ValueError("RUL labels must align with official test engine IDs.")


def endpoint_windows(test, window_size):
    windows, padded = [], []
    for unit, group in test.groupby("unit", sort=True):
        values = group.sort_values("cycle")[helpers.SELECTED_SENSORS].to_numpy(np.float32)
        if len(values) < window_size:
            padded.append(int(unit))
            values = np.concatenate([np.repeat(values[:1], window_size-len(values), axis=0), values])
        windows.append(values[-window_size:])
    return np.stack(windows), padded


def paired_engine_bootstrap(truth, proposed, baseline, seed=20261012, samples=2000):
    """Conditional fixed-fit RMSE difference; engines, not windows, are resampled."""
    truth, proposed, baseline = [finite_vector(v, "bootstrap vector")
                                 for v in (truth, proposed, baseline)]
    if not (truth.shape == proposed.shape == baseline.shape):
        raise ValueError("Paired bootstrap requires aligned engine predictions.")
    indices = np.random.default_rng(seed).integers(0, len(truth), (samples, len(truth)))
    delta = (np.sqrt(np.mean((proposed[indices]-truth[indices])**2, axis=1))
             - np.sqrt(np.mean((baseline[indices]-truth[indices])**2, axis=1)))
    return {"delta_rmse": metrics(truth, proposed)["rmse"]-metrics(truth, baseline)["rmse"],
            "percentile_95": np.quantile(delta, [0.025, 0.975]).tolist(),
            "resamples": samples, "bootstrap_seed": seed,
            "interpretation": "paired test-engine bootstrap, conditional on this single trained fit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--data-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=300)
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > 20 or not 0 < args.max_seconds <= 300:
        parser.error("Bounded validation allows 1–20 epochs and at most 300 training seconds.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use an empty output directory; existing results are never overwritten.")
    manifest = json.loads(args.data_manifest.read_text())
    hashes = {}
    for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"):
        digest = hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
        expected = [row["sha256"] for row in manifest["files"]
                    if pathlib.Path(row["path"]).name == name]
        if expected != [digest]:
            raise ValueError(f"Dataset hash is absent, ambiguous or mismatched: {name}.")
        hashes[name] = digest
    train = pd.read_csv(args.data_dir / "train_FD001.txt", sep=r"\s+", names=helpers.COL_NAMES)
    test = pd.read_csv(args.data_dir / "test_FD001.txt", sep=r"\s+", names=helpers.COL_NAMES)
    uncapped = np.atleast_1d(np.loadtxt(args.data_dir / "RUL_FD001.txt"))
    validate_frames(train, test, uncapped)
    truth = np.minimum(uncapped, helpers.RUL_MAX)
    fit, validation, test_norm = helpers.normalize_roles(train, test, 1, args.seed)
    xs, ys = helpers.make_engine_windows(fit, helpers.compute_rul(fit), 32)
    xv, yv = helpers.make_engine_windows(validation, helpers.compute_rul(validation), 32)
    xt, padded = endpoint_windows(test_norm, 32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = [pathlib.Path(__file__), REPO / "benchmarks/run_cmapss_rul_normalized.py",
               REPO / "src/ai_cta/rul_estimator.py"]
    snapshot = args.output_dir / "source_snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for source in sources:
        relative = source.resolve().relative_to(REPO)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "tensorflow"]
    config = {"status": "started", "args": {k: str(v) if isinstance(v, pathlib.Path) else v
                                               for k, v in vars(args).items()},
              "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "source_sha256": source_hashes, "dataset_sha256": hashes,
              "data_provenance": "matches local acquisition manifest; no independent download in this run",
              "python": platform.python_version(), "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in packages},
              "numerical_environment": {name: os.environ.get(name) for name in (
                  "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_ENABLE_ONEDNN_OPTS")},
              "tensorflow_op_determinism": True,
              "fit_engines": sorted(map(int, fit.unit.unique())),
              "early_stop_engines": sorted(map(int, validation.unit.unique())),
              "official_test_engines": sorted(map(int, test.unit.unique())),
              "roles_note": "train and official test IDs belong to different official partitions",
              "fit_windows": len(xs), "validation_windows": len(xv),
              "padded_test_engines": padded, "target": "min(RUL,125)", "window": [32, 14],
              "unit_of_evaluation": "one endpoint per official test engine",
              "hyperparameters": {"ridge_alpha": 10, "rf_trees": 64, "rf_depth": 16,
                                    "rf_max_features": "sqrt", "lstm_units": [64, 32],
                                    "dropout": 0.2, "learning_rate": 0.001, "batch_size": 64,
                                    "early_stopping_patience": 8},
              "limits": ["one seed, no tuning", "uniform training-window weights", "conditional bootstrap",
                         "cooperative budget checked between neural training batches, not a process hard timeout",
                         "no claim of independent industrial validation", "all methods share input window context"]}
    config_file = args.output_dir / "run.json"
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    predictions, results = {}, {}
    started = time.perf_counter()
    for name, model in [("Ridge", Ridge(alpha=10)),
                        ("RandomForest", RandomForestRegressor(n_estimators=64, max_depth=16,
                             max_features="sqrt", random_state=args.seed, n_jobs=2))]:
        before = time.perf_counter()
        model.fit(xs.reshape(len(xs), -1), ys)
        predictions[name] = clip_finite_predictions(model.predict(xt.reshape(len(xt), -1)))
        results[name] = {"training_and_prediction_sec": time.perf_counter()-before,
                         "capped_target": metrics(truth, predictions[name])}
        print(name, results[name], flush=True)

    import tensorflow as tf

    from ai_cta.rul_estimator import RULEstimator

    class Budget(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if time.perf_counter()-started > args.max_seconds:
                raise TimeoutError("Bounded training budget exceeded; incomplete run.")

    tf.config.experimental.enable_op_determinism()
    quantiles = None
    for name in ["LSTM-MSE", "LSTM-quantile"]:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        estimator = RULEstimator(window_size=32, lstm_units=(64, 32), dropout=0.2)
        model = estimator._build_model(14)
        if name == "LSTM-MSE":
            model = tf.keras.Model(model.input, tf.keras.layers.Dense(1)(model.layers[-2].output))
            model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
        before = time.perf_counter()
        history = model.fit(xs, ys, validation_data=(xv, yv), epochs=args.epochs,
                            batch_size=64, callbacks=[Budget(), tf.keras.callbacks.EarlyStopping(
                                patience=8, restore_best_weights=True)], verbose=0)
        output = model.predict(xt, batch_size=64, verbose=0)
        if name == "LSTM-quantile":
            quantiles = np.sort(clip_finite_predictions(output), axis=1)
            predictions[name] = quantiles[:, 1]
            for index, q in enumerate([10, 50, 90]):
                predictions[f"raw_q{q}"] = output[:, index]
                predictions[f"q{q}"] = quantiles[:, index]
        else:
            predictions[name] = clip_finite_predictions(output[:, 0])
        results[name] = {"training_and_prediction_sec": time.perf_counter()-before,
                         "epochs_ran": len(history.history["loss"]),
                         "best_validation_epoch": int(np.argmin(history.history["val_loss"])) + 1,
                         "history": history.history, "capped_target": metrics(truth, predictions[name])}
        print(name, results[name]["capped_target"], "sec", results[name]["training_and_prediction_sec"], flush=True)
    results["interval"] = {"nominal_coverage": 0.8,
                           "empirical_capped_coverage": float(((quantiles[:, 0] <= truth) & (truth <= quantiles[:, 2])).mean()),
                           "empirical_uncapped_coverage": float(((quantiles[:, 0] <= uncapped) & (uncapped <= quantiles[:, 2])).mean()),
                           "mean_width": float((quantiles[:, 2]-quantiles[:, 0]).mean()),
                           "raw_crossing_engine_count": int((np.diff(output, axis=1) < 0).any(axis=1).sum()),
                           "policy": "clip [0,125] then sort quantiles; no coverage guarantee"}
    for name in ["Ridge", "RandomForest", "LSTM-MSE", "LSTM-quantile"]:
        results[name]["uncapped_target_sensitivity"] = metrics(uncapped, predictions[name])
    results["paired_bootstrap"] = {name: paired_engine_bootstrap(truth, predictions["LSTM-quantile"], predictions[name])
                                   for name in ["Ridge", "RandomForest", "LSTM-MSE"]}
    pd.DataFrame({"engine_id": sorted(test.unit.unique()), "rul_uncapped": uncapped,
                  "rul_capped": truth, **predictions}).to_csv(
                      args.output_dir / "predictions.csv", index=False, float_format="%.17g")
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    config.update(status="complete", total_training_and_prediction_sec=time.perf_counter()-started)
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    print("Complete", config["total_training_and_prediction_sec"], flush=True)


if __name__ == "__main__":
    main()
