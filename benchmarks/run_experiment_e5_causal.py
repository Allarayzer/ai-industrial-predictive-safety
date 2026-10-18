"""E5: causal, scheduled drift-alert evaluation on synthetic Dataset S2.

KS + Bonferroni is compared with the historical ADWIN-style KS heuristic
(NOT the ADWIN algorithm) and a fixed periodic retraining schedule (NOT a
drift detector). Windows are available only at their inclusive final sample.
No alert is right-censored at the last evaluated detector decision, not
converted into an event at the end of the stream. The observation end is
reported separately, including any unevaluated trailing samples.

S2 uses an ABSTRACT simulation clock: one existing generator sample is
labelled one minute. Relabelling the generator's second timestamps does not
resample signals or establish physical sampling units. The historical +2-sd
mean-shift ramp over 2880 samples is retained; this is covariate drift, not
evidence of a changed conditional target distribution.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import pathlib
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from ai_cta.data import generate_synthetic_stream
from ai_cta.drift_detector import DriftDetector

SAMPLES_PER_DAY = 1440
PROTOCOL = "e5-causal-observation-v1"


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_stream(stream: pd.DataFrame, drift_onset: int) -> list[str]:
    if (isinstance(drift_onset, bool) or not isinstance(drift_onset, (int, np.integer))
            or not 0 <= drift_onset < len(stream)):
        raise ValueError("drift_onset must be an observed sample index")
    if "timestamp" in stream:
        try:
            timestamps = pd.to_datetime(stream["timestamp"], errors="raise")
            valid = (not timestamps.isna().any()
                     and timestamps.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all())
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise ValueError("Supplied timestamps must be complete, increasing, one minute apart")
    cols = [c for c in stream.columns if c != "timestamp"]
    # Both detectors must see the same real numeric channels: DriftDetector
    # selects numeric dtypes, whereas scipy would also accept numeric strings.
    real_numeric = all(pd.api.types.is_numeric_dtype(stream[c].dtype)
                       and not pd.api.types.is_complex_dtype(stream[c].dtype)
                       and not pd.api.types.is_bool_dtype(stream[c].dtype) for c in cols)
    if not cols or not real_numeric:
        raise ValueError("At least one finite real numeric signal channel is required")
    try:
        finite = np.isfinite(stream[cols].to_numpy(dtype=float)).all()
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError("All signal channels must be finite real numeric values")
    return cols


def build_dataset_s2(
    pre_drift_days: int = 40, post_drift_days: int = 20, seed: int = 0,
) -> tuple[pd.DataFrame, int]:
    """Retain historical signal generation and shift; declare the abstract clock.

    A new generation from the same seed is not proof of byte identity with an
    unsaved historical stream. Save the returned stream and its hash on each run.
    """
    pre_drift_days = _positive_integer(pre_drift_days, "pre_drift_days")
    post_drift_days = _positive_integer(post_drift_days, "post_drift_days")
    total = (pre_drift_days + post_drift_days) * SAMPLES_PER_DAY
    stream = generate_synthetic_stream(n_samples=total, random_state=seed)
    stream["timestamp"] = pd.date_range("2024-01-01", periods=total, freq="min")
    drift_onset = pre_drift_days * SAMPLES_PER_DAY
    temp_std = float(stream["temperature"].iloc[:drift_onset].std())
    vib_std = float(stream["vibration"].iloc[:drift_onset].std())
    ramp_samples = 48 * 60
    for i in range(drift_onset, total):
        progress = min(1.0, (i - drift_onset) / ramp_samples)
        stream.at[i, "temperature"] += progress * 2.0 * temp_std
        stream.at[i, "vibration"] += progress * 2.0 * vib_std
    return stream, drift_onset


@dataclass
class DetectionResult:
    """All indices are zero-based positional indices with inclusive endpoints."""

    event_delay_hours: float | None
    false_alarms: int
    fields: dict
    trace: list[dict]


def _timestamp(stream: pd.DataFrame, index: int | None) -> str | None:
    if index is None or "timestamp" not in stream:
        return None
    return pd.Timestamp(stream["timestamp"].iloc[index]).isoformat()


def _result(stream: pd.DataFrame, onset: int, trace: list[dict],
            event_kind: str) -> DetectionResult:
    detector = event_kind == "drift_alert"
    alerts = [r for r in trace if r["alert"]]
    pre = [r for r in trace if r["decision_index"] < onset]
    post = [r for r in trace if r["decision_index"] >= onset]
    events = [r for r in post if r["alert"] or not detector]
    event = events[0] if events else None
    event_index = event["decision_index"] if event else None
    observation_end = len(stream) - 1
    first = trace[0]["decision_index"] if trace else None
    last = trace[-1]["decision_index"] if trace else None
    # A detector gives no evidence about untested trailing samples. A calendar
    # schedule, in contrast, is known to have no event between its boundaries.
    censor_index = None if event else (
        (last if post else None) if detector else observation_end)
    stop_index = event_index if event else censor_index
    delay = (event_index - onset) / 60.0 if event else None
    false_alarms = sum(r["alert"] for r in pre) if detector else 0
    first_alert = alerts[0]["decision_index"] if alerts else None
    fields = {
        "event_kind": event_kind, "event_observed": event is not None,
        "detected": event is not None if detector else None,
        "censored": event is None,
        "post_onset_evaluable": bool(post) if detector else True,
        "time_to_event_hours": delay,
        "time_to_detect_hours": delay if detector else None,
        "false_alarms": false_alarms if detector else None,
        "pre_onset_alert_opportunities": len(pre) if detector else None,
        "false_alarms_per_opportunity": false_alarms / len(pre) if detector and pre else None,
        "pre_onset_decision_span_hours": (
            (pre[-1]["decision_index"] - pre[0]["decision_index"]) / 60.0
            if detector and pre else None),
        "drift_onset_index": onset, "observation_end_index": observation_end,
        "observation_followup_hours": (observation_end - onset) / 60.0,
        "first_decision_index": first, "last_decision_index": last,
        "first_post_onset_decision_index": post[0]["decision_index"] if post else None,
        "last_decision_followup_hours": (last - onset) / 60.0 if post else None,
        "first_alert_index": first_alert, "event_index": event_index,
        "event_window_start_index": event["window_start_index"] if event else None,
        "event_window_end_index": event["window_end_index"] if event else None,
        "censor_index": censor_index,
        "censor_basis": "last_evaluated_detector_decision" if detector else "known_calendar_observation_end",
        "observed_followup_hours": (stop_index - onset) / 60.0 if stop_index is not None else None,
        "clock": "abstract_one_sample_per_minute",
    }
    for name in ("drift_onset", "observation_end", "first_decision", "last_decision",
                 "first_alert", "event", "censor"):
        fields[f"{name}_timestamp"] = _timestamp(stream, fields[f"{name}_index"])
    return DetectionResult(delay, int(false_alarms), fields, trace)


def _run_detector(stream: pd.DataFrame, drift_onset: int, reference_window_days: int,
                  detection_window_hours: int, step_hours: int, heuristic: bool) -> DetectionResult:
    cols = _validate_stream(stream, drift_onset)
    ref_end = _positive_integer(reference_window_days, "reference_window_days") * SAMPLES_PER_DAY
    window_samples = _positive_integer(detection_window_hours, "detection_window_hours") * 60
    step_samples = _positive_integer(step_hours, "step_hours") * 60
    if ref_end > drift_onset:
        raise ValueError("The exclusive reference end must not exceed drift_onset")
    reference = stream.iloc[:ref_end][cols].to_numpy()
    reference_start = 0
    det = None
    if not heuristic:
        det = DriftDetector(method="ks", threshold=1e-12, bonferroni=True)
        det.fit(stream.iloc[:ref_end][cols])
    trace = []
    # First decision uses a complete window disjoint from the reference. The
    # range includes the final scheduled complete window, but no partial one.
    for end in range(ref_end + window_samples - 1, len(stream), step_samples):
        start = end - window_samples + 1
        win = stream.iloc[start:end + 1][cols]
        before_start = reference_start
        if heuristic:
            values = win.to_numpy()
            alert = any(stats.ks_2samp(reference[:, c], values[:, c])[0] > 0.25
                        for c in range(len(cols)))
            # Historical fixed-reference truncation. This is not ADWIN and
            # does not adapt using new observations. Update on the alert alone,
            # without consulting the known simulated onset or a future window.
            if alert:
                reference = reference[-3 * SAMPLES_PER_DAY:]
                reference_start = ref_end - len(reference)
        else:
            alert = bool(det.detect(win).drift_detected)
        trace.append({
            "decision_index": end, "decision_timestamp": _timestamp(stream, end),
            "window_start_index": start, "window_end_index": end,
            "reference_start_index": before_start, "reference_end_exclusive": ref_end,
            "reference_start_after_alert": reference_start,
            "alert": bool(alert), "pre_onset": end < drift_onset,
        })
        if alert and end >= drift_onset:
            break
    return _result(stream, drift_onset, trace, "drift_alert")


def run_ks_bonferroni(stream: pd.DataFrame, drift_onset: int,
                     reference_window_days: int = 7, detection_window_hours: int = 12,
                     step_hours: int = 1) -> DetectionResult:
    """Per-feature KS p < 1e-12, Bonferroni correction; fixed reference."""
    return _run_detector(stream, drift_onset, reference_window_days,
                         detection_window_hours, step_hours, heuristic=False)


def run_adwin_approx(stream: pd.DataFrame, drift_onset: int,
                    detection_window_hours: int = 12, step_hours: int = 1,
                    reference_window_days: int = 7) -> DetectionResult:
    """Historical ADWIN-style KS-D > .25 heuristic; not actual ADWIN.

    An alert truncates the original reference to its final three days.
    No claim about ADWIN's characteristic false alarms or speed is implied.
    """
    return _run_detector(stream, drift_onset, reference_window_days,
                         detection_window_hours, step_hours, heuristic=True)


def run_periodic(stream: pd.DataFrame, drift_onset: int,
                 period_days: int = 7) -> DetectionResult:
    """Calendar event at the first observed period boundary >= onset."""
    _validate_stream(stream, drift_onset)
    period = _positive_integer(period_days, "period_days") * SAMPLES_PER_DAY
    trace = []
    # The calendar includes time zero; a boundary exactly at onset has delay 0.
    for index in range(0, len(stream), period):
        trace.append({
            "decision_index": index, "decision_timestamp": _timestamp(stream, index),
            "window_start_index": None, "window_end_index": None,
            "reference_start_index": None, "reference_end_exclusive": None,
            "reference_start_after_alert": None, "alert": False,
            "pre_onset": index < drift_onset,
        })
        if index >= drift_onset:
            break
    return _result(stream, drift_onset, trace, "scheduled_retraining")


METHODS = {
    "KS + Bonferroni": run_ks_bonferroni,
    "ADWIN-style KS heuristic": run_adwin_approx,
    "Periodic (7d)": run_periodic,
}


def _write_csv(path: pathlib.Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--pre-drift-days", type=int, default=40)
    parser.add_argument("--post-drift-days", type=int, default=20)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True,
                        help="New directory; existing directories are never overwritten")
    args = parser.parse_args()
    _positive_integer(args.n_seeds, "n_seeds")
    _positive_integer(args.pre_drift_days, "pre_drift_days")
    _positive_integer(args.post_drift_days, "post_drift_days")
    if args.pre_drift_days < 7:
        parser.error("pre-drift-days must cover the seven-day reference")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "protocol": PROTOCOL, "status": "incomplete", "seeds": list(range(args.n_seeds)),
        "pre_drift_days": args.pre_drift_days, "post_drift_days": args.post_drift_days,
        "clock": "abstract sample-to-minute relabelling; no physical resampling",
        "historical_stream_identity": "not established: historical raw-stream hashes absent",
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "source_sha256": {str(path): _sha256(path) for path in (
            pathlib.Path(__file__).resolve(),
            pathlib.Path(__file__).resolve().parents[1] / "src/ai_cta/data.py",
            pathlib.Path(__file__).resolve().parents[1] / "src/ai_cta/drift_detector.py")},
        "files_sha256": {}, "completed_results": [],
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    trace_fields = ["method", "decision_index", "decision_timestamp", "window_start_index",
                    "window_end_index", "reference_start_index", "reference_end_exclusive",
                    "reference_start_after_alert", "alert", "pre_onset"]
    try:
        for seed in range(args.n_seeds):
            metadata.update(active_seed=seed, active_method=None)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            stream, onset = build_dataset_s2(args.pre_drift_days, args.post_drift_days, seed)
            raw_path = args.output_dir / f"s2_seed_{seed}.csv"
            stream.to_csv(raw_path, index=False, mode="x", float_format="%.17g")
            metadata["files_sha256"][raw_path.name] = _sha256(raw_path)
            trace_path = args.output_dir / f"decisions_seed_{seed}.csv"
            _write_csv(trace_path, [], trace_fields)
            for name, fn in METHODS.items():
                metadata["active_method"] = name
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                t0 = time.perf_counter()
                result = fn(stream, onset)
                row = {"seed": seed, "method": name, **result.fields}
                rows.append(row)
                with trace_path.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=trace_fields)
                    writer.writerows({"method": name, **record} for record in result.trace)
                metadata["files_sha256"][trace_path.name] = _sha256(trace_path)
                metadata["completed_results"].append(row)
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"seed={seed} {name}: event={result.event_delay_hours}h "
                      f"censored={result.fields['censored']} ({time.perf_counter() - t0:.1f}s)")
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__, error=str(error))
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        raise
    out = args.output_dir / "experiment_e5_causal.csv"
    _write_csv(out, rows, list(rows[0]))
    metadata["files_sha256"][out.name] = _sha256(out)
    metadata["status"] = "complete"
    metadata.update(active_seed=None, active_method=None)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    # Censored observations are not substituted with a finite delay in a mean.
    for name in METHODS:
        subset = [r for r in rows if r["method"] == name]
        events = [r["time_to_event_hours"] for r in subset if r["event_observed"]]
        print(f"{name}: observed={len(events)}/{len(subset)}, "
              f"censored={len(subset) - len(events)}, "
              f"mean_delay_given_observed={float(np.mean(events)) if events else None}")
    print(f"Saved causal series: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
