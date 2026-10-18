"""Experiment E5 (monograph § 13.9): drift detection on Dataset S2.

Compares three drift-detection methods on a synthetic stream with
injected concept drift at day 40:

    KS + Bonferroni - per-feature Kolmogorov-Smirnov test with
                      Bonferroni correction (provided by DriftDetector).
    ADWIN           - adaptive windowing (approximated via rolling KS
                      with dynamic window size; the full ADWIN algorithm
                      from Bifet & Gavalda 2007 is not shipped by the
                      reference package, this is a lightweight stand-in).
    Periodic        - retrain every 7 days regardless of detected drift.

Metrics:
    time_to_detect_hours - simulated hours from drift onset to an observed
                           drift alert; missing if right-censored.
    event_delay_hours    - also records scheduled retraining for the periodic
                           proxy, which is not an empirical drift detector.
    false_alarms_per_30d - number of false drift alerts over 30 days
                           of drift-free data (lower is better).

Dataset S2: 3-channel synthetic stream at 1 sample/minute (1440/day).
Drift: mean shift of +2σ on two of the three channels starting day 40.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import platform
import subprocess
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy import stats

from ai_cta.data import generate_synthetic_stream
from ai_cta.drift_detector import DriftDetector

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "maintained" / "results"

SAMPLES_PER_DAY = 1440  # 1 sample / minute


def build_dataset_s2(
    pre_drift_days: int = 40, post_drift_days: int = 20, seed: int = 0,
    drift_multiplier: float = 2.0,
) -> tuple[pd.DataFrame, int]:
    """Return (stream, drift_onset_idx). First pre_drift_days are clean.

    To isolate the intentional drift, we generate ONE long stream with a
    single seed and then additively shift the post-drift half — the only
    differential between pre and post is the mean shift of magnitude
    drift_multiplier standard deviations on two channels. Zero gives a null
    control. The legacy synthetic sample sequence is interpreted as abstract
    minute-indexed observations; timestamps explicitly use that cadence.
    """
    if pre_drift_days < 1 or post_drift_days < 1:
        raise ValueError("pre_drift_days and post_drift_days must be positive.")
    if not math.isfinite(drift_multiplier) or drift_multiplier < 0:
        raise ValueError("drift_multiplier must be finite and nonnegative.")
    total = (pre_drift_days + post_drift_days) * SAMPLES_PER_DAY
    stream = generate_synthetic_stream(n_samples=total, random_state=seed)
    stream["timestamp"] = pd.date_range(stream.timestamp.iloc[0], periods=total, freq="min")
    drift_onset = pre_drift_days * SAMPLES_PER_DAY
    # Apply drift only to post-drift window. Ramp linearly over 48h from
    # 0 to final shift magnitude — matches the monograph assumption that
    # concept drift is gradual, not an instantaneous jump.
    temp_std = float(stream["temperature"].iloc[:drift_onset].std())
    vib_std = float(stream["vibration"].iloc[:drift_onset].std())
    ramp_samples = 48 * 60  # 48 hours to reach full shift
    for i in range(drift_onset, total):
        progress = min(1.0, (i - drift_onset) / ramp_samples)
        stream.at[i, "temperature"] = (
            stream.at[i, "temperature"] + progress * drift_multiplier * temp_std
        )
        stream.at[i, "vibration"] = (
            stream.at[i, "vibration"] + progress * drift_multiplier * vib_std
        )
    return stream, drift_onset


# -----------------------------------------------------------------
#  Drift-detection methods
# -----------------------------------------------------------------
def _validate_observation(stream, drift_onset, reference_samples=0):
    if not 0 < drift_onset < len(stream):
        raise ValueError("drift_onset must leave nonempty pre-onset and follow-up observations.")
    if reference_samples < 0 or reference_samples > drift_onset:
        raise ValueError("The reference window must end before drift onset.")
    if "timestamp" in stream:
        stamps = pd.to_datetime(stream["timestamp"], errors="raise")
        if not stamps.diff().dropna().eq(pd.Timedelta(minutes=1)).all():
            raise ValueError("E5 requires timestamps spaced exactly one minute apart.")


def run_ks_bonferroni(
    stream: pd.DataFrame, drift_onset: int, reference_window_days: int = 7,
    detection_window_hours: int = 12, step_hours: int = 1,
) -> tuple[float, int]:
    """Run KS+Bonferroni drift detection via DriftDetector.

    Uses a 7-day reference window and a 12-hour rolling detection window
    stepped every hour. The rolling window size is chosen so the KS test
    has enough samples to give a meaningful p-value while still reacting
    in minutes once drift begins.
    """
    if min(reference_window_days, detection_window_hours, step_hours) <= 0:
        raise ValueError("Reference, detection window and step sizes must be positive.")
    ref_end = reference_window_days * SAMPLES_PER_DAY
    _validate_observation(stream, drift_onset, ref_end)
    cols = [c for c in stream.columns if c != "timestamp"]
    ref = stream.iloc[: ref_end][cols]
    # Extreme significance threshold (p < 1e-12 / 3 channels after Bonferroni)
    # to reject false alarms induced by small-sample KS fluctuations.
    det = DriftDetector(method="ks", threshold=1e-12, bonferroni=True)
    det.fit(ref)

    window_samples = detection_window_hours * 60
    step_samples = step_hours * 60
    false_alarms = 0
    detection_time = None
    # A decision is available only at the final sample of its trailing window.
    for end in range(ref_end + window_samples - 1, len(stream), step_samples):
        win = stream.iloc[end - window_samples + 1 : end + 1][cols]
        report = det.detect(win)
        if report.drift_detected:
            if end < drift_onset:
                false_alarms += 1
            elif detection_time is None:
                detection_time = (end - drift_onset) / 60.0
                break

    if detection_time is None:
        detection_time = float("inf")  # no event observed before right censoring
    return detection_time, false_alarms


def run_adwin_approx(
    stream: pd.DataFrame, drift_onset: int, detection_window_hours: int = 12,
    step_hours: int = 1, reference_window_days: int = 7,
) -> tuple[float, int]:
    """Lightweight ADWIN-style detector: rolling KS with adaptive threshold.

    Historical KS surrogate, not the ADWIN algorithm. Its results cannot
    establish a comparison against a genuine ADWIN implementation.
    """
    if min(reference_window_days, detection_window_hours, step_hours) <= 0:
        raise ValueError("Reference, detection window and step sizes must be positive.")
    ref_end = reference_window_days * SAMPLES_PER_DAY
    _validate_observation(stream, drift_onset, ref_end)
    cols = [c for c in stream.columns if c != "timestamp"]
    reference = stream.iloc[: ref_end][cols].to_numpy()
    window_samples = detection_window_hours * 60
    step_samples = step_hours * 60
    false_alarms = 0
    detection_time = None
    for end in range(ref_end + window_samples - 1, len(stream), step_samples):
        win = stream.iloc[end - window_samples + 1 : end + 1][cols].to_numpy()
        drift = False
        for c in range(len(cols)):
            d, _ = stats.ks_2samp(reference[:, c], win[:, c])
            if d > 0.25:  # ADWIN-ish d-statistic threshold
                drift = True
                break
        if drift:
            if end < drift_onset:
                false_alarms += 1
                # ADWIN shrinks the window on false alarm.
                reference = reference[-3 * SAMPLES_PER_DAY :]
            elif detection_time is None:
                detection_time = (end - drift_onset) / 60.0
                break
    if detection_time is None:
        detection_time = float("inf")
    return detection_time, false_alarms


def run_periodic(stream: pd.DataFrame, drift_onset: int, period_days: int = 7) -> tuple[float, int]:
    """Time to a scheduled retraining inside follow-up, not measured detection."""
    if period_days <= 0:
        raise ValueError("period_days must be positive.")
    _validate_observation(stream, drift_onset)
    period_samples = period_days * SAMPLES_PER_DAY
    next_retrain = ((drift_onset + period_samples - 1) // period_samples) * period_samples
    if next_retrain >= len(stream):
        return float("inf"), 0
    time_to_detect_hours = (next_retrain - drift_onset) / 60.0
    return time_to_detect_hours, 0


METHODS = {
    "KS + Bonferroni": run_ks_bonferroni,
    "ADWIN (approx)": run_adwin_approx,
    "Periodic (7d)": run_periodic,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--pre-drift-days", type=int, default=40)
    parser.add_argument("--post-drift-days", type=int, default=20)
    parser.add_argument("--drift-multiplier", type=float, default=2.0,
                        help="Final mean shift in pre-onset standard deviations; 0 is a null control.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=RESULTS_DIR)
    args = parser.parse_args()
    if args.n_seeds < 1 or args.pre_drift_days < 7 or args.post_drift_days < 1:
        parser.error("Use at least one seed, seven pre-drift reference days and one post-drift day.")
    if not math.isfinite(args.drift_multiplier) or args.drift_multiplier < 0:
        parser.error("--drift-multiplier must be finite and nonnegative.")

    rows: list[dict] = []
    stream_hashes = {}
    for seed in range(args.n_seeds):
        stream, drift_onset = build_dataset_s2(
            args.pre_drift_days, args.post_drift_days, seed=seed,
            drift_multiplier=args.drift_multiplier,
        )
        stream_hashes[str(seed)] = hashlib.sha256(stream.to_csv(
            index=False, float_format="%.17g", date_format="%Y-%m-%dT%H:%M:%S",
        ).encode()).hexdigest()
        horizon = (len(stream) - 1 - drift_onset) / 60.0
        first_decision = 7 * SAMPLES_PER_DAY + 12 * 60 - 1
        monitoring_days = max(0.0, (drift_onset - first_decision) / SAMPLES_PER_DAY)
        print(f"\n=== seed {seed}: {len(stream)} samples, drift at index {drift_onset} ===")
        for name, fn in METHODS.items():
            t0 = time.perf_counter()
            ttd, fa = fn(stream, drift_onset)
            elapsed = time.perf_counter() - t0
            periodic = fn is run_periodic
            observed = math.isfinite(ttd)
            detected = observed and not periodic and args.drift_multiplier > 0
            event_delay = ttd if observed else None
            detection_delay = ttd if detected else None
            print(f"  {name:<20} event_delay={event_delay}h  censored={not observed} "
                  f"false_alarms={fa}  ({elapsed:.1f}s)")
            rows.append({
                "seed": seed, "method": name,
                "event_kind": "scheduled_retraining" if periodic else "drift_alert",
                "drift_injected": args.drift_multiplier > 0,
                "event_observed": observed, "detected": detected,
                "censored": not observed,
                "time_to_detect_hours": detection_delay,
                "event_delay_hours": event_delay,
                "censor_horizon_hours": horizon,
                "false_alarms": None if periodic else int(fa),
                # The first available monitoring window ends after 7d + 12h.
                "pre_drift_monitoring_days": None if periodic else monitoring_days,
                "false_alarms_per_30d": (
                    round(fa * 30 / monitoring_days, 3)
                    if monitoring_days > 0 and not periodic else None
                ),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "experiment_e5_drift.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    df = pd.DataFrame(rows)
    agg = df.groupby("method").agg(
        n_runs=("seed", "count"),
        event_fraction=("event_observed", "mean"),
        detection_fraction=("detected", "mean"),
        n_censored=("censored", "sum"),
        conditional_delay_mean_hours=("time_to_detect_hours", "mean"),
        conditional_delay_std_hours=("time_to_detect_hours", "std"),
        conditional_event_delay_mean_hours=("event_delay_hours", "mean"),
        mean_false_alarms=("false_alarms", "mean"),
    )
    # Scheduled retraining and null controls are not detection opportunities.
    inapplicable = df.loc[(df.event_kind == "scheduled_retraining") | ~df.drift_injected, "method"]
    agg.loc[inapplicable.unique(), "detection_fraction"] = np.nan
    summary = args.output_dir / "experiment_e5_drift_summary.csv"
    agg.to_csv(summary)
    repo = pathlib.Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                              text=True, check=False)
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=repo, check=False)
    sources = [pathlib.Path(__file__), repo / "src/ai_cta/data.py", repo / "src/ai_cta/drift_detector.py"]
    metadata = {
        "protocol": "E5 minute-indexed, right-censored follow-up; periodic is a scheduling proxy",
        "arguments": {key: str(value) if isinstance(value, pathlib.Path) else value
                      for key, value in vars(args).items()},
        "seeds": list(range(args.n_seeds)),
        "sample_interval_seconds": 60, "reference_days": 7,
        "detection_window_hours": 12, "step_hours": 1, "period_days": 7,
        "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_worktree_dirty": dirty.returncode == 1,
        "source_sha256": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "stream_sha256": stream_hashes,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **{name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")}},
        "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (out, summary)},
    }
    (args.output_dir / "experiment_e5_drift.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\n=== Aggregated ===")
    print(agg.to_string())
    print("Delay means are conditional on observed detections; censored runs are excluded. "
          "Null-control alerts and periodic retraining are reported as events, not true detections.")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
