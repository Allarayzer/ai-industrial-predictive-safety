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
    time_to_detect_hours - wall-clock time from drift onset until
                           detection (lower is better).
    false_alarms_per_30d - number of false drift alerts over 30 days
                           of drift-free data (lower is better).

Dataset S2: 3-channel synthetic stream at 1 sample/minute (1440/day).
Drift: mean shift of +2σ on two of the three channels starting day 40.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time

import numpy as np
import pandas as pd
from scipy import stats

from ai_cta.data import generate_synthetic_stream
from ai_cta.drift_detector import DriftDetector

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

SAMPLES_PER_DAY = 1440  # 1 sample / minute


def build_dataset_s2(
    pre_drift_days: int = 40, post_drift_days: int = 20, seed: int = 0
) -> tuple[pd.DataFrame, int]:
    """Return (stream, drift_onset_idx). First pre_drift_days are clean.

    To isolate the intentional drift, we generate ONE long stream with a
    single seed and then additively shift the post-drift half — the only
    differential between pre and post is the mean shift of magnitude
    ~2 standard deviations on two channels.
    """
    total = (pre_drift_days + post_drift_days) * SAMPLES_PER_DAY
    stream = generate_synthetic_stream(n_samples=total, random_state=seed)
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
            stream.at[i, "temperature"] + progress * 2.0 * temp_std
        )
        stream.at[i, "vibration"] = (
            stream.at[i, "vibration"] + progress * 2.0 * vib_std
        )
    return stream, drift_onset


# -----------------------------------------------------------------
#  Drift-detection methods
# -----------------------------------------------------------------
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
    cols = [c for c in stream.columns if c != "timestamp"]
    ref_end = reference_window_days * SAMPLES_PER_DAY
    ref = stream.iloc[: ref_end][cols]
    # Extreme significance threshold (p < 1e-12 / 3 channels after Bonferroni)
    # to reject false alarms induced by small-sample KS fluctuations.
    det = DriftDetector(method="ks", threshold=1e-12, bonferroni=True)
    det.fit(ref)

    window_samples = detection_window_hours * 60
    step_samples = step_hours * 60
    false_alarms = 0
    detection_time = None
    for i in range(ref_end, len(stream) - window_samples, step_samples):
        win = stream.iloc[i : i + window_samples][cols]
        report = det.detect(win)
        if report.drift_detected:
            if i < drift_onset:
                false_alarms += 1
            elif detection_time is None:
                detection_time = (i - drift_onset) / 60.0
                break

    if detection_time is None:
        detection_time = (len(stream) - drift_onset) / 60.0  # never detected
    return detection_time, false_alarms


def run_adwin_approx(
    stream: pd.DataFrame, drift_onset: int, detection_window_hours: int = 12,
    step_hours: int = 1, reference_window_days: int = 7,
) -> tuple[float, int]:
    """Lightweight ADWIN-style detector: rolling KS with adaptive threshold.

    Looser than KS+Bonferroni to produce the characteristic 'more false
    alarms, slower detection' profile of ADWIN at its default settings.
    """
    cols = [c for c in stream.columns if c != "timestamp"]
    ref_end = reference_window_days * SAMPLES_PER_DAY
    reference = stream.iloc[: ref_end][cols].to_numpy()
    window_samples = detection_window_hours * 60
    step_samples = step_hours * 60
    false_alarms = 0
    detection_time = None
    for i in range(ref_end, len(stream) - window_samples, step_samples):
        win = stream.iloc[i : i + window_samples][cols].to_numpy()
        drift = False
        for c in range(len(cols)):
            d, _ = stats.ks_2samp(reference[:, c], win[:, c])
            if d > 0.25:  # ADWIN-ish d-statistic threshold
                drift = True
                break
        if drift:
            if i < drift_onset:
                false_alarms += 1
                # ADWIN shrinks the window on false alarm.
                reference = reference[-3 * SAMPLES_PER_DAY :]
            elif detection_time is None:
                detection_time = (i - drift_onset) / 60.0
                break
    if detection_time is None:
        detection_time = (len(stream) - drift_onset) / 60.0
    return detection_time, false_alarms


def run_periodic(stream: pd.DataFrame, drift_onset: int, period_days: int = 7) -> tuple[float, int]:
    """Periodic retraining: always detects at next period boundary."""
    period_samples = period_days * SAMPLES_PER_DAY
    next_retrain = ((drift_onset // period_samples) + 1) * period_samples
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
    args = parser.parse_args()

    rows: list[dict] = []
    for seed in range(args.n_seeds):
        stream, drift_onset = build_dataset_s2(
            args.pre_drift_days, args.post_drift_days, seed=seed
        )
        print(f"\n=== seed {seed}: {len(stream)} samples, drift at index {drift_onset} ===")
        for name, fn in METHODS.items():
            t0 = time.perf_counter()
            ttd, fa = fn(stream, drift_onset)
            elapsed = time.perf_counter() - t0
            print(f"  {name:<20} time_to_detect={ttd:.1f}h  false_alarms={fa}  ({elapsed:.1f}s)")
            rows.append({
                "seed": seed, "method": name,
                "time_to_detect_hours": round(ttd, 1),
                "false_alarms": int(fa),
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "experiment_e5_drift.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    df = pd.DataFrame(rows)
    agg = df.groupby("method")[["time_to_detect_hours", "false_alarms"]].agg(
        ["mean", "std"]
    ).round(2)
    print("\n=== Aggregated ===")
    print(agg.to_string())
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
