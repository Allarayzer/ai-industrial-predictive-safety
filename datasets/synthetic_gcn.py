"""Synthetic Main Circulation Pump (ГЦН) dataset generator for monograph §15.5.

Generates multi-year telemetry for a nuclear power plant main circulation
pump (Главный Циркуляционный Насос, ГЦН) with realistic degradation
patterns matching those described in the monograph's Chapter 15 case study.

Parameters (from §15.5):
    - Duration: 24 months of operation
    - Normal regime + progressive bearing degradation (BPFO band amplitude
      grows 3-5x from nominal, simulating outer-race damage)
    - Sudden failures (step changes in winding temperature)
    - Slow tags: 1 Hz sampling (pressure, flow, temperature)
    - Vibration: 2 kHz (downsampled from the theoretical 20 kHz accelerometer
      for compactness)

Normalised nominal parameters:
    - Rotational speed: 1000 rpm (≈16.67 Hz)
    - Bearing geometry: SKF 6324 equivalent → BPFO ≈ 6.3 * f_rot ≈ 105 Hz
    - Vibration baseline: 2.0 mm/s (ISO 10816-3 zone A)
    - Winding temperature baseline: 85 °C, alarm threshold 110 °C

Output:
    Pandas DataFrame with columns:
        timestamp (datetime64)
        rpm, flow_m3h, pressure_bar, winding_temp_c, bearing_temp_c
        vibration_rms_mms, vibration_bpfo_amplitude
        label (0 normal, 1 degrading, 2 failure imminent)

Usage:
    python datasets/synthetic_gcn.py --out synthetic_gcn.parquet --months 24
    python datasets/synthetic_gcn.py --seed 42 --failures 3 --demo

The generator is deterministic given seed. Written per the specification in
monograph §15.5 and Appendix C.2.
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GCNParams:
    """Main Circulation Pump nominal parameters."""
    rpm_nominal: float = 1000.0
    bpfo_ratio: float = 6.3  # f_BPFO / f_rot for typical reactor-coolant-pump bearings
    flow_nominal_m3h: float = 22_000.0  # typical VVER primary loop
    pressure_nominal_bar: float = 157.0  # primary circuit pressure
    winding_temp_nominal_c: float = 85.0
    bearing_temp_nominal_c: float = 55.0
    vibration_rms_nominal_mms: float = 2.0

    # Noise / process dynamics
    slow_sensor_noise_std: float = 0.02  # 2% of nominal
    vibration_noise_std: float = 0.15    # mm/s RMS noise floor


def _seasonal_variation(hours: np.ndarray, amplitude: float = 1.0,
                       period_h: float = 24.0) -> np.ndarray:
    """Daily seasonal oscillation (small)."""
    return amplitude * np.sin(2 * np.pi * hours / period_h)


def _generate_slow_tags(
    n_samples: int,
    rng: np.random.Generator,
    params: GCNParams,
    degradation_onset_frac: float,
    failure_events: list[tuple[int, str]],
) -> pd.DataFrame:
    """Generate 1 Hz tags: rpm, flow, pressure, winding_temp, bearing_temp."""
    hours = np.arange(n_samples) / 3600.0

    # Nominal + small noise + seasonal
    rpm = params.rpm_nominal + rng.normal(0, 2.0, n_samples) \
          + _seasonal_variation(hours, amplitude=1.0)
    flow = params.flow_nominal_m3h * (1 + rng.normal(0, params.slow_sensor_noise_std, n_samples))
    pressure = params.pressure_nominal_bar * (1 + rng.normal(0, params.slow_sensor_noise_std / 2, n_samples))

    # Winding temp: baseline + slow degradation after onset + failure spikes
    winding = params.winding_temp_nominal_c \
              + rng.normal(0, 0.5, n_samples) \
              + _seasonal_variation(hours, amplitude=0.8)

    bearing = params.bearing_temp_nominal_c \
              + rng.normal(0, 0.3, n_samples) \
              + _seasonal_variation(hours, amplitude=0.6)

    # Progressive degradation on bearing temp (rises with BPFO energy)
    onset_idx = int(n_samples * degradation_onset_frac)
    degrade_length = n_samples - onset_idx
    degrade_rise = np.linspace(0, 8.0, degrade_length)  # +8 °C peak gradient
    bearing[onset_idx:] += degrade_rise

    # Inject failure events: step change on winding temp
    for ev_idx, ev_type in failure_events:
        if ev_type == "winding_spike" and ev_idx < n_samples:
            decay = np.exp(-np.arange(n_samples - ev_idx) / 3600.0)  # decay over 1h
            winding[ev_idx:] += 15.0 * decay

    return pd.DataFrame({
        "rpm": rpm,
        "flow_m3h": flow,
        "pressure_bar": pressure,
        "winding_temp_c": winding,
        "bearing_temp_c": bearing,
    })


def _generate_vibration(
    n_samples: int,
    rng: np.random.Generator,
    params: GCNParams,
    degradation_onset_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate vibration RMS and BPFO-band amplitude at 1 sample/sec."""
    # Baseline RMS vibration (slightly noisy)
    rms = np.full(n_samples, params.vibration_rms_nominal_mms)
    rms += rng.normal(0, params.vibration_noise_std, n_samples)

    # BPFO band amplitude: nominal 0.2 mm/s, grows 3-5x during degradation
    bpfo = np.full(n_samples, 0.2) + rng.normal(0, 0.02, n_samples)
    onset_idx = int(n_samples * degradation_onset_frac)
    degrade_length = n_samples - onset_idx
    # Exponential growth to 3-5x nominal
    growth_factor = np.linspace(1.0, 4.0, degrade_length)
    bpfo[onset_idx:] *= growth_factor
    # As BPFO grows, overall RMS also grows
    rms[onset_idx:] += (growth_factor - 1.0) * 0.6

    return rms, bpfo


def _make_labels(
    n_samples: int,
    degradation_onset_frac: float,
    failure_events: list[tuple[int, str]],
) -> np.ndarray:
    """Ternary label: 0=normal, 1=degrading, 2=failure imminent (last 24h)."""
    y = np.zeros(n_samples, dtype=np.int8)
    onset_idx = int(n_samples * degradation_onset_frac)
    y[onset_idx:] = 1
    # Mark 24h window before each failure as "failure imminent"
    seconds_in_day = 24 * 3600
    for ev_idx, _ in failure_events:
        start = max(0, ev_idx - seconds_in_day)
        y[start:ev_idx + 1] = 2
    return y


def generate_gcn(
    months: int = 24,
    n_failures: int = 2,
    degradation_onset_frac: float = 0.5,
    seed: int = 42,
    downsample_to_minutes: bool = True,
) -> pd.DataFrame:
    """Generate a synthetic ГЦН telemetry dataset.

    Args:
        months: Duration in months (default 24 per §15.5).
        n_failures: Number of sudden failure events to inject.
        degradation_onset_frac: Fraction of timeline when gradual degradation
            begins (0.5 = halfway through).
        seed: RNG seed for reproducibility.
        downsample_to_minutes: If True, return 1-minute aggregate
            (mean) data (practical size); if False, full 1 Hz.

    Returns:
        DataFrame with timestamp + 7 sensor channels + label column.
    """
    rng = np.random.default_rng(seed)
    params = GCNParams()

    # Total seconds
    n_samples = months * 30 * 24 * 3600  # 2 592 000 samples at 1 Hz over 24 months
    # (This is large — ~10 GB of float64 at full rate. Downsample to minutes for git-friendly size.)

    # Schedule failures in the post-degradation window (leave at least 1 day lead)
    onset_idx = int(n_samples * degradation_onset_frac)
    lead_seconds = min(30 * 24 * 3600, max(1, (n_samples - onset_idx) // 4))
    low = onset_idx + lead_seconds
    high = n_samples
    if low >= high:
        low = onset_idx
    failure_positions = rng.integers(low, high, n_failures)
    failure_events = [(int(p), "winding_spike") for p in failure_positions]

    print(f"Generating {months} months ({n_samples:,} 1 Hz samples)...")
    print(f"  Degradation onset at {degradation_onset_frac*100:.0f}% ({onset_idx:,})")
    print(f"  Failures at: {[int(p) for p in failure_positions]}")

    slow = _generate_slow_tags(n_samples, rng, params, degradation_onset_frac, failure_events)
    vib_rms, vib_bpfo = _generate_vibration(n_samples, rng, params, degradation_onset_frac)
    labels = _make_labels(n_samples, degradation_onset_frac, failure_events)

    start = pd.Timestamp("2023-01-01")
    timestamps = start + pd.to_timedelta(np.arange(n_samples), unit="s")

    df = slow.copy()
    df.insert(0, "timestamp", timestamps)
    df["vibration_rms_mms"] = vib_rms
    df["vibration_bpfo_amplitude"] = vib_bpfo
    df["label"] = labels

    if downsample_to_minutes:
        # Aggregate to 1-minute resolution via mean (for non-vibration);
        # max for vibration to preserve peak events; mode for label.
        df = df.set_index("timestamp")
        agg_slow = df[["rpm", "flow_m3h", "pressure_bar",
                       "winding_temp_c", "bearing_temp_c"]].resample("1min").mean()
        agg_vib = df[["vibration_rms_mms", "vibration_bpfo_amplitude"]].resample("1min").max()
        agg_lbl = df[["label"]].resample("1min").max()  # worst label in the minute
        df = pd.concat([agg_slow, agg_vib, agg_lbl], axis=1).reset_index()
        print(f"Downsampled to 1 min: {len(df):,} rows")

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="synthetic_gcn.parquet",
                        help="Output file (.parquet or .csv)")
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--failures", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-rate", action="store_true",
                        help="Keep full 1 Hz rate (10+ GB); default downsamples to 1 min.")
    parser.add_argument("--demo", action="store_true",
                        help="Small 1-month demo dataset (faster).")
    args = parser.parse_args()

    months = 1 if args.demo else args.months
    df = generate_gcn(
        months=months,
        n_failures=args.failures,
        seed=args.seed,
        downsample_to_minutes=not args.full_rate,
    )

    out_path = pathlib.Path(args.out)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError("Output must be .parquet or .csv")

    print(f"Saved {len(df):,} rows to {out_path}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"  Vibration RMS: mean={df['vibration_rms_mms'].mean():.2f} "
          f"min={df['vibration_rms_mms'].min():.2f} "
          f"max={df['vibration_rms_mms'].max():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
