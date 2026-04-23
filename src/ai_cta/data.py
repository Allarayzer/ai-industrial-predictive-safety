"""Synthetic data utilities for testing and demonstration."""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["generate_synthetic_stream", "inject_anomalies"]

def generate_synthetic_stream(
    n_samples: int = 1000,
    sampling_rate: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulate a multivariate industrial sensor stream.
    Generates three correlated channels representative of rotating machinery
    under stable operation: temperature, vibration (with a dominant
    frequency), and pressure.
    Parameters
    ----------
    n_samples : int
        Number of time steps.
    sampling_rate : float
        Sampling rate in Hz (affects the vibration waveform).
    random_state : int
    Returns
    -------
    df : DataFrame
        Columns: timestamp, temperature, vibration, pressure.
    """
    rng = np.random.default_rng(random_state)
    t = np.arange(n_samples) / sampling_rate
    timestamps = pd.date_range("2025-01-01", periods=n_samples, freq="1s")
    temperature = 50.0 + 2.0 * np.sin(2 * np.pi * t / 600.0) + rng.normal(0, 0.4, n_samples)
    vibration = 0.3 + 0.1 * np.sin(2 * np.pi * 0.1 * t) + rng.normal(0, 0.02, n_samples)
    # Pressure slightly anti-correlated with temperature.
    pressure = 1.0 - 0.003 * (temperature - 50.0) + rng.normal(0, 0.01, n_samples)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temperature,
            "vibration": np.abs(vibration),
            "pressure": pressure,
        }
    )

def inject_anomalies(
    df: pd.DataFrame,
    n_anomalies: int = 20,
    anomaly_types: tuple[str, ...] = ("spike", "drift", "oscillation"),
    channels: tuple[str, ...] = ("temperature", "vibration", "pressure"),
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Inject three kinds of anomalies into a sensor DataFrame.
    Returns
    -------
    df_out : DataFrame
        Copy of `df` with anomalies injected.
    labels : ndarray of shape (len(df),)
        Binary label (1 = anomalous timestep, 0 = normal).
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()
    labels = np.zeros(len(df), dtype=int)
    for _ in range(n_anomalies):
        kind = rng.choice(anomaly_types)
        channel = rng.choice(channels)
        if channel not in out.columns:
            continue
        if kind == "spike":
            idx = int(rng.integers(1, len(out) - 1))
            magnitude = float(rng.uniform(3.0, 6.0)) * float(np.std(out[channel]))
            sign = rng.choice([-1.0, 1.0])
            out.loc[idx, channel] += sign * magnitude
            labels[idx] = 1
        elif kind == "drift":
            start = int(rng.integers(0, len(out) - 50))
            end = min(start + int(rng.integers(30, 80)), len(out))
            magnitude = float(rng.uniform(2.0, 4.0)) * float(np.std(out[channel]))
            ramp = np.linspace(0, magnitude, end - start)
            out.loc[start : end - 1, channel] = out.loc[start : end - 1, channel].to_numpy() + ramp
            labels[start:end] = 1
        elif kind == "oscillation":
            start = int(rng.integers(0, len(out) - 50))
            end = min(start + int(rng.integers(20, 50)), len(out))
            base_amp = float(np.std(out[channel]))
            t = np.arange(end - start)
            wave = 2.0 * base_amp * np.sin(2 * np.pi * t / 5.0)
            out.loc[start : end - 1, channel] = out.loc[start : end - 1, channel].to_numpy() + wave
            labels[start:end] = 1
    return out, labels
