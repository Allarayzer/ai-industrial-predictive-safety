"""Industrial telemetry simulator (monograph § 12.3.1).
Generates multi-channel synthetic sensor data following the model
specified in Chapter 12.3.1 of the accompanying monograph:
    x_i(t) = μ_i + A_i · sin(2π f_i t + φ_i) + D_i(t) + Σ_k δ_ik(t) + ε_i(t)
where:
- μ_i is the channel baseline,
- A_i sin(...) is a periodic seasonal component,
- D_i(t) is degradation drift (linear, exponential, or stepwise),
- δ_ik(t) are anomaly events (spike, level shift, drift),
- ε_i(t) ~ N(0, σ_i²) is observation noise.
Anomaly events follow a Poisson process with intensity λ; each event
is associated (with non-zero probability) with a future failure at a
delay drawn from an exponential distribution.
The simulator is fully reproducible via a single integer seed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd

__all__ = ["IndustrialSimulator", "SimulationEvent"]

@dataclass
class SimulationEvent:
    """Record of one anomaly event injected by the simulator."""
    timestep: int
    channel: str
    kind: str  # "spike" | "level_shift" | "drift"
    magnitude: float
    failure_at: int | None = None  # absolute timestep of related failure, if any

@dataclass
class _ChannelSpec:
    """Per-channel parameters of the generative model."""
    name: str
    baseline: float
    seasonal_amplitude: float
    seasonal_period_minutes: float
    seasonal_phase: float
    degradation_rate: float
    degradation_kind: Literal["linear", "exponential", "stepwise", "none"]
    noise_std: float

class IndustrialSimulator:
    """Reproducible multi-sensor telemetry generator.
    Parameters
    ----------
    n_sensors : int, default=10
        Number of sensor channels.
    duration_days : float, default=30
        Total simulation length in days.
    frequency_minutes : float, default=1
        Sampling interval in minutes.
    anomaly_intensity : float, default=0.005
        Poisson intensity for anomaly events (events per timestep).
    degradation_rate : float, default=0.02
        Average per-day degradation rate (channel-specific values are
        drawn around this mean).
    failure_probability : float, default=0.6
        Probability that an anomaly event escalates to a failure later.
    failure_delay_mean : float, default=720
        Mean delay (in timesteps) from triggering anomaly to failure.
    seed : int, default=42
    Examples
    --------
    >>> sim = IndustrialSimulator(n_sensors=5, duration_days=7, seed=0)
    >>> df, events = sim.generate()
    >>> df.shape
    (10080, 6)        # 7 days × 1440 minutes/day, +1 column for timestamp
    """
    def __init__(
        self,
        n_sensors: int = 10,
        duration_days: float = 30.0,
        frequency_minutes: float = 1.0,
        anomaly_intensity: float = 0.005,
        degradation_rate: float = 0.02,
        failure_probability: float = 0.6,
        failure_delay_mean: float = 720.0,
        seed: int = 42,
    ):
        if n_sensors < 1:
            raise ValueError("n_sensors must be >= 1.")
        if duration_days <= 0:
            raise ValueError("duration_days must be positive.")
        if frequency_minutes <= 0:
            raise ValueError("frequency_minutes must be positive.")
        if not 0.0 <= anomaly_intensity <= 1.0:
            raise ValueError("anomaly_intensity must be in [0, 1].")
        if not 0.0 <= failure_probability <= 1.0:
            raise ValueError("failure_probability must be in [0, 1].")
        self.n_sensors = n_sensors
        self.duration_days = duration_days
        self.frequency_minutes = frequency_minutes
        self.anomaly_intensity = anomaly_intensity
        self.degradation_rate = degradation_rate
        self.failure_probability = failure_probability
        self.failure_delay_mean = failure_delay_mean
        self.seed = seed
    # ----------------------------------------------------- channel specs
    def _build_channel_specs(self, rng: np.random.Generator) -> list[_ChannelSpec]:
        """Randomize per-channel parameters around documented defaults."""
        kinds: list[str] = list(rng.choice(
            ["linear", "exponential", "stepwise", "none"],
            size=self.n_sensors,
            p=[0.45, 0.25, 0.15, 0.15],
        ))
        specs: list[_ChannelSpec] = []
        for i in range(self.n_sensors):
            baseline = float(rng.uniform(20.0, 80.0))
            seasonal_period = float(rng.choice([60.0, 360.0, 720.0, 1440.0]))
            specs.append(
                _ChannelSpec(
                    name=f"sensor_{i+1:02d}",
                    baseline=baseline,
                    seasonal_amplitude=float(rng.uniform(0.5, 4.0)),
                    seasonal_period_minutes=seasonal_period,
                    seasonal_phase=float(rng.uniform(0, 2 * np.pi)),
                    degradation_rate=float(
                        rng.uniform(0.5, 1.5) * self.degradation_rate
                    ),
                    degradation_kind=kinds[i],  # type: ignore[arg-type]
                    noise_std=float(rng.uniform(0.2, 1.0)),
                )
            )
        return specs
    # --------------------------------------------------------- generate
    def generate(self) -> tuple[pd.DataFrame, list[SimulationEvent]]:
        """Run the simulation.
        Returns
        -------
        df : DataFrame
            One row per timestep. Columns: ``timestamp`` plus one per
            sensor (``sensor_01``, ``sensor_02``, ...).
        events : list of SimulationEvent
            All injected anomaly events with their metadata.
        """
        rng = np.random.default_rng(self.seed)
        specs = self._build_channel_specs(rng)
        n_steps = int(round(self.duration_days * 24 * 60 / self.frequency_minutes))
        timestamps = pd.date_range(
            "2025-01-01",
            periods=n_steps,
            freq=f"{self.frequency_minutes:g}min",
        )
        t_minutes = np.arange(n_steps, dtype=float) * self.frequency_minutes
        # Base signal: μ + seasonal + degradation + noise
        data: dict[str, np.ndarray] = {}
        for spec in specs:
            seasonal = spec.seasonal_amplitude * np.sin(
                2 * np.pi * t_minutes / spec.seasonal_period_minutes
                + spec.seasonal_phase
            )
            degradation = self._degradation_curve(t_minutes, spec)
            noise = rng.normal(0.0, spec.noise_std, n_steps)
            data[spec.name] = spec.baseline + seasonal + degradation + noise
        # Inject anomalies via Poisson process
        events: list[SimulationEvent] = []
        n_events = int(rng.poisson(self.anomaly_intensity * n_steps))
        for _ in range(n_events):
            t_event = int(rng.integers(0, n_steps))
            spec = specs[int(rng.integers(0, len(specs)))]
            kind = str(
                rng.choice(["spike", "level_shift", "drift"], p=[0.5, 0.3, 0.2])
            )
            base_amp = max(spec.noise_std * 4.0, 1.0)
            magnitude = float(rng.uniform(2.0, 5.0)) * base_amp * float(rng.choice([-1, 1]))
            self._apply_event(data[spec.name], t_event, kind, magnitude, n_steps)
            failure_at: int | None = None
            if rng.random() < self.failure_probability:
                delay = int(rng.exponential(self.failure_delay_mean))
                failure_at = min(t_event + delay, n_steps - 1)
            events.append(
                SimulationEvent(
                    timestep=t_event,
                    channel=spec.name,
                    kind=kind,
                    magnitude=magnitude,
                    failure_at=failure_at,
                )
            )
        df = pd.DataFrame({"timestamp": timestamps, **data})
        return df, events
    # ---------------------------------------------------- internal helpers
    @staticmethod
    def _degradation_curve(t_minutes: np.ndarray, spec: _ChannelSpec) -> np.ndarray:
        days = t_minutes / (60.0 * 24.0)
        if spec.degradation_kind == "linear":
            return spec.degradation_rate * days * spec.baseline / 100.0
        if spec.degradation_kind == "exponential":
            return spec.baseline * (np.exp(spec.degradation_rate * days / 100.0) - 1.0) * 0.1
        if spec.degradation_kind == "stepwise":
            n_steps = max(int(days.max() / 7.0), 1)
            step_height = spec.degradation_rate * spec.baseline / 100.0
            return np.floor(days / 7.0) * step_height * 0.5
        return np.zeros_like(t_minutes)
    @staticmethod
    def _apply_event(
        signal: np.ndarray,
        t_event: int,
        kind: str,
        magnitude: float,
        n_steps: int,
    ) -> None:
        if kind == "spike":
            signal[t_event] += magnitude
        elif kind == "level_shift":
            signal[t_event:] += magnitude
        elif kind == "drift":
            duration = min(int(0.05 * n_steps) + 30, n_steps - t_event)
            ramp = np.linspace(0, magnitude, duration)
            signal[t_event : t_event + duration] += ramp

def make_failure_labels(
    n_steps: int,
    events: list[SimulationEvent],
    horizon: int = 720,
) -> np.ndarray:
    """Convert simulator events into a binary label per timestep.
    Label at time t is 1 if any event with `failure_at` value within
    the next `horizon` timesteps has been triggered up to time t.
    """
    labels = np.zeros(n_steps, dtype=int)
    for ev in events:
        if ev.failure_at is None:
            continue
        start = ev.timestep
        end = min(ev.failure_at + 1, n_steps)
        labels[start:end] = 1
    return labels
