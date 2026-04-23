"""Tests for the IndustrialSimulator."""
import numpy as np
import pandas as pd
from ai_cta.utils import (
    IndustrialSimulator,
    SimulationEvent,
    make_failure_labels,
)

def test_simulator_basic_shape():
    sim = IndustrialSimulator(
        n_sensors=4, duration_days=1.0, frequency_minutes=5.0, seed=0
    )
    df, events = sim.generate()
    expected_steps = int(round(1.0 * 24 * 60 / 5.0))
    assert len(df) == expected_steps
    assert "timestamp" in df.columns
    assert df.shape[1] == 1 + 4  # timestamp + 4 sensors
    assert isinstance(events, list)

def test_simulator_reproducibility():
    sim1 = IndustrialSimulator(n_sensors=3, duration_days=0.5, seed=42)
    sim2 = IndustrialSimulator(n_sensors=3, duration_days=0.5, seed=42)
    df1, events1 = sim1.generate()
    df2, events2 = sim2.generate()
    pd.testing.assert_frame_equal(df1, df2)
    assert len(events1) == len(events2)
    for e1, e2 in zip(events1, events2):
        assert e1 == e2

def test_simulator_different_seeds_differ():
    sim1 = IndustrialSimulator(n_sensors=3, duration_days=0.5, seed=1)
    sim2 = IndustrialSimulator(n_sensors=3, duration_days=0.5, seed=2)
    df1, _ = sim1.generate()
    df2, _ = sim2.generate()
    # Should differ in numeric values
    diff = (df1.drop(columns=["timestamp"]) - df2.drop(columns=["timestamp"])).abs().sum().sum()
    assert diff > 0.0

def test_simulator_validates_inputs():
    import pytest
    with pytest.raises(ValueError):
        IndustrialSimulator(n_sensors=0)
    with pytest.raises(ValueError):
        IndustrialSimulator(duration_days=0)
    with pytest.raises(ValueError):
        IndustrialSimulator(frequency_minutes=0)
    with pytest.raises(ValueError):
        IndustrialSimulator(anomaly_intensity=2.0)
    with pytest.raises(ValueError):
        IndustrialSimulator(failure_probability=1.5)

def test_simulator_events_have_metadata():
    sim = IndustrialSimulator(
        n_sensors=3, duration_days=2.0, anomaly_intensity=0.05, seed=7
    )
    df, events = sim.generate()
    assert len(events) > 0
    for e in events:
        assert isinstance(e, SimulationEvent)
        assert 0 <= e.timestep < len(df)
        assert e.channel in df.columns
        assert e.kind in {"spike", "level_shift", "drift"}

def test_make_failure_labels():
    events = [
        SimulationEvent(timestep=10, channel="s", kind="spike", magnitude=1.0, failure_at=20),
        SimulationEvent(timestep=50, channel="s", kind="spike", magnitude=1.0, failure_at=None),
    ]
    labels = make_failure_labels(n_steps=100, events=events)
    assert labels[10:21].sum() == 11  # period from event to failure
    assert labels[50:].sum() == 0  # no failure_at = no labels
