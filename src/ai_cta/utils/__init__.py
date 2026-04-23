"""Utility functions for dataset loading, simulation, and evaluation."""
from ai_cta.utils.data import (
    generate_synthetic_stream,
    inject_anomalies,
)
from ai_cta.utils.evaluation import evaluate_binary_detector
from ai_cta.utils.simulator import (
    IndustrialSimulator,
    SimulationEvent,
    make_failure_labels,
)
__all__ = [
    "generate_synthetic_stream",
    "inject_anomalies",
    "evaluate_binary_detector",
    "IndustrialSimulator",
    "SimulationEvent",
    "make_failure_labels",
]
