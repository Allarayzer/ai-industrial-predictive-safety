"""Tests for the streaming SafetyPipeline."""
import numpy as np
import pandas as pd

from ai_cta.data import generate_synthetic_stream
from ai_cta.pipeline import SafetyPipeline
from ai_cta.risk_model import RiskScorer


class _DummyDetector:
    """Detector that returns a constant anomaly score per window."""
    def __init__(self, score: float = 0.3):
        self.score = score
    def decision_function(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.score)

def test_pipeline_emits_events():
    df = generate_synthetic_stream(n_samples=100, random_state=0)
    stream = df.to_dict(orient="records")
    pipeline = SafetyPipeline(
        detector=_DummyDetector(0.3),
        risk_scorer=RiskScorer(ml_weight=1.0),
        window_size=32,
        alert_threshold=0.95,
    )
    events = list(pipeline.run(stream))
    assert len(events) == len(stream) - 32 + 1
    for event in events:
        assert 0.0 <= event.risk_score <= 1.0
        assert event.risk_level in {"OK", "Warning", "Critical", "Emergency"}

def test_pipeline_triggers_alerts():
    df = generate_synthetic_stream(n_samples=50, random_state=0)
    alerts = []
    pipeline = SafetyPipeline(
        detector=_DummyDetector(0.9),
        risk_scorer=RiskScorer(ml_weight=1.0),
        window_size=10,
        alert_threshold=0.5,
        alert_callback=alerts.append,
    )
    list(pipeline.run(df.to_dict(orient="records")))
    assert len(alerts) > 0

def test_pipeline_skips_warmup():
    df = generate_synthetic_stream(n_samples=20, random_state=0)
    pipeline = SafetyPipeline(
        detector=_DummyDetector(0.1),
        risk_scorer=RiskScorer(ml_weight=1.0),
        window_size=30,  # larger than stream
    )
    events = list(pipeline.run(df.to_dict(orient="records")))
    # Stream shorter than window: no events produced.
    assert events == []
