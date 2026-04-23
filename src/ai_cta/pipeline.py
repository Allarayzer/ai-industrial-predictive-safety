"""End-to-end streaming pipeline.
The SafetyPipeline orchestrates feature extraction, anomaly detection, risk
scoring, and alerting over a stream of sensor readings. It is designed for
demonstration and research use; production deployments should integrate it
with industrial-grade streaming backends (Kafka, MQTT brokers, OPC UA
servers) via the provided adapters.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional
import numpy as np
import pandas as pd
from ai_cta.risk_model import RiskScorer

__all__ = ["SafetyPipeline", "PipelineEvent"]

logger = logging.getLogger(__name__)

@dataclass
class PipelineEvent:
    """Single monitoring event emitted by the pipeline."""
    timestamp: pd.Timestamp
    anomaly_score: float
    risk_score: float
    risk_level: str
    channel_values: dict = field(default_factory=dict)
    def to_dict(self) -> dict:
        return {
            "timestamp": str(self.timestamp),
            "anomaly_score": float(self.anomaly_score),
            "risk_score": float(self.risk_score),
            "risk_level": self.risk_level,
            "channel_values": self.channel_values,
        }

class SafetyPipeline:
    """Orchestrate detection, scoring, and alerting over a sensor stream.
    Parameters
    ----------
    detector : object
        Any fitted detector exposing a `decision_function(pd.DataFrame) -> np.ndarray`.
    risk_scorer : RiskScorer
    window_size : int, default=32
        Samples accumulated before the first prediction is issued.
    alert_threshold : float, default=0.6
        Risk scores at or above this value trigger `alert_callback`.
    alert_callback : callable, optional
        Invoked with a PipelineEvent whenever `alert_threshold` is exceeded.
        Default is a no-op that simply logs at WARNING level.
    Examples
    --------
    >>> pipeline = SafetyPipeline(
    ...     detector=my_detector,
    ...     risk_scorer=my_scorer,
    ...     alert_threshold=0.7,
    ... )
    >>> for event in pipeline.run(stream_of_dicts):
    ...     print(event.risk_level)
    """
    def __init__(
        self,
        detector,
        risk_scorer: RiskScorer,
        window_size: int = 32,
        alert_threshold: float = 0.6,
        alert_callback: Callable[[PipelineEvent], None] | None = None,
    ):
        self.detector = detector
        self.risk_scorer = risk_scorer
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.alert_callback = alert_callback or self._default_alert
    @staticmethod
    def _default_alert(event: PipelineEvent) -> None:
        logger.warning(
            "Alert: risk=%.3f level=%s at %s",
            event.risk_score,
            event.risk_level,
            event.timestamp,
        )
    def run(self, stream: Iterable[dict]) -> Iterable[PipelineEvent]:
        """Process each incoming reading, yielding PipelineEvent objects."""
        buffer: list[dict] = []
        for reading in stream:
            buffer.append(reading)
            if len(buffer) < self.window_size:
                continue
            buffer = buffer[-self.window_size :]
            df = pd.DataFrame(buffer)
            numeric_df = df.select_dtypes(include="number")
            try:
                scores = self.detector.decision_function(numeric_df)
            except Exception as exc:  # noqa: BLE001 - we re-raise after logging
                logger.error("Detector error: %s", exc)
                raise
            anomaly_score = float(scores[-1])
            last_reading = reading
            channel_values = {
                k: float(v)
                for k, v in last_reading.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            risk_score = self.risk_scorer.score(anomaly_score, channel_values)
            level = self.risk_scorer.level(risk_score)
            timestamp = last_reading.get("timestamp")
            if not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.Timestamp(timestamp) if timestamp else pd.Timestamp.now()
            event = PipelineEvent(
                timestamp=timestamp,
                anomaly_score=anomaly_score,
                risk_score=risk_score,
                risk_level=level,
                channel_values=channel_values,
            )
            if risk_score >= self.alert_threshold:
                self.alert_callback(event)
            yield event
    # ----------------------------------------------- webhook integration
    @staticmethod
    def make_webhook_callback(
        webhook_url: str,
        timeout: float = 5.0,
    ) -> Callable[[PipelineEvent], None]:
        """Build a callback that POSTs events to a webhook (e.g., n8n).
        Import of `requests` is deferred so that the rest of the package
        does not require a network library for purely offline use.
        """
        def _callback(event: PipelineEvent) -> None:
            import requests  # noqa: WPS433 - intentional lazy import
            try:
                requests.post(
                    webhook_url,
                    data=json.dumps(event.to_dict()),
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                logger.error("Webhook POST failed: %s", exc)
        return _callback
