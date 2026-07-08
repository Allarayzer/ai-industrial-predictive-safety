"""Operator-auditable online pipeline for asynchronous risk fusion."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd

from ai_cta.adaptive_calibration import (
    GuardedRecalibrationController,
    RecalibrationDecision,
)
from ai_cta.drift_detector import DriftDetector, DriftReport
from ai_cta.online_fusion import AsynchronousRiskFusion, FusionSnapshot

__all__ = ["AdaptivePipelineEvent", "AdaptiveSafetyPipeline"]


@dataclass(frozen=True)
class AdaptivePipelineEvent:
    timestamp: pd.Timestamp
    fusion: FusionSnapshot
    threshold: float
    alarm: bool
    drift_report: DriftReport | None
    recalibration: RecalibrationDecision | None


class AdaptiveSafetyPipeline:
    """Fuse asynchronous channels, monitor drift, and guard recalibration.

    The caller supplies precomputed risk-channel updates.  This separation is
    deliberate: channel models can run at different cadences or even in
    different services, while this class owns only temporal alignment,
    thresholding, drift monitoring, and the audit trail.
    """

    def __init__(
        self,
        fusion: AsynchronousRiskFusion,
        drift_detector: DriftDetector,
        recalibration: GuardedRecalibrationController,
        reference_features: pd.DataFrame,
        drift_window_size: int = 240,
        drift_check_interval: int = 60,
    ):
        if drift_window_size < 2:
            raise ValueError("drift_window_size must be >= 2.")
        if drift_check_interval < 1:
            raise ValueError("drift_check_interval must be >= 1.")
        self.fusion = fusion
        self.drift_detector = drift_detector.fit(reference_features)
        self.recalibration = recalibration
        self.drift_window_size = drift_window_size
        self.drift_check_interval = drift_check_interval
        self._features: deque[dict] = deque(maxlen=drift_window_size)
        self._seen = 0

    def process(
        self,
        timestamp: pd.Timestamp | str,
        features: dict[str, float],
        channel_updates: dict[str, float],
        trusted_normal: bool = False,
    ) -> AdaptivePipelineEvent:
        ts = pd.Timestamp(timestamp)
        self.fusion.update_many(channel_updates, ts)
        snapshot = self.fusion.fuse(ts)
        self._features.append(features)
        self._seen += 1

        report: DriftReport | None = None
        decision: RecalibrationDecision | None = None
        if (
            len(self._features) == self.drift_window_size
            and self._seen % self.drift_check_interval == 0
        ):
            report = self.drift_detector.detect(pd.DataFrame(self._features))
            self.recalibration.update_drift(report.drift_detected)

        if trusted_normal:
            self.recalibration.observe_normal(snapshot.risk_score)
        if report is not None and report.drift_detected:
            decision = self.recalibration.maybe_recalibrate()

        threshold = self.recalibration.threshold
        return AdaptivePipelineEvent(
            timestamp=ts,
            fusion=snapshot,
            threshold=threshold,
            alarm=snapshot.risk_score > threshold,
            drift_report=report,
            recalibration=decision,
        )
