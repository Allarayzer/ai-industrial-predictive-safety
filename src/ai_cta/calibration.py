"""Online recalibration of detector thresholds.
Implements Chapter 9.10 of the monograph: a bounded-window online
calibration loop that periodically updates conformal thresholds and
risk-aggregator weights as new ground-truth labels arrive from the
operations team (e.g., confirmed alarms, missed events tagged
post-hoc).
The calibrator maintains:
- A sliding buffer of (score, label) pairs.
- A schedule (time-based or event-count-based) on which to re-fit.
When triggered, it produces fresh threshold and weight updates that
the consuming pipeline can adopt atomically.
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from ai_cta.risk_model import ConformalThresholdCalibrator

__all__ = ["OnlineCalibrator", "CalibrationUpdate"]

@dataclass
class CalibrationUpdate:
    """Output of one calibration cycle."""
    threshold: float
    n_samples_used: int
    timestamp: float = field(default_factory=time.time)

class OnlineCalibrator:
    """Periodic recalibration of an anomaly threshold.
    Maintains a sliding window of recently observed scores from
    known-normal data points; periodically uses split conformal
    prediction to recompute the operational threshold.
    Parameters
    ----------
    alpha : float, default=0.05
        Target false-alarm rate.
    buffer_size : int, default=2000
        Maximum number of recent observations retained.
    min_recalibrate : int, default=200
        Minimum number of buffered observations before the first
        recalibration is allowed.
    schedule_seconds : float, default=3600
        Minimum time between consecutive recalibrations.
    schedule_samples : int, default=500
        Minimum number of new samples observed since the last
        recalibration before the next one is allowed.
    Examples
    --------
    >>> calibrator = OnlineCalibrator(alpha=0.05)
    >>> for score in incoming_normal_scores:
    ...     calibrator.add(score)
    ...     update = calibrator.maybe_recalibrate()
    ...     if update is not None:
    ...         pipeline.threshold = update.threshold
    """
    def __init__(
        self,
        alpha: float = 0.05,
        buffer_size: int = 2000,
        min_recalibrate: int = 200,
        schedule_seconds: float = 3600.0,
        schedule_samples: int = 500,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if buffer_size < min_recalibrate:
            raise ValueError("buffer_size must be >= min_recalibrate.")
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.min_recalibrate = min_recalibrate
        self.schedule_seconds = schedule_seconds
        self.schedule_samples = schedule_samples
        self._buffer: deque[float] = deque(maxlen=buffer_size)
        self._last_calibration_time: Optional[float] = None
        self._last_calibration_count: int = 0
        self._observations_seen: int = 0
    # -------------------------------------------------------- ingestion
    def add(self, score: float) -> None:
        """Append a single normal-data score to the calibration buffer."""
        self._buffer.append(float(score))
        self._observations_seen += 1
    def add_batch(self, scores: np.ndarray) -> None:
        """Append multiple scores at once."""
        for s in scores:
            self.add(float(s))
    # ----------------------------------------------------- calibration
    def maybe_recalibrate(self) -> Optional[CalibrationUpdate]:
        """Return a fresh threshold if the schedule allows, else None.
        Recalibration happens when:
        - The buffer has at least `min_recalibrate` entries.
        - At least `schedule_samples` new entries have been seen since
          the last recalibration.
        - At least `schedule_seconds` have elapsed since the last
          recalibration (or it is the first call).
        """
        if len(self._buffer) < self.min_recalibrate:
            return None
        now = time.time()
        new_samples = self._observations_seen - self._last_calibration_count
        if (
            self._last_calibration_time is not None
            and now - self._last_calibration_time < self.schedule_seconds
        ):
            return None
        if new_samples < self.schedule_samples and self._last_calibration_time is not None:
            return None
        scores = np.asarray(list(self._buffer), dtype=float)
        cal = ConformalThresholdCalibrator(alpha=self.alpha).calibrate(scores)
        self._last_calibration_time = now
        self._last_calibration_count = self._observations_seen
        return CalibrationUpdate(
            threshold=cal.threshold_,
            n_samples_used=len(scores),
        )
    # ------------------------------------------------------- inspection
    def current_buffer_size(self) -> int:
        """Number of scores currently held in the buffer."""
        return len(self._buffer)
    def total_observations(self) -> int:
        """Total number of scores ever added (may exceed buffer capacity)."""
        return self._observations_seen
