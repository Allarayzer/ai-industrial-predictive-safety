"""Regime-aware conformal calibration and guarded drift-triggered updates."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Hashable

import numpy as np

from ai_cta.risk_model import ConformalThresholdCalibrator

__all__ = [
    "GuardedRecalibrationController",
    "RecalibrationDecision",
    "RegimeConformalCalibrator",
]


@dataclass(frozen=True)
class RecalibrationDecision:
    """Audit record for a proposed threshold update."""

    attempted: bool
    accepted: bool
    old_threshold: float
    candidate_threshold: float | None
    validation_far: float | None
    n_calibration: int
    n_validation: int
    reason: str


class RegimeConformalCalibrator:
    """Fit one split-conformal threshold per operating regime.

    A pooled fallback threshold is always fitted and is used for unseen regimes.
    """

    def __init__(self, alpha: float = 0.05, min_regime_samples: int | None = None):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        self.alpha = alpha
        self.min_regime_samples = (
            int(np.ceil(1 / alpha)) if min_regime_samples is None else min_regime_samples
        )

    def calibrate(
        self,
        scores: np.ndarray,
        regimes: np.ndarray,
    ) -> "RegimeConformalCalibrator":
        values = np.asarray(scores, dtype=float)
        groups = np.asarray(regimes, dtype=object)
        if values.ndim != 1 or groups.ndim != 1 or len(values) != len(groups):
            raise ValueError("scores and regimes must be aligned 1-D arrays.")
        self.pooled_ = ConformalThresholdCalibrator(self.alpha).calibrate(values).threshold_
        self.thresholds_: dict[Hashable, float] = {}
        for regime in np.unique(groups):
            subset = values[groups == regime]
            if len(subset) >= self.min_regime_samples:
                self.thresholds_[regime] = ConformalThresholdCalibrator(
                    self.alpha
                ).calibrate(subset).threshold_
        return self

    def threshold_for(self, regime: Hashable) -> float:
        if not hasattr(self, "pooled_"):
            raise RuntimeError("Call calibrate() before threshold_for().")
        return float(self.thresholds_.get(regime, self.pooled_))

    def apply(self, scores: np.ndarray, regimes: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        groups = np.asarray(regimes, dtype=object)
        if len(values) != len(groups):
            raise ValueError("scores and regimes must have equal length.")
        thresholds = np.asarray([self.threshold_for(g) for g in groups], dtype=float)
        return (values > thresholds).astype(int)


class GuardedRecalibrationController:
    """Propose and validate a new conformal threshold after persistent drift.

    The controller intentionally does not retrain component models.  It waits
    for ``drift_persistence`` consecutive drift reports, accumulates trusted
    normal fused scores, splits them chronologically into calibration and
    validation portions, and accepts a candidate threshold only when:

    * the candidate change is not larger than ``max_relative_change``; and
    * the validation false-alarm rate is at most ``far_tolerance * alpha``.

    Rejected proposals leave the current threshold unchanged, providing a
    simple rollback safeguard.
    """

    def __init__(
        self,
        initial_threshold: float,
        alpha: float = 0.05,
        buffer_size: int = 2000,
        min_samples: int = 300,
        validation_fraction: float = 0.25,
        drift_persistence: int = 3,
        max_relative_change: float = 0.75,
        far_tolerance: float = 1.5,
        cooldown_samples: int = 500,
        conservative_factor: float = 1.0,
    ):
        if not 0.0 <= initial_threshold <= 1.0:
            raise ValueError("initial_threshold must be in [0, 1].")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if not 0.0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5).")
        if buffer_size < min_samples:
            raise ValueError("buffer_size must be >= min_samples.")
        if not 0.0 < conservative_factor <= 1.0:
            raise ValueError("conservative_factor must be in (0, 1].")
        self.threshold = float(initial_threshold)
        self.alpha = alpha
        self.min_samples = min_samples
        self.validation_fraction = validation_fraction
        self.drift_persistence = drift_persistence
        self.max_relative_change = max_relative_change
        self.far_tolerance = far_tolerance
        self.cooldown_samples = cooldown_samples
        self.conservative_factor = conservative_factor
        self._normal_scores: deque[float] = deque(maxlen=buffer_size)
        self._drift_streak = 0
        self._collecting_after_drift = False
        self._samples_since_attempt = cooldown_samples
        self.history: list[RecalibrationDecision] = []

    def observe_normal(self, score: float) -> None:
        value = float(score)
        if np.isfinite(value):
            self._normal_scores.append(float(np.clip(value, 0.0, 1.0)))
        self._samples_since_attempt += 1

    def update_drift(self, drift_detected: bool) -> None:
        previous = self._drift_streak
        self._drift_streak = self._drift_streak + 1 if drift_detected else 0
        # Once drift becomes persistent, discard pre-shift calibration scores.
        # The candidate threshold must be estimated from the new operating
        # distribution rather than a mixture of old and new regimes.
        if (
            drift_detected
            and previous < self.drift_persistence <= self._drift_streak
            and not self._collecting_after_drift
        ):
            self._normal_scores.clear()
            self._collecting_after_drift = True
            self._samples_since_attempt = 0

    def maybe_recalibrate(self) -> RecalibrationDecision:
        if self._drift_streak < self.drift_persistence:
            decision = RecalibrationDecision(
                attempted=False,
                accepted=False,
                old_threshold=self.threshold,
                candidate_threshold=None,
                validation_far=None,
                n_calibration=0,
                n_validation=0,
                reason="drift_not_persistent",
            )
            return decision
        if self._samples_since_attempt < self.cooldown_samples:
            return RecalibrationDecision(
                attempted=False,
                accepted=False,
                old_threshold=self.threshold,
                candidate_threshold=None,
                validation_far=None,
                n_calibration=0,
                n_validation=0,
                reason="cooldown",
            )
        if len(self._normal_scores) < self.min_samples:
            return RecalibrationDecision(
                attempted=False,
                accepted=False,
                old_threshold=self.threshold,
                candidate_threshold=None,
                validation_far=None,
                n_calibration=0,
                n_validation=0,
                reason="insufficient_trusted_normal_scores",
            )

        scores = np.asarray(self._normal_scores, dtype=float)
        split = int(np.floor(len(scores) * (1.0 - self.validation_fraction)))
        calibration, validation = scores[:split], scores[split:]
        candidate = ConformalThresholdCalibrator(
            self.alpha * self.conservative_factor
        ).calibrate(
            calibration
        ).threshold_
        validation_far = float(np.mean(validation > candidate))
        relative_change = abs(candidate - self.threshold) / max(self.threshold, 1e-6)
        accepted = (
            relative_change <= self.max_relative_change
            and validation_far <= self.far_tolerance * self.alpha
        )
        reason = "accepted"
        if relative_change > self.max_relative_change:
            reason = "threshold_jump_too_large"
        elif validation_far > self.far_tolerance * self.alpha:
            reason = "validation_far_too_high"

        old = self.threshold
        if accepted:
            self.threshold = float(candidate)
            self._drift_streak = 0
            self._collecting_after_drift = False
        self._samples_since_attempt = 0
        decision = RecalibrationDecision(
            attempted=True,
            accepted=accepted,
            old_threshold=old,
            candidate_threshold=float(candidate),
            validation_far=validation_far,
            n_calibration=len(calibration),
            n_validation=len(validation),
            reason=reason,
        )
        self.history.append(decision)
        return decision
