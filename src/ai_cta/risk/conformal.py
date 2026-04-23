"""Conformal calibration of anomaly-detection thresholds.
Classical anomaly detectors produce scores whose distribution depends on the
training set and is rarely directly interpretable as a probability. For
safety-critical deployment we need a principled way to pick the decision
threshold so that a target false-alarm rate is guaranteed on the kind of
data the system will see in production.
Split conformal prediction provides such a guarantee under exchangeability:
given a held-out calibration set of scores produced on known-normal data,
the empirical (1 - alpha) quantile serves as a threshold with marginal
coverage at least 1 - alpha in expectation.
References
----------
Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a
Random World.
Angelopoulos, A. N., & Bates, S. (2023). Conformal Prediction: A Gentle
Introduction. Foundations and Trends in Machine Learning.
"""
from __future__ import annotations
import numpy as np

__all__ = ["ConformalThresholdCalibrator"]

class ConformalThresholdCalibrator:
    """Split conformal threshold for anomaly detectors.
    Parameters
    ----------
    alpha : float, default=0.05
        Target false-alarm rate. The calibrated threshold flags at most a
        fraction alpha of the known-normal calibration set as anomalous.
    random_state : int, default=42
    Attributes
    ----------
    threshold_ : float
        Calibrated threshold; populated after `calibrate`.
    calibration_size_ : int
        Number of calibration samples used.
    """
    def __init__(self, alpha: float = 0.05, random_state: int = 42):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        self.alpha = alpha
        self.random_state = random_state
    def calibrate(self, calibration_scores: np.ndarray) -> "ConformalThresholdCalibrator":
        """Compute the (1 - alpha)-quantile on a held-out calibration set.
        Parameters
        ----------
        calibration_scores : ndarray
            Scores produced by the detector on known-normal data. Should NOT
            overlap with the detector's training data.
        """
        scores = np.asarray(calibration_scores, dtype=float)
        if scores.ndim != 1:
            raise ValueError("calibration_scores must be 1-D.")
        n = len(scores)
        if n < int(1 / self.alpha):
            raise ValueError(
                f"Need at least {int(1 / self.alpha)} calibration points "
                f"for alpha={self.alpha}; got {n}."
            )
        # The finite-sample-corrected quantile guarantees the coverage.
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = float(np.clip(level, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, level))
        self.calibration_size_ = n
        return self
    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Return binary decisions using the calibrated threshold."""
        if not hasattr(self, "threshold_"):
            raise RuntimeError("Call calibrate() before apply().")
        return (np.asarray(scores) > self.threshold_).astype(int)
