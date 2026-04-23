"""Hybrid ensemble of Isolation Forest and LSTM detectors.
The two base detectors capture complementary anomaly modes:
- IsolationForestDetector flags point-wise structural anomalies (values that
  are unlike anything seen in training) using engineered features over windows.
- LSTMDetector flags dynamical anomalies (values that break the expected
  temporal continuation of a process) using prediction residuals.
The hybrid combines them via a convex combination whose weight can be fixed
or learned on a held-out validation set.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from ai_cta.detection.isolation_forest_detector import (
    IsolationForestDetector,
)
from ai_cta.detection.lstm_detector import LSTMDetector

__all__ = ["HybridDetector"]

class HybridDetector(BaseEstimator):
    """Weighted combination of Isolation Forest and LSTM anomaly scores.
    Parameters
    ----------
    if_weight : float, default=0.5
        Weight assigned to the Isolation Forest score. The LSTM weight is
        1 - if_weight. Use `tune_weights` to fit this on labeled data.
    if_params : dict, optional
        Keyword arguments forwarded to IsolationForestDetector.
    lstm_params : dict, optional
        Keyword arguments forwarded to LSTMDetector.
    Notes
    -----
    The two detectors operate on different window/stride conventions; the
    hybrid aligns their output lengths by resampling the shorter stream
    using nearest-neighbor indexing.
    """
    def __init__(
        self,
        if_weight: float = 0.5,
        if_params: dict | None = None,
        lstm_params: dict | None = None,
    ):
        self.if_weight = if_weight
        self.if_params = if_params or {}
        self.lstm_params = lstm_params or {}
    def fit(self, X: pd.DataFrame, y=None) -> "HybridDetector":
        self._if = IsolationForestDetector(**self.if_params).fit(X)
        self._lstm = LSTMDetector(**self.lstm_params).fit(X)
        return self
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        if_scores = self._if.decision_function(X)
        lstm_scores = self._lstm.decision_function(X)
        if_scores_aligned, lstm_scores_aligned = self._align(if_scores, lstm_scores)
        weight = float(np.clip(self.if_weight, 0.0, 1.0))
        return weight * if_scores_aligned + (1.0 - weight) * lstm_scores_aligned
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.decision_function(X) >= threshold).astype(int)
    # ------------------------------------------------------------ alignment
    @staticmethod
    def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resample two score arrays to the same length via nearest neighbor."""
        if len(a) == len(b):
            return a, b
        target_len = min(len(a), len(b))
        def resample(x: np.ndarray, n: int) -> np.ndarray:
            idx = np.linspace(0, len(x) - 1, num=n).round().astype(int)
            return x[idx]
        return resample(a, target_len), resample(b, target_len)
    # --------------------------------------------------------- weight tuning
    def tune_weights(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        n_grid: int = 21,
    ) -> float:
        """Find the if_weight that maximizes F1 on a labeled validation set.
        Parameters
        ----------
        X_val : DataFrame
            Validation inputs.
        y_val : ndarray of 0/1
            Ground-truth labels, aligned to the detector output length.
        n_grid : int
            Number of weight values to try (uniformly spaced in [0, 1]).
        Returns
        -------
        best_weight : float
            Assigned to `self.if_weight` as a side effect.
        """
        from sklearn.metrics import f1_score
        original_weight = self.if_weight
        best_w, best_f1 = original_weight, -1.0
        for w in np.linspace(0.0, 1.0, n_grid):
            self.if_weight = float(w)
            preds = self.predict(X_val)
            n = min(len(preds), len(y_val))
            f1 = f1_score(y_val[:n], preds[:n], zero_division=0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_w = float(w)
        self.if_weight = best_w
        return best_w
