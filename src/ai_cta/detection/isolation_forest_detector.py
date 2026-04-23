"""Isolation Forest anomaly detector with integrated feature engineering.
This module wraps scikit-learn's IsolationForest with additional capabilities
useful for industrial time-series data:
- Automatic feature extraction from raw multivariate sensor streams
- Calibrated anomaly scores (mapped to [0, 1] via logistic transformation)
- Sliding-window inference for streaming data
- Optional contamination estimation via the classical approach of Liu et al.
References
----------
Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest.
In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422).
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from ai_cta.features.extractors import (
    StatisticalFeatureExtractor,
    FrequencyDomainFeatureExtractor,
)

__all__ = ["IsolationForestDetector"]

class IsolationForestDetector(BaseEstimator):
    """Unsupervised anomaly detector for industrial sensor streams.
    The detector builds a feature pipeline (statistical + optionally spectral
    descriptors) over sliding windows, scales features robustly, and trains
    an Isolation Forest on the result. At inference time, anomaly scores are
    mapped to the unit interval through a logistic transform calibrated on
    training scores.
    Parameters
    ----------
    window_size : int, default=64
        Length of each analysis window, in samples.
    stride : int, default=32
        Step between consecutive windows.
    sampling_rate : float, default=1.0
        Sampling rate of the input signal; used for spectral features.
    use_spectral : bool, default=True
        Whether to include frequency-domain features.
    contamination : float or "auto", default=0.05
        Expected fraction of anomalies in training data. Passed to
        IsolationForest.
    n_estimators : int, default=200
        Number of base trees in the Isolation Forest.
    random_state : int, default=42
        Seed for reproducibility.
    Attributes
    ----------
    feature_names_ : list of str
        Names of engineered features, populated after `fit`.
    calibration_mean_, calibration_std_ : float
        Mean and standard deviation of raw scores on training data,
        used to calibrate output scores.
    """
    def __init__(
        self,
        window_size: int = 64,
        stride: int = 32,
        sampling_rate: float = 1.0,
        use_spectral: bool = True,
        contamination: float | str = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.window_size = window_size
        self.stride = stride
        self.sampling_rate = sampling_rate
        self.use_spectral = use_spectral
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
    # ------------------------------------------------------------------ utils
    def _windows(self, X: pd.DataFrame) -> list[pd.DataFrame]:
        """Split a time series into (possibly overlapping) windows."""
        n = len(X)
        if n < self.window_size:
            raise ValueError(
                f"Input has {n} samples but window_size={self.window_size}."
            )
        return [
            X.iloc[i : i + self.window_size].reset_index(drop=True)
            for i in range(0, n - self.window_size + 1, self.stride)
        ]
    def _engineer(self, X: pd.DataFrame) -> np.ndarray:
        """Apply feature extractors to all windows of X."""
        windows = self._windows(X)
        rows: list[np.ndarray] = []
        for w in windows:
            parts = [self._stat.transform(w)]
            if self.use_spectral:
                parts.append(self._spec.transform(w))
            rows.append(np.concatenate(parts, axis=1))
        return np.vstack(rows)
    # ------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y=None) -> "IsolationForestDetector":
        """Fit feature extractors, scaler, and Isolation Forest on X."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        channels = list(X.select_dtypes(include="number").columns)
        if not channels:
            raise ValueError("No numeric channels found in input.")
        self._stat = StatisticalFeatureExtractor(channels=channels).fit(X)
        if self.use_spectral:
            self._spec = FrequencyDomainFeatureExtractor(
                sampling_rate=self.sampling_rate, channels=channels
            ).fit(X)
        features = self._engineer(X)
        self._scaler = RobustScaler().fit(features)
        features_scaled = self._scaler.transform(features)
        self._iforest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(features_scaled)
        # Calibrate raw scores on the training set.
        raw = -self._iforest.score_samples(features_scaled)
        self.calibration_mean_ = float(np.mean(raw))
        self.calibration_std_ = float(np.std(raw) + 1e-9)
        self.feature_names_ = list(self._stat.get_feature_names_out())
        if self.use_spectral:
            self.feature_names_ += list(self._spec.get_feature_names_out())
        self.n_windows_fit_ = len(features)
        return self
    # --------------------------------------------------------------- predict
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated anomaly scores in [0, 1] per window.
        A score close to 1 indicates a strong anomaly; close to 0 indicates
        normal operation.
        """
        self._check_fitted()
        features = self._engineer(X)
        features_scaled = self._scaler.transform(features)
        raw = -self._iforest.score_samples(features_scaled)
        z = (raw - self.calibration_mean_) / self.calibration_std_
        # Logistic squashing: keeps the score in (0, 1).
        return 1.0 / (1.0 + np.exp(-z))
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction: 1 for anomaly, 0 for normal."""
        scores = self.decision_function(X)
        return (scores >= threshold).astype(int)
    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """Alias for `decision_function` matching scikit-learn conventions."""
        return self.decision_function(X)
    # ---------------------------------------------------------------- utils
    def _check_fitted(self) -> None:
        if not hasattr(self, "_iforest"):
            raise RuntimeError("Call fit() before predicting.")
