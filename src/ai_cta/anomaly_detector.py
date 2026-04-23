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

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler

from ai_cta.preprocess import (
    FrequencyDomainFeatureExtractor,
    StatisticalFeatureExtractor,
)

# Lazy tensorflow import. Kept at module scope so that all methods of
# LSTMDetector (including _build_model) can reference `tf` without
# shadowing via a method-local re-import. When tensorflow is not
# installed, `tf` is None and LSTMDetector.fit() raises ImportError
# on first use.
try:
    import tensorflow as tf  # noqa: F401  (re-exposed for _build_model)
except ImportError:  # pragma: no cover - optional dependency
    tf = None  # type: ignore[assignment]


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
    def fit(self, X: pd.DataFrame, y=None) -> IsolationForestDetector:
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



"""LSTM-based anomaly detector via one-step-ahead prediction residuals.
The detector trains a stacked LSTM to predict the next sample of each sensor
channel given the preceding window. At inference time, anomalies are flagged
based on the magnitude of the prediction residual: a poorly-predicted future
sample indicates that the process has departed from the patterns seen during
training.
This formulation is particularly well-suited to slowly-degrading equipment
where a subtle shift in dynamics (rather than an outlier spike) marks the
onset of a fault.
References
----------
Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G.
(2016). LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.
ICML Anomaly Detection Workshop.
"""
class LSTMDetector(BaseEstimator):
    """Recurrent forecasting model for anomaly detection.
    Parameters
    ----------
    window_size : int, default=32
        Number of past samples used to predict the next one.
    lstm_units : tuple of int, default=(64, 32)
        Hidden size of each stacked LSTM layer.
    dropout : float, default=0.1
        Dropout applied after each LSTM layer.
    learning_rate : float, default=1e-3
        Adam optimizer learning rate.
    epochs : int, default=20
        Training epochs.
    batch_size : int, default=64
    validation_split : float, default=0.1
    patience : int, default=5
        Early-stopping patience on validation loss.
    random_state : int, default=42
    Notes
    -----
    TensorFlow is imported lazily so that the rest of the package remains
    usable without a deep-learning backend installed. An ImportError is
    raised only when `fit` is first called.
    """
    def __init__(
        self,
        window_size: int = 32,
        lstm_units: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 64,
        validation_split: float = 0.1,
        patience: int = 5,
        random_state: int = 42,
    ):
        self.window_size = window_size
        self.lstm_units = tuple(lstm_units)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.random_state = random_state
    # ------------------------------------------------------------- windowing
    def _windowize(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert (n_samples, n_channels) into sliding (X, y) pairs."""
        n = len(X)
        if n <= self.window_size:
            raise ValueError(
                f"Need at least {self.window_size + 1} samples; got {n}."
            )
        n_win = n - self.window_size
        xs = np.empty((n_win, self.window_size, X.shape[1]), dtype=np.float32)
        ys = np.empty((n_win, X.shape[1]), dtype=np.float32)
        for i in range(n_win):
            xs[i] = X[i : i + self.window_size]
            ys[i] = X[i + self.window_size]
        return xs, ys
    # ------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y=None) -> LSTMDetector:
        if tf is None:
            raise ImportError(
                "LSTMDetector requires tensorflow. "
                "Install it via `pip install tensorflow`."
            )
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        self.channels_ = list(X.select_dtypes(include="number").columns)
        if not self.channels_:
            raise ValueError("No numeric channels found in input.")
        values = X[self.channels_].to_numpy(dtype=np.float32)
        self._scaler = StandardScaler().fit(values)
        scaled = self._scaler.transform(values).astype(np.float32)
        xs, ys = self._windowize(scaled)
        self.model_ = self._build_model(n_channels=len(self.channels_))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=self.patience, restore_best_weights=True
            ),
        ]
        self.history_ = self.model_.fit(
            xs,
            ys,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=0,
        ).history
        # Calibrate residual distribution on training data (post-fit).
        residuals = self._residuals(scaled)
        self.residual_median_ = float(np.median(residuals))
        self.residual_mad_ = float(np.median(np.abs(residuals - self.residual_median_)) + 1e-9)
        return self
    def _build_model(self, n_channels: int):
        inputs = tf.keras.Input(shape=(self.window_size, n_channels))
        x = inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = tf.keras.layers.LSTM(
                units, return_sequences=return_sequences, dropout=self.dropout
            )(x)
        outputs = tf.keras.layers.Dense(n_channels)(x)
        model = tf.keras.Model(inputs, outputs, name="lstm_anomaly_forecaster")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mse",
        )
        return model
    # ----------------------------------------------------------- scoring
    def _residuals(self, scaled: np.ndarray) -> np.ndarray:
        """Per-sample L2 residual between predicted and observed values."""
        xs, ys = self._windowize(scaled)
        preds = self.model_.predict(xs, verbose=0)
        return np.linalg.norm(preds - ys, axis=1)
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores in [0, 1] for each predicted sample.
        Scores are a logistic transform of robust-standardized residuals
        (modified z-score based on the median absolute deviation).
        """
        self._check_fitted()
        values = X[self.channels_].to_numpy(dtype=np.float32)
        scaled = self._scaler.transform(values).astype(np.float32)
        residuals = self._residuals(scaled)
        # Modified z-score (Iglewicz & Hoaglin, 1993); robust to outliers.
        mz = 0.6745 * (residuals - self.residual_median_) / self.residual_mad_
        return 1.0 / (1.0 + np.exp(-mz))
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.decision_function(X) >= threshold).astype(int)
    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before predicting.")



"""Hybrid ensemble of Isolation Forest and LSTM detectors.
The two base detectors capture complementary anomaly modes:
- IsolationForestDetector flags point-wise structural anomalies (values that
  are unlike anything seen in training) using engineered features over windows.
- LSTMDetector flags dynamical anomalies (values that break the expected
  temporal continuation of a process) using prediction residuals.
The hybrid combines them via a convex combination whose weight can be fixed
or learned on a held-out validation set.
"""
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
    def fit(self, X: pd.DataFrame, y=None) -> HybridDetector:
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


__all__ = ["IsolationForestDetector", "LSTMDetector", "HybridDetector"]
