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
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

__all__ = ["LSTMDetector"]

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
    def fit(self, X: pd.DataFrame, y=None) -> "LSTMDetector":
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "LSTMDetector requires tensorflow. "
                "Install it via `pip install tensorflow`."
            ) from exc
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
        import tensorflow as tf
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
