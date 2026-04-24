"""Remaining Useful Life (RUL) estimation via quantile regression.
Implements Model B of the three-model ensemble specified in Chapter 9.6
and 10.6 of the accompanying monograph: an LSTM-based quantile regressor
that produces a predictive distribution over the remaining useful life
of a unit at each timestep, rather than a single point estimate.

Companion implementations of Models A (XGBoost quantile) and C (Weibull
physics-guided) live in ai_cta.rul_ensemble; the full stacked ensemble
is :class:`ai_cta.rul_ensemble.RULEnsemble`. The benchmark driver
`benchmarks/run_cmapss_rul_normalized.py --full-ensemble` runs all three
jointly and reports pinball-optimised stacked predictions.

The monograph's headline RUL figures (Table 13.3) are produced by this
module in isolation (Model B only). The full 3-model ensemble is
reported separately in `docs/benchmarks.md` and Appendix 13.A.

The quantile formulation provides naturally calibrated confidence
intervals — the 10th and 90th percentile predictions bracket the true
RUL with the configured nominal coverage when the model is well-fit.
Mathematical formulation
------------------------
Given a feature vector x_t at time t, predict the conditional RUL
distribution F_RUL(τ | x_{1:t}). The model is trained to minimize the
pinball loss for a set of quantiles {α_1, ..., α_K}:
    L(y, ŷ) = Σ_k max(α_k · (y - ŷ_k), (α_k - 1) · (y - ŷ_k))
The risk component R_RUL(t; τ) defined in the monograph (§ 8.3.2) is
recovered from the quantile predictions via empirical CDF interpolation.
References
----------
Koenker, R., & Bassett, G. (1978). Regression quantiles. Econometrica.
Tagasovska, N., & Lopez-Paz, D. (2019). Single-Model Uncertainties for
Deep Learning. NeurIPS.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

__all__ = ["RULEstimator"]

class RULEstimator(BaseEstimator):
    """LSTM quantile regressor for remaining useful life prediction.
    Parameters
    ----------
    window_size : int, default=32
        Number of past timesteps used as model input.
    quantiles : sequence of float, default=(0.1, 0.5, 0.9)
        Quantiles to estimate. Must be in (0, 1).
    rul_max : float, default=125.0
        Upper bound on RUL values during training (the standard C-MAPSS
        convention; values above this are clipped before fitting).
    lstm_units : tuple of int, default=(64, 32)
        Hidden size of each stacked LSTM layer.
    dropout : float, default=0.2
        Dropout applied between LSTM layers.
    learning_rate : float, default=1e-3
    epochs : int, default=30
    batch_size : int, default=64
    validation_split : float, default=0.15
    patience : int, default=8
        Early-stopping patience on validation loss.
    random_state : int, default=42
    Attributes
    ----------
    quantile_index_ : dict
        Maps quantile value to its column index in model output.
    Notes
    -----
    TensorFlow is imported lazily so the package remains importable
    without a deep-learning backend.
    """
    def __init__(
        self,
        window_size: int = 32,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        rul_max: float = 125.0,
        lstm_units: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 64,
        validation_split: float = 0.15,
        patience: int = 8,
        random_state: int = 42,
    ):
        self.window_size = window_size
        self.quantiles = tuple(quantiles)
        self.rul_max = rul_max
        self.lstm_units = tuple(lstm_units)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.random_state = random_state
    # ----------------------------------------------------- preprocessing
    def _validate_quantiles(self) -> None:
        for q in self.quantiles:
            if not 0.0 < q < 1.0:
                raise ValueError(f"Quantiles must be in (0, 1); got {q}.")
        if list(self.quantiles) != sorted(self.quantiles):
            raise ValueError("Quantiles must be sorted ascending.")
    def _make_windows(
        self, X: np.ndarray, y: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Convert a (T, d) trajectory into sliding (window_size, d) inputs.
        For each window ending at time t, the target is the RUL at t.
        """
        n = len(X)
        if n < self.window_size:
            raise ValueError(
                f"Need at least {self.window_size} samples; got {n}."
            )
        n_win = n - self.window_size + 1
        xs = np.empty((n_win, self.window_size, X.shape[1]), dtype=np.float32)
        for i in range(n_win):
            xs[i] = X[i : i + self.window_size]
        if y is None:
            return xs, None
        ys = y[self.window_size - 1 :].astype(np.float32)
        return xs, ys
    # ------------------------------------------------------------- fit
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> RULEstimator:
        """Fit the quantile regressor.
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_channels)
            Numeric sensor channels.
        y : array-like of shape (n_samples,)
            RUL value for each timestep (in cycles or time units).
        """
        self._validate_quantiles()
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "RULEstimator requires tensorflow. "
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
        y_arr = np.asarray(y, dtype=np.float32)
        if len(y_arr) != len(values):
            raise ValueError(
                f"X has {len(values)} rows but y has {len(y_arr)}."
            )
        # Clip RUL — large values are not useful for prediction and bias
        # the loss toward the easy-to-predict early portion of life.
        y_arr = np.minimum(y_arr, self.rul_max)
        self._scaler = StandardScaler().fit(values)
        scaled = self._scaler.transform(values).astype(np.float32)
        xs, ys = self._make_windows(scaled, y_arr)
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
        self.quantile_index_ = {q: i for i, q in enumerate(self.quantiles)}
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
        outputs = tf.keras.layers.Dense(len(self.quantiles))(x)
        model = tf.keras.Model(inputs, outputs, name="lstm_quantile_rul")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=self._pinball_loss(),
        )
        return model
    def _pinball_loss(self):
        """Multi-quantile pinball loss.
        For each quantile α, the loss for a single observation is
        max(α·(y − ŷ), (α−1)·(y − ŷ)). Total loss is the mean over all
        quantiles and observations.
        """
        import tensorflow as tf
        alphas = tf.constant(self.quantiles, dtype=tf.float32)
        def loss(y_true, y_pred):
            # y_true: (batch,); y_pred: (batch, n_quantiles)
            y_true = tf.expand_dims(y_true, axis=-1)
            err = y_true - y_pred
            return tf.reduce_mean(
                tf.maximum(alphas * err, (alphas - 1.0) * err)
            )
        return loss
    # --------------------------------------------------------- predict
    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        """Return per-quantile RUL predictions.
        Returns
        -------
        preds : dict mapping quantile -> ndarray
            For each configured quantile, an array of predicted RUL
            values (one per window).
        """
        self._check_fitted()
        values = X[self.channels_].to_numpy(dtype=np.float32)
        scaled = self._scaler.transform(values).astype(np.float32)
        xs, _ = self._make_windows(scaled, None)
        raw = self.model_.predict(xs, verbose=0)
        return {q: raw[:, i] for i, q in enumerate(self.quantiles)}
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return median (50th percentile) RUL prediction per window.
        If the median quantile is not in `self.quantiles`, returns the
        average of all configured quantiles.
        """
        preds = self.predict_quantiles(X)
        if 0.5 in preds:
            return preds[0.5]
        return np.mean(list(preds.values()), axis=0)
    def risk_at_horizon(
        self,
        X: pd.DataFrame,
        horizon: float,
    ) -> np.ndarray:
        """Compute R_RUL(t; horizon) — the probability that RUL ≤ horizon.
        Implements the risk component R_RUL described in monograph § 8.3.2.
        Estimated via piecewise-linear interpolation of the empirical CDF
        defined by the predicted quantiles.
        Parameters
        ----------
        X : DataFrame
            Sensor input.
        horizon : float
            Time horizon (in same units as the training RUL labels).
        Returns
        -------
        risk : ndarray of shape (n_windows,)
            Probabilities in [0, 1] that the unit will fail within
            `horizon` time units.
        """
        if horizon < 0:
            raise ValueError("horizon must be non-negative.")
        preds = self.predict_quantiles(X)
        sorted_qs = sorted(preds.keys())
        n = len(next(iter(preds.values())))
        risk = np.zeros(n, dtype=np.float32)
        for i in range(n):
            # quantile values for this window, in ascending quantile order
            quant_vals = np.array([preds[q][i] for q in sorted_qs])
            # P(RUL <= horizon) is interpolated against the inverse CDF.
            # quant_vals are RUL values at quantile levels sorted_qs.
            # If horizon is below the smallest quant_val, P~0; above the
            # largest, P~1; in between, linear interpolation.
            if horizon <= quant_vals[0]:
                p = float(sorted_qs[0]) * (horizon / max(quant_vals[0], 1e-9))
            elif horizon >= quant_vals[-1]:
                p = 1.0 - (1.0 - float(sorted_qs[-1])) * (
                    1.0 - min(horizon / max(quant_vals[-1], 1e-9), 1.0)
                )
                p = min(p, 1.0)
            else:
                p = float(np.interp(horizon, quant_vals, sorted_qs))
            risk[i] = float(np.clip(p, 0.0, 1.0))
        return risk
    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before predicting.")
