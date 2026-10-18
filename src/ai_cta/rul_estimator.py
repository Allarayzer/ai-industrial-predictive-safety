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
        self, X: np.ndarray, y: np.ndarray | None, groups: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Convert a (T, d) trajectory into sliding (window_size, d) inputs.
        For each window ending at time t, the target is the RUL at t.
        With groups, row order within each unit must already be chronological.
        """
        if groups is not None:
            groups = np.asarray(groups)
            if groups.ndim != 1 or len(groups) != len(X) or pd.isna(groups).any():
                raise ValueError("groups must contain one non-missing unit ID per row.")
            windows, targets = [], []
            for group in pd.unique(groups):
                mask = groups == group
                if mask.sum() < self.window_size:
                    continue
                xs, ys = self._make_windows(X[mask], None if y is None else y[mask])
                windows.append(xs)
                if ys is not None:
                    targets.append(ys)
            if not windows:
                raise ValueError(f"No group has at least {self.window_size} samples.")
            return np.concatenate(windows), np.concatenate(targets) if y is not None else None
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
        *,
        groups: np.ndarray | None = None,
        validation_data: tuple | None = None,
    ) -> RULEstimator:
        """Fit the quantile regressor.
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_channels)
            Numeric sensor channels.
        y : array-like of shape (n_samples,)
            RUL value for each timestep (in cycles or time units).
        groups : array-like, optional
            Unit IDs aligned with X. Windows never cross unit boundaries.
            Without explicit validation_data, validation_split reserves whole
            units before fitting the scaler and constructing windows.
        validation_data : tuple, optional
            (X_val, y_val) for a single trajectory, or (X_val, y_val, groups_val).
            Grouped fitting requires disjoint validation unit IDs. Explicit
            validation replaces validation_split and never fits the scaler.
        """
        self._validate_quantiles()
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "RULEstimator requires tensorflow. "
                "Install it via `pip install tensorflow`."
            ) from exc
        tf.keras.utils.set_random_seed(self.random_state)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        self.channels_ = list(X.select_dtypes(include="number").columns)
        if not self.channels_:
            raise ValueError("No numeric channels found in input.")
        values = X[self.channels_].to_numpy(dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr[:, 0]
        if y_arr.ndim != 1 or not np.isfinite(y_arr).all() or not np.isfinite(values).all():
            raise ValueError("X and one-dimensional y must contain finite values.")
        if len(y_arr) != len(values):
            raise ValueError(
                f"X has {len(values)} rows but y has {len(y_arr)}."
            )
        if groups is not None:
            groups = np.asarray(groups)
            if groups.ndim != 1 or len(groups) != len(values) or pd.isna(groups).any():
                raise ValueError("groups must contain one non-missing unit ID per row.")
        val_values = val_y = val_groups = None
        if validation_data is not None:
            if len(validation_data) not in (2, 3):
                raise ValueError("validation_data must be (X_val, y_val[, groups_val]).")
            val_X, val_y = validation_data[:2]
            if not isinstance(val_X, pd.DataFrame) or not set(self.channels_).issubset(val_X.columns):
                raise ValueError("Validation X must be a DataFrame with the training channels.")
            val_values = val_X[self.channels_].to_numpy(dtype=np.float32)
            val_y = np.asarray(val_y, dtype=np.float32)
            if val_y.ndim == 2 and val_y.shape[1] == 1:
                val_y = val_y[:, 0]
            if val_y.ndim != 1 or not np.isfinite(val_y).all() or not np.isfinite(val_values).all():
                raise ValueError("Validation X and one-dimensional y must contain finite values.")
            if len(val_y) != len(val_values):
                raise ValueError("Validation X and y must have the same number of rows.")
            if len(validation_data) == 3:
                val_groups = np.asarray(validation_data[2])
                if (val_groups.ndim != 1 or len(val_groups) != len(val_values)
                        or pd.isna(val_groups).any()):
                    raise ValueError("Validation groups need one non-missing unit ID per row.")
            if groups is not None:
                if val_groups is None:
                    raise ValueError("Grouped validation needs disjoint validation group IDs.")
                if np.isin(groups, val_groups).any():
                    raise ValueError("Training and validation groups must be disjoint.")
        elif groups is not None and self.validation_split > 0:
            if not 0 < self.validation_split < 1:
                raise ValueError("validation_split must be between 0 and 1.")
            units = pd.unique(groups).copy()
            if len(units) < 2:
                raise ValueError("Grouped validation needs at least two disjoint units.")
            np.random.default_rng(self.random_state).shuffle(units)
            n_val = min(len(units) - 1, max(1, int(np.ceil(len(units) * self.validation_split))))
            is_val = np.isin(groups, units[:n_val])
            val_values, val_y, val_groups = values[is_val], y_arr[is_val], groups[is_val]
            values, y_arr, groups = values[~is_val], y_arr[~is_val], groups[~is_val]
        # Clip RUL — large values are not useful for prediction and bias
        # the loss toward the easy-to-predict early portion of life.
        y_arr = np.minimum(y_arr, self.rul_max)
        self._scaler = StandardScaler().fit(values)
        scaled = self._scaler.transform(values).astype(np.float32)
        xs, ys = self._make_windows(scaled, y_arr, groups)
        fit_validation = {}
        if val_values is not None:
            val_scaled = self._scaler.transform(val_values).astype(np.float32)
            val_windows, val_targets = self._make_windows(
                val_scaled, np.minimum(val_y, self.rul_max), val_groups
            )
            fit_validation["validation_data"] = (val_windows, val_targets)
        else:
            fit_validation["validation_split"] = self.validation_split
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
            **fit_validation,
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
            # Keras can supply (batch,) or (batch, 1) targets. Preserve
            # sample pairing rather than broadcasting across the batch.
            y_true = tf.reshape(tf.cast(y_true, y_pred.dtype), (-1, 1))
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
        """Compute a CDF-like R_RUL(t; horizon) score from predicted quantiles.
        Implements the risk component R_RUL described in monograph § 8.3.2.
        Interpolate between finite, nonnegative, nondecreasing quantile
        predictions. Below the first quantile, interpolate from (0, 0).
        At or above the last quantile, keep its probability level: the
        unmodelled upper tail does not justify jumping to probability one.
        Tied predictions use the rightmost quantile level at that value.
        These tail conventions do not provide probability calibration.
        Parameters
        ----------
        X : DataFrame
            Sensor input.
        horizon : float
            Time horizon (in same units as the training RUL labels).
        Returns
        -------
        risk : ndarray of shape (n_windows,)
            CDF-like scores in [0, 1] at `horizon` time units.
        """
        if not np.isfinite(horizon) or horizon < 0:
            raise ValueError("horizon must be finite and non-negative.")
        preds = self.predict_quantiles(X)
        sorted_qs = sorted(preds.keys())
        if not sorted_qs or any(not 0 < q < 1 for q in sorted_qs):
            raise ValueError("Predicted quantile levels must be in (0, 1).")
        columns = []
        for q in sorted_qs:
            values = np.asarray(preds[q], dtype=float)
            if values.ndim == 2 and values.shape[1] == 1:
                values = values[:, 0]
            if values.ndim != 1:
                raise ValueError("Quantile predictions must have shape (n,) or (n, 1).")
            columns.append(values)
        if len({len(values) for values in columns}) != 1:
            raise ValueError("Quantile predictions must have equal length.")
        quantile_values = np.column_stack(columns)
        if not np.all(np.isfinite(quantile_values)) or np.any(quantile_values < 0):
            raise ValueError("Quantile predictions must be finite and non-negative.")
        if np.any(np.diff(quantile_values, axis=1) < 0):
            raise ValueError("Quantile predictions must be nondecreasing; found crossing quantiles.")
        n = len(quantile_values)
        risk = np.zeros(n, dtype=np.float32)
        for i, quant_vals in enumerate(quantile_values):
            if horizon < quant_vals[0]:
                p = float(sorted_qs[0]) * horizon / quant_vals[0]
            else:
                # np.interp uses the rightmost level for ties and keeps the
                # last level beyond the observed quantiles (upper plateau).
                p = float(np.interp(horizon, quant_vals, sorted_qs))
            risk[i] = float(np.clip(p, 0.0, 1.0))
        return risk
    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before predicting.")
