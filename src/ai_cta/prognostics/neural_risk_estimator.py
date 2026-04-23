"""Neural risk estimator for context-aware integrated risk.
Implements the third risk component R_NN defined in Chapter 8.3.3 and
implemented in Chapter 9.7 of the monograph: a feedforward neural
network trained on labeled historical data to map (recent telemetry,
operational context, asset history) to the probability of failure
within a fixed horizon.
This component complements the unsupervised anomaly score (which fires
on out-of-distribution states) and the RUL estimator (which captures
gradual degradation) by exploiting the supervised signal of past
incidents — patterns that may be subtle in the raw telemetry but become
informative when combined with operating mode, asset age, and recent
maintenance history.
Loss function
-------------
Class-asymmetric binary cross-entropy as in monograph § 8.3.3:
    L = − Σ [c_FN · y · log(g(x))  +  c_FP · (1 − y) · log(1 − g(x))]
with c_FN ≫ c_FP, reflecting the higher cost of missed accidents.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

__all__ = ["NeuralRiskEstimator"]

class NeuralRiskEstimator(BaseEstimator):
    """Feedforward NN that estimates contextualized failure probability.
    Parameters
    ----------
    hidden_units : tuple of int, default=(128, 64, 32)
        Hidden layer sizes.
    dropout : float, default=0.3
    learning_rate : float, default=1e-3
    epochs : int, default=40
    batch_size : int, default=128
    validation_split : float, default=0.15
    patience : int, default=10
    cost_fn : float, default=10.0
        Cost weight for false negatives in the asymmetric BCE loss.
    cost_fp : float, default=1.0
        Cost weight for false positives.
    random_state : int, default=42
    """
    def __init__(
        self,
        hidden_units: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        epochs: int = 40,
        batch_size: int = 128,
        validation_split: float = 0.15,
        patience: int = 10,
        cost_fn: float = 10.0,
        cost_fp: float = 1.0,
        random_state: int = 42,
    ):
        self.hidden_units = tuple(hidden_units)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.random_state = random_state
    # ------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "NeuralRiskEstimator":
        """Fit the neural risk estimator on labeled data.
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Per-sample feature vector. Typically includes engineered
            telemetry features, operational context (mode, conditions),
            and asset history (age, time since last maintenance).
        y : array-like of 0/1
            Binary label: 1 if failure occurred within the target horizon
            after this sample, 0 otherwise.
        """
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "NeuralRiskEstimator requires tensorflow. "
                "Install it via `pip install tensorflow`."
            ) from exc
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        self.feature_names_ = list(X.select_dtypes(include="number").columns)
        if not self.feature_names_:
            raise ValueError("No numeric features found in input.")
        values = X[self.feature_names_].to_numpy(dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if len(y_arr) != len(values):
            raise ValueError(
                f"X has {len(values)} rows but y has {len(y_arr)}."
            )
        if not set(np.unique(y_arr)).issubset({0.0, 1.0}):
            raise ValueError("y must contain only 0/1 labels.")
        self._scaler = StandardScaler().fit(values)
        scaled = self._scaler.transform(values).astype(np.float32)
        self.model_ = self._build_model(n_features=len(self.feature_names_))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=self.patience,
                restore_best_weights=True,
                monitor="val_loss",
            ),
        ]
        self.history_ = self.model_.fit(
            scaled,
            y_arr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=0,
        ).history
        return self
    def _build_model(self, n_features: int):
        import tensorflow as tf
        inputs = tf.keras.Input(shape=(n_features,))
        x = inputs
        for units in self.hidden_units:
            x = tf.keras.layers.Dense(units, activation="relu")(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs, name="neural_risk_estimator")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=self._asymmetric_bce(),
            metrics=["binary_accuracy"],
        )
        return model
    def _asymmetric_bce(self):
        """Cost-asymmetric binary cross-entropy (monograph § 8.3.3)."""
        import tensorflow as tf
        cost_fn = float(self.cost_fn)
        cost_fp = float(self.cost_fp)
        eps = 1e-7
        def loss(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
            pos = cost_fn * y_true * tf.math.log(y_pred)
            neg = cost_fp * (1.0 - y_true) * tf.math.log(1.0 - y_pred)
            return -tf.reduce_mean(pos + neg)
        return loss
    # --------------------------------------------------------- predict
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return failure probability per sample, in [0, 1]."""
        self._check_fitted()
        values = X[self.feature_names_].to_numpy(dtype=np.float32)
        scaled = self._scaler.transform(values).astype(np.float32)
        return self.model_.predict(scaled, verbose=0).flatten()
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before predicting.")
