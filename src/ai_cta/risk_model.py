"""Composite risk scoring for industrial monitoring.
The RiskScorer aggregates a data-driven anomaly score with interpretable
physical indicators (threshold exceedances for temperature, vibration,
pressure, and similar channels) into a single risk level.
This two-track design reflects the requirements of safety-critical
deployments: the ML component captures subtle patterns that rules miss,
while the rule-based component remains auditable and aligns with existing
operational limits.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

# Lazy tensorflow import at module scope so NeuralRiskEstimator's
# private methods (`_build_model`, `_asymmetric_bce`) can reference `tf`
# without needing to re-import inside each method. Set to None when
# tensorflow is not installed; methods that require it check and
# raise ImportError at call time.
try:
    import tensorflow as tf  # noqa: F401  (used in NeuralRiskEstimator methods)
except ImportError:  # pragma: no cover - optional dependency
    tf = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ChannelLimits:
    """Operational limits for a single sensor channel.
    Values below `warn_low` or above `warn_high` raise the rule-based risk;
    values outside `alarm_low`/`alarm_high` raise it further.
    """
    warn_low: float
    warn_high: float
    alarm_low: float
    alarm_high: float
    def exceedance(self, value: float) -> float:
        """Return a normalized rule-based risk contribution in [0, 1].
        0.0 means within warning band; values up to 1.0 indicate proximity to
        or violation of the alarm thresholds.
        """
        if self.alarm_low < value < self.alarm_high:
            # Inside alarm band: linear ramp from warn to alarm.
            if value < self.warn_low:
                span = max(self.warn_low - self.alarm_low, 1e-9)
                return float(np.clip((self.warn_low - value) / span, 0.0, 1.0))
            if value > self.warn_high:
                span = max(self.alarm_high - self.warn_high, 1e-9)
                return float(np.clip((value - self.warn_high) / span, 0.0, 1.0))
            return 0.0
        # Outside alarm band: saturated at 1.0.
        return 1.0

@dataclass(frozen=True)
class RiskLevel:
    """Discretized risk category."""
    name: str
    lower: float
    upper: float

class RiskScorer:
    """Combine ML anomaly scores with physical threshold exceedances.
    Parameters
    ----------
    ml_weight : float, default=0.6
        Weight assigned to the ML anomaly score in [0, 1]. The remaining
        weight is distributed uniformly across the configured channel limits.
    limits : mapping of str -> ChannelLimits, optional
        Operational limits per channel. If empty, the risk score equals the
        ML score.
    levels : sequence of RiskLevel, optional
        Discrete categories. Defaults to LOW / MEDIUM / HIGH / CRITICAL.
    Examples
    --------
    >>> scorer = RiskScorer(
    ...     ml_weight=0.6,
    ...     limits={
    ...         "temperature": ChannelLimits(40, 80, 20, 100),
    ...         "vibration": ChannelLimits(0.1, 0.5, 0.0, 0.8),
    ...     },
    ... )
    >>> score = scorer.score(
    ...     anomaly_score=0.3,
    ...     channel_values={"temperature": 85.0, "vibration": 0.6},
    ... )
    """
    # Risk-level bands follow the four-level convention defined in the
    # accompanying monograph (§ 8.4.3): OK / Warning / Critical / Emergency.
    DEFAULT_LEVELS = (
        RiskLevel("OK", 0.0, 0.3),
        RiskLevel("Warning", 0.3, 0.6),
        RiskLevel("Critical", 0.6, 0.85),
        RiskLevel("Emergency", 0.85, 1.01),
    )
    def __init__(
        self,
        ml_weight: float = 0.6,
        limits: Mapping[str, ChannelLimits] | None = None,
        levels: tuple[RiskLevel, ...] = DEFAULT_LEVELS,
    ):
        if not 0.0 <= ml_weight <= 1.0:
            raise ValueError("ml_weight must be in [0, 1].")
        self.ml_weight = ml_weight
        self.limits = dict(limits) if limits else {}
        self.levels = tuple(levels)
    def score(
        self,
        anomaly_score: float,
        channel_values: Mapping[str, float] | None = None,
    ) -> float:
        """Compute a composite risk score in [0, 1]."""
        score = float(self.ml_weight) * float(np.clip(anomaly_score, 0.0, 1.0))
        if self.limits and channel_values:
            rule_weight = 1.0 - self.ml_weight
            active = [
                self.limits[name].exceedance(value)
                for name, value in channel_values.items()
                if name in self.limits
            ]
            if active:
                score += rule_weight * float(np.mean(active))
        return float(np.clip(score, 0.0, 1.0))
    def level(self, score: float) -> str:
        """Map a numeric score to a discrete risk level name."""
        for lvl in self.levels:
            if lvl.lower <= score < lvl.upper:
                return lvl.name
        return self.levels[-1].name



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
    def calibrate(self, calibration_scores: np.ndarray) -> ConformalThresholdCalibrator:
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



"""Three-component risk aggregator with SLSQP weight calibration.
Implements the hybrid risk function R_final defined in Chapter 8.4 of
the monograph and the corresponding `RiskAggregator` class shown in
§ 10.7. Combines three risk signals:
- R_anom: instantaneous anomaly score (e.g., from IsolationForestDetector
          calibrated to a probability via Platt scaling).
- R_RUL:  remaining-useful-life-derived risk in a fixed horizon
          (from `RULEstimator.risk_at_horizon`).
- R_NN:   contextualized neural risk
          (from `NeuralRiskEstimator.predict_proba`).
Final score:
    R_final(t) = w_1 R_anom(t) + w_2 R_RUL(t; τ) + w_3 R_NN(t)
with w_i ≥ 0, Σ w_i = 1.
Weights are calibrated by minimizing an asymmetric BCE loss on a
labeled validation set, subject to the simplex constraint, via SLSQP.
"""
class RiskAggregator:
    """Linear combination of three risk components with calibrated weights.
    Parameters
    ----------
    weights : sequence of float, default=(1/3, 1/3, 1/3)
        Initial weights for (R_anom, R_RUL, R_NN). Re-fit by
        `calibrate_weights`.
    thresholds : tuple of float, default=(0.3, 0.6, 0.85)
        Boundary values mapping R_final to alert levels
        (OK / Warning / Critical / Emergency), per § 8.4.3.
    cost_fn : float, default=10.0
        False-negative cost in the weight-calibration loss.
    cost_fp : float, default=1.0
        False-positive cost.
    Attributes
    ----------
    w : ndarray of shape (3,)
        Current aggregation weights.
    """
    LEVELS = ("OK", "Warning", "Critical", "Emergency")
    def __init__(
        self,
        weights: tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3),
        thresholds: tuple[float, float, float] = (0.3, 0.6, 0.85),
        cost_fn: float = 10.0,
        cost_fp: float = 1.0,
    ):
        w = np.asarray(weights, dtype=float)
        if w.shape != (3,):
            raise ValueError("weights must have length 3.")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")
        s = w.sum()
        if s <= 0:
            raise ValueError("weights must sum to a positive value.")
        self.w = w / s
        if len(thresholds) != 3:
            raise ValueError("thresholds must have length 3.")
        if not (0.0 < thresholds[0] < thresholds[1] < thresholds[2] < 1.0):
            raise ValueError("thresholds must be strictly increasing in (0, 1).")
        self.thresholds = tuple(float(t) for t in thresholds)
        if cost_fn <= 0 or cost_fp <= 0:
            raise ValueError("Cost weights must be positive.")
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
    # --------------------------------------------------- aggregation
    def aggregate(
        self,
        R_anom: np.ndarray,
        R_RUL: np.ndarray,
        R_NN: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute R_final and per-sample alert level.
        Returns
        -------
        R_final : ndarray of shape (n,)
            Aggregated risk in [0, 1].
        alert : ndarray of shape (n,) with dtype object
            Alert level per sample.
        """
        anom, rul, nn = self._align_three(R_anom, R_RUL, R_NN)
        stacked = np.stack([anom, rul, nn], axis=1)
        R_final = stacked @ self.w
        R_final = np.clip(R_final, 0.0, 1.0)
        return R_final, self._classify(R_final)
    def _classify(self, R: np.ndarray) -> np.ndarray:
        labels = np.full(R.shape, self.LEVELS[0], dtype=object)
        labels[self.thresholds[0] <= R] = self.LEVELS[1]
        labels[self.thresholds[1] <= R] = self.LEVELS[2]
        labels[self.thresholds[2] <= R] = self.LEVELS[3]
        return labels
    @staticmethod
    def _align_three(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
        target = min(len(a) for a in arrays)
        def resample(x: np.ndarray, n: int) -> np.ndarray:
            if len(x) == n:
                return x
            idx = np.linspace(0, len(x) - 1, n).round().astype(int)
            return x[idx]
        return tuple(resample(np.asarray(a, dtype=float), target) for a in arrays)
    # --------------------------------------------- weight calibration
    def calibrate_weights(
        self,
        R_anom_val: np.ndarray,
        R_RUL_val: np.ndarray,
        R_NN_val: np.ndarray,
        y_val: np.ndarray,
    ) -> np.ndarray:
        """Find w on the 2-simplex that minimizes asymmetric BCE.
        Parameters
        ----------
        R_anom_val, R_RUL_val, R_NN_val : ndarray
            Per-sample risk components on a labeled validation set.
        y_val : ndarray of 0/1
            Binary failure labels.
        Returns
        -------
        w : ndarray of shape (3,)
            Calibrated weights, also assigned to `self.w`.
        """
        anom, rul, nn = self._align_three(R_anom_val, R_RUL_val, R_NN_val)
        y = np.asarray(y_val, dtype=float)
        if len(y) != len(anom):
            idx = np.linspace(0, len(y) - 1, len(anom)).round().astype(int)
            y = y[idx]
        stacked = np.stack([anom, rul, nn], axis=1)
        eps = 1e-9
        def loss(w: np.ndarray) -> float:
            w = np.clip(w, 0, None)
            s = w.sum() + eps
            w = w / s
            R = np.clip(stacked @ w, eps, 1.0 - eps)
            loss_per_sample = -(
                self.cost_fn * y * np.log(R)
                + self.cost_fp * (1.0 - y) * np.log(1.0 - R)
            )
            return float(loss_per_sample.mean())
        cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1)},)
        bnds = [(0.0, 1.0)] * 3
        x0 = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        res = minimize(loss, x0, method="SLSQP", bounds=bnds, constraints=cons)
        if not res.success:
            # Keep prior weights but warn via attribute; do not raise
            # because gradient methods occasionally fail on flat regions.
            self.last_optimization_warning_ = res.message
        else:
            self.last_optimization_warning_ = None
        self.w = res.x / res.x.sum()
        return self.w



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
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> NeuralRiskEstimator:
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
        if tf is None:
            raise ImportError(
                "NeuralRiskEstimator requires tensorflow. "
                "Install it via `pip install tensorflow`."
            )
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


__all__ = ["RiskScorer", "ChannelLimits", "RiskLevel", "ConformalThresholdCalibrator", "RiskAggregator", "NeuralRiskEstimator"]
