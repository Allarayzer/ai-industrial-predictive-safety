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
from fractions import Fraction
from math import ceil

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


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Normalize a vector or single-column array without cross-sample broadcasting."""
    result = np.asarray(values, dtype=float)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a nonempty array of shape (n,) or (n, 1).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


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
        if not np.isfinite(value):
            raise ValueError("Channel values must be finite.")
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
    missing_channels : {"raise", "legacy_ignore"}, default="raise"
        Require all configured rule channels. The explicit legacy policy
        ignores missing channels and averages only the rules that are present;
        that changes the score's meaning and requires separate calibration.
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
        missing_channels: str = "raise",
    ):
        if not 0.0 <= ml_weight <= 1.0:
            raise ValueError("ml_weight must be in [0, 1].")
        if missing_channels not in {"raise", "legacy_ignore"}:
            raise ValueError("missing_channels must be 'raise' or 'legacy_ignore'.")
        self.ml_weight = ml_weight
        self.limits = dict(limits) if limits else {}
        self.levels = tuple(levels)
        self.missing_channels = missing_channels
    def score(
        self,
        anomaly_score: float,
        channel_values: Mapping[str, float] | None = None,
    ) -> float:
        """Compute a composite risk score in [0, 1]."""
        if not np.isfinite(anomaly_score):
            raise ValueError("anomaly_score must be finite.")
        if not self.limits:
            return float(np.clip(anomaly_score, 0.0, 1.0))
        channel_values = channel_values if channel_values is not None else {}
        missing = self.limits.keys() - channel_values.keys()
        if missing and self.missing_channels == "raise":
            raise ValueError(f"Missing configured rule channels: {sorted(missing)}.")
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
        Target marginal false-alarm rate for a new normal observation under
        exchangeability with the held-out calibration scores. The threshold
        is the k-th observed order statistic, k=ceil((n+1)*(1-alpha)); ties
        are covered because only strictly larger scores trigger an alarm.
        The rank uses exact arithmetic on the decimal representation
        ``str(alpha)``, without a tolerance that could round the rank down.
        A finite threshold requires k <= n; smaller samples are rejected.
        No conditional, time-series, or adaptive guarantee is implied.
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
        """Select the finite-sample conformal order statistic without interpolation.
        Parameters
        ----------
        calibration_scores : ndarray
            Scores produced by the detector on known-normal data. Should NOT
            overlap with the detector's training data.
        """
        scores = np.asarray(calibration_scores, dtype=float)
        if scores.ndim != 1:
            raise ValueError("calibration_scores must be 1-D.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("calibration_scores must contain only finite values.")
        n = len(scores)
        # Binary float arithmetic can move an exact integer rank upward
        # (e.g. alpha=.95, n=19). Use the documented decimal alpha exactly;
        # a level just below a boundary must still receive the higher rank.
        rank = ceil((n + 1) * (1 - Fraction(str(self.alpha))))
        if rank > n:
            raise ValueError(
                f"Insufficient calibration points for a finite threshold: "
                f"alpha={self.alpha}, n={n}, required order-statistic rank={rank}."
            )
        threshold = float(np.partition(scores, rank - 1)[rank - 1])
        if not np.isfinite(threshold):
            raise ValueError("Calibration produced a nonfinite threshold.")
        self.threshold_ = threshold
        self.calibration_size_ = n
        return self
    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Return binary decisions using the calibrated threshold."""
        if not hasattr(self, "threshold_"):
            raise RuntimeError("Call calibrate() before apply().")
        values = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("scores must contain only finite values.")
        return (values > self.threshold_).astype(int)



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
        Finite positive false-negative cost in the weight-calibration loss.
    cost_fp : float, default=1.0
        Finite positive false-positive cost.
    alignment_policy : {"strict", "legacy_resample"}, default="strict"
        Require equal-length, row-aligned component vectors. The explicit
        legacy policy resamples each vector over normalized row positions;
        it does not establish timestamp alignment and can discard events.
        Weight calibration always requires exact row alignment.
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
        alignment_policy: str = "strict",
    ):
        if alignment_policy not in {"strict", "legacy_resample"}:
            raise ValueError("alignment_policy must be 'strict' or 'legacy_resample'.")
        self.alignment_policy = alignment_policy
        w = np.asarray(weights, dtype=float)
        if w.shape != (3,):
            raise ValueError("weights must have length 3.")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite.")
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
        if not (np.isfinite(cost_fn) and np.isfinite(cost_fp)) or cost_fn <= 0 or cost_fp <= 0:
            raise ValueError("Cost weights must be finite and positive.")
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
    def _align_three(self, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
        arrays = tuple(_finite_vector(a, "risk component") for a in arrays)
        if self.alignment_policy == "strict" and len({len(a) for a in arrays}) != 1:
            raise ValueError(
                "Risk components must have equal length and aligned rows. "
                "Use alignment_policy='legacy_resample' only to reproduce the old "
                "normalized-position resampling behavior."
            )
        target = min(len(a) for a in arrays)
        def resample(x: np.ndarray, n: int) -> np.ndarray:
            if len(x) == n:
                return x
            idx = np.linspace(0, len(x) - 1, n).round().astype(int)
            return x[idx]
        return tuple(resample(a, target) for a in arrays)
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
            Finite per-sample risk components on the same validation rows.
            Shape (n,) or (n, 1); unequal lengths are rejected, not resampled.
        y_val : ndarray of 0/1
            Binary failure labels.
        Returns
        -------
        w : ndarray of shape (3,)
            Calibrated weights, also assigned to `self.w`.
        Raises
        ------
        RuntimeError
            Optimization failed or returned nonfinite/invalid weights or loss.
            The previous weights remain unchanged; `last_optimization_warning_`
            records the reason. Successful weights must be nonnegative and sum
            to one within absolute tolerance 1e-6 before normalization.
        """
        anom, rul, nn = (
            _finite_vector(values, name)
            for values, name in zip(
                (R_anom_val, R_RUL_val, R_NN_val),
                ("R_anom_val", "R_RUL_val", "R_NN_val"), strict=True,
            )
        )
        y = _finite_vector(y_val, "y_val")
        if len({len(anom), len(rul), len(nn), len(y)}) != 1:
            raise ValueError("Risk components and y_val must have equal length and aligned rows.")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y_val must contain only binary 0/1 labels.")
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
        error = None
        if not res.success:
            error = str(res.message)
        else:
            candidate = np.asarray(res.x, dtype=float)
            if (candidate.shape != (3,) or not np.all(np.isfinite(candidate))
                    or np.any(candidate < 0)
                    or not np.isclose(candidate.sum(), 1.0, rtol=0, atol=1e-6)
                    or not np.isfinite(res.fun)):
                error = "Optimizer returned invalid simplex weights or a nonfinite objective."
        if error is not None:
            self.last_optimization_warning_ = error
            raise RuntimeError(f"Weight calibration failed: {error}")
        self.w = candidate / candidate.sum()
        self.last_optimization_warning_ = None
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
            y_true = tf.reshape(tf.cast(y_true, y_pred.dtype), tf.shape(y_pred))
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
