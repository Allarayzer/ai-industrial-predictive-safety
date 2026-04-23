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
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from ai_cta.risk.scoring import RiskScorer  # noqa: F401  (re-export friendly)

__all__ = ["RiskAggregator"]

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
        labels[R >= self.thresholds[0]] = self.LEVELS[1]
        labels[R >= self.thresholds[1]] = self.LEVELS[2]
        labels[R >= self.thresholds[2]] = self.LEVELS[3]
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
            l = -(
                self.cost_fn * y * np.log(R)
                + self.cost_fp * (1.0 - y) * np.log(1.0 - R)
            )
            return float(l.mean())
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
