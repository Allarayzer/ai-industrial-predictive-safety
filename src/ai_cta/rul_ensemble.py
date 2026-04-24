"""Three-model RUL ensemble per monograph §9.6.

Implements the full ensemble specification described in the book:
    Model A — Gradient Boosting (XGBoost) quantile regression
    Model B — LSTM quantile regression (provided by RULEstimator)
    Model C — Physics-guided fallback (Weibull hazard model)

The final RUL prediction is a calibrated combination of all three,
with ensemble weights learnable on a held-out validation set.

See also:
    - ai_cta.rul_estimator.RULEstimator (Model B, LSTM quantile).
    - benchmarks/run_cmapss_rul_normalized.py --full-ensemble.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ===========================================================================
# Model A — XGBoost quantile regression
# ===========================================================================
class XGBoostQuantileRegressor:
    """Per-quantile XGBoost regressor trained with pinball loss.

    Fits one independent XGBRegressor per target quantile using the
    native `quantile` objective introduced in XGBoost 2.0. Hand-crafted
    (flat) features are the input — upstream pipeline extracts statistical
    + spectral + rolling-window descriptors per sliding window.

    Parameters
    ----------
    quantiles : sequence of float, default=(0.1, 0.5, 0.9)
    n_estimators : int, default=500
    max_depth : int, default=6
    learning_rate : float, default=0.05
    early_stopping_rounds : int, default=20
    rul_max : float, default=125.0 (clip target, standard C-MAPSS convention).
    random_state : int, default=42
    """

    def __init__(
        self,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 20,
        rul_max: float = 125.0,
        random_state: int = 42,
    ):
        self.quantiles = tuple(quantiles)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.rul_max = rul_max
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostQuantileRegressor":
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "XGBoostQuantileRegressor requires xgboost. "
                'Install it via: pip install "xgboost>=2.0"'
            ) from exc

        y = np.minimum(np.asarray(y, dtype=np.float32), self.rul_max)

        self.models_: dict[float, Any] = {}
        for q in self.quantiles:
            mdl = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                tree_method="hist",
                random_state=self.random_state,
                verbosity=0,
            )
            mdl.fit(X, y)
            self.models_[q] = mdl
        return self

    def predict_quantiles(self, X: np.ndarray) -> dict[float, np.ndarray]:
        return {q: np.clip(m.predict(X), 0.0, self.rul_max)
                for q, m in self.models_.items()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return median quantile prediction."""
        return self.predict_quantiles(X)[0.5]


# ===========================================================================
# Model C — Physics-guided Weibull fallback
# ===========================================================================
@dataclass
class WeibullParams:
    """Two-parameter Weibull life-distribution parameters.

    PDF: f(t; shape k, scale λ) = (k/λ)(t/λ)^{k-1} exp(−(t/λ)^k)
    Expected remaining life given elapsed time t_elapsed is
    E[RUL | survived t_elapsed] = E[T] − t_elapsed (approx for small t).
    Fitted by method-of-moments on a training set of observed failure times.
    """
    shape: float = 2.0  # k; shape 2 = typical bearing wear, 1 = exponential
    scale: float = 100.0  # λ; median life time
    observed_times: np.ndarray | None = None

    def mean_life(self) -> float:
        """Mean of Weibull distribution: λ·Γ(1+1/k). Uses gamma function."""
        from scipy.special import gamma
        return self.scale * gamma(1 + 1.0 / self.shape)


class PhysicsGuidedRUL:
    """Weibull-based physics-informed RUL sanity check.

    When data-driven models (XGBoost, LSTM) produce RUL estimates that
    diverge sharply from the Weibull-derived engineering estimate, a
    warning is emitted. This serves as an anomaly-detector for the
    predictions themselves, catching extrapolation failures or
    distribution-shift induced miscalibration.

    The Weibull parameters are fit by method-of-moments on training
    failure times (complete data) — no censoring support in this
    simplified reference implementation.

    Parameters
    ----------
    rul_max : float, default=125.0
    default_shape : float, default=2.0
        Used as a fallback if MoM fit fails (e.g. too few samples).
    warning_threshold_cycles : float, default=30.0
        If |data-driven prediction − physics prediction| > this, issue a
        warning (surface to human operator).
    """

    def __init__(
        self,
        rul_max: float = 125.0,
        default_shape: float = 2.0,
        warning_threshold_cycles: float = 30.0,
    ):
        self.rul_max = rul_max
        self.default_shape = default_shape
        self.warning_threshold_cycles = warning_threshold_cycles
        self.params_: WeibullParams | None = None

    def fit(self, failure_times: np.ndarray) -> "PhysicsGuidedRUL":
        """Fit Weibull parameters by method-of-moments on failure times."""
        failure_times = np.asarray(failure_times, dtype=np.float64)
        failure_times = failure_times[failure_times > 0]

        if len(failure_times) < 5:
            # Not enough samples — use defaults
            self.params_ = WeibullParams(
                shape=self.default_shape,
                scale=float(failure_times.mean()) if len(failure_times) else self.rul_max,
                observed_times=failure_times,
            )
            return self

        # Method of moments: (mean/std)^2 ~ function of shape
        from scipy.special import gamma
        from scipy.optimize import brentq

        mean = float(failure_times.mean())
        std = float(failure_times.std(ddof=1))
        cv = std / (mean + 1e-9)  # coefficient of variation

        def cv_of_k(k: float) -> float:
            g1 = gamma(1 + 1 / k)
            g2 = gamma(1 + 2 / k)
            return float(np.sqrt(g2 / g1**2 - 1) - cv)

        try:
            shape = brentq(cv_of_k, 0.5, 15.0, xtol=1e-4)
        except Exception:
            shape = self.default_shape

        scale = mean / gamma(1 + 1 / shape)
        self.params_ = WeibullParams(shape=shape, scale=scale, observed_times=failure_times)
        return self

    def predict(self, elapsed_cycles: np.ndarray) -> np.ndarray:
        """Predict RUL given elapsed operational cycles.

        Returns E[RUL | elapsed] ≈ E[T] − elapsed (simple approximation;
        the correct conditional expectation requires integration but
        differs little for t_elapsed ≪ E[T]).
        """
        if self.params_ is None:
            raise RuntimeError("PhysicsGuidedRUL must be fit before prediction.")
        mean_life = self.params_.mean_life()
        elapsed = np.asarray(elapsed_cycles, dtype=np.float64)
        rul = mean_life - elapsed
        return np.clip(rul, 0.0, self.rul_max)

    def sanity_check(
        self,
        data_driven_rul: np.ndarray,
        elapsed_cycles: np.ndarray,
    ) -> dict[str, Any]:
        """Compare data-driven predictions with physics-derived estimate.

        Returns a dict with:
            physics_rul     : Weibull prediction
            diff            : |data-driven − physics| per-sample
            warning_fraction: share of samples with diff > threshold
        """
        phys = self.predict(elapsed_cycles)
        data = np.asarray(data_driven_rul, dtype=np.float64)
        diff = np.abs(data - phys)
        warn_frac = float((diff > self.warning_threshold_cycles).mean())
        return {
            "physics_rul": phys.tolist(),
            "mean_abs_diff": float(diff.mean()),
            "warning_fraction": warn_frac,
        }


# ===========================================================================
# Full RUL Ensemble (stacked)
# ===========================================================================
class RULEnsemble:
    """Stacked ensemble of Models A, B, C per monograph §9.6.

    Combines:
        Model A: XGBoost quantile  (tabular features)
        Model B: LSTM quantile     (sequential raw signal)
        Model C: Weibull physics   (prior on life distribution)

    At inference the final RUL is a convex combination of A, B, and
    (optionally) C. Weights are either:
        1. Fixed equal   (default if no validation data supplied)
        2. Optimised by grid search on a held-out validation set
           against pinball loss.

    This class is a thin orchestrator — underlying Models A/B/C are
    trained independently and passed in after fit.

    Parameters
    ----------
    quantiles : sequence of float, default=(0.1, 0.5, 0.9)
    rul_max : float, default=125.0
    use_physics : bool, default=True
        Whether Model C participates as a third ensemble member.
    weight_grid : int, default=11
        Grid resolution for weight search. Ignored if a validation
        set is not passed to fit().
    """

    def __init__(
        self,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        rul_max: float = 125.0,
        use_physics: bool = True,
        weight_grid: int = 11,
    ):
        self.quantiles = tuple(quantiles)
        self.rul_max = rul_max
        self.use_physics = use_physics
        self.weight_grid = weight_grid

    def fit(
        self,
        model_a: Any,        # Must expose predict_quantiles(X_tabular)
        model_b: Any,        # RULEstimator or equivalent with predict_quantiles(X_df)
        model_c: Any | None, # PhysicsGuidedRUL or None
        X_tabular_val: np.ndarray | None = None,
        X_sequential_val: pd.DataFrame | None = None,
        elapsed_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "RULEnsemble":
        self.model_a_ = model_a
        self.model_b_ = model_b
        self.model_c_ = model_c if self.use_physics else None

        # Default weights: equal over active models
        n_active = 2 + (1 if self.model_c_ is not None else 0)
        self.weights_ = np.full(3, 1.0 / n_active, dtype=float)
        if self.model_c_ is None:
            self.weights_[2] = 0.0
            self.weights_[:2] = 0.5

        # Optional: grid-search optimal weights on validation data
        if y_val is not None and X_tabular_val is not None \
                and X_sequential_val is not None and elapsed_val is not None:
            self._optimise_weights(X_tabular_val, X_sequential_val, elapsed_val, y_val)

        return self

    def _pinball_loss(self, y_true: np.ndarray, preds: dict[float, np.ndarray]) -> float:
        loss = 0.0
        for q, p in preds.items():
            err = y_true - p
            loss += float(np.mean(np.maximum(q * err, (q - 1) * err)))
        return loss / len(preds)

    def _optimise_weights(
        self,
        X_tabular_val: np.ndarray,
        X_sequential_val: pd.DataFrame,
        elapsed_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Simple 3-D grid search on the 2-simplex of ensemble weights."""
        q_a = self.model_a_.predict_quantiles(X_tabular_val)
        q_b = self.model_b_.predict_quantiles(X_sequential_val)
        q_c = None
        if self.model_c_ is not None:
            c_pred = self.model_c_.predict(elapsed_val)
            # Model C only gives a point estimate; fake quantiles by adding
            # a small spread proportional to Weibull std.
            spread = self.model_c_.params_.scale * 0.15 if self.model_c_.params_ else 10.0
            q_c = {
                0.1: np.clip(c_pred - spread, 0, self.rul_max),
                0.5: c_pred,
                0.9: np.clip(c_pred + spread, 0, self.rul_max),
            }

        grid = np.linspace(0, 1, self.weight_grid)
        best_loss = float("inf")
        best_w = self.weights_.copy()
        for wa in grid:
            for wb in grid:
                if wa + wb > 1.0 + 1e-6:
                    continue
                wc = 1.0 - wa - wb if (q_c is not None) else 0.0
                if q_c is None and wc != 0:
                    continue
                combined = {}
                for q in self.quantiles:
                    pred = wa * q_a[q] + wb * q_b[q]
                    if q_c is not None:
                        pred = pred + wc * q_c[q]
                    combined[q] = pred
                loss = self._pinball_loss(y_val, combined)
                if loss < best_loss:
                    best_loss = loss
                    best_w = np.array([wa, wb, wc])
        self.weights_ = best_w

    def predict_quantiles(
        self,
        X_tabular: np.ndarray,
        X_sequential: pd.DataFrame,
        elapsed: np.ndarray | None = None,
    ) -> dict[float, np.ndarray]:
        q_a = self.model_a_.predict_quantiles(X_tabular)
        q_b = self.model_b_.predict_quantiles(X_sequential)
        w = self.weights_
        combined: dict[float, np.ndarray] = {}
        for q in self.quantiles:
            pred = w[0] * q_a[q] + w[1] * q_b[q]
            if self.model_c_ is not None and w[2] > 0 and elapsed is not None:
                c_pred = self.model_c_.predict(elapsed)
                spread = self.model_c_.params_.scale * 0.15
                c_quant = c_pred if q == 0.5 else (
                    np.clip(c_pred - spread, 0, self.rul_max) if q < 0.5
                    else np.clip(c_pred + spread, 0, self.rul_max)
                )
                pred = pred + w[2] * c_quant
            combined[q] = np.clip(pred, 0, self.rul_max)
        return combined

    def predict(
        self,
        X_tabular: np.ndarray,
        X_sequential: pd.DataFrame,
        elapsed: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.predict_quantiles(X_tabular, X_sequential, elapsed)[0.5]


__all__ = ["XGBoostQuantileRegressor", "PhysicsGuidedRUL", "WeibullParams", "RULEnsemble"]
