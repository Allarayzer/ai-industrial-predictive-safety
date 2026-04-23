"""Distribution drift detection for streaming sensor data.
Implements the algorithm described in Chapter 9.9 of the monograph:
detect when the distribution of incoming data has shifted away from the
distribution the models were trained on, signaling that retraining or
recalibration is needed.
Two methods are provided:
1. **Population Stability Index (PSI)** — a per-channel summary
   commonly used in industrial monitoring; flags drift when PSI exceeds
   a configurable threshold (typical thresholds: 0.1 = small shift,
   0.25 = significant shift).
2. **Kolmogorov-Smirnov test** — a non-parametric two-sample test on
   each channel; flags drift when the p-value falls below a significance
   level (with optional Bonferroni correction for multiple channels).
Both methods operate on a sliding reference/current window comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["DriftDetector", "DriftReport"]

@dataclass(frozen=True)
class DriftReport:
    """Outcome of a single drift check."""
    drift_detected: bool
    method: str
    per_channel_score: dict[str, float]
    per_channel_drift: dict[str, bool]
    threshold: float
    def channels_with_drift(self) -> list[str]:
        return [c for c, flag in self.per_channel_drift.items() if flag]

class DriftDetector:
    """Distribution-drift monitor over multivariate sensor channels.
    Parameters
    ----------
    method : {"psi", "ks"}, default="psi"
        Drift-detection method.
    threshold : float, default depends on method
        For PSI: per-channel PSI value above which drift is flagged
        (default 0.25). For KS: per-channel p-value below which drift
        is flagged (default 0.01, Bonferroni-corrected internally).
    n_bins : int, default=10
        Number of bins used for the PSI histogram.
    bonferroni : bool, default=True
        For KS only — apply Bonferroni correction to control the
        family-wise error rate when testing many channels.
    Examples
    --------
    >>> detector = DriftDetector(method="psi", threshold=0.25)
    >>> detector.fit(reference_window)
    >>> report = detector.detect(current_window)
    >>> if report.drift_detected:
    ...     print(f"Drift in: {report.channels_with_drift()}")
    """
    def __init__(
        self,
        method: str = "psi",
        threshold: float | None = None,
        n_bins: int = 10,
        bonferroni: bool = True,
    ):
        if method not in {"psi", "ks"}:
            raise ValueError("method must be 'psi' or 'ks'.")
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2.")
        self.method = method
        self.n_bins = n_bins
        self.bonferroni = bonferroni
        self.threshold = threshold if threshold is not None else (0.25 if method == "psi" else 0.01)
    # ------------------------------------------------------------- fit
    def fit(self, reference: pd.DataFrame) -> "DriftDetector":
        """Establish the reference distribution.
        Parameters
        ----------
        reference : DataFrame
            Reference window; typically the last known-stable period of
            normal operation, or the original training data.
        """
        if not isinstance(reference, pd.DataFrame):
            raise TypeError("reference must be a pandas DataFrame.")
        self.channels_ = list(reference.select_dtypes(include="number").columns)
        if not self.channels_:
            raise ValueError("No numeric channels found in reference.")
        self._reference = reference[self.channels_].copy()
        if self.method == "psi":
            self._bin_edges = {
                ch: self._compute_bin_edges(self._reference[ch].to_numpy())
                for ch in self.channels_
            }
        return self
    def _compute_bin_edges(self, values: np.ndarray) -> np.ndarray:
        """Quantile-based bin edges with sentinels at ±inf for robustness."""
        if len(values) < self.n_bins:
            raise ValueError(
                f"Reference must have at least n_bins ({self.n_bins}) samples."
            )
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.unique(np.quantile(values, quantiles))
        if len(edges) < 2:
            edges = np.array([values.min(), values.max() + 1e-9])
        # Sentinel edges so out-of-range new values are bucketed safely.
        edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])
        return edges
    # ----------------------------------------------------------- detect
    def detect(self, current: pd.DataFrame) -> DriftReport:
        """Check the current window against the fitted reference."""
        self._check_fitted()
        if not isinstance(current, pd.DataFrame):
            raise TypeError("current must be a pandas DataFrame.")
        missing = set(self.channels_) - set(current.columns)
        if missing:
            raise ValueError(f"Channels missing in current window: {missing}")
        if self.method == "psi":
            return self._detect_psi(current)
        return self._detect_ks(current)
    def _detect_psi(self, current: pd.DataFrame) -> DriftReport:
        scores: dict[str, float] = {}
        flags: dict[str, bool] = {}
        for ch in self.channels_:
            ref_vals = self._reference[ch].to_numpy()
            cur_vals = current[ch].to_numpy()
            edges = self._bin_edges[ch]
            psi = self._psi(ref_vals, cur_vals, edges)
            scores[ch] = float(psi)
            flags[ch] = psi > self.threshold
        return DriftReport(
            drift_detected=any(flags.values()),
            method="psi",
            per_channel_score=scores,
            per_channel_drift=flags,
            threshold=self.threshold,
        )
    @staticmethod
    def _psi(ref: np.ndarray, cur: np.ndarray, edges: np.ndarray) -> float:
        """Population Stability Index between two samples."""
        ref_hist, _ = np.histogram(ref, bins=edges)
        cur_hist, _ = np.histogram(cur, bins=edges)
        # Avoid zeros to keep log finite. Standard correction: 1 / total.
        ref_p = (ref_hist + 1) / (ref_hist.sum() + len(ref_hist))
        cur_p = (cur_hist + 1) / (cur_hist.sum() + len(cur_hist))
        return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    def _detect_ks(self, current: pd.DataFrame) -> DriftReport:
        n_channels = len(self.channels_)
        effective_threshold = (
            self.threshold / n_channels if self.bonferroni else self.threshold
        )
        scores: dict[str, float] = {}
        flags: dict[str, bool] = {}
        for ch in self.channels_:
            ref_vals = self._reference[ch].to_numpy()
            cur_vals = current[ch].to_numpy()
            stat_, p_value = stats.ks_2samp(ref_vals, cur_vals)
            scores[ch] = float(p_value)
            flags[ch] = p_value < effective_threshold
        return DriftReport(
            drift_detected=any(flags.values()),
            method="ks",
            per_channel_score=scores,
            per_channel_drift=flags,
            threshold=effective_threshold,
        )
    def _check_fitted(self) -> None:
        if not hasattr(self, "_reference"):
            raise RuntimeError("Call fit(reference) before detect().")
