"""Feature extraction for industrial sensor time series.
This module provides three families of feature extractors commonly used in
condition monitoring and predictive maintenance:
1. Statistical features (mean, std, skewness, kurtosis, peak, rms)
2. Rolling-window features (sliding-window statistics over time)
3. Frequency-domain features (FFT-based spectral descriptors)
All extractors follow the scikit-learn transformer interface (fit/transform)
so they can be composed with standard sklearn Pipelines.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "StatisticalFeatureExtractor",
    "RollingWindowFeatureExtractor",
    "FrequencyDomainFeatureExtractor",
]

class StatisticalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract per-channel statistical descriptors.
    For each input channel produces the following features:
    - mean: arithmetic mean
    - std: standard deviation
    - min, max: extrema
    - rms: root-mean-square amplitude
    - peak_to_peak: range
    - skewness: distribution asymmetry
    - kurtosis: distribution tailedness
    - crest_factor: peak / rms ratio (sensitive to impulsive faults)
    Parameters
    ----------
    channels : sequence of str, optional
        Column names to extract features from. If None, uses all columns.
    Notes
    -----
    The crest factor is particularly informative for bearing fault detection
    in rotating machinery; an increasing crest factor often precedes
    surface-level degradation of rolling elements.
    """
    FEATURE_NAMES = (
        "mean",
        "std",
        "min",
        "max",
        "rms",
        "peak_to_peak",
        "skewness",
        "kurtosis",
        "crest_factor",
    )
    def __init__(self, channels: Sequence[str] | None = None):
        self.channels = channels
    def fit(self, X: pd.DataFrame, y=None) -> "StatisticalFeatureExtractor":
        """Record channel names (no fitted parameters in this transformer)."""
        if self.channels is None:
            self.channels_ = list(X.columns)
        else:
            self.channels_ = list(self.channels)
        self.n_features_out_ = len(self.channels_) * len(self.FEATURE_NAMES)
        return self
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Extract a single feature vector per input segment.
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_channels)
            Input segment; features are computed over the entire segment.
        Returns
        -------
        features : ndarray of shape (1, n_channels * 9)
        """
        feats: list[float] = []
        for ch in self.channels_:
            x = X[ch].to_numpy()
            rms = float(np.sqrt(np.mean(x**2)))
            peak = float(np.max(np.abs(x)))
            crest = peak / rms if rms > 1e-12 else 0.0
            feats.extend(
                [
                    float(np.mean(x)),
                    float(np.std(x)),
                    float(np.min(x)),
                    float(np.max(x)),
                    rms,
                    float(np.max(x) - np.min(x)),
                    float(stats.skew(x)) if len(x) > 2 else 0.0,
                    float(stats.kurtosis(x)) if len(x) > 3 else 0.0,
                    crest,
                ]
            )
        return np.asarray(feats).reshape(1, -1)
    def get_feature_names_out(self) -> list[str]:
        return [f"{ch}_{name}" for ch in self.channels_ for name in self.FEATURE_NAMES]

@dataclass
class _RollingSpec:
    """Specification of a rolling aggregation."""
    name: str
    fn: str  # pandas rolling method name

class RollingWindowFeatureExtractor(BaseEstimator, TransformerMixin):
    """Sliding-window statistical features over a multivariate time series.
    For each channel and each row of the input, computes statistics over the
    preceding window_size samples. Useful for online monitoring where features
    must be produced incrementally rather than on fixed segments.
    Parameters
    ----------
    window_size : int
        Number of samples in the rolling window.
    step : int, default=1
        Stride between consecutive windows (1 means one feature row per input
        row, with NaNs at the beginning until the window is filled).
    channels : sequence of str, optional
        Columns to process. If None, uses all numeric columns.
    aggregations : sequence of str, default=("mean", "std", "min", "max")
        Rolling aggregations to compute.
    """
    def __init__(
        self,
        window_size: int = 32,
        step: int = 1,
        channels: Sequence[str] | None = None,
        aggregations: Sequence[str] = ("mean", "std", "min", "max"),
    ):
        self.window_size = window_size
        self.step = step
        self.channels = channels
        self.aggregations = aggregations
    def fit(self, X: pd.DataFrame, y=None) -> "RollingWindowFeatureExtractor":
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if self.channels is None:
            self.channels_ = list(X.select_dtypes(include="number").columns)
        else:
            self.channels_ = list(self.channels)
        self.aggregations_ = tuple(self.aggregations)
        self.n_features_out_ = len(self.channels_) * len(self.aggregations_)
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling features.
        Returns
        -------
        features : DataFrame of shape (n_samples // step, n_channels * n_aggs)
        """
        frames: list[pd.DataFrame] = []
        for ch in self.channels_:
            series = X[ch]
            rolling = series.rolling(window=self.window_size, min_periods=self.window_size)
            for agg in self.aggregations_:
                frames.append(
                    getattr(rolling, agg)().rename(f"{ch}_roll_{agg}_{self.window_size}")
                )
        combined = pd.concat(frames, axis=1)
        if self.step > 1:
            combined = combined.iloc[:: self.step].reset_index(drop=True)
        return combined
    def get_feature_names_out(self) -> list[str]:
        return [
            f"{ch}_roll_{agg}_{self.window_size}"
            for ch in self.channels_
            for agg in self.aggregations_
        ]

class FrequencyDomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """Frequency-domain features extracted via FFT.
    For each channel produces:
    - spectral_centroid: amplitude-weighted mean frequency
    - spectral_bandwidth: amplitude-weighted std of frequency
    - spectral_energy: total spectral energy
    - dominant_frequency: frequency of the highest-amplitude bin
    Parameters
    ----------
    sampling_rate : float
        Sampling rate of the input signal, in Hz.
    channels : sequence of str, optional
        Columns to process.
    Notes
    -----
    Spectral features are particularly informative for vibration-based
    diagnostics, since many mechanical faults (unbalance, misalignment,
    bearing defects) manifest as distinctive frequency components.
    """
    FEATURE_NAMES = (
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_energy",
        "dominant_frequency",
    )
    def __init__(self, sampling_rate: float = 1.0, channels: Sequence[str] | None = None):
        self.sampling_rate = sampling_rate
        self.channels = channels
    def fit(self, X: pd.DataFrame, y=None) -> "FrequencyDomainFeatureExtractor":
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if self.channels is None:
            self.channels_ = list(X.select_dtypes(include="number").columns)
        else:
            self.channels_ = list(self.channels)
        self.n_features_out_ = len(self.channels_) * len(self.FEATURE_NAMES)
        return self
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Compute per-segment spectral features."""
        feats: list[float] = []
        for ch in self.channels_:
            x = X[ch].to_numpy()
            n = len(x)
            # Remove DC component to avoid dominating the centroid.
            x_centered = x - np.mean(x)
            fft_vals = np.abs(np.fft.rfft(x_centered))
            freqs = np.fft.rfftfreq(n, d=1.0 / self.sampling_rate)
            energy = float(np.sum(fft_vals**2))
            if energy > 1e-12:
                amp_norm = fft_vals / np.sum(fft_vals + 1e-12)
                centroid = float(np.sum(freqs * amp_norm))
                bandwidth = float(
                    np.sqrt(np.sum(((freqs - centroid) ** 2) * amp_norm))
                )
                dominant = float(freqs[np.argmax(fft_vals)])
            else:
                centroid = bandwidth = dominant = 0.0
            feats.extend([centroid, bandwidth, energy, dominant])
        return np.asarray(feats).reshape(1, -1)
    def get_feature_names_out(self) -> list[str]:
        return [f"{ch}_{name}" for ch in self.channels_ for name in self.FEATURE_NAMES]
