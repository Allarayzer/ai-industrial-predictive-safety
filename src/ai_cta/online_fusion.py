"""Asynchronous online fusion of anomaly, RUL, and neural risk channels.

The offline :class:`~ai_cta.risk_model.RiskAggregator` assumes aligned arrays.
Real streaming systems rarely receive all channels at the same cadence: an
anomaly detector may update every sample, a neural model every few windows,
and an RUL model only after a longer history is available.  This module keeps
the most recent observation from each channel, rejects stale values, and
renormalizes the calibrated simplex weights over the channels that are
currently usable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ai_cta.risk_model import RiskAggregator

__all__ = [
    "AsynchronousRiskFusion",
    "ChannelObservation",
    "FusionSnapshot",
]

CHANNELS = ("anomaly", "rul", "neural")


@dataclass(frozen=True)
class ChannelObservation:
    """One timestamped channel score."""

    score: float
    timestamp: pd.Timestamp


@dataclass(frozen=True)
class FusionSnapshot:
    """Result of one asynchronous fusion decision."""

    timestamp: pd.Timestamp
    risk_score: float
    risk_level: str
    effective_weights: dict[str, float]
    channel_scores: dict[str, float]
    channel_age_seconds: dict[str, float]
    stale_channels: tuple[str, ...]
    missing_channels: tuple[str, ...]
    used_zero_weight_fallback: bool = False

    @property
    def n_available(self) -> int:
        return len(self.channel_scores)


class AsynchronousRiskFusion:
    """Fuse the latest non-stale scores from three risk channels.

    Parameters
    ----------
    aggregator:
        The calibrated three-component aggregator.  Its simplex weights and
        alert thresholds are reused online.
    max_age_seconds:
        Per-channel maximum age.  A scalar applies to all channels.
    fallback:
        ``"renormalize"`` (default) redistributes weight over available
        channels.  ``"strict"`` requires all channels to be present and fresh.
    min_channels:
        Minimum number of usable channels for a decision in renormalize mode.
    """

    def __init__(
        self,
        aggregator: RiskAggregator,
        max_age_seconds: float | dict[str, float] = 300.0,
        fallback: Literal["renormalize", "strict"] = "renormalize",
        min_channels: int = 1,
    ):
        if fallback not in {"renormalize", "strict"}:
            raise ValueError("fallback must be 'renormalize' or 'strict'.")
        if not 1 <= min_channels <= 3:
            raise ValueError("min_channels must be between 1 and 3.")
        self.aggregator = aggregator
        self.fallback = fallback
        self.min_channels = min_channels
        if isinstance(max_age_seconds, dict):
            missing = set(CHANNELS) - set(max_age_seconds)
            if missing:
                raise ValueError(f"max_age_seconds missing channels: {sorted(missing)}")
            self.max_age_seconds = {
                ch: float(max_age_seconds[ch]) for ch in CHANNELS
            }
        else:
            self.max_age_seconds = {ch: float(max_age_seconds) for ch in CHANNELS}
        if any(v <= 0 for v in self.max_age_seconds.values()):
            raise ValueError("All max_age_seconds values must be positive.")
        self._latest: dict[str, ChannelObservation] = {}

    def update(
        self,
        channel: str,
        score: float,
        timestamp: pd.Timestamp | str | None = None,
    ) -> None:
        """Store the newest score for one channel."""
        if channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}; got {channel!r}.")
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("score must be finite.")
        ts = pd.Timestamp.now() if timestamp is None else pd.Timestamp(timestamp)
        previous = self._latest.get(channel)
        if previous is not None and ts < previous.timestamp:
            raise ValueError("Out-of-order channel update.")
        self._latest[channel] = ChannelObservation(float(np.clip(value, 0.0, 1.0)), ts)

    def update_many(
        self,
        scores: dict[str, float],
        timestamp: pd.Timestamp | str | None = None,
    ) -> None:
        """Store several scores with one timestamp."""
        for channel, score in scores.items():
            self.update(channel, score, timestamp)

    def fuse(self, timestamp: pd.Timestamp | str | None = None) -> FusionSnapshot:
        """Fuse all currently available, non-stale channel observations."""
        ts = pd.Timestamp.now() if timestamp is None else pd.Timestamp(timestamp)
        ages: dict[str, float] = {}
        available: list[str] = []
        stale: list[str] = []
        missing: list[str] = []
        for ch in CHANNELS:
            obs = self._latest.get(ch)
            if obs is None:
                missing.append(ch)
                continue
            age = max((ts - obs.timestamp).total_seconds(), 0.0)
            ages[ch] = float(age)
            if age <= self.max_age_seconds[ch]:
                available.append(ch)
            else:
                stale.append(ch)

        if self.fallback == "strict" and len(available) != 3:
            raise RuntimeError("All three channels must be fresh in strict mode.")
        if len(available) < self.min_channels:
            raise RuntimeError(
                f"Only {len(available)} usable channel(s); min_channels={self.min_channels}."
            )

        base_weights = dict(zip(CHANNELS, self.aggregator.w, strict=True))
        effective = {ch: (base_weights[ch] if ch in available else 0.0) for ch in CHANNELS}
        weight_sum = sum(effective.values())
        used_zero_weight_fallback = False
        if weight_sum <= np.finfo(float).eps:
            # A calibrated simplex can concentrate all mass on one channel. If
            # that channel is temporarily unavailable, the remaining fresh
            # channels would otherwise be unusable despite min_channels being
            # satisfied. In renormalize mode, degrade gracefully to an equal
            # mixture over the available channels and expose the fallback in
            # the returned snapshot for auditability.
            if self.fallback == "strict":
                raise RuntimeError("Available channels have zero total aggregation weight.")
            equal = 1.0 / len(available)
            effective = {ch: (equal if ch in available else 0.0) for ch in CHANNELS}
            used_zero_weight_fallback = True
        else:
            effective = {ch: w / weight_sum for ch, w in effective.items()}

        channel_scores = {ch: self._latest[ch].score for ch in available}
        risk = float(sum(effective[ch] * channel_scores[ch] for ch in available))
        risk = float(np.clip(risk, 0.0, 1.0))
        level = str(self.aggregator._classify(np.asarray([risk], dtype=float))[0])
        return FusionSnapshot(
            timestamp=ts,
            risk_score=risk,
            risk_level=level,
            effective_weights=effective,
            channel_scores=channel_scores,
            channel_age_seconds=ages,
            stale_channels=tuple(stale),
            missing_channels=tuple(missing),
            used_zero_weight_fallback=used_zero_weight_fallback,
        )

    def reset(self) -> None:
        """Forget all stored channel observations."""
        self._latest.clear()
