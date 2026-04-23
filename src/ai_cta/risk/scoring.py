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
from dataclasses import dataclass
from typing import Mapping
import numpy as np

__all__ = ["RiskScorer", "ChannelLimits", "RiskLevel"]

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
