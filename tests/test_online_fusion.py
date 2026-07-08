import numpy as np
import pandas as pd
import pytest

from ai_cta.online_fusion import AsynchronousRiskFusion
from ai_cta.risk_model import RiskAggregator


def test_asynchronous_fusion_renormalizes_available_weights():
    agg = RiskAggregator(weights=(0.2, 0.3, 0.5))
    fusion = AsynchronousRiskFusion(
        agg,
        max_age_seconds={"anomaly": 10, "rul": 10, "neural": 10},
        min_channels=1,
    )
    ts = pd.Timestamp("2026-01-01")
    fusion.update("anomaly", 0.4, ts)
    fusion.update("neural", 0.8, ts)
    snap = fusion.fuse(ts)
    expected = (0.2 / 0.7) * 0.4 + (0.5 / 0.7) * 0.8
    assert np.isclose(snap.risk_score, expected)
    assert snap.missing_channels == ("rul",)
    assert np.isclose(sum(snap.effective_weights.values()), 1.0)


def test_asynchronous_fusion_no_usable_channel_raises():
    agg = RiskAggregator()
    fusion = AsynchronousRiskFusion(agg, max_age_seconds=30, min_channels=1)
    ts = pd.Timestamp("2026-01-01")
    fusion.update("anomaly", 0.5, ts)
    with pytest.raises(RuntimeError):
        fusion.fuse(ts + pd.Timedelta(seconds=31))


def test_strict_mode_requires_all_channels():
    fusion = AsynchronousRiskFusion(RiskAggregator(), fallback="strict")
    ts = pd.Timestamp("2026-01-01")
    fusion.update("anomaly", 0.2, ts)
    with pytest.raises(RuntimeError):
        fusion.fuse(ts)


def test_zero_weight_available_subset_falls_back_to_equal_weights():
    agg = RiskAggregator(weights=(0.0, 0.0, 1.0))
    fusion = AsynchronousRiskFusion(agg, max_age_seconds=10, min_channels=1)
    ts = pd.Timestamp("2026-01-01")
    fusion.update("anomaly", 0.2, ts)
    fusion.update("rul", 0.8, ts)
    snap = fusion.fuse(ts)
    assert np.isclose(snap.risk_score, 0.5)
    assert snap.used_zero_weight_fallback
    assert np.isclose(snap.effective_weights["anomaly"], 0.5)
    assert np.isclose(snap.effective_weights["rul"], 0.5)
    assert snap.effective_weights["neural"] == 0.0
