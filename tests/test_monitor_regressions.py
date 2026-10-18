"""Small checks for degenerate reference data and causal online scoring."""
import numpy as np
import pandas as pd
import pytest

from ai_cta.drift_detector import DriftDetector
from ai_cta.online_fusion import AsynchronousRiskFusion
from ai_cta.risk_model import RiskAggregator


@pytest.mark.parametrize("shift", [-100000.0, 100000.0])
def test_psi_constant_reference_detects_shift_in_either_direction(shift):
    reference = pd.DataFrame({"sensor": np.zeros(100)})
    detector = DriftDetector(method="psi").fit(reference)
    assert not detector.detect(reference).drift_detected
    shifted = pd.DataFrame({"sensor": np.full(100, shift)})
    assert detector.detect(shifted).drift_detected


@pytest.mark.parametrize("method", ["psi", "ks"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_drift_rejects_nonfinite_reference_and_current(method, invalid):
    reference = pd.DataFrame({"sensor": np.linspace(0, 1, 100)})
    invalid_frame = reference.copy()
    invalid_frame.loc[10, "sensor"] = invalid
    with pytest.raises(ValueError, match="finite"):
        DriftDetector(method=method).fit(invalid_frame)
    detector = DriftDetector(method=method).fit(reference)
    with pytest.raises(ValueError, match="finite"):
        detector.detect(invalid_frame)


@pytest.mark.parametrize("method", ["psi", "ks"])
def test_drift_rejects_empty_current_window(method):
    reference = pd.DataFrame({"sensor": np.arange(20.0)})
    detector = DriftDetector(method=method).fit(reference)
    with pytest.raises(ValueError, match="empty"):
        detector.detect(reference.iloc[:0])


def test_fusion_rejects_future_observations_and_preserves_state():
    fusion = AsynchronousRiskFusion(RiskAggregator())
    ts = pd.Timestamp("2026-01-01T12:00:00Z")
    fusion.update("anomaly", 0.9, ts)
    with pytest.raises(ValueError, match="future"):
        fusion.fuse(ts - pd.Timedelta(hours=1))
    assert fusion.fuse(ts).risk_score == pytest.approx(0.9)


@pytest.mark.parametrize("operation", ["update", "fuse"])
def test_fusion_rejects_nat_timestamp(operation):
    fusion = AsynchronousRiskFusion(RiskAggregator())
    if operation == "update":
        with pytest.raises(ValueError, match="timestamp"):
            fusion.update("anomaly", 0.5, pd.NaT)
    else:
        fusion.update("anomaly", 0.5, "2026-01-01")
        with pytest.raises(ValueError, match="timestamp"):
            fusion.fuse(pd.NaT)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_fusion_requires_finite_maximum_age(invalid):
    with pytest.raises(ValueError, match="finite"):
        AsynchronousRiskFusion(RiskAggregator(), max_age_seconds=invalid)
