"""Tests for the DriftDetector."""
import numpy as np
import pandas as pd
import pytest

from ai_cta.drift_detector import DriftDetector, DriftReport


def _make_df(rng_seed: int, mean: float = 0.0, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    return pd.DataFrame(
        {
            "a": rng.normal(mean, 1.0, n),
            "b": rng.normal(mean + 1.0, 0.5, n),
        }
    )

def test_drift_detector_no_drift_psi():
    ref = _make_df(0)
    cur = _make_df(1)  # same distribution, different draw
    detector = DriftDetector(method="psi", threshold=0.25).fit(ref)
    report = detector.detect(cur)
    assert isinstance(report, DriftReport)
    assert report.method == "psi"
    # No drift expected at threshold 0.25 for same distribution
    assert not report.drift_detected

def test_drift_detector_detects_shift_psi():
    ref = _make_df(0, mean=0.0)
    cur = _make_df(1, mean=3.0)  # large mean shift
    detector = DriftDetector(method="psi", threshold=0.25).fit(ref)
    report = detector.detect(cur)
    assert report.drift_detected
    assert "a" in report.channels_with_drift()

def test_drift_detector_no_drift_ks():
    ref = _make_df(0)
    cur = _make_df(1)
    detector = DriftDetector(method="ks", threshold=0.01).fit(ref)
    report = detector.detect(cur)
    assert not report.drift_detected

def test_drift_detector_detects_shift_ks():
    ref = _make_df(0, mean=0.0, n=300)
    cur = _make_df(1, mean=2.0, n=300)
    detector = DriftDetector(method="ks", threshold=0.01).fit(ref)
    report = detector.detect(cur)
    assert report.drift_detected

def test_drift_detector_invalid_method():
    with pytest.raises(ValueError):
        DriftDetector(method="invalid")

def test_drift_detector_invalid_bins():
    with pytest.raises(ValueError):
        DriftDetector(n_bins=1)

def test_drift_detector_requires_fit():
    detector = DriftDetector()
    with pytest.raises(RuntimeError):
        detector.detect(_make_df(0))

def test_drift_detector_missing_channel():
    ref = _make_df(0)
    cur = pd.DataFrame({"a": np.random.randn(100)})  # missing 'b'
    detector = DriftDetector().fit(ref)
    with pytest.raises(ValueError):
        detector.detect(cur)
