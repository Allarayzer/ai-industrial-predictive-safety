"""Tests for the Isolation Forest detector."""
import numpy as np
import pandas as pd
import pytest
from ai_cta.detection import IsolationForestDetector
from ai_cta.utils.data import (
    generate_synthetic_stream,
    inject_anomalies,
)

@pytest.fixture
def normal_data() -> pd.DataFrame:
    return generate_synthetic_stream(n_samples=500, random_state=0)

def test_detector_fit_and_predict(normal_data):
    detector = IsolationForestDetector(
        window_size=32, stride=16, use_spectral=False
    ).fit(normal_data.drop(columns=["timestamp"]))
    scores = detector.decision_function(normal_data.drop(columns=["timestamp"]))
    assert scores.shape[0] > 0
    assert np.all((scores >= 0.0) & (scores <= 1.0))

def test_detector_rejects_non_dataframe():
    detector = IsolationForestDetector()
    with pytest.raises(TypeError):
        detector.fit(np.array([[1, 2, 3]]))

def test_detector_rejects_short_input(normal_data):
    detector = IsolationForestDetector(window_size=32).fit(
        normal_data.drop(columns=["timestamp"])
    )
    short = normal_data.drop(columns=["timestamp"]).iloc[:10]
    with pytest.raises(ValueError):
        detector.decision_function(short)

def test_detector_prediction_threshold(normal_data):
    detector = IsolationForestDetector(
        window_size=32, stride=16, use_spectral=False
    ).fit(normal_data.drop(columns=["timestamp"]))
    preds = detector.predict(normal_data.drop(columns=["timestamp"]), threshold=0.5)
    assert preds.dtype.kind in ("i", "u")
    assert set(np.unique(preds)).issubset({0, 1})

def test_detector_detects_anomaly_regions():
    df = generate_synthetic_stream(n_samples=800, random_state=1)
    contaminated, labels = inject_anomalies(
        df, n_anomalies=10, random_state=1
    )
    numeric = contaminated.drop(columns=["timestamp"])
    # Train on the first portion which we hope is mostly normal.
    detector = IsolationForestDetector(
        window_size=32, stride=8, use_spectral=False
    ).fit(numeric.iloc[:300])
    scores = detector.decision_function(numeric)
    # With any working detector, at least the mean score over contaminated
    # regions should be higher than over normal regions.
    # Map labels back onto window indexed scores via nearest alignment.
    label_indices = np.linspace(0, len(labels) - 1, len(scores)).astype(int)
    aligned = labels[label_indices]
    if aligned.sum() > 0 and (1 - aligned).sum() > 0:
        assert np.mean(scores[aligned == 1]) > np.mean(scores[aligned == 0])

def test_detector_without_spectral_features(normal_data):
    detector = IsolationForestDetector(
        window_size=32, stride=16, use_spectral=False
    ).fit(normal_data.drop(columns=["timestamp"]))
    expected = 3 * 9  # 3 channels * 9 statistical features
    assert len(detector.feature_names_) == expected
