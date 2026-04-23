"""Tests for synthetic data utilities."""
import numpy as np
import pandas as pd
from ai_cta.data import (
    generate_synthetic_stream,
    inject_anomalies,
)
from ai_cta.evaluation import evaluate_binary_detector

def test_generate_synthetic_stream_shape():
    df = generate_synthetic_stream(n_samples=500, random_state=0)
    assert len(df) == 500
    assert set(df.columns) == {"timestamp", "temperature", "vibration", "pressure"}

def test_generate_synthetic_stream_reproducible():
    df1 = generate_synthetic_stream(n_samples=200, random_state=42)
    df2 = generate_synthetic_stream(n_samples=200, random_state=42)
    pd.testing.assert_frame_equal(df1, df2)

def test_inject_anomalies_labels_match_length():
    df = generate_synthetic_stream(n_samples=500, random_state=0)
    contaminated, labels = inject_anomalies(df, n_anomalies=10, random_state=0)
    assert len(contaminated) == len(df)
    assert len(labels) == len(df)
    assert labels.sum() > 0

def test_inject_anomalies_modifies_data():
    df = generate_synthetic_stream(n_samples=500, random_state=0)
    contaminated, _ = inject_anomalies(df, n_anomalies=20, random_state=0)
    diff = (df.drop(columns=["timestamp"]) - contaminated.drop(columns=["timestamp"])).abs().sum().sum()
    assert diff > 0.0

def test_evaluate_binary_detector_perfect():
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0])
    m = evaluate_binary_detector(y_true, y_pred)
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.false_alarm_rate == 0.0

def test_evaluate_binary_detector_worst():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    m = evaluate_binary_detector(y_true, y_pred)
    assert m.accuracy == 0.0
    assert m.precision == 0.0
    assert m.recall == 0.0

def test_evaluate_binary_detector_mismatched_lengths():
    import pytest
    with pytest.raises(ValueError):
        evaluate_binary_detector(np.array([0, 1]), np.array([0, 1, 0]))
