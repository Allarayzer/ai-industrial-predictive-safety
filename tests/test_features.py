"""Tests for feature extractors."""
import numpy as np
import pandas as pd
import pytest

from ai_cta.preprocess import (
    FrequencyDomainFeatureExtractor,
    RollingWindowFeatureExtractor,
    StatisticalFeatureExtractor,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, 100),
            "b": rng.normal(5, 2, 100),
        }
    )

def test_statistical_extractor_feature_count(sample_df):
    extractor = StatisticalFeatureExtractor().fit(sample_df)
    features = extractor.transform(sample_df)
    assert features.shape == (1, 2 * 9)
    assert len(extractor.get_feature_names_out()) == 2 * 9

def test_statistical_extractor_selected_channel(sample_df):
    extractor = StatisticalFeatureExtractor(channels=["a"]).fit(sample_df)
    features = extractor.transform(sample_df)
    assert features.shape == (1, 9)

def test_statistical_extractor_known_values():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    extractor = StatisticalFeatureExtractor().fit(df)
    features = extractor.transform(df)[0]
    names = extractor.get_feature_names_out()
    feat_map = dict(zip(names, features, strict=False))
    assert feat_map["a_mean"] == pytest.approx(3.0)
    assert feat_map["a_min"] == pytest.approx(1.0)
    assert feat_map["a_max"] == pytest.approx(5.0)
    assert feat_map["a_peak_to_peak"] == pytest.approx(4.0)

def test_rolling_window_extractor_shape(sample_df):
    extractor = RollingWindowFeatureExtractor(
        window_size=10, step=1
    ).fit(sample_df)
    features = extractor.transform(sample_df)
    assert features.shape == (100, 2 * 4)

def test_rolling_window_extractor_stride(sample_df):
    extractor = RollingWindowFeatureExtractor(
        window_size=10, step=5
    ).fit(sample_df)
    features = extractor.transform(sample_df)
    assert features.shape == (20, 2 * 4)

def test_rolling_window_invalid_window():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        RollingWindowFeatureExtractor(window_size=1).fit(df)

def test_frequency_extractor_shape(sample_df):
    extractor = FrequencyDomainFeatureExtractor(sampling_rate=10.0).fit(sample_df)
    features = extractor.transform(sample_df)
    assert features.shape == (1, 2 * 4)

def test_frequency_extractor_dominant_frequency():
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    signal = np.sin(2 * np.pi * 5.0 * t)  # 5 Hz pure tone
    df = pd.DataFrame({"ch": signal})
    extractor = FrequencyDomainFeatureExtractor(sampling_rate=fs).fit(df)
    names = extractor.get_feature_names_out()
    features = dict(zip(names, extractor.transform(df)[0], strict=False))
    assert features["ch_dominant_frequency"] == pytest.approx(5.0, abs=0.1)

def test_frequency_extractor_invalid_rate(sample_df):
    with pytest.raises(ValueError):
        FrequencyDomainFeatureExtractor(sampling_rate=0).fit(sample_df)
