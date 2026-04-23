"""AI-CTA: AI-Industrial Predictive Safety.
Reference implementation of predictive safety AI for high-hazard industrial
facilities. Companion code to the monograph:
    Serebriakov, I. (2026). Artificial Intelligence for Preventing Accidents
    at High-Risk Industrial Facilities. Ridero.
Main components
---------------
- Feature engineering for industrial sensor time series
- Unsupervised anomaly detection (Isolation Forest, LSTM)
- Hybrid ensemble of structural and dynamical anomaly views
- Remaining Useful Life (RUL) estimation via quantile regression
- Neural risk estimator with cost-asymmetric BCE loss
- Three-component risk aggregation (R_anom + R_RUL + R_NN)
- Conformal calibration and online recalibration
- Distribution drift detection (PSI, KS)
- Streaming pipeline with automated alerting
- Industrial telemetry simulator
"""
from ai_cta._version import __version__
# Detection
from ai_cta.anomaly_detector import IsolationForestDetector
from ai_cta.anomaly_detector import LSTMDetector
from ai_cta.anomaly_detector import HybridDetector
# Features
from ai_cta.preprocess import (
    StatisticalFeatureExtractor,
    RollingWindowFeatureExtractor,
    FrequencyDomainFeatureExtractor,
)
# Risk
from ai_cta.risk_model import RiskScorer, ChannelLimits, RiskLevel
from ai_cta.risk_model import ConformalThresholdCalibrator
from ai_cta.risk_model import RiskAggregator
# Prognostics
from ai_cta.rul_estimator import RULEstimator
from ai_cta.risk_model import NeuralRiskEstimator
from ai_cta.drift_detector import DriftDetector, DriftReport
from ai_cta.calibration import OnlineCalibrator, CalibrationUpdate
# Streaming
from ai_cta.pipeline import SafetyPipeline, PipelineEvent
__all__ = [
    "__version__",
    # Detection
    "IsolationForestDetector",
    "LSTMDetector",
    "HybridDetector",
    # Features
    "StatisticalFeatureExtractor",
    "RollingWindowFeatureExtractor",
    "FrequencyDomainFeatureExtractor",
    # Risk
    "RiskScorer",
    "ChannelLimits",
    "RiskLevel",
    "ConformalThresholdCalibrator",
    "RiskAggregator",
    # Prognostics
    "RULEstimator",
    "NeuralRiskEstimator",
    "DriftDetector",
    "DriftReport",
    "OnlineCalibrator",
    "CalibrationUpdate",
    # Streaming
    "SafetyPipeline",
    "PipelineEvent",
]
