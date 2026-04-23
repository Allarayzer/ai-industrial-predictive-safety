"""FastAPI service exposing the AI-CTA pipeline as REST endpoints.
Implements the API surface described in Chapter 10.8 of the monograph:
- ``POST /predict``      — score a single reading (book § 10.8 canonical form)
- ``POST /score``        — alias for /predict with legacy payload schema
- ``POST /score-batch``  — score a batch of readings
- ``GET  /health``       — health probe
- ``GET  /version``      — package version and runtime info
The service expects a fitted detector and risk scorer to be loaded at
startup. By default it constructs a demo pipeline using the synthetic
stream generator so the service is immediately runnable.
Run locally::
    pip install "fastapi" "uvicorn[standard]"
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
API documentation (Swagger UI) is auto-generated at ``/docs``.
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd
from ai_cta import (
    IsolationForestDetector,
    RiskScorer,
    SafetyPipeline,
    __version__,
)
from ai_cta.risk_model import ChannelLimits
from ai_cta.data import generate_synthetic_stream

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "The API service requires fastapi and pydantic. "
        'Install them with: pip install "fastapi" "uvicorn[standard]"'
    ) from exc

logger = logging.getLogger(__name__)

# --------------------------------------------------------- request/response
class SensorReading(BaseModel):
    """Single sensor reading payload."""
    timestamp: Optional[str] = Field(
        None, description="ISO-8601 timestamp; auto-filled if omitted."
    )
    temperature: float
    vibration: float
    pressure: float

class BatchRequest(BaseModel):
    """Batch of sensor readings."""
    readings: list[SensorReading]

class ScoreResponse(BaseModel):
    """One row of pipeline output."""
    timestamp: str
    anomaly_score: float
    risk_score: float
    risk_level: str

class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool

class VersionResponse(BaseModel):
    package_version: str

# ----------------------------------------------------------- service object
class _ServiceState:
    """Mutable holder for the loaded detector + pipeline."""
    detector = None
    scorer: Optional[RiskScorer] = None
    pipeline: Optional[SafetyPipeline] = None
    window_size: int = 64

state = _ServiceState()

def build_default_pipeline() -> None:
    """Build a demo pipeline so the service is runnable out of the box."""
    train_df = generate_synthetic_stream(n_samples=2000, random_state=0)
    detector = IsolationForestDetector(window_size=64, stride=32, use_spectral=False)
    detector.fit(train_df.drop(columns=["timestamp"]))
    scorer = RiskScorer(
        ml_weight=0.6,
        limits={
            "temperature": ChannelLimits(45, 55, 30, 70),
            "vibration": ChannelLimits(0.1, 0.5, 0.0, 0.8),
            "pressure": ChannelLimits(0.9, 1.1, 0.7, 1.3),
        },
    )
    state.detector = detector
    state.scorer = scorer
    state.pipeline = SafetyPipeline(
        detector=detector,
        risk_scorer=scorer,
        window_size=64,
        alert_threshold=0.7,
    )
    state.window_size = 64
    logger.info("Default pipeline built and loaded.")

# ----------------------------------------------------------------- app
app = FastAPI(
    title="AI-CTA Predictive Safety API",
    description=(
        "Reference REST service for the AI-Industrial-Safety pipeline. "
        "See accompanying monograph, Chapter 10.8."
    ),
    version=__version__,
)

@app.on_event("startup")
def _startup() -> None:
    build_default_pipeline()

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        pipeline_ready=state.pipeline is not None,
    )

@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(package_version=__version__)

@app.post("/score-batch", response_model=list[ScoreResponse])
def score_batch(req: BatchRequest) -> list[ScoreResponse]:
    """Score a batch of readings as a single window."""
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    if len(req.readings) < state.window_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"At least {state.window_size} readings required; "
                f"got {len(req.readings)}."
            ),
        )
    rows = [r.model_dump() for r in req.readings]
    for row in rows:
        if row.get("timestamp") is None:
            row["timestamp"] = pd.Timestamp.now().isoformat()
    events = list(state.pipeline.run(rows))
    return [
        ScoreResponse(
            timestamp=str(e.timestamp),
            anomaly_score=float(e.anomaly_score),
            risk_score=float(e.risk_score),
            risk_level=e.risk_level,
        )
        for e in events
    ]

@app.post("/predict", response_model=ScoreResponse)
@app.post("/score", response_model=ScoreResponse)
def score_single(reading: SensorReading) -> ScoreResponse:
    """Score a single reading by appending it to a synthetic prefix.

    This is the canonical single-sample endpoint referenced in monograph
    § 10.8. Registered under both ``/predict`` (book-canonical) and
    ``/score`` (legacy alias) so integrations written against either
    spelling continue to work.

    Useful for smoke testing. For real streaming scoring, send batches
    of at least ``window_size`` readings via ``/score-batch``.
    """
    if state.detector is None or state.scorer is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    # Build a window using a recent synthetic baseline as prefix.
    prefix = generate_synthetic_stream(n_samples=state.window_size, random_state=0)
    payload = reading.model_dump()
    payload.setdefault("timestamp", str(pd.Timestamp.now()))
    df = pd.concat(
        [
            prefix.drop(columns=["timestamp"]),
            pd.DataFrame([{
                "temperature": payload["temperature"],
                "vibration": payload["vibration"],
                "pressure": payload["pressure"],
            }]),
        ],
        ignore_index=True,
    )
    scores = state.detector.decision_function(df)
    anomaly_score = float(scores[-1])
    channel_values = {
        "temperature": float(payload["temperature"]),
        "vibration": float(payload["vibration"]),
        "pressure": float(payload["pressure"]),
    }
    risk_score = state.scorer.score(anomaly_score, channel_values)
    return ScoreResponse(
        timestamp=str(payload["timestamp"]),
        anomaly_score=anomaly_score,
        risk_score=risk_score,
        risk_level=state.scorer.level(risk_score),
    )
