"""Exercise the public HTTP endpoints with the real demo detector."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ai_cta.data import generate_synthetic_stream
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as http:
        yield http


@pytest.mark.parametrize("endpoint", ["/predict", "/score"])
def test_single_reading_reaches_the_detector(client, endpoint):
    normal = {"temperature": 50.0, "vibration": 0.3, "pressure": 1.0}
    extreme = {"temperature": 500.0, "vibration": 100.0, "pressure": 10.0}
    first = client.post(endpoint, json=normal)
    second = client.post(endpoint, json=extreme)
    assert first.status_code == second.status_code == 200
    assert second.json()["anomaly_score"] > first.json()["anomaly_score"]
    assert pd.Timestamp(first.json()["timestamp"]).tzinfo is not None


def test_single_reading_preserves_timestamp(client):
    timestamp = "2026-01-01T12:00:00Z"
    response = client.post("/predict", json={
        "timestamp": timestamp, "temperature": 50, "vibration": 0.3, "pressure": 1,
    })
    assert response.status_code == 200
    assert response.json()["timestamp"] == timestamp


def test_batch_requires_real_history_and_emits_one_event_per_full_window(client):
    rows = generate_synthetic_stream(65, random_state=3).drop(columns="timestamp")
    records = rows.to_dict(orient="records")
    assert client.post("/score-batch", json={"readings": records[:63]}).status_code == 400
    response = client.post("/score-batch", json={"readings": records})
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert all(0 <= event["risk_score"] <= 1 for event in response.json())
