"""Input and delivery failures must be visible without external HTTP calls."""
import json
import logging

import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient

from ai_cta.pipeline import PipelineEvent, SafetyPipeline
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=False) as http:
        yield http


def _post_reading(client, endpoint, reading):
    if endpoint == "/score-batch":
        return client.post(endpoint, json={"readings": [reading] * 64})
    return client.post(endpoint, json=reading)


@pytest.mark.parametrize("endpoint", ["/predict", "/score", "/score-batch"])
@pytest.mark.parametrize("field,value", [
    ("timestamp", "not-a-date"),
    ("timestamp", "NaT"),
    ("timestamp", ""),
    ("timestamp", "2026-W01-1"),
    ("timestamp", "2026-01-01X12:00:00"),
    ("temperature", "inf"),
    ("vibration", "-inf"),
    ("pressure", "nan"),
])
def test_invalid_readings_are_client_errors(client, endpoint, field, value):
    reading = {"temperature": 50, "vibration": 0.3, "pressure": 1, field: value}
    response = _post_reading(client, endpoint, reading)
    assert response.status_code == 422
    assert any(error["loc"][-1] == field for error in response.json()["detail"])


@pytest.mark.parametrize("endpoint", ["/predict", "/score", "/score-batch"])
def test_overflowing_json_number_has_serializable_validation_error(client, endpoint):
    # A syntactically valid JSON number can overflow the JSON decoder's float.
    reading = '{"temperature":1e999,"vibration":0.3,"pressure":1}'
    payload = '{"readings":[' + ','.join([reading] * 64) + ']}' if endpoint == "/score-batch" else reading
    response = client.post(endpoint, content=payload, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "temperature"


@pytest.mark.parametrize("endpoint", ["/predict", "/score-batch"])
@pytest.mark.parametrize("timestamp", [
    "2026-01-01T12:00:00Z",
    "2026-01-01T12:00:00+03:00",
    "2026-01-01T12:00:00",
    "2026-01-01",
])
def test_valid_timestamp_formats_keep_existing_behavior(client, endpoint, timestamp):
    reading = {"temperature": 50, "vibration": 0.3, "pressure": 1,
               "timestamp": timestamp}
    response = _post_reading(client, endpoint, reading)
    assert response.status_code == 200
    if endpoint == "/score-batch":
        assert pd.Timestamp(response.json()[0]["timestamp"]) == pd.Timestamp(timestamp)
    else:
        assert response.json()["timestamp"] == timestamp


def _event():
    return PipelineEvent(pd.Timestamp("2026-01-01T12:00:00Z"), 0.9, 0.9, "Emergency")


@pytest.mark.parametrize("status", [400, 503])
def test_webhook_http_failure_is_logged(monkeypatch, caplog, status):
    response = requests.Response()
    response.status_code = status
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)
    with caplog.at_level(logging.ERROR, logger="ai_cta.pipeline"):
        SafetyPipeline.make_webhook_callback("https://example.invalid/alert")(_event())
    assert "Webhook POST failed" in caplog.text
    assert str(status) in caplog.text


def test_webhook_timeout_is_logged(monkeypatch, caplog):
    def post(*args, **kwargs):
        raise requests.Timeout("test timeout")

    monkeypatch.setattr(requests, "post", post)
    with caplog.at_level(logging.ERROR, logger="ai_cta.pipeline"):
        SafetyPipeline.make_webhook_callback("https://example.invalid/alert")(_event())
    assert "Webhook POST failed" in caplog.text


def test_webhook_success_preserves_payload_and_timeout(monkeypatch, caplog):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        response = requests.Response()
        response.status_code = 204
        return response

    monkeypatch.setattr(requests, "post", post)
    event = _event()
    SafetyPipeline.make_webhook_callback("https://example.invalid/alert", timeout=2)(event)
    assert len(calls) == 1
    assert calls[0][0] == "https://example.invalid/alert"
    assert calls[0][1]["timeout"] == 2
    assert json.loads(calls[0][1]["data"]) == event.to_dict()
    assert "Webhook POST failed" not in caplog.text
