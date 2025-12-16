import json
import requests


def send_alert(
    risk: float,
    data: dict,
    webhook_url: str = "http://localhost:5678/webhook/accident_alert",
) -> None:
    """
    Send a risk alert to an external automation system (e.g., n8n) via webhook.

    Parameters:
        risk: overall risk score in [0, 1]
        data: raw sensor reading (dict)
        webhook_url: n8n webhook endpoint
    """
    payload = {
        "risk_score": float(risk),
        "timestamp": data.get("timestamp"),
        "temperature": data.get("temperature"),
        "vibration": data.get("vibration"),
        "pressure": data.get("pressure"),
    }

    requests.post(
        webhook_url,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
