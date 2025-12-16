def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """
    Limit a numeric value to the [low, high] interval.
    """
    return max(low, min(high, value))


def calculate_risk(anomaly_score: float, vibration: float, temperature: float) -> float:
    """
    Compute an overall risk score in [0, 1].

    Notes:
    - This is an interpretable heuristic used for the demonstrational prototype.
    - In an industrial deployment, weights should be calibrated on historical data
      and validated by subject-matter experts.
    """
    risk = 0.6 * (1 - anomaly_score) + 0.2 * vibration + 0.2 * temperature
    return float(clamp(risk, 0.0, 1.0))
