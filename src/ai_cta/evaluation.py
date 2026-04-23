"""Evaluation helpers for anomaly detectors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["evaluate_binary_detector", "DetectorMetrics"]

@dataclass
class DetectorMetrics:
    """Summary metrics for a binary anomaly detector."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    false_alarm_rate: float

def evaluate_binary_detector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> DetectorMetrics:
    """Compute standard binary classification metrics.
    Parameters
    ----------
    y_true : ndarray of 0/1
    y_pred : ndarray of 0/1
        Must have the same length as y_true.
    Returns
    -------
    metrics : DetectorMetrics
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / max(len(y_true), 1)
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return DetectorMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        false_alarm_rate=float(false_alarm_rate),
    )
