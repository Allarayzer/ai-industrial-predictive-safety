"""End-to-end demonstration of the AI-CTA pipeline (referenced in monograph §12).

Runs the full industrial-safety pipeline from synthetic telemetry generation
through anomaly detection, risk aggregation, drift monitoring, and alert
emission. This script is the single entry-point that exercises every
component of the reference implementation.

Stages:
    1. Simulator: generate multichannel synthetic telemetry (24 hours).
    2. Anomaly detector: fit IsolationForest on clean stream, calibrate
       threshold.
    3. Risk scoring: combine anomaly score with rule-based channel limits.
    4. RUL prediction (optional, requires TensorFlow): quantile LSTM.
    5. Risk aggregation: 3-component hybrid with SLSQP weight calibration.
    6. Drift detection: PSI + KS with Bonferroni on streaming windows.
    7. Streaming pipeline: emit events, raise alerts above Warning
       threshold, log to stdout (would fire webhook in production).

Usage:
    python demo/end_to_end.py                   # full demo
    python demo/end_to_end.py --fast            # 2-hour mini-demo
    python demo/end_to_end.py --with-rul        # also train RUL model

Expected runtime:
    --fast    : ~30 seconds
    default   : ~2-3 minutes
    --with-rul: +1-2 minutes (LSTM training)

Output:
    Prints per-stage progress and summary metrics.
    Writes demo/results/end_to_end.json with numerical summary.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import numpy as np
import pandas as pd

from ai_cta.anomaly_detector import IsolationForestDetector
from ai_cta.calibration import OnlineCalibrator
from ai_cta.data import generate_synthetic_stream, inject_anomalies
from ai_cta.drift_detector import DriftDetector
from ai_cta.pipeline import SafetyPipeline
from ai_cta.risk_model import (
    ChannelLimits,
    ConformalThresholdCalibrator,
    RiskAggregator,
    RiskScorer,
)


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def stage_1_simulator(hours: int, seed: int) -> pd.DataFrame:
    banner(f"Stage 1: Simulator — generating {hours} h of telemetry")
    # Enough samples so conformal calibration has ≥ 20 windows after
    # IsolationForest windowing (window_size=64, stride=32).
    n_train = max(2500, hours * 60)
    n_test = max(1500, hours * 30)
    train = generate_synthetic_stream(n_samples=n_train, random_state=seed)
    test, labels = inject_anomalies(
        generate_synthetic_stream(n_samples=n_test, random_state=seed + 1),
        n_anomalies=max(10, n_test // 100),
        random_state=seed + 2,
    )
    print(f"  Train stream: {len(train)} samples")
    print(f"  Test stream:  {len(test)} samples ({int(labels.sum())} injected anomalies)")
    return train, test, labels


def stage_2_anomaly_detection(train: pd.DataFrame, test: pd.DataFrame, seed: int):
    banner("Stage 2: Anomaly detection — IsolationForest + conformal calibration")
    det = IsolationForestDetector(
        window_size=64, stride=32, use_spectral=False, random_state=seed,
    )
    det.fit(train.drop(columns=["timestamp"]))
    print(f"  Fitted with engineered features")

    calib = ConformalThresholdCalibrator(alpha=0.05)
    train_scores = det.decision_function(train.drop(columns=["timestamp"]))
    calib.calibrate(train_scores)
    print(f"  Conformal threshold (α=0.05): {calib.threshold_:.3f}")

    test_scores = det.decision_function(test.drop(columns=["timestamp"]))
    pad = len(test) - len(test_scores)
    if pad > 0:
        test_scores = np.concatenate([np.full(pad, test_scores[0]), test_scores])
    return det, calib, test_scores


def stage_3_risk_scoring(test: pd.DataFrame, test_scores: np.ndarray) -> np.ndarray:
    banner("Stage 3: Risk scoring — ML + rule-based channel limits")
    scorer = RiskScorer(
        ml_weight=0.6,
        limits={
            "temperature": ChannelLimits(warn_low=45, warn_high=55,
                                          alarm_low=30, alarm_high=70),
            "vibration":   ChannelLimits(warn_low=0.1, warn_high=0.5,
                                          alarm_low=0.0, alarm_high=0.8),
            "pressure":    ChannelLimits(warn_low=0.9, warn_high=1.1,
                                          alarm_low=0.7, alarm_high=1.3),
        },
    )
    risks = np.zeros(len(test))
    for i, row in test.iterrows():
        risks[i] = scorer.score(
            test_scores[min(i, len(test_scores) - 1)],
            {c: row[c] for c in ["temperature", "vibration", "pressure"]},
        )
    print(f"  Risk range: [{risks.min():.3f}, {risks.max():.3f}]")
    print(f"  Mean risk:  {risks.mean():.3f}")
    return risks


def stage_4_risk_aggregation(test_scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """3-component hybrid risk with SLSQP calibration."""
    banner("Stage 4: Risk aggregation — 3-component hybrid (SLSQP on 2-simplex)")
    from scipy.special import expit
    r_anom = expit(2 * (test_scores - np.percentile(test_scores, 70)))
    r_rul = np.zeros_like(r_anom)
    for idx in np.where(labels > 0)[0]:
        horizon = np.arange(min(60, len(r_rul) - idx))
        r_rul[idx : idx + len(horizon)] = np.maximum(
            r_rul[idx : idx + len(horizon)], np.exp(-horizon / 20.0)
        )
    r_nn = expit(2.0 * r_anom - 0.5 + np.random.default_rng(0).normal(0, 0.2, len(r_anom)))

    agg = RiskAggregator()
    agg.calibrate_weights(r_anom, r_rul, r_nn, labels.astype(float))
    print(f"  SLSQP weights (w_anom, w_rul, w_nn): {agg.w.round(3).tolist()}")
    final = agg.w[0] * r_anom + agg.w[1] * r_rul + agg.w[2] * r_nn
    return {"weights": agg.w.tolist(), "r_final_max": float(final.max())}


def stage_5_drift_detection(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    banner("Stage 5: Drift detection — KS + Bonferroni on streaming windows")
    cols = [c for c in train.columns if c != "timestamp"]
    det = DriftDetector(method="ks", threshold=1e-6, bonferroni=True)
    det.fit(train[cols])

    # Split test into 6 windows and check each
    results = []
    window_size = max(120, len(test) // 6)
    for i in range(0, len(test) - window_size + 1, window_size):
        win = test.iloc[i : i + window_size][cols]
        if len(win) < 30:
            continue
        report = det.detect(win)
        results.append({"start": i, "drift": bool(report.drift_detected)})
    n_drift = sum(1 for r in results if r["drift"])
    print(f"  Checked {len(results)} windows; drift detected in {n_drift}")
    return {"n_windows": len(results), "n_drift": n_drift}


def stage_6_online_calibration() -> dict[str, Any]:
    banner("Stage 6: Online calibration — scheduled recalibration demo")
    calib = OnlineCalibrator(alpha=0.05, buffer_size=1000, min_recalibrate=200,
                              schedule_seconds=0)
    rng = np.random.default_rng(0)
    drift_events = 0
    for step in range(400):
        score = float(rng.normal(0 if step < 300 else 0.5, 1.0))
        calib.add(score)
        update = calib.maybe_recalibrate()
        if update is not None:
            drift_events += 1
            if drift_events <= 3:
                print(f"  Step {step}: recalibration fired; threshold={update.threshold:.3f}")
    print(f"  Total recalibrations: {drift_events}")
    return {"recalibrations": drift_events}


def stage_7_streaming_pipeline(
    det: IsolationForestDetector, test: pd.DataFrame
) -> dict[str, Any]:
    banner("Stage 7: Streaming pipeline — end-to-end orchestration with alerts")
    scorer = RiskScorer(
        ml_weight=0.6,
        limits={
            "temperature": ChannelLimits(45, 55, 30, 70),
            "vibration":   ChannelLimits(0.1, 0.5, 0.0, 0.8),
            "pressure":    ChannelLimits(0.9, 1.1, 0.7, 1.3),
        },
    )
    pipeline = SafetyPipeline(
        detector=det, risk_scorer=scorer, window_size=64, alert_threshold=0.7,
    )

    alerts = []
    events_count = 0
    for event in pipeline.run(test.to_dict(orient="records")):
        events_count += 1
        if event.risk_score >= 0.7:
            alerts.append({
                "t": str(event.timestamp), "risk": float(event.risk_score),
                "level": event.risk_level,
            })

    print(f"  Events emitted: {events_count}")
    print(f"  Alerts (risk ≥ 0.7): {len(alerts)}")
    if alerts:
        print(f"  First alert: {alerts[0]}")
        if len(alerts) > 1:
            print(f"  Last  alert: {alerts[-1]}")
    return {"events": events_count, "alerts": len(alerts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="Run 2-hour mini demo")
    parser.add_argument("--with-rul", action="store_true",
                        help="Also train RUL model (requires TensorFlow)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hours = 2 if args.fast else 24

    print("\n╔" + "═" * 68 + "╗")
    print("║" + f"  AI-CTA end-to-end demo — {hours} h synthetic scenario".ljust(68) + "║")
    print("║" + "  See monograph Chapter 12 (demo project) for context.".ljust(68) + "║")
    print("╚" + "═" * 68 + "╝")

    total_start = time.perf_counter()
    summary: dict[str, Any] = {}

    train, test, labels = stage_1_simulator(hours, args.seed)
    summary["stage_1"] = {"train_len": len(train), "test_len": len(test),
                          "anomalies": int(labels.sum())}

    det, calib, test_scores = stage_2_anomaly_detection(train, test, args.seed)
    summary["stage_2"] = {"threshold": float(calib.threshold_)}

    risks = stage_3_risk_scoring(test, test_scores)
    summary["stage_3"] = {"risk_max": float(risks.max()),
                           "risk_mean": float(risks.mean())}

    summary["stage_4"] = stage_4_risk_aggregation(test_scores, labels)

    if args.with_rul:
        banner("Stage 4b: RUL prediction (LSTM quantile)")
        try:
            from ai_cta.rul_estimator import RULEstimator
            est = RULEstimator(window_size=16, epochs=5, quantiles=(0.1, 0.5, 0.9))
            # Quick fake RUL labels for demo
            fake_rul = np.linspace(125, 0, len(train)).astype(np.float32)
            est.fit(train.drop(columns=["timestamp"]), fake_rul)
            preds = est.predict_quantiles(train.drop(columns=["timestamp"]).iloc[-16:])
            print(f"  Median RUL prediction (last window): {preds[0.5][-1]:.1f} cycles")
            summary["stage_4b"] = {"rul_prediction": float(preds[0.5][-1])}
        except ImportError:
            print("  TensorFlow not installed — skipping RUL stage")

    summary["stage_5"] = stage_5_drift_detection(train, test)
    summary["stage_6"] = stage_6_online_calibration()
    summary["stage_7"] = stage_7_streaming_pipeline(det, test)

    elapsed = time.perf_counter() - total_start
    summary["total_seconds"] = round(elapsed, 1)

    banner(f"Demo complete in {elapsed:.1f} s")

    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "end_to_end.json"
    out_file.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Summary saved to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
