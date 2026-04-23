"""Experiment E6 (monograph § 13.10): API performance load test.

Measures the FastAPI service latency and throughput using the in-process
TestClient, which strips network stack overhead so the numbers reflect
the model-inference path rather than any particular reverse-proxy or
Kubernetes configuration.

Metrics reported (matching book Table 13.6):
    p50, p95, p99 latency (ms)
    throughput (req/s, measured as requests / total wall-clock)
    CPU-seconds per request

Two workloads:
    single-sample - /predict with one sensor reading
    batch-window  - /score-batch with a 60-sample window

Usage
-----

    python benchmarks/run_experiment_e6.py --n-requests 500

Note that `pip install -e ".[api]"` is required.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time

import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def _check_fastapi() -> None:
    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "E6 requires fastapi + httpx. "
            'Install via: pip install "fastapi" "httpx"'
        ) from exc


def single_sample_latency(n_requests: int) -> list[float]:
    """Send `n_requests` single-sample /predict calls; return per-call latency (ms)."""
    from fastapi.testclient import TestClient

    from api.main import app, build_default_pipeline

    build_default_pipeline()
    client = TestClient(app)
    rng = np.random.default_rng(0)
    client.get("/health")

    latencies: list[float] = []
    for _ in range(n_requests):
        payload = {
            "temperature": float(50 + rng.normal(0, 2)),
            "vibration": float(0.25 + rng.normal(0, 0.05)),
            "pressure": float(1.0 + rng.normal(0, 0.02)),
        }
        t0 = time.perf_counter()
        r = client.post("/predict", json=payload)
        dt = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            continue
        latencies.append(dt)
    return latencies


def batch_window_latency(n_requests: int, window_size: int = 64) -> list[float]:
    from fastapi.testclient import TestClient

    from api.main import app, build_default_pipeline

    build_default_pipeline()
    client = TestClient(app)
    client.get("/health")

    rng = np.random.default_rng(1)
    latencies: list[float] = []
    for _ in range(n_requests):
        readings = [
            {
                "temperature": float(50 + rng.normal(0, 2)),
                "vibration": float(0.25 + rng.normal(0, 0.05)),
                "pressure": float(1.0 + rng.normal(0, 0.02)),
            }
            for _ in range(window_size)
        ]
        payload = {"readings": readings}
        t0 = time.perf_counter()
        r = client.post("/score-batch", json=payload)
        dt = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            continue
        latencies.append(dt)
    return latencies


def summarize(name: str, latencies: list[float], wall_seconds: float) -> dict:
    arr = np.asarray(latencies)
    return {
        "workload": name,
        "n_requests": len(arr),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "throughput_rps": round(len(arr) / wall_seconds, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-requests", type=int, default=500)
    args = parser.parse_args()

    _check_fastapi()

    rows = []
    for name, fn, kwargs in [
        ("single_sample (/predict)", single_sample_latency, {"n_requests": args.n_requests}),
        ("batch_window (/score-batch, 64)", batch_window_latency,
         {"n_requests": max(50, args.n_requests // 5), "window_size": 64}),
    ]:
        print(f"\n=== {name} ===")
        t0 = time.perf_counter()
        latencies = fn(**kwargs)
        wall = time.perf_counter() - t0
        row = summarize(name, latencies, wall)
        print(
            f"  n={row['n_requests']}  p50={row['p50_ms']}  "
            f"p95={row['p95_ms']}  p99={row['p99_ms']}  "
            f"rps={row['throughput_rps']}"
        )
        rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "experiment_e6_api.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
