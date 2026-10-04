"""Leakage-safe CWRU bearing-fault detection benchmark (Phase 2 corrected protocol).

Why this script exists
----------------------
The shipped ``benchmarks/run_cwru_benchmark.py`` is NOT leakage-safe: it fits the
detector on *all* normal windows and then evaluates on the full mixed set, so every
normal training window also appears in the evaluation set, and windows overlap
(stride = window/2). The CWRU methodological literature (Hendriks et al. 2022;
Smith & Randall 2015; Rosa et al. 2024) shows such constructions inflate accuracy.

This corrected runner enforces a leakage-safe, cross-load protocol:

* Windows are NON-overlapping (stride == window).
* The four normal baseline recordings 97/98/99/100 correspond to motor loads
  0/1/2/3 hp. They are assigned to DISJOINT roles:
      train  : 97, 98   (loads 0, 1)  -> fit IsolationForest on normal only
      calib  : 99        (load 2)      -> split-conformal threshold (held-out normal)
      test   : 100       (load 3)      -> normal test windows (measures false-alarm rate)
             + ALL fault recordings    -> fault test windows (test-only, never seen)
  Train, calibration, and test normal data come from different physical recordings
  AND different load conditions, so there is no recording-level or window-overlap leak.
* The decision threshold is set by ai_cta.ConformalThresholdCalibrator at nominal
  alpha on the held-out calibration recording, directly exercising the C3 component
  on real data. We also report threshold-free ROC-AUC / PR-AUC.

Outputs (per seed, to analysis/results/):
    cwru_leakage_safe.csv   columns: seed, method, alpha, n_train, n_calib,
        n_test_normal, n_test_fault, threshold, precision, recall, f1, far,
        roc_auc, pr_auc, fit_time_sec

The IsolationForest random_state is varied over seeds 0..n_seeds-1; the split is
fixed and deterministic. Everything is fit on train-normal only; the calibration
recording sets the threshold; metrics are computed on the held-out test set.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time

import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

from ai_cta.risk_model import ConformalThresholdCalibrator

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "benchmarks" / "data" / "cwru"
RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"

TRAIN_FILES = ["97.mat", "98.mat"]   # loads 0, 1 hp
CALIB_FILES = ["99.mat"]             # load 2 hp (held-out normal for conformal)
TESTNORMAL_FILES = ["100.mat"]       # load 3 hp (held-out normal for FAR)


def load_de_signal(path: pathlib.Path) -> np.ndarray:
    """Return the drive-end vibration channel, preferring the key whose number
    matches the file stem (file 99.mat ships an extra X098 key)."""
    mat = loadmat(path)
    de_keys = [k for k in mat.keys() if k.endswith("_DE_time")]
    if not de_keys:
        raise ValueError(f"No _DE_time channel in {path}")
    stem_digits = "".join(ch for ch in path.stem if ch.isdigit())
    preferred = [k for k in de_keys if stem_digits in k]
    key = preferred[0] if preferred else max(de_keys, key=lambda k: mat[k].size)
    return mat[key].flatten().astype(np.float64)


def windows(signal: np.ndarray, window: int) -> np.ndarray:
    """Non-overlapping windows (stride == window)."""
    n = (len(signal) // window) * window
    return signal[:n].reshape(-1, window)


def extract_features(win: np.ndarray, sampling_rate: float) -> np.ndarray:
    """13 standard vibration features per window (time + frequency domain)."""
    x = win
    mean = x.mean()
    std = x.std() + 1e-12
    rms = np.sqrt(np.mean(x**2)) + 1e-12
    peak = np.max(np.abs(x))
    p2p = x.max() - x.min()
    crest = peak / rms
    shape = rms / (np.mean(np.abs(x)) + 1e-12)
    impulse = peak / (np.mean(np.abs(x)) + 1e-12)
    kurt = float(kurtosis(x))
    sk = float(skew(x))
    # Frequency domain: spectral energy in three bands + spectral entropy.
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sampling_rate)
    mag = np.abs(np.fft.rfft(x))
    psd = mag**2
    total = psd.sum() + 1e-12
    nyq = sampling_rate / 2.0
    band_lo = psd[freqs < nyq / 3].sum() / total
    band_mid = psd[(freqs >= nyq / 3) & (freqs < 2 * nyq / 3)].sum() / total
    p = psd / total
    spec_entropy = float(-np.sum(p * np.log(p + 1e-12)))
    return np.array([
        mean, std, rms, peak, p2p, crest, shape, impulse, kurt, sk,
        band_lo, band_mid, spec_entropy,
    ], dtype=np.float64)


def build_feature_matrix(files, subdir, window, sampling_rate):
    feats = []
    for fname in files:
        sig = load_de_signal(DATA_DIR / subdir / fname)
        for w in windows(sig, window):
            feats.append(extract_features(w, sampling_rate))
    return np.vstack(feats) if feats else np.empty((0, 13))


def run_seed(seed, window, sampling_rate, alphas):
    fault_files = sorted(p.name for p in (DATA_DIR / "fault").glob("*.mat"))

    t0 = time.perf_counter()
    X_train = build_feature_matrix(TRAIN_FILES, "normal", window, sampling_rate)
    X_calib = build_feature_matrix(CALIB_FILES, "normal", window, sampling_rate)
    X_testN = build_feature_matrix(TESTNORMAL_FILES, "normal", window, sampling_rate)
    X_testF = build_feature_matrix(fault_files, "fault", window, sampling_rate)

    scaler = RobustScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xcal = scaler.transform(X_calib)
    XtN = scaler.transform(X_testN)
    XtF = scaler.transform(X_testF)

    iforest = IsolationForest(n_estimators=200, contamination="auto",
                              random_state=seed, n_jobs=-1).fit(Xtr)
    fit_time = time.perf_counter() - t0

    # Anomaly score: higher == more anomalous. Calibrate to a stable scale
    # using train statistics, then logistic-squash (matches the package's
    # IsolationForestDetector calibration convention).
    raw_tr = -iforest.score_samples(Xtr)
    mu, sd = float(raw_tr.mean()), float(raw_tr.std() + 1e-9)

    def score(Z):
        z = (-iforest.score_samples(Z) - mu) / sd
        return 1.0 / (1.0 + np.exp(-z))

    s_cal = score(Xcal)
    s_testN = score(XtN)
    s_testF = score(XtF)

    y_test = np.concatenate([np.zeros(len(s_testN)), np.ones(len(s_testF))])
    s_test = np.concatenate([s_testN, s_testF])

    roc = float(roc_auc_score(y_test, s_test))
    pr = float(average_precision_score(y_test, s_test))

    rows = []
    for alpha in alphas:
        cal = ConformalThresholdCalibrator(alpha=alpha).calibrate(s_cal)
        thr = cal.threshold_
        pred = (s_test > thr).astype(int)
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        tn = int(((pred == 0) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        far = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append({
            "seed": seed, "method": "IsolationForest+conformal", "alpha": alpha,
            "n_train": len(Xtr), "n_calib": len(Xcal),
            "n_test_normal": len(s_testN), "n_test_fault": len(s_testF),
            "threshold": round(thr, 5), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4), "far": round(far, 4),
            "roc_auc": round(roc, 4), "pr_auc": round(pr, 4),
            "fit_time_sec": round(fit_time, 2),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--sampling-rate", type=float, default=12000.0)
    ap.add_argument("--alphas", type=str, default="0.01,0.05,0.10")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for seed in range(args.n_seeds):
        rows = run_seed(seed, args.window, args.sampling_rate, alphas)
        all_rows.extend(rows)
        head = rows[1] if len(rows) > 1 else rows[0]  # alpha=0.05 row
        print(f"seed={seed}  ROC-AUC={head['roc_auc']:.3f}  PR-AUC={head['pr_auc']:.3f}  "
              f"(@a=0.05) F1={head['f1']:.3f} recall={head['recall']:.3f} FAR={head['far']:.3f}")

    out = RESULTS_DIR / "cwru_leakage_safe.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows -> {out}")
    print(f"Protocol: train={TRAIN_FILES} calib={CALIB_FILES} testN={TESTNORMAL_FILES} "
          f"+ all fault files; non-overlapping {args.window}-sample windows.")


if __name__ == "__main__":
    main()
