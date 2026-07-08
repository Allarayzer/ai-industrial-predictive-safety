"""W3 fix: extended physical validation on the NASA Randomized Battery Usage dataset (RW1-RW12,
room-temperature groups) — Supplementary external replication of the battery replay.

Pipeline mirrors the four-cell replay (e3_battery_noleak.py) with the same leak-free principle:
- unit of analysis = reference-discharge event (fixed-current capacity benchmark performed
  after fixed intervals of randomized usage), capacity = coulomb-counted Ah over the step;
- features: age, sqrt(age), lagged capacity, discharge duration + lag + delta, voltage stats,
  late-discharge voltage, temperature stats and rise, energy proxy + lag + delta —
  current-cycle capacity and its first difference are EXCLUDED (not reconstructible);
- endpoint: remaining reference cycles, near-end label at ceil(0.15 * series length) per cell
  (the 4-cell replay's 20-cycle horizon equals ~13-17% of those cells' series);
- folds: within each 4-cell group (part 1: RW9-12; part 2: RW3-6; part 3: RW1/2/7/8), the same
  rotation as the primary replay (test 1 / calibrate 1 / train 2), so 12 cells are each tested
  exactly once; channels, asynchronous delivery seeds, policies, and the governed controller
  are byte-identical to the primary replay.

Usage: python e7_rw_battery.py [parse|run]
  parse — extract reference discharges from the .mat archives -> rw_reference_cycles.csv
  run   — replay all policies -> rw_fold_results.csv / rw_summary.csv
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RW = ROOT / "analysis2" / "data" / "rw"
OUT = ROOT / "analysis2" / "results"

spec = importlib.util.spec_from_file_location("bat2", ROOT / "analysis2" / "scripts" / "e3_battery_noleak.py")
bat = importlib.util.module_from_spec(spec)
sys.modules["bat2"] = bat
spec.loader.exec_module(bat)

GROUPS = {
    "part1": ["RW9", "RW10", "RW11", "RW12"],
    "part2": ["RW3", "RW4", "RW5", "RW6"],
    "part3": ["RW1", "RW2", "RW7", "RW8"],
}
ZIPS = {
    "part1": "1.Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post.zip",
    "part2": "2.Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post.zip",
    "part3": "3.Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post.zip",
}


def load_steps(matpath: pathlib.Path):
    from scipy.io import loadmat
    m = loadmat(str(matpath), squeeze_me=True, struct_as_record=False)
    data = m["data"]
    steps = data.step if hasattr(data, "step") else data
    return steps


def parse_cell(matpath: pathlib.Path, battery: str) -> pd.DataFrame:
    steps = load_steps(matpath)
    rows = []
    ref_idx = 0
    for st in np.atleast_1d(steps):
        comment = str(getattr(st, "comment", "")).strip().lower()
        if comment != "reference discharge":
            continue
        t = np.atleast_1d(np.asarray(st.relativeTime, dtype=float))
        v = np.atleast_1d(np.asarray(st.voltage, dtype=float))
        c = np.atleast_1d(np.asarray(st.current, dtype=float))
        temp = np.atleast_1d(np.asarray(st.temperature, dtype=float))
        if len(t) < 10:
            continue
        order = np.argsort(t); t, v, c, temp = t[order], v[order], c[order], temp[order]
        dt = np.diff(t, prepend=t[0])
        cap_ah = float(np.sum(np.abs(c) * dt) / 3600)
        if cap_ah < 0.2:  # discard degenerate/aborted reference steps
            continue
        k = max(3, len(v) // 10)
        rows.append(dict(
            Battery=battery, cycle=ref_idx,
            capacity=cap_ah, duration=float(t.max() - t.min()),
            v_mean=float(v.mean()), v_min=float(v.min()), v_max=float(v.max()),
            v_std=float(v.std()), v_early=float(v[:k].mean()), v_late=float(v[-k:].mean()),
            i_mean=float(np.abs(c).mean()), i_std=float(c.std()),
            temp_mean=float(temp.mean()), temp_max=float(temp.max()),
            temp_rise=float(temp.max() - temp[0]),
            energy_proxy=float(np.sum(np.abs(v * c) * dt) / 3600)))
        ref_idx += 1
    return pd.DataFrame(rows)


def parse_all() -> pd.DataFrame:
    frames = []
    for part, cells in GROUPS.items():
        zp = RW / ZIPS[part]
        exdir = RW / part
        if not exdir.exists():
            print(f"extracting {zp.name}", flush=True)
            with zipfile.ZipFile(zp) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".mat")]
                z.extractall(exdir, members=names)
        mats = sorted(exdir.rglob("*.mat"))
        for cell in cells:
            cand = [p for p in mats if p.stem.upper().endswith(cell) or cell in p.stem.upper()]
            assert cand, f"no .mat for {cell} in {part}: {[p.name for p in mats]}"
            df = parse_cell(cand[0], cell)
            print(f"{cell}: {len(df)} reference discharges (file {cand[0].name})", flush=True)
            frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(OUT / "rw_reference_cycles.csv", index=False)
    return d


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    z = d.sort_values(["Battery", "cycle"]).reset_index(drop=True)
    for col in ["capacity", "duration", "v_mean", "v_min", "v_std", "v_late",
                "temp_mean", "temp_max", "temp_rise", "energy_proxy"]:
        z["lag_" + col] = z.groupby("Battery")[col].shift(1)
        z["delta_" + col] = z.groupby("Battery")[col].diff()
    z["age"] = z.groupby("Battery").cumcount().astype(float)
    z["age_sqrt"] = np.sqrt(z.age)
    z = z.dropna().reset_index(drop=True)
    max_cycle = z.groupby("Battery").cycle.transform("max")
    n_cycles = z.groupby("Battery").cycle.transform("count")
    z["rul"] = (max_cycle - z.cycle).clip(lower=0)
    z["horizon"] = np.ceil(0.15 * n_cycles).astype(int)
    z["label"] = (z.rul <= z.horizon).astype(int)
    return z


def run_replay(d: pd.DataFrame) -> None:
    rows = []
    healthy_q = 0.30  # healthy reference = first (1-0.30) of life by RUL fraction, mirrors rul>40
    for part, cells in GROUPS.items():
        folds = [(cells[0], cells[1], [cells[2], cells[3]]),
                 (cells[1], cells[2], [cells[3], cells[0]]),
                 (cells[2], cells[3], [cells[0], cells[1]]),
                 (cells[3], cells[0], [cells[1], cells[2]])]
        for fi, (testb, calb, trainbs) in enumerate(folds):
            tr = d[d.Battery.isin(trainbs)]; cal = d[d.Battery == calb]; te = d[d.Battery == testb]
            if min(len(tr), len(cal), len(te)) < 12 or te.label.nunique() < 2 or cal.label.nunique() < 2:
                print(f"skip {part} fold {fi} ({testb}): too small/degenerate", flush=True)
                continue
            m = fit_models_rw(tr)
            cc = predict_models_rw(m, cal); tc = predict_models_rw(m, te)
            w = bat.optimize_weights(cc, cal.label.to_numpy())
            cl, ca = bat.delivered_cache(cc, 100 + fi); tl, ta = bat.delivered_cache(tc, 200 + fi)
            cs = bat.fused_from_cache(cl, ca, w); ts = bat.fused_from_cache(tl, ta, w)
            q0 = bat.conformal_q(cs[cal.label.to_numpy() == 0])
            y = te.label.to_numpy(int)
            for meth in ["Static", "Scheduled", "ACI", "Quantile tracking", "SF-OGD"]:
                if meth == "Static":
                    p = (ts > q0).astype(int); upd = 0
                else:
                    p, th, upd = bat.run_threshold(meth, ts, y, q0)
                mm = bat.metrics(y, p, ts)
                rows.append(dict(part=part, fold=fi, test_battery=testb, method=meth,
                                 updates=upd, **mm))
            g = governed_rw(tc, tl, ta, w, q0, y)
            rows.append(dict(part=part, fold=fi, test_battery=testb,
                             method="Governed escalation", **g))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "rw_fold_results.csv", index=False)
    S = R.groupby("method").agg(
        mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"), far_mean=("far", "mean"),
        recall_mean=("recall", "mean"), cost_mean=("expected_cost", "mean"),
        auc_mean=("roc_auc", "mean"), updates_mean=("updates", "mean"),
        n_folds=("mcc", "size")).reset_index()
    S.to_csv(OUT / "rw_summary.csv", index=False)
    print(S.round(4).to_string(index=False))


def fit_models_rw(train):
    return bat.fit_models_generic(train) if hasattr(bat, "fit_models_generic") else _fit(train)


def _fit(train):
    from dataclasses import dataclass
    from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = train[FEATURES].to_numpy(float)
    sc = StandardScaler().fit(X); Z = sc.transform(X)
    healthy = (train.rul / train.groupby("Battery").rul.transform("max").clip(lower=1)) > healthy_frac
    if healthy.sum() < 10:
        healthy = train.rul > train.rul.median()
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=11).fit(Z[healthy.to_numpy()])
    ref = np.sort(-iso.score_samples(Z[healthy.to_numpy()]))
    rul = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=.035,
                                    loss="huber", random_state=12).fit(Z, train.rul)
    risk = LogisticRegression(max_iter=2000, class_weight={0: 1, 1: 10}, C=.5,
                              random_state=13).fit(Z, train.label)
    return bat.BatteryModels(sc, iso, rul, risk, ref)


FEATURES = ["age", "age_sqrt", "lag_capacity", "duration", "lag_duration", "delta_duration",
            "v_mean", "v_min", "v_std", "v_late", "lag_v_mean", "delta_v_mean",
            "temp_mean", "temp_max", "temp_rise", "lag_temp_mean",
            "energy_proxy", "lag_energy_proxy", "delta_energy_proxy"]
healthy_frac = 0.35


def predict_models_rw(m, d):
    from scipy.special import expit
    Z = m.scaler.transform(d[FEATURES].to_numpy(float))
    raw = -m.iso.score_samples(Z)
    anom = np.searchsorted(m.healthy_ref, raw, side="right") / len(m.healthy_ref)
    h = d.horizon.to_numpy(float)
    rh = np.clip(m.rul.predict(Z), 0, 200)
    rr = expit((h - rh) / np.maximum(h / 4.0, 1.0))
    risk = m.risk.predict_proba(Z)[:, 1]
    return np.column_stack([anom, rr, risk])


def governed_rw(tc, latest, avail, w, q0, y):
    from sklearn.metrics import roc_auc_score
    curw = w.copy(); q = q0
    p = np.zeros(len(y), int); w_hist = []; buf = []; wu = tu = esc = 0
    for i in range(len(y)):
        ww = curw * avail[i]; ww = ww / ww.sum() if ww.sum() > 0 else np.ones(3) / 3
        p[i] = float(latest[i] @ ww) > q; w_hist.append(curw.copy())
        r = i - 2
        if r >= 0: buf.append((tc[r].copy(), int(y[r]))); buf = buf[-70:]
        if i > 0 and i % 12 == 0 and len(buf) >= 24:
            B = np.array([x for x, _ in buf]); Y = np.array([z for _, z in buf])
            sp = max(16, int(.7 * len(B))); Bt, Yt = B[:sp], Y[:sp]; Bv, Yv = B[sp:], Y[sp:]
            if len(np.unique(Yt)) == 2 and len(np.unique(Yv)) == 2:
                old_s = Bv @ curw
                old_cost = bat.metrics(Yv, (old_s > q).astype(int), old_s)["expected_cost"]
                cand = bat.optimize_weights_grid(Bt, Yt, curw, step=.1, shrink=.08)
                new_s = Bv @ cand; normals = Bv[Yv == 0] @ cand
                if len(normals) >= 5:
                    candq = bat.conformal_q(normals)
                    new_cost = bat.metrics(Yv, (new_s > candq).astype(int), new_s)["expected_cost"]
                    aucs = []
                    for j in range(3):
                        try: aucs.append(roc_auc_score(Yv, Bv[:, j]))
                        except Exception: aucs.append(.5)
                    if max(aucs) < .56: esc += 1
                    elif new_cost <= old_cost * .95 and np.sum(abs(cand - curw)) <= 1.2:
                        curw = cand; q = candq; wu += 1; tu += 1
            normal_scores = np.array([x @ curw for x, yy in buf if yy == 0])
            if len(normal_scores) >= 15:
                candq = bat.conformal_q(normal_scores)
                if abs(candq - q) / max(abs(q), .05) < .6: q = candq; tu += 1
    score_dyn = np.array([latest[i] @ (w_hist[i] * avail[i] / max((w_hist[i] * avail[i]).sum(), 1e-12))
                          for i in range(len(y))])
    mm = bat.metrics(y, p, score_dyn)
    return dict(updates=tu, weight_updates=wu, model_escalations=esc, **mm)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("parse", "all"):
        d = parse_all()
    else:
        d = pd.read_csv(OUT / "rw_reference_cycles.csv")
    if mode in ("run", "all"):
        z = prepare(d)
        print(z.groupby("Battery").agg(n=("cycle", "size"), cap0=("capacity", "first"),
                                       capN=("capacity", "last")).to_string())
        run_replay(z)
