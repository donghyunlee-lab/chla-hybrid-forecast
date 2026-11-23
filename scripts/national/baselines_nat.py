"""Baselines of the national benchmark, on CPU.
Per station and record (6, 12, 24, 36 months to 2023-12-31): climatology and damped anomaly, ARIMA and harmonic ARIMA (the
functions of scripts/clean_revision/bench_baselines_v7.py with the national record end), and residual XGBoost trained on the
record (seeds 42, 123, 456, 1, 2). Per fold and seed: residual XGBoost trained on the fold's source stations (36-month record)
and applied to every station of the held-out basin (zero-shot). Also writes the persistence forecasts of all stations
(persistence.csv) for the analysis. Idempotent (DONE markers). Usage: PINN_RESULTS_DIR=<results folder> baselines_nat.py"""
from __future__ import annotations
import os, sys, time
from multiprocessing import Pool
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "clean_revision")); sys.path.insert(0, str(ROOT / "scripts" / "national"))
import nat as N                                                                  # noqa: E402
C = N.patch()
import bench_baselines_v7 as B7                                                  # noqa: E402
from baselines import save_run, xgboost_fit_predict                              # noqa: E402
B7.RECORD_END = N.RECORD_END
_df, _sc, _per = None, {}, None


def done(tag):
    return (C.RESULTS / "runs" / tag / "DONE").exists()


def init():
    global _df, _per
    _df = N.load_daily(); _per = pd.read_csv(C.RESULTS / "persistence.csv", dtype={"station": str}, parse_dates=["target_date"])


def scaler(fold):
    if fold not in _sc:
        N.set_fold(fold)
        m = ((_df["date"] >= C.SPLITS["train"][0]) & (_df["date"] <= C.SPLITS["train"][1])).to_numpy()
        _sc[fold] = C.Scaler().fit(_df, m, C.SOURCE)
    N.set_fold(fold)
    return _sc[fold]


def task_stat(code, months):
    out = []
    if not (done(f"bl_climatology_{code}_r{months}") and done(f"bl_damped_{code}_r{months}")):
        keys = _per[(_per.station == code) & (pd.to_datetime(_per.anchor_date) >= C.SPLITS["test"][0])][["station", "anchor_date", "horizon", "target_date", "y_true_raw"]]
        cl, dam = B7.seasonal(_df, code, months, keys)
        save_run(f"bl_climatology_{code}_r{months}", cl, {"method": "climatology", "stations": [code], "record_months": months, "fit": "annual Fourier K=2 on the record"})
        save_run(f"bl_damped_{code}_r{months}", dam, {"method": "damped anomaly", "stations": [code], "record_months": months, "fit": "climatology and anomaly ACF on the record"})
        out.append("seasonal")
    for harmonic, name in ((False, "arima"), (True, "harmonic")):
        tag = f"bl_{name}_{code}_r{months}"
        if not done(tag):
            t0 = time.time(); f, info = B7.arima_like(_df, code, months, harmonic)
            save_run(tag, f, {"method": "harmonic ARIMA" if harmonic else "ARIMA", "stations": [code], "record_months": months, "record_end": N.RECORD_END,
                              "chosen": info, "time_s": time.time() - t0})
            out.append(name)
    return code, months, out


def task_xgb(code, months, seed):
    tag = f"bl_xgboost_{code}_r{months}_s{seed}"
    if done(tag):
        return tag
    sc = scaler(N.BASIN_OF[code]); arr = C.build_arrays(_df, sc, [code], C.SEQ_LEN, C.HORIZONS)
    trt, tet = C.split_indices(arr, "train", [code]), C.split_indices(arr, "test", [code])
    trm, vam, tr_end, rec_end = C.record_window(arr, trt, months, N.RECORD_END, N.VAL_FRAC)
    masks = (C.mask_targets_after(arr, trm, tr_end), C.mask_targets_after(arr, vam, rec_end))
    t0 = time.time(); f, bi, ntr = xgboost_fit_predict(arr, sc, trm, vam, tet, seed, True, masks=masks)
    save_run(tag, f, {"method": "XGBoost-residual", "mode": "target_only", "stations": [code], "record_months": months, "record_end": N.RECORD_END,
                      "val_frac_in_record": N.VAL_FRAC, "seed": seed, "best_iteration": bi, "n_train": ntr, "time_s": time.time() - t0})
    return tag


def task_xgb_zero_shot(fold, seed):
    tags = [f"bl_xgbzs_{c}_s{seed}" for c in N.FOLDS[fold]]
    if all(done(t) for t in tags):
        return fold, seed
    sc = scaler(fold); src, tgt = list(C.SOURCE), list(C.TARGETS)
    arr = C.build_arrays(_df, sc, src + tgt, C.SEQ_LEN, C.HORIZONS)
    trs = C.split_indices(arr, "train", src)
    trm, vam, tr_end, rec_end = C.record_window(arr, trs, N.SOURCE_RECORD, N.RECORD_END, N.VAL_FRAC)
    masks = (C.mask_targets_after(arr, trm, tr_end), C.mask_targets_after(arr, vam, rec_end))
    tet = C.split_indices(arr, "test", tgt)
    t0 = time.time(); f, bi, ntr = xgboost_fit_predict(arr, sc, trm, vam, tet, seed, True, masks=masks)
    for c in tgt:
        save_run(f"bl_xgbzs_{c}_s{seed}", f[f.station == c].reset_index(drop=True),
                 {"method": "XGBoost-residual", "mode": "zero_shot", "fold": fold, "source_stations": src, "stations": [c], "record_months": N.SOURCE_RECORD,
                  "seed": seed, "best_iteration": bi, "n_train": ntr, "time_s": time.time() - t0})
    return fold, seed


def run(args):
    kind, *rest = args
    try:
        return {"stat": task_stat, "xgb": task_xgb, "xgbzs": task_xgb_zero_shot}[kind](*rest)
    except Exception as e:                                                       # report and continue; the scheduler log shows failures
        print("FAILED", args, repr(e), flush=True); return ("FAILED", args)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    per_path = C.RESULTS / "persistence.csv"
    if not per_path.exists():
        per = __import__("persistence_v4").persistence_frame()                 # all stations; last measured Chl-a in the 30-day window
        per_path.parent.mkdir(parents=True, exist_ok=True); per.to_csv(per_path, index=False)
    tasks = [("xgbzs", f, s) for s in N.SEEDS for f in N.BASINS]
    tasks += [("stat", c, m) for c in N.CODES for m in N.RECORDS]
    tasks += [("xgb", c, m, s) for s in N.SEEDS for c in N.CODES for m in N.RECORDS]
    with Pool(int(os.environ.get("NAT_CPU_WORKERS", "12")), initializer=init) as pool:
        for i, r in enumerate(pool.imap_unordered(run, tasks), 1):
            if i % 50 == 0 or (isinstance(r, tuple) and r and r[0] == "FAILED"):
                print(time.strftime("%H:%M:%S"), i, "/", len(tasks), r, flush=True)
    print("baselines done", time.strftime("%F %T"), flush=True)
