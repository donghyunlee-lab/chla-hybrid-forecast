"""Baselines of the shortened-record benchmark, fitted on the same shortened records as the networks: records of 6, 12 and 24
months that end on 2022-12-31 (rivers and reservoirs). Methods as in the paper, only the fitting data change:
  ARIMA (non-seasonal orders chosen by AIC) and harmonic ARIMA (annual Fourier terms, K = 2, ARIMA errors chosen by AIC),
  fitted on the record and extended day by day through the test years; climatology (annual Fourier, K = 2) and damped
  anomaly (climatology + anomaly of the last measurement times the anomaly autocorrelation) fitted on the record; residual
  XGBoost (pooled river stations, or the reservoir alone) trained on the record and validated on its last 20 %.
Persistence does not depend on the record and is taken from persistence_v4 in the analysis.
Writes to $PINN_RESULTS_DIR (any results folder). Idempotent (DONE markers)."""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from baselines import save_run, xgboost_fit_predict
from baselines_v4 import fourier as fourier_t
warnings.filterwarnings("ignore")
H = C.HORIZONS
RECORD_END, VAL_FRAC, RECORDS = "2022-12-31", 0.2, (6, 12, 24)
ORDERS = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (1, 0, 1), (2, 0, 1), (1, 0, 2)]


def done(tag):
    return (C.RESULTS / "runs" / tag / "DONE").exists()


def record_start(months):
    return pd.Timestamp(RECORD_END) - pd.DateOffset(months=months) + pd.Timedelta(days=1)


def arima_like(df, st, months, harmonic, K=2):
    """ARIMA or harmonic ARIMA fitted on the record of one station, then extended day by day through the test years."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    d = df[df.station == st].set_index("date")["chlorophyll_a"]; y = np.log1p(d.clip(lower=0))
    y_fit = y[(y.index >= record_start(months)) & (y.index <= RECORD_END)]
    ex = (lambda idx: fourier_t(idx, K)) if harmonic else (lambda idx: None)
    best = None
    for o in ORDERS:
        try:
            r = SARIMAX(y_fit, exog=ex(y_fit.index), order=o, trend="c" if o[1] == 0 else "n", enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
            if best is None or r.aic < best[0].aic:
                best = (r, o)
        except Exception as e:
            print("fail", st, months, harmonic, o, e, flush=True)
    res, o = best; cur = res; rows = []; test_end = pd.Timestamp(C.SPLITS["test"][1])
    # the gap between the record end and the test start is filled day by day as well (measured values, no refit)
    for day, val in y[(y.index > pd.Timestamp(RECORD_END)) & (y.index <= test_end)].items():
        cur = cur.extend(pd.Series([val], index=[day]), exog=ex([day]))
        if day < pd.Timestamp(C.SPLITS["test"][0]):
            continue
        fut = pd.date_range(day + pd.Timedelta(days=1), periods=max(H), freq="D")
        fc = np.asarray(cur.get_forecast(steps=max(H), exog=ex(fut)).predicted_mean)
        for h in H:
            td = day + pd.Timedelta(days=h); yt = d.get(td, np.nan) if td <= test_end else np.nan
            rows.append({"station": st, "anchor_date": str(day.date()), "horizon": h, "target_date": td, "observed": not np.isnan(yt), "y_true_raw": yt,
                         "y_pred_raw": float(np.expm1(fc[h - 1]))})
    return pd.DataFrame(rows), {"order": o, "aic": float(res.aic), "fourier_K": K if harmonic else 0}


def seasonal(df, st, months, keys):
    """Climatology and damped anomaly fitted on the record (v5_additions.clim_tables with the record as fit window)."""
    from v5_additions import fourier
    d = df[df.station == st].set_index("date").sort_index(); full = pd.date_range(d.index.min(), d.index.max(), freq="D"); d = d.reindex(full)
    y = np.log1p(d.chlorophyll_a.clip(lower=0)).to_numpy(); X = fourier(full.dayofyear.to_numpy())
    m = (full >= record_start(months)) & (full <= RECORD_END) & ~np.isnan(y); beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    t = pd.DataFrame({"y": y, "clim": X @ beta, "a": y - X @ beta}, index=full)
    s = t.a[(t.index >= record_start(months)) & (t.index <= RECORD_END)]
    acf = {L: float(pd.concat([s, s.shift(-L)], axis=1).dropna().corr().iloc[0, 1]) for L in range(1, 91)}
    rows = []
    for r in keys.itertuples(index=False):
        a0 = pd.Timestamp(r.anchor_date); h = int(r.horizon); tgt = a0 + pd.Timedelta(days=h)
        win = t.loc[a0 - pd.Timedelta(days=C.SEQ_LEN - 1):a0]; obs = win[win.y.notna()]; c = t.clim.get(tgt, np.nan)
        rho = acf.get(min(h + (a0 - obs.index[-1]).days, 90), 0.0) if len(obs) else 0.0
        rho = 0.0 if not np.isfinite(rho) else rho
        damped = c + (rho * obs.a.iloc[-1] if len(obs) else 0.0)
        rows.append({"station": st, "anchor_date": r.anchor_date, "horizon": h, "target_date": r.target_date, "observed": True, "y_true_raw": r.y_true_raw,
                     "clim": float(np.expm1(c)), "damped": float(np.expm1(damped))})
    b = pd.DataFrame(rows)
    cols = ["station", "anchor_date", "horizon", "target_date", "observed", "y_true_raw"]
    return b.assign(y_pred_raw=b.clim)[cols + ["y_pred_raw"]], b.assign(y_pred_raw=b.damped)[cols + ["y_pred_raw"]]


if __name__ == "__main__":
    df = C.load_daily()
    train_mask = ((df["date"] >= C.SPLITS["train"][0]) & (df["date"] <= C.SPLITS["train"][1])).to_numpy()
    scaler = C.Scaler().fit(df, train_mask, C.SOURCE)
    arr = C.build_arrays(df, scaler, list(C.STATION_IDS), C.SEQ_LEN, H)
    per = __import__("persistence_v4").persistence_frame()
    stations = list(C.SOURCE) + list(C.TARGETS)
    for months in RECORDS:
        for group, sts in (("rivers", list(C.SOURCE)),) + tuple((tg, [tg]) for tg in C.TARGETS):     # XGBoost: pooled rivers or one reservoir
            trt, tet = C.split_indices(arr, "train", sts), C.split_indices(arr, "test", sts)
            trm, vam, tr_end, rec_end = C.record_window(arr, trt, months, RECORD_END, VAL_FRAC)
            masks = (C.mask_targets_after(arr, trm, tr_end), C.mask_targets_after(arr, vam, rec_end))
            for seed in (42, 123, 456):
                tag = f"bl_xgboost_{group}_r{months}_s{seed}"
                if done(tag):
                    continue
                t0 = time.time(); f, bi, ntr = xgboost_fit_predict(arr, scaler, trm, vam, tet, seed, True, masks=masks)
                save_run(tag, f, {"method": "XGBoost-residual", "stations": sts, "record_months": months, "record_end": RECORD_END, "val_frac_in_record": VAL_FRAC,
                                  "seed": seed, "best_iteration": bi, "n_train": ntr, "time_s": time.time() - t0})
        for st in stations:
            keys = per[(per.station == st) & (pd.to_datetime(per.anchor_date) >= C.SPLITS["test"][0])][["station", "anchor_date", "horizon", "target_date", "y_true_raw"]]
            if not (done(f"bl_climatology_{st}_r{months}") and done(f"bl_damped_{st}_r{months}")):
                cl, dam = seasonal(df, st, months, keys)
                save_run(f"bl_climatology_{st}_r{months}", cl, {"method": "climatology", "stations": [st], "record_months": months, "fit": "annual Fourier K=2 on the record"})
                save_run(f"bl_damped_{st}_r{months}", dam, {"method": "damped anomaly", "stations": [st], "record_months": months, "fit": "climatology and anomaly ACF on the record"})
            for harmonic, name in ((False, "arima"), (True, "harmonic")):
                tag = f"bl_{name}_{st}_r{months}"
                if done(tag):
                    continue
                t0 = time.time(); f, info = arima_like(df, st, months, harmonic)
                save_run(tag, f, {"method": "harmonic ARIMA" if harmonic else "ARIMA", "stations": [st], "record_months": months, "record_end": RECORD_END,
                                  "orders_tried": [list(o) for o in ORDERS], "chosen": info, "time_s": time.time() - t0})
                print(tag, info, flush=True)
