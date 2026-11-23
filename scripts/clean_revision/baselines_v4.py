"""Baselines of the physics runs. Same anchors, splits, source-only scaler and metrics as baselines.py.
1. Seasonal harmonic regression: annual Fourier terms (K = 2) with ARIMA errors chosen by AIC, fitted on 2021-2023 and
   extended day by day through the test years. The "SARIMA" of baselines.py chose non-seasonal orders at every station (weekly seasonal
   candidates only) and is reported as ARIMA.
2. Residual XGBoost without the day-of-year terms, so its inputs match the networks' (the XGBoost of baselines.py had them, the networks did not).
3. Target-only residual XGBoost for the short records that end on 2022-12-31 (6 and 12 months), validated on the last 20 %
   of the record, paired with the short-record transfer runs."""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from baselines import save_run, xgboost_fit_predict
warnings.filterwarnings("ignore")
H = C.HORIZONS
RECORD_END, VAL_FRAC = "2022-12-31", 0.2
ORIGIN = pd.Timestamp("2021-01-01")


def fourier(dates, K=2, period=365.25):
    t = (pd.DatetimeIndex(dates) - ORIGIN).days.to_numpy(float)
    return np.column_stack([f(2 * np.pi * k * t / period) for k in range(1, K + 1) for f in (np.sin, np.cos)])


def harmonic(df, stations, tag, K=2):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    orders = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (1, 0, 1), (2, 0, 1), (1, 0, 2)]
    rows, chosen, t0 = [], {}, time.time(); test_end = pd.Timestamp(C.SPLITS["test"][1])
    for st in stations:
        d = df[df.station == st].set_index("date")["chlorophyll_a"]; y = np.log1p(d.clip(lower=0))
        y_fit = y[(y.index >= C.SPLITS["train"][0]) & (y.index <= C.SPLITS["val"][1])]; best = None
        for o in orders:
            try:
                r = SARIMAX(y_fit, exog=fourier(y_fit.index, K), order=o, trend="c" if o[1] == 0 else "n",
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
                if best is None or r.aic < best[0].aic: best = (r, o)
            except Exception as e:
                print("harmonic fail", st, o, e)
        res, o = best; chosen[st] = {"order": o, "fourier_K": K, "aic": float(res.aic)}; cur = res
        for day, val in y[(y.index >= C.SPLITS["test"][0]) & (y.index <= C.SPLITS["test"][1])].items():
            cur = cur.extend(pd.Series([val], index=[day]), exog=fourier([day], K))
            fut = pd.date_range(day + pd.Timedelta(days=1), periods=max(H), freq="D")
            fc = np.asarray(cur.get_forecast(steps=max(H), exog=fourier(fut, K)).predicted_mean)
            for h in H:
                td = day + pd.Timedelta(days=h); yt = d.get(td, np.nan) if td <= test_end else np.nan
                rows.append({"station": st, "anchor_date": str(day.date()), "horizon": h, "target_date": td, "observed": not np.isnan(yt), "y_true_raw": yt, "y_pred_raw": float(np.expm1(fc[h - 1]))})
        print(tag, st, chosen[st], flush=True)
    save_run(tag, pd.DataFrame(rows), {"method": "Harmonic-ARIMA", "orders_tried": [list(o) for o in orders], "fourier": {"K": K, "period_days": 365.25, "origin": str(ORIGIN.date())},
                                       "chosen": chosen, "fit_window": "2021-01-01..2023-12-31", "forecast": "NaN-aware Kalman filter extended daily; direct multi-step forecast with known Fourier terms",
                                       "stations": stations, "target": stations[0] if len(stations) == 1 else "", "time_s": time.time() - t0})


if __name__ == "__main__":
    df = C.load_daily()
    train_mask = ((df["date"] >= C.SPLITS["train"][0]) & (df["date"] <= C.SPLITS["train"][1])).to_numpy()
    SCALER = C.Scaler().fit(df, train_mask, C.SOURCE)
    arr = C.build_arrays(df, SCALER, list(C.STATION_IDS), C.SEQ_LEN, H)
    tr, va, te = C.split_indices(arr, "train", C.SOURCE), C.split_indices(arr, "val", C.SOURCE), C.split_indices(arr, "test", C.SOURCE)
    for seed in (42, 123, 456):                                        # 2. no calendar terms (source)
        t0 = time.time(); f, bi, ntr = xgboost_fit_predict(arr, SCALER, tr, va, te, seed, True, season=False)
        save_run(f"baseline_xgboost_residual_nodoy_s{seed}", f, {"method": "XGBoost-residual-no-calendar", "seed": seed, "residual_target": True, "best_iteration": bi,
                 "features": "as v3 without day-of-year sin/cos", "n_train": ntr, "stations": C.SOURCE, "time_s": time.time() - t0})
    for tg in C.TARGETS:                                               # 3. short records ending 2022-12-31, validated inside the record
        trt, tet = C.split_indices(arr, "train", [tg]), C.split_indices(arr, "test", [tg])
        for months in (6, 12):
            trm, vam, tr_end, rec_end = C.record_window(arr, trt, months, RECORD_END, VAL_FRAC)
            masks = (C.mask_targets_after(arr, trm, tr_end), C.mask_targets_after(arr, vam, rec_end))
            for seed in (42, 123, 456):
                t0 = time.time(); f, bi, ntr = xgboost_fit_predict(arr, SCALER, trm, vam, tet, seed, True, masks=masks)
                save_run(f"baseline_xgboost_target_{tg}_end12m{months}_s{seed}", f, {"method": "XGBoost-residual-target", "mode": "target_only", "target": tg, "record_months": months,
                         "record_end": RECORD_END, "val_frac_in_record": VAL_FRAC, "train_targets_end": str(tr_end.date()), "seed": seed, "residual_target": True,
                         "best_iteration": bi, "n_train": ntr, "stations": [tg], "time_s": time.time() - t0})
    harmonic(df, C.SOURCE, "baseline_harmonic_source")                 # 1. seasonal harmonic regression
    for tg in C.TARGETS:
        harmonic(df, [tg], f"baseline_harmonic_{tg}")
