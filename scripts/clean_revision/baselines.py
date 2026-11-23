"""SARIMA and XGBoost baselines of the pure-network runs: same anchors, splits, source-only scaler and metrics as the DL runs.
XGBoost is run on the source stations (level and residual targets) and on each target reservoir with
record-length training windows, to pair with the transfer experiments."""
from __future__ import annotations
import sys, time, warnings, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
warnings.filterwarnings("ignore")
INV = {v: k for k, v in C.STATION_IDS.items()}
H = C.HORIZONS


def save_run(tag, pred, cfg):
    rd = C.RESULTS / "runs" / tag; rd.mkdir(parents=True, exist_ok=True)
    pred.to_csv(rd / "predictions_test.csv", index=False)
    obs = pred[pred.observed & pred.y_pred_raw.notna()]
    m = {"test": C.evaluate_predictions(obs)}
    for yr in sorted(obs.target_date.dt.year.unique()):
        m[f"test_{yr}"] = C.evaluate_predictions(obs[obs.target_date.dt.year == yr])
    C.write_json(m, rd / "metrics.json"); C.write_json(cfg, rd / "config.json"); (rd / "DONE").write_text("ok")
    pool = lambda s, h: m.get(s, {}).get(f"h{h}", {}).get("pooled", {})
    C.append_index({"tag": tag, "backbone": cfg["method"], "physics": "off", "mode": cfg.get("mode", "baseline"), "target": cfg.get("target", ""), "record_months": cfg.get("record_months", 24),
                    "residual": int(bool(cfg.get("residual_target", False))), "horizons": ",".join(map(str, H)), "seed": cfg.get("seed", 0), "kn_max": np.nan, "light": 0, "lr_selected": np.nan,
                    "n_train": cfg.get("n_train", 0), "best_epoch": np.nan, "train_time_s": cfg.get("time_s", 0), "params": np.nan,
                    **{f"test_h{h}_r2": pool("test", h).get("r2") for h in H}, **{f"test2025_h{h}_r2": pool("test_2025", h).get("r2") for h in H},
                    "K_N": np.nan, "K_P": np.nan, "finished": dt.datetime.now().isoformat(timespec="seconds")})
    print(tag, {f"h{h}": round(pool("test", h).get("r2", np.nan), 3) for h in H}, flush=True)


def frame(arr, idx, ym, pred_raw):
    rows = []
    for n, i in enumerate(idx):
        for hi, h in enumerate(H):
            rows.append({"station": INV[int(arr["sid"][i])], "anchor_date": str(arr["anchor"][i]), "horizon": h, "target_date": str(arr["target"][i][hi]),
                         "observed": bool(ym[n, hi] > 0), "y_true_raw": float(arr["y_raw"][i, hi]) if ym[n, hi] > 0 else np.nan,
                         "y_pred_raw": float(pred_raw[n, hi]) if not np.isnan(pred_raw[n, hi]) else np.nan})
    df = pd.DataFrame(rows); df["target_date"] = pd.to_datetime(df["target_date"]); return df


def sarima(df, stations, tag):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    orders = [((1, 1, 1), (0, 0, 0, 0)), ((2, 1, 1), (0, 0, 0, 0)), ((1, 1, 2), (0, 0, 0, 0)), ((1, 1, 1), (1, 0, 1, 7)), ((2, 1, 1), (1, 0, 1, 7))]
    rows, chosen, t0 = [], {}, time.time(); test_end = pd.Timestamp(C.SPLITS["test"][1])
    for st in stations:
        d = df[df.station == st].set_index("date")["chlorophyll_a"]; y = np.log1p(d.clip(lower=0))
        y_fit = y[(y.index >= C.SPLITS["train"][0]) & (y.index <= C.SPLITS["val"][1])]; best = None
        for o, so in orders:
            try:
                r = SARIMAX(y_fit, order=o, seasonal_order=so, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
                if best is None or r.aic < best[0].aic: best = (r, o, so)
            except Exception as e:
                print("sarima fail", st, o, so, e)
        res, o, so = best; chosen[st] = {"order": o, "seasonal_order": so, "aic": float(res.aic)}; cur = res
        for day, val in y[(y.index >= C.SPLITS["test"][0]) & (y.index <= C.SPLITS["test"][1])].items():
            cur = cur.extend(pd.Series([val], index=[day])); fc = np.asarray(cur.get_forecast(steps=max(H)).predicted_mean)
            for h in H:
                td = day + pd.Timedelta(days=h); yt = d.get(td, np.nan) if td <= test_end else np.nan
                rows.append({"station": st, "anchor_date": str(day.date()), "horizon": h, "target_date": td, "observed": not np.isnan(yt), "y_true_raw": yt, "y_pred_raw": float(np.expm1(fc[h - 1]))})
    save_run(tag, pd.DataFrame(rows), {"method": "SARIMA", "orders_tried": [list(map(list, o)) for o in orders], "chosen": chosen, "fit_window": "2021-01-01..2023-12-31",
                                       "forecast": "NaN-aware Kalman filter extended daily; direct multi-step forecast", "stations": stations, "target": stations[0] if len(stations) == 1 else "", "time_s": time.time() - t0})


def xgb_features(arr, idx, season=True):
    X = arr["X"][idx]; sid = arr["sid"][idx]; names = arr["feat_names"]; chl = names.index("chlorophyll_a")
    last = X[:, -1, :]; lags = np.stack([X[:, -1 - l, chl] for l in (1, 2, 3, 6, 13, 29)], 1)
    means = np.stack([X[:, -7:, chl].mean(1), X[:, :, chl].mean(1), X[:, -7:, names.index("era5_ssrd_mj")].mean(1), X[:, -7:, names.index("era5_t2m")].mean(1)], 1)
    doy = np.array([(x - x.astype("datetime64[Y]")).astype(int) for x in arr["anchor"][idx]]) / 365.0
    cal = [np.stack([np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)], 1)] if season else []
    return np.concatenate([last, lags, means, *cal, np.eye(len(C.STATION_IDS))[sid]], 1)


def xgboost_fit_predict(arr, scaler, tr, va, te, seed, residual, season=True, masks=None):
    """masks: optional (train, validation) target masks; default masks cut targets at the split ends (v3)."""
    import xgboost as xgb
    ymtr, ymva = masks if masks is not None else (C.mask_targets_to_split(arr, tr, "train"), C.mask_targets_to_split(arr, va, "val"))
    ymte = C.mask_targets_to_split(arr, te, "test")
    Ftr, Fva, Fte = xgb_features(arr, tr, season), xgb_features(arr, va, season), xgb_features(arr, te, season); pred = np.full((len(te), len(H)), np.nan); best_it = {}
    off = lambda idx: arr["base"][idx] if residual else 0.0
    for k, h in enumerate(H):
        mtr, mva = ymtr[:, k] > 0, ymva[:, k] > 0
        if mtr.sum() < 5:
            continue
        model = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                                 random_state=seed, early_stopping_rounds=100, n_jobs=8)
        model.fit(Ftr[mtr], (arr["y"][tr][:, k] - off(tr))[mtr], eval_set=[(Fva[mva], (arr["y"][va][:, k] - off(va))[mva])], verbose=False)
        pred[:, k] = model.predict(Fte) + off(te); best_it[f"h{h}"] = int(model.best_iteration)
    return frame(arr, te, ymte, scaler.y_unscale(pred)), best_it, int((ymtr > 0).any(1).sum())


if __name__ == "__main__":
    df = C.load_daily()
    train_mask = ((df["date"] >= C.SPLITS["train"][0]) & (df["date"] <= C.SPLITS["train"][1])).to_numpy()
    SCALER = C.Scaler().fit(df, train_mask, C.SOURCE)
    arr = C.build_arrays(df, SCALER, list(C.STATION_IDS), C.SEQ_LEN, H)
    sarima(df, C.SOURCE, "baseline_sarima_source")
    for tg in C.TARGETS:
        sarima(df, [tg], f"baseline_sarima_{tg}")
    tr, va, te = C.split_indices(arr, "train", C.SOURCE), C.split_indices(arr, "val", C.SOURCE), C.split_indices(arr, "test", C.SOURCE)
    for residual in (False, True):
        for seed in (42, 123, 456):
            t0 = time.time(); f, bi, ntr = xgboost_fit_predict(arr, SCALER, tr, va, te, seed, residual)
            save_run(f"baseline_xgboost{'_residual' if residual else ''}_s{seed}", f, {"method": "XGBoost" + ("-residual" if residual else ""), "seed": seed, "residual_target": residual, "best_iteration": bi,
                     "features": "anchor-day channels + chl lags(1,2,3,6,13,29) + 7/30-day means + ERA5 7-day means + season + station one-hot", "n_train": ntr, "stations": C.SOURCE, "time_s": time.time() - t0})
    for tg in C.TARGETS:                                   # target-only XGBoost at each record length (pairs with the transfer table)
        trt, vat, tet = C.split_indices(arr, "train", [tg]), C.split_indices(arr, "val", [tg]), C.split_indices(arr, "test", [tg])
        for months in (1, 3, 6, 12, 24):
            trm = C.record_subset(arr, trt, months)
            for seed in (42, 123, 456):
                t0 = time.time(); f, bi, ntr = xgboost_fit_predict(arr, SCALER, trm, vat, tet, seed, True)
                save_run(f"baseline_xgboost_target_{tg}_m{months}_s{seed}", f, {"method": "XGBoost-residual-target", "mode": "target_only", "target": tg, "record_months": months, "seed": seed,
                         "residual_target": True, "best_iteration": bi, "n_train": ntr, "stations": [tg], "time_s": time.time() - t0})
