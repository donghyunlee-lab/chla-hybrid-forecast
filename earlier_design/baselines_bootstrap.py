"""Baselines and block-bootstrap comparisons for one rerun folder of the original pipeline (1-day forecasts, test 2024).

Networks : the saved phase-1 models (pure step 1, PINN step 2), forecasts averaged over the three seeds.
Baselines on the same test samples and the same dataset (log1p Chl-a):
  persistence      y(t+1) = y(t)
  ARIMA            orders chosen by AIC among (1,1,1), (2,1,1), (1,1,2) and weekly seasonal (1,0,1,7) variants
  harmonic ARIMA   two pairs of annual Fourier terms with ARIMA errors chosen by AIC
                   (both fitted on 2020-01-01..2023-12-31, then one-step forecasts through 2024 with the fitted parameters)
  XGBoost          residual target y(t+1) - y(t); anchor-day values of all inputs, Chl-a lags (1,2,3,6,13,29),
                   7/30-day means, day of year and station; trained on 2020-2022, early stopping on 2023; three seeds
R2 is computed on log1p(Chl-a) (the networks' normalized scale is an affine map of it, so R2 is the same).
Bootstrap: 30-day circular moving blocks over target dates, the same dates for all stations, 2,000 draws.
Lead time: PINN_HORIZON=h (default 1) forecasts h days ahead with the same rules (persistence y(t+h) = y(t), ARIMA h-step
  forecasts from each origin with the fitted parameters, XGBoost residual target y(t+h) - y(t)); h = 1 is the original setup.
Usage: [PINN_HORIZON=h] [BASELINE_SEEDS=42,123,456] python baselines_bootstrap.py <rerun folder>
       (writes <folder>/experiments/baselines_bootstrap_*.csv)"""
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
RUN = Path(sys.argv[1]).resolve()
os.chdir(RUN)
sys.path.insert(0, str(RUN / "src")); sys.path.insert(0, str(RUN / "scripts"))
import run_phase1_backbone_contest as p1                                       # noqa: E402

CHL = "클로로필-a (mg/㎥)"
SEEDS = tuple(int(s) for s in os.environ.get("BASELINE_SEEDS", "42,123,456").split(","))
H = int(os.environ.get("PINN_HORIZON", "1"))                                  # forecast lead time in days
FIT = ("2020-01-01", "2023-12-31"); TRAIN = ("2020-01-01", "2022-12-31"); VAL = ("2023-01-01", "2023-12-31"); TEST = ("2024-01-01", "2024-12-31")

# ---- test samples of the original pipeline and the networks' forecasts --------------------------------------------
data, feature_names, prep = p1.load_data()
sites = p1.DATA_CONFIG["site_names"]
t_dates = pd.to_datetime(pd.Series(data["test_dates"])).dt.normalize().to_numpy()
t_st = np.asarray([sites[int(i)] for i in data["test_station_ids"]])
ti = prep.target_idx; lo, hi = prep.scaler.data_min_[ti], prep.scaler.data_max_[ti]
y_norm = np.asarray(data["y_test"]).reshape(-1)
y = y_norm * (hi - lo) + lo                                                     # back to log1p(Chl-a)
import json                                                                      # noqa: E402
(RUN / "experiments").mkdir(exist_ok=True)
(RUN / "experiments" / "target_scale.json").write_text(json.dumps({"lo": float(lo), "hi": float(hi)}))   # for all_metrics_bootstrap.py

flags = pd.read_csv(RUN / "data" / "causal_daily_data_measured.csv"); flags["측정일"] = pd.to_datetime(flags["측정일"])
meas_set = set(zip(flags.loc[flags[CHL] == 1, "측정소명"], flags.loc[flags[CHL] == 1, "측정일"]))
measured = np.array([(s, pd.Timestamp(d)) in meas_set for s, d in zip(t_st, t_dates)])


def predict(model):
    X = torch.as_tensor(np.asarray(data["X_test"]), dtype=torch.float32); sid = torch.as_tensor(np.asarray(data["test_station_ids"]), dtype=torch.long)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            m, _ = model(X[i:i + 512].to(p1.device), sid[i:i + 512].to(p1.device)); out.append(m.detach().cpu().numpy().reshape(-1))
    return np.concatenate(out) * (hi - lo) + lo


preds = {}
for kind, mt, fname, extra in (("pure", "{b}", "step1_{b}_seed{s}.pth", {}), ("PINN", "{b}_PINN", "step2_{b}_PINN_seed{s}.pth", p1.PHYSICS_CONFIG)):
    for b in p1.BACKBONES:
        per_seed = []
        for s in SEEDS:
            ck = RUN / "experiments" / "phase1" / fname.format(b=b, s=s)
            model = p1.get_model({"model_type": mt.format(b=b), **p1.MODEL_DEFAULTS, **extra}, n_features=data["X_train"].shape[2],
                                 n_stations=len(sites), feature_names=feature_names)
            model.load_state_dict(torch.load(str(ck), map_location=p1.device)); model.to(p1.device).eval(); per_seed.append(predict(model))
        preds[f"{b} ({kind})"] = np.mean(per_seed, axis=0)

# ---- baselines on the same dataset --------------------------------------------------------------------------------
raw = pd.read_csv(RUN / "data" / "imputed_daily_data.csv"); raw["측정일"] = pd.to_datetime(raw["측정일"])
VARS = [c for c in raw.columns if c not in ("수계", "측정소", "측정일", "시작연도", "종료연도", "파일명", "측정소명") and "[ECD]" not in c]
series = {st: raw[raw["측정소명"] == st].set_index("측정일").sort_index() for st in sites}
key = pd.MultiIndex.from_arrays([t_st, pd.DatetimeIndex(t_dates)])


def lookup(frame_by_station):
    return np.array([frame_by_station[s].get(pd.Timestamp(d), np.nan) for s, d in zip(t_st, t_dates)])


# persistence
logc = {st: np.log1p(series[st][CHL].clip(lower=0)) for st in sites}
preds["Persistence"] = lookup({st: logc[st].shift(H) for st in sites})
print("target check: max |network target - log1p(Chl-a) of the dataset| =", float(np.nanmax(np.abs(y - lookup(logc)))), flush=True)


def fourier(idx, K=2):
    t = (pd.DatetimeIndex(idx) - pd.Timestamp("2020-01-01")).days.to_numpy(float)
    return np.column_stack([f(2 * np.pi * k * t / 365.25) for k in range(1, K + 1) for f in (np.sin, np.cos)])


def arima_family(harmonic):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if harmonic:
        grid = [((1, 1, 1), (0, 0, 0, 0)), ((2, 1, 1), (0, 0, 0, 0)), ((1, 1, 2), (0, 0, 0, 0)), ((1, 0, 1), (0, 0, 0, 0)), ((2, 0, 1), (0, 0, 0, 0)), ((1, 0, 2), (0, 0, 0, 0))]
    else:
        grid = [((1, 1, 1), (0, 0, 0, 0)), ((2, 1, 1), (0, 0, 0, 0)), ((1, 1, 2), (0, 0, 0, 0)), ((1, 1, 1), (1, 0, 1, 7)), ((2, 1, 1), (1, 0, 1, 7))]
    fcst, chosen = {}, {}
    for st in sites:
        yy = logc[st].asfreq("D"); fit = yy[FIT[0]:FIT[1]]; best = None
        for o, so in grid:
            kw = dict(order=o, seasonal_order=so, enforce_stationarity=False, enforce_invertibility=False)
            if harmonic:
                kw.update(exog=fourier(fit.index), trend="c" if o[1] == 0 else "n")
            try:
                r = SARIMAX(fit, **kw).fit(disp=False, maxiter=200)
                if best is None or r.aic < best[0].aic:
                    best = (r, o, so)
            except Exception as e:                                                   # noqa: BLE001
                print("fit failed", st, o, so, e)
        r, o, so = best; chosen[st] = (o, so)
        full = yy[FIT[0]:TEST[1]]
        app = r.apply(full, exog=fourier(full.index) if harmonic else None)       # same parameters, no refit
        if H == 1:
            fcst[st] = app.predict(start=pd.Timestamp(TEST[0]), end=pd.Timestamp(TEST[1]), dynamic=False)
        else:                                                                     # H-step forecast from each origin (target - H)
            targets = pd.DatetimeIndex(sorted({pd.Timestamp(d) for s_, d in zip(t_st, t_dates) if s_ == st}))
            first = [d - pd.Timedelta(days=H - 1) for d in targets]
            fcst[st] = pd.Series([app.get_prediction(start=f0, end=d, dynamic=f0).predicted_mean.iloc[-1] for f0, d in zip(first, targets)],
                                 index=targets)
    print("harmonic ARIMA" if harmonic else "ARIMA", chosen, flush=True)
    return lookup(fcst), chosen


preds["ARIMA"], arima_orders = arima_family(False)
preds["Harmonic ARIMA"], harm_orders = arima_family(True)


def xgb_frame(st):
    s = series[st]; c = logc[st]; f = pd.DataFrame(index=s.index)
    for v in VARS:
        f[f"x_{v}"] = s[v]
    for k in (1, 2, 3, 6, 13, 29):
        f[f"lag{k}"] = c.shift(k)
    f["mean7"] = c.rolling(7, min_periods=1).mean(); f["mean30"] = c.rolling(30, min_periods=1).mean()
    doy = s.index.dayofyear.to_numpy(); f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25); f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for other in sites:
        f[f"st_{other}"] = float(other == st)
    f["base"] = c; f["target"] = c.shift(-H) - c; f["target_date"] = f.index + pd.Timedelta(days=H)
    return f


import xgboost as xgb                                                            # noqa: E402
frames = pd.concat([xgb_frame(st).assign(station=st) for st in sites])
feat = [c for c in frames.columns if c not in ("base", "target", "target_date", "station")]
part = lambda a, b: frames[(frames.target_date >= pd.Timestamp(a) + pd.Timedelta(days=29 + H)) & (frames.target_date <= b) & (frames.target_date >= a) & frames.target.notna()]
tr, va = part(*TRAIN), part(*VAL)
te = frames.set_index(["station", "target_date"]).reindex(key)
xp = []
for s in SEEDS:
    m = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                         random_state=s, early_stopping_rounds=100, n_jobs=8)
    m.fit(tr[feat], tr.target, eval_set=[(va[feat], va.target)], verbose=False)
    xp.append(te["base"].to_numpy() + m.predict(te[feat]))
preds["XGBoost"] = np.mean(xp, axis=0)

# ---- metrics and bootstrap ------------------------------------------------------------------------------------------
dates_u, d_idx = np.unique(t_dates, return_inverse=True); D = len(dates_u)
rng = np.random.default_rng(20261013); B, L = 2000, 30
starts = rng.integers(0, D, size=(B, int(np.ceil(D / L))))
counts = np.zeros((B, D))
for b in range(B):
    idx = ((starts[b][:, None] + np.arange(L)[None, :]) % D).ravel()[:D]
    np.add.at(counts[b], idx, 1.0)


def per_date(v, mask):
    out = np.zeros(D); np.add.at(out, d_idx[mask], v[mask]); return out


def r2_boot(p, mask):
    n = per_date(np.ones_like(y), mask); sy = per_date(y, mask); syy = per_date(y * y, mask); se = per_date((y - p) ** 2, mask)
    W, SY, SYY, SE = counts @ n, counts @ sy, counts @ syy, counts @ se
    return 1 - SE / (SYY - SY ** 2 / W)


def r2(p, mask):
    return 1 - np.sum((y[mask] - p[mask]) ** 2) / np.sum((y[mask] - y[mask].mean()) ** 2)


subsets = {"all days": np.ones_like(measured), "measured days": measured}
ok = np.all([np.isfinite(p) for p in preds.values()], axis=0)
met = pd.DataFrame({name: {f"R2 {k}": r2(p, m & ok) for k, m in subsets.items()} for name, p in preds.items()}).T
met.to_csv(RUN / "experiments" / "baselines_bootstrap_metrics.csv")
pairs = [("xLSTM (pure)", "Harmonic ARIMA"), ("xLSTM (pure)", "ARIMA"), ("xLSTM (pure)", "XGBoost"), ("xLSTM (pure)", "Persistence"),
         ("xLSTM (PINN)", "xLSTM (pure)"), ("xLSTM (PINN)", "Harmonic ARIMA"), ("XGBoost", "Harmonic ARIMA"), ("Harmonic ARIMA", "Persistence"),
         ("ModernTCN (pure)", "Harmonic ARIMA"), ("iTransformer (pure)", "Harmonic ARIMA"), ("Mamba (pure)", "Harmonic ARIMA")]
rows = []
for a, b in pairs:
    r = {"comparison": f"{a} - {b}"}
    for k, m in subsets.items():
        mm = m & ok; diff = r2_boot(preds[a], mm) - r2_boot(preds[b], mm); point = r2(preds[a], mm) - r2(preds[b], mm)
        lo_, hi_ = np.percentile(diff, [2.5, 97.5])
        r[k] = f"{point:+.4f} [{lo_:+.4f}, {hi_:+.4f}]{' *' if lo_ > 0 or hi_ < 0 else ''}"
    rows.append(r)
cmp = pd.DataFrame(rows); cmp.to_csv(RUN / "experiments" / "baselines_bootstrap_comparisons.csv", index=False)
pd.DataFrame({"station": t_st, "target_date": t_dates, "measured": measured, "y_log1p": y, **{k: v for k, v in preds.items()}}).to_csv(
    RUN / "experiments" / "baselines_bootstrap_predictions.csv", index=False)
print(met.round(4).to_string()); print(cmp.to_string(index=False))
print(f"samples {len(y)} (finite in all models {int(ok.sum())}), measured {int(measured.sum())}; ARIMA orders {arima_orders}; harmonic orders {harm_orders}")
