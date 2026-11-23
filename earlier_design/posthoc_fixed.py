"""Post-hoc checks of the corrected original-design rerun (inference only, no training).

For one lead-time folder (PINN_HORIZON = h, folder hNN of reruns/causal_fixed_20261017):
  response   change one input at a time in all 30 days of each test window whose target falls in June-September 2024
             (water temperature +2 / -2 degC, TP x1.3 / x0.7, TN x0.7) and record the change of the forecast Chl-a for
             the xLSTM without (pure, phase 1 step 1) and with the physics loss (PINN, phase 1 step 2), three seeds each.
For h = 1 also:
  kinetic    share of the observed daily change in ln C explained by the fitted net growth rate (kinetic R2), on pairs of
             consecutive days with measured Chl-a, for the phase-1 PINN and the three constraint modes of phase 3 step 4,
             in the training years (2020-2022) and the test year (2024)
  netrate    daily net growth rate of the phase-1 PINN (mean of the seeds) at each station in 2024, with the measured Chl-a
Usage: PINN_HORIZON=h python posthoc_fixed.py <folder>   (writes <folder>/experiments/posthoc/*.csv)"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

RUN = Path(sys.argv[1]).resolve()
os.chdir(RUN)
sys.path.insert(0, str(RUN / "src")); sys.path.insert(0, str(RUN / "scripts"))
import run_phase1_backbone_contest as p1                                       # noqa: E402
from pinn_hab.physics.biogeochemical import PHYSICS_FEATURE_PATTERNS          # noqa: E402

H = int(os.environ.get("PINN_HORIZON", "1"))
OUT = RUN / "experiments" / "posthoc"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 123, 456)
data, feature_names, prep = p1.load_data()
sites = p1.DATA_CONFIG["site_names"]
lo_all, hi_all = prep.scaler.data_min_.astype(float), prep.scaler.data_max_.astype(float)
logset = set(prep.log_transform_cols)
ti = prep.target_idx


def find(var):
    for i, name in enumerate(feature_names):
        if any(p.lower() in name.lower() for p in PHYSICS_FEATURE_PATTERNS[var]):
            return i
    raise KeyError(var)


IDX = {v: find(v) for v in ("temperature", "nitrogen", "phosphorus", "chlorophyll")}
print("physics columns:", {v: feature_names[i] for v, i in IDX.items()}, flush=True)


def to_phys(x, i):
    z = x * (hi_all[i] - lo_all[i]) + lo_all[i]
    return np.expm1(z) if feature_names[i] in logset else z


def to_scaled(v, i):
    z = np.log1p(np.maximum(v, 0.0)) if feature_names[i] in logset else v
    return (z - lo_all[i]) / (hi_all[i] - lo_all[i])


def load(kind, seed, folder=RUN):
    if kind == "pure":
        mt, ck, extra = "xLSTM", folder / "experiments" / "phase1" / f"step1_xLSTM_seed{seed}.pth", {}
    else:
        mt, ck, extra = "xLSTM_PINN", folder / "experiments" / "phase1" / f"step2_xLSTM_PINN_seed{seed}.pth", p1.PHYSICS_CONFIG
    m = p1.get_model({"model_type": mt, **p1.MODEL_DEFAULTS, **extra}, n_features=data["X_train"].shape[2],
                     n_stations=len(sites), feature_names=feature_names)
    m.load_state_dict(torch.load(str(ck), map_location=p1.device)); return m.to(p1.device).eval()


def predict(model, X):
    out = []
    sid = torch.as_tensor(np.asarray(data["test_station_ids"]), dtype=torch.long)
    with torch.no_grad():
        for i in range(0, len(X), 512):
            mu, _ = model(torch.as_tensor(X[i:i + 512], dtype=torch.float32).to(p1.device), sid[i:i + 512].to(p1.device))
            out.append(mu.detach().cpu().numpy().reshape(-1))
    z = np.concatenate(out) * (hi_all[ti] - lo_all[ti]) + lo_all[ti]          # log1p(Chl-a)
    return np.expm1(z)


# ---- response test --------------------------------------------------------------------------------------------------
X0 = np.asarray(data["X_test"], dtype=np.float64)
tdates = pd.to_datetime(pd.Series(data["test_dates"])).dt.normalize()
summer = tdates.dt.month.between(6, 9).to_numpy()
SCEN = {"T+2": ("temperature", lambda v: v + 2.0, +1), "T-2": ("temperature", lambda v: v - 2.0, -1),
        "TPx1.3": ("phosphorus", lambda v: v * 1.3, +1), "TPx0.7": ("phosphorus", lambda v: v * 0.7, -1),
        "TNx0.7": ("nitrogen", lambda v: v * 0.7, -1)}
rows = []
for kind in ("pure", "PINN"):
    for seed in SEEDS:
        model = load(kind, seed)
        base = predict(model, X0)[summer]
        for name, (var, fn, sign) in SCEN.items():
            i = IDX[var]; X = X0.copy()
            X[..., i] = to_scaled(fn(to_phys(X0[..., i], i)), i)
            new = predict(model, X)[summer]
            rows.append({"horizon": H, "model": kind, "seed": seed, "scenario": name, "expected_sign": sign,
                         "mean_change_pct": float(100 * np.mean(new / np.maximum(base, 1e-6) - 1)), "n_windows": int(summer.sum())})
resp = pd.DataFrame(rows); resp.to_csv(OUT / "response_test.csv", index=False)
agg = resp.groupby(["model", "scenario"]).agg(mean=("mean_change_pct", "mean"), lo=("mean_change_pct", "min"), hi=("mean_change_pct", "max"),
                                             sign=("expected_sign", "first")).reset_index()
agg["pass"] = (np.sign(agg["mean"]) == agg["sign"]) & (np.sign(agg["lo"]) == np.sign(agg["hi"])) & (np.sign(agg["lo"]) == agg["sign"])
agg.insert(0, "horizon", H); agg.to_csv(OUT / "response_summary.csv", index=False)
print(agg.round(3).to_string(index=False), flush=True)

if H == 1:
    # ---- kinetic parameters of each checkpoint ------------------------------------------------------------------------
    def params(model):
        ph = model.physics
        with torch.no_grad():
            mu = ph.get_mu_max(torch.arange(len(sites), device=p1.device)).cpu().numpy()
            return {"mu": mu, "K_N": float(ph.get_K_N()), "K_P": float(ph.get_K_P()), "r": float(ph.get_r()), "m": float(ph.get_m()),
                    "theta": float(ph.theta)}
    fits = []
    for seed in SEEDS:
        fits.append(("PINN (phase 1)", seed, params(load("PINN", seed))))
    MODE = {"sigmoid": "sigmoid", "no_constraint": "none", "penalty": "soft"}
    for fname, mode in MODE.items():
        for seed in SEEDS:
            m = p1.get_model({"model_type": "xLSTM_PINN", **p1.MODEL_DEFAULTS, **p1.PHYSICS_CONFIG}, n_features=data["X_train"].shape[2],
                             n_stations=len(sites), feature_names=feature_names)
            m.physics.set_constraint_mode(mode, 10.0)
            m.load_state_dict(torch.load(str(RUN / "experiments" / "phase3" / f"step4_{fname}_seed{seed}.pth"), map_location=p1.device))
            fits.append((f"constraint: {fname}", seed, params(m.to(p1.device).eval())))
    pd.DataFrame([{"source": s, "seed": sd, **{k: v for k, v in p.items() if k != "mu"}, **{f"mu_max_{sites[i]}": float(p["mu"][i]) for i in range(len(sites))}}
                  for s, sd, p in fits]).to_csv(OUT / "kinetic_parameters.csv", index=False)

    # ---- kinetic R2 on measured consecutive days -----------------------------------------------------------------------
    raw = pd.read_csv(RUN / "data" / "imputed_daily_data.csv"); raw["측정일"] = pd.to_datetime(raw["측정일"])
    flg = pd.read_csv(RUN / "data" / "causal_daily_data_measured.csv"); flg["측정일"] = pd.to_datetime(flg["측정일"])
    TC, NC, PC, CC = "수온 (℃)", "총질소 (mg/L)", "총인 (mg/L)", "클로로필-a (mg/㎥)"
    eps = 1e-3

    def net_rate(p, st_i, T, N, P):
        return p["mu"][st_i] * p["theta"] ** (np.clip(T, 0, 40) - 20) * N / (p["K_N"] + N) * P / (p["K_P"] + P) - p["r"] - p["m"]
    pairs = {}
    for st_i, st in enumerate(sites):
        d = raw[raw["측정소명"] == st].set_index("측정일").sort_index(); f = flg[flg["측정소명"] == st].set_index("측정일").reindex(d.index)
        nxt = d.shift(-1, freq="D").reindex(d.index); fn = f.shift(-1, freq="D").reindex(d.index)
        ok = (f[CC] == 1) & (fn[CC] == 1)
        y = np.log(nxt[CC] + eps) - np.log(d[CC] + eps)
        pairs[st_i] = pd.DataFrame({"date": d.index, "y": y.values, "ok": ok.values, "T": d[TC].values, "N": np.maximum(d[NC].values, 0),
                                    "P": np.maximum(d[PC].values, 0)})
    krows = []
    for src, seed, p in fits:
        for period, (a, b) in {"train 2020-2022": ("2020-01-01", "2022-12-30"), "test 2024": ("2024-01-01", "2024-12-30")}.items():
            ys, ns = [], []
            for st_i, df in pairs.items():
                q = df[(df.date >= a) & (df.date <= b) & df.ok & df.y.notna()]
                ys.append(q.y.values); ns.append(net_rate(p, st_i, q["T"].values, q["N"].values, q["P"].values))
            y, n = np.concatenate(ys), np.concatenate(ns)
            krows.append({"source": src, "seed": seed, "period": period, "n_pairs": len(y), "kinetic_r2": float(1 - np.sum((y - n) ** 2) / np.sum((y - y.mean()) ** 2)),
                          "corr": float(np.corrcoef(n, y)[0, 1]) if n.std() > 0 else float("nan"), "mean_obs": float(y.mean()), "mean_net": float(n.mean()),
                          "sd_obs": float(y.std()), "sd_net": float(n.std())})
    kin = pd.DataFrame(krows); kin.to_csv(OUT / "kinetic_r2.csv", index=False)
    print(kin.groupby(["source", "period"])[["n_pairs", "kinetic_r2", "corr", "mean_obs", "mean_net", "sd_obs", "sd_net"]].mean().round(4).to_string(), flush=True)

    # ---- daily net growth rate of the phase-1 PINN in 2024 -------------------------------------------------------------
    nrows = []
    for st_i, df in pairs.items():
        q = df[(df.date >= "2024-01-01") & (df.date <= "2024-12-31")]
        rates = np.mean([net_rate(p, st_i, q["T"].values, q["N"].values, q["P"].values) for src, _, p in fits if src == "PINN (phase 1)"], axis=0)
        d = raw[raw["측정소명"] == sites[st_i]].set_index("측정일").reindex(q.date); f = flg[flg["측정소명"] == sites[st_i]].set_index("측정일").reindex(q.date)
        nrows.append(pd.DataFrame({"station": sites[st_i], "date": q.date.values, "net_rate": rates, "obs_change": q.y.values, "pair_measured": q.ok.values,
                                   "chla": d[CC].values, "chla_measured": (f[CC] == 1).values, "T": q["T"].values, "N": q["N"].values, "P": q["P"].values}))
    pd.concat(nrows).to_csv(OUT / "net_rate_2024.csv", index=False)
print("done", flush=True)
