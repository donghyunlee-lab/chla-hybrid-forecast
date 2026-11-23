"""Response of the forecast to rain and water temperature (no training).

The saved models (`model.pt`) of the 36-month record are loaded and run on every evaluation anchor of 2024-2025, first as they are
(check against the saved forecasts) and then with one input changed:
  rain    10 mm a day added to each of the last 7 days: the 7-day mean rain of the flushing term +10 mm/d, and the precipitation
          channel of the network +10 mm on the last 7 days of the input window (scaled unit of the record)
  temp    +3 or -3 degrees C on the last 7 days: the 7-day mean water temperature of the rate term and the water-temperature channel
          of the network (measured flags unchanged)
For models with process terms and a network the response is split: `process` changes the inputs of the process terms only (the
network's correction keeps its unperturbed value), `network` is the rest.
Models: main and core (stage 6), bb-nut (stage 5b), free-eps and free-const (stage 8).
Writes results/input_design_20261114/response/: response.csv (27 stations pooled and by basin; seed-mean forecast and mean over the
single runs), response_runs.csv (per run), response_check.csv (agreement with the saved forecasts and with the raw inputs).
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Names: bb-C1 and the other bb-* models are the joint network of the paper (one xLSTM trained on the stations together,
without process terms; 'bb' for black box); core (P-mcf-all-kT) is the process model and hybrid (H-res-all) the hybrid
model.
Usage: newinput_response.py [--models main,core,bb-nut]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "xpinn"))
import newinput as NI                                                              # noqa: E402

OUT, S8, REC, SPLIT, HS, TOL = NI.STAGE5 / "response", NI.STAGE5 / "stage8", 36, "test", (3, 7, 14), 1e-3
MODELS = {"main": (NI.HYB, "", None), "core": (NI.CORE, "", None), "bb-nut": ("bb-C1", "", None),
          "free-eps": (NI.HYB, "eps", S8 / "free-eps"), "free-const": (NI.HYB, "const", S8 / "free-const")}
STAGE6 = {"main": "hybrid", "core": "core", "bb-nut": "bb-nut"}
PERT = {"rain +10 mm/d": ("rain", 10.0), "water temperature +3 C": ("temp", 3.0), "water temperature -3 C": ("temp", -3.0)}


def setup():
    """The settings of stage 6 (stations, data build, 17 input channels), then the model code."""
    for k in NI.SWITCHES:
        os.environ.pop(k, None)
    os.environ.update(NI.env())
    for p in ("clean_revision", "national", "xpinn"):
        sys.path.insert(0, str(ROOT / "scripts" / p))
    import torch
    import core as K1
    import core2 as X2
    return X2, K1, torch


def run_dir(model, net, seed):
    spec, _, base = MODELS[model]
    return NI.run_dir(STAGE6[model], net, SPLIT, REC, seed) if base is None else base / "runs" / NI.tag(spec, net, SPLIT, REC, seed)


class patched:
    """The two module settings that stage 8 replaced for its `free` variants (newinput_ablate.child)."""
    def __init__(self, X2, patch):
        self.X2, self.patch = X2, patch

    def __enter__(self):
        X2 = self.X2; self.keep = (X2.EPS_MAX, X2.bound)
        if self.patch == "eps":
            X2.EPS_MAX = 1.0
        elif self.patch == "const":
            def free(z, name):
                lo, hi = X2.RANGE[name]; p = (lo + hi) / 2 + (hi - lo) * z / 4
                return p.clamp_min(0.5) if name == "thk" else p
            X2.bound = free

    def __exit__(self, *a):
        self.X2.EPS_MAX, self.X2.bound = self.keep


def scaler(K1, net):
    """The scaler of core.prepare (fitted inside the record of the basin's stations)."""
    N, C = K1.N, K1.C; stations, _ = K1.NETWORKS[net]; rec_end = pd.Timestamp(K1.SPLITS[SPLIT][0])
    start = rec_end - pd.DateOffset(months=REC) + pd.Timedelta(days=1)
    df = N.load_daily(); df = df[df.station.isin(stations)].reset_index(drop=True)
    return C.Scaler().fit(df, ((df.date >= start) & (df.date <= rec_end)).to_numpy(), stations), df


def forecast(X2, torch, m, s, D, idx, mod=None, bs=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = X2.batch(D, idx[i:i + bs], s, "cpu")
            if mod is not None:
                mod(b)
            out.append(m(b)[0].numpy())
    return np.maximum(np.expm1(np.concatenate(out)), 1e-3)                        # (anchors, stations, 4) in mg m-3


def stats(r, toward=None):
    r = r[np.isfinite(r)]; mv = np.abs(r) > TOL
    d = {"n": len(r), "median": float(np.median(r)), "mean": float(r.mean()), "p05": float(np.quantile(r, 0.05)), "p95": float(np.quantile(r, 0.95)),
         "share_decrease": float((r < -TOL).mean()), "share_increase": float((r > TOL).mean()), "share_no_change": float((~mv).mean())}
    if toward is not None:
        t = toward[np.isfinite(toward)]; d["share_toward_record_mean"] = float(t.mean()) if len(t) else np.nan
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="main,core,bb-nut"); a = ap.parse_args(); models = a.models.split(",")
    X2, K1, torch = setup(); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4))); OUT.mkdir(parents=True, exist_ok=True)
    seeds, hk = NI.SEEDS[SPLIT], [K1.H.index(h) for h in HS]; runs, checks, store = [], [], {}
    for net in NI.NETS:
        D = X2.prepare(net, SPLIT, REC); ev = D["ev"]; sc, df = scaler(K1, net); feat = D["feat"]; ctp, cwt = feat.index("era5_tp_mm"), feat.index("water_temperature")
        # the scaled inputs give back the raw values: precipitation on the anchor day
        raw = df.pivot(index="date", columns="station", values="era5_tp_mm").reindex(pd.DatetimeIndex(D["anc"][ev]))[D["stations"]].to_numpy(float)
        back = D["X"][ev][:, :, -1, ctp] * sc.std["era5_tp_mm"] + sc.mean["era5_tp_mm"]; ok = np.isfinite(raw)
        chk_in = float(np.abs(back - raw)[ok].max())
        mask = D["m_ev"][ev] > 0; lm = D["mmean"][ev]                               # scored forecasts; record mean of ln C

        def mods(kind, val, what):
            def f(b):
                if what in ("all", "process"):
                    if kind == "rain":
                        b["rain"] = b["rain"] + val
                    else:
                        b["drv"] = b["drv"].clone(); b["drv"][..., 0] = b["drv"][..., 0] + val
                if what in ("all", "network"):
                    b["x"] = b["x"].clone()
                    if kind == "rain":
                        b["x"][:, :, -7:, ctp] += val / sc.std["era5_tp_mm"]
                    else:
                        b["x"][:, :, -7:, cwt] += val / sc.std["water_temperature"]
            return f

        for model in models:
            spec, patch, _ = MODELS[model]; s = X2.Spec.parse(spec); hasnet = bool(s.bb) or s.n_daily() > 0; hasproc = not s.bb
            for seed in seeds:
                d = run_dir(model, net, seed)
                with patched(X2, patch):
                    m = X2.XNet2(s, D["X"].shape[-1], D["stations"], D["up"], D["dist"], D["drec"]); m.load_state_dict(torch.load(d / "model.pt", map_location="cpu")); m.eval()
                    base = forecast(X2, torch, m, s, D, ev)
                    P = pd.read_csv(d / "predictions_eval.csv", dtype={"station": str}); P = P[P.y_pred_raw.notna()]
                    P["anchor_date"] = P.anchor_date.astype(str).str[:10]
                    ai = {str(x)[:10]: i for i, x in enumerate(D["anc"][ev])}; si = {x: i for i, x in enumerate(D["stations"])}; hi = {h: i for i, h in enumerate(K1.H)}
                    P = P[P.anchor_date.isin(ai)]
                    mine = base[P.anchor_date.map(ai).to_numpy(), P.station.map(si).to_numpy(), P.horizon.map(hi).to_numpy()]
                    rel = float(np.abs(mine / np.maximum(P.y_pred_raw.to_numpy(float), 1e-3) - 1).max())
                    checks.append({"model": model, "net": net, "seed": seed, "forecasts_compared": len(P), "max_rel_diff_to_saved_forecast": rel,
                                   "max_abs_diff_raw_rain_mm": chk_in})
                    store[(model, net, seed, "base", "all")] = base
                    for pname, (kind, val) in PERT.items():
                        parts = ["all"] + (["process"] if hasnet and hasproc else [])
                        for what in parts:
                            store[(model, net, seed, pname, what)] = forecast(X2, torch, m, s, D, ev, mods(kind, val, what))
                print(f"[{net}] {model} seed {seed}: max rel diff to saved {rel:.2e}", flush=True)
        store[("mask", net)], store[("lm", net)] = mask, lm
    pd.DataFrame(checks).to_csv(OUT / "response_check.csv", index=False)
    # responses: change of ln(forecast); parts all, process, network (= all - process)
    rows = []
    for model in models:
        spec, _, _ = MODELS[model]; s = X2.Spec.parse(spec); split = (bool(s.bb) or s.n_daily() > 0) and not s.bb
        for pname, (kind, _) in PERT.items():
            for k, h in zip(hk, HS):
                per_scope = {}
                for net in NI.NETS:
                    mk = store[("mask", net)][:, :, k]; lm = store[("lm", net)]
                    B = np.stack([store[(model, net, sd, "base", "all")][:, :, k] for sd in seeds]); A = np.stack([store[(model, net, sd, pname, "all")][:, :, k] for sd in seeds])
                    Pp = np.stack([store[(model, net, sd, pname, "process")][:, :, k] for sd in seeds]) if split else None
                    cur = {"all": (np.log(A) - np.log(B))}
                    if split:
                        cur["process"] = np.log(Pp) - np.log(B); cur["network"] = np.log(A) - np.log(Pp)
                    ens = {"all": np.log(A.mean(0)) - np.log(B.mean(0))}
                    if split:
                        ens["process"] = np.log(Pp.mean(0)) - np.log(B.mean(0)); ens["network"] = np.log(A.mean(0)) - np.log(Pp.mean(0))
                    side = np.sign(lm - np.log(B.mean(0)))                              # +1 if the record mean is above the forecast
                    per_scope[net] = (cur, ens, mk, side)
                    for part, r in cur.items():
                        for i, sd in enumerate(seeds):
                            runs.append({"model": model, "perturbation": pname, "part": part, "horizon": h, "net": net, "seed": sd, **stats(r[i][mk])})
                for scope in ("all", *NI.NETS):
                    nets = NI.NETS if scope == "all" else (scope,)
                    for part in per_scope[nets[0]][1]:
                        r = np.concatenate([per_scope[n][1][part][per_scope[n][2]] for n in nets])
                        sidev = np.concatenate([per_scope[n][3][per_scope[n][2]] for n in nets]); mv = np.abs(r) > TOL
                        toward = (np.sign(r[mv]) == sidev[mv]).astype(float) if kind == "temp" else None
                        one = np.concatenate([per_scope[n][0][part][:, per_scope[n][2]] for n in nets], axis=1)     # (seeds, forecasts)
                        rows.append({"model": model, "perturbation": pname, "part": part, "horizon": h, "scope": scope, "forecast": "seed mean", **stats(r, toward),
                                     "single_runs_median_mean": float(np.median(one, axis=1).mean()), "single_runs_median_sd": float(np.median(one, axis=1).std(ddof=1)),
                                     "single_runs_share_decrease_mean": float((one < -TOL).mean(1).mean()), "single_runs_share_increase_mean": float((one > TOL).mean(1).mean())})
    R = pd.DataFrame(rows); R.to_csv(OUT / "response.csv", index=False); pd.DataFrame(runs).to_csv(OUT / "response_runs.csv", index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60)
    print(pd.DataFrame(checks).groupby("model")[["max_rel_diff_to_saved_forecast", "max_abs_diff_raw_rain_mm"]].max().to_string())
    print(R[R.scope == "all"].drop(columns=["scope", "forecast", "n", "mean"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
