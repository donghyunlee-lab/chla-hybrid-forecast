"""Scoring of the automatic search of core.py. Accuracy against the references
and the explainability checks X1-X5, per stage. Nothing is trained here; the statistical references of the development
year (damped anomaly, ARIMA, harmonic ARIMA) are fitted with the national benchmark's functions on the same record."""
from __future__ import annotations
import json, sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
for p in ("clean_revision", "national", "xpinn"):
    sys.path.insert(0, str(ROOT / "scripts" / p))
import core as X                                                                 # noqa: E402
import nat as N                                                                  # noqa: E402
import stats_nat as S                                                            # noqa: E402

KEY = ["station", "anchor_date", "horizon"]; FH = (3, 7, 14); RUNS = X.OUT / "runs"; REFS = X.OUT / "refs"
DEV_REFS = ["persistence", "damped anomaly", "ARIMA", "harmonic ARIMA", "bb-C1", "bb-C2"]
MIN_EVAL, THR = 60, {"X1": 0.8, "X2": 0.10, "X3": 0.10, "X4": 2 / 3, "X5": 0.005, "agree": 0.7, "goal": -0.02}


def load(t):
    d = RUNS / t
    if not (d / "DONE").exists():
        return None
    x = pd.read_csv(d / "predictions_eval.csv", dtype={"station": str})
    x = x[x.observed & x.y_pred_raw.notna()].copy(); x["anchor_date"] = x.anchor_date.astype(str).str[:10]
    return x[KEY + ["target_date", "y_true_raw", "y_pred_raw", "base_raw"]]


def ensemble(v, net, split, rec, seeds, placebo=False):
    """Mean forecast of the seeds (None unless every seed is finished)."""
    fr = [load(X.tag(v, net, split, rec, s, placebo)) for s in seeds]
    if any(f is None for f in fr):
        return None
    e = fr[0][KEY + ["target_date", "y_true_raw", "base_raw"]].copy()
    for i, f in enumerate(fr):
        e = e.merge(f[KEY + ["y_pred_raw"]].rename(columns={"y_pred_raw": f"p{i}"}), on=KEY)
    e["y_pred_raw"] = e[[f"p{i}" for i in range(len(fr))]].mean(1)
    return e[KEY + ["target_date", "y_true_raw", "base_raw", "y_pred_raw"]]


def config(net, split, rec):
    """Configuration of any finished run of this net, split and record (the data and counts are the same for all)."""
    for d in sorted(RUNS.glob(f"*__{net}_{split}_r{rec}_s*")):
        if (d / "DONE").exists() and not d.name.endswith("__placebo"):
            return json.loads((d / "config.json").read_text())
    raise FileNotFoundError(f"no finished run for {net} {split} {rec}")


def eligible(net, split, rec):
    c = config(net, split, rec)["counts"]
    return [s for s, k in c.items() if k["chl_train"] >= N.MIN_CHL_TRAIN and k["chl_val"] >= N.MIN_CHL_VAL and min(k["eval_targets"]) >= MIN_EVAL]


# ------------------------------------------------------------------------------------------------ development references
def _ref_task(a):
    st, rec, keys = a
    import warnings; warnings.filterwarnings("ignore")
    import bench_baselines_v7 as B7, common
    B7.RECORD_END = X.SPLITS["dev"][0]; common.SPLITS = {**common.SPLITS, "test": (X.SPLITS["dev"][1], X.SPLITS["dev"][2])}
    df = N.load_daily(); out = []
    _, dam = B7.seasonal(df, st, rec, keys); out.append(dam.assign(model="damped anomaly"))
    for harmonic, name in ((False, "ARIMA"), (True, "harmonic ARIMA")):
        f, _ = B7.arima_like(df, st, rec, harmonic); out.append(f.assign(model=name))
    return pd.concat(out, ignore_index=True)


def dev_refs(net, rec, workers=12):
    """Damped anomaly, ARIMA and harmonic ARIMA fitted on the record ending 2022-12-31 and run through 2023 (cached)."""
    f = REFS / f"{net}_dev_r{rec}.csv"
    if f.exists():
        r = pd.read_csv(f, dtype={"station": str}); r["anchor_date"] = r.anchor_date.astype(str).str[:10]
        return r
    REFS.mkdir(parents=True, exist_ok=True)
    base = next(load(d.name) for d in sorted(RUNS.glob(f"*__{net}_dev_r{rec}_s*")) if (d / "DONE").exists() and not d.name.endswith("__placebo"))
    tasks = [(st, rec, base[base.station == st][["station", "anchor_date", "horizon", "target_date", "y_true_raw"]]) for st in X.NETWORKS[net][0]]
    with Pool(min(workers, len(tasks))) as pool:
        r = pd.concat(pool.map(_ref_task, tasks), ignore_index=True)
    r["anchor_date"] = r.anchor_date.astype(str).str[:10]
    r = r[r.observed.astype(bool)][KEY + ["y_pred_raw", "model"]]; r.to_csv(f, index=False, float_format="%.6g")
    return r


def r2_table(frames: dict, stations, truth):
    """R^2 (and the other metrics) per station and horizon, on the forecasts every model shares."""
    rows = []
    for st in stations:
        for h in X.H:
            keys = truth[(truth.station == st) & (truth.horizon == h)][KEY + ["y_true_raw"]]
            for f in frames.values():
                keys = keys.merge(f[(f.station == st) & (f.horizon == h)][KEY], on=KEY)
            if len(keys) < 10:
                continue
            y = keys.y_true_raw.to_numpy(float)
            q = keys[KEY].merge(frames["persistence"][KEY + ["y_pred_raw"]], on=KEY, how="left").y_pred_raw.to_numpy(float)
            for name, f in frames.items():
                p = keys[KEY].merge(f[KEY + ["y_pred_raw"]], on=KEY, how="left").y_pred_raw.to_numpy(float)
                rows.append({"station": st, "horizon": h, "model": name, **S.metrics(y, p, q)})
    return pd.DataFrame(rows)


def dev_table(variants, net, rec, seeds):
    frames = {v: e for v in list(variants) + ["bb-C1", "bb-C2"] if (e := ensemble(v, net, "dev", rec, seeds)) is not None}
    truth = next(iter(frames.values()))
    frames["persistence"] = truth.assign(y_pred_raw=truth.base_raw)
    refs = dev_refs(net, rec)
    for name, g in refs.groupby("model"):
        frames[name] = g
    return r2_table(frames, eligible(net, "dev", rec), truth)


def accuracy(T, variants, refs=DEV_REFS):
    """A = mean over 3, 7 and 14 days of the median over stations of R^2(variant) - R^2(best reference at that horizon).
    Also the mean rank among every model in T (1 = best)."""
    med = T.groupby(["model", "horizon"]).r2.median().unstack()
    best = {h: med.loc[[r for r in refs if r in med.index], h].idxmax() for h in X.H}
    W = T.pivot_table(index=["station", "horizon"], columns="model", values="r2")
    rk = W.rank(axis=1, ascending=False).groupby(level="horizon").mean()
    rows = []
    for v in variants:
        if v not in W.columns:
            continue
        d = {h: float((W.xs(h, level="horizon")[v] - W.xs(h, level="horizon")[best[h]]).median()) for h in X.H}
        rows.append({"variant": v, **{f"r2_h{h}": float(med.loc[v, h]) for h in X.H}, **{f"dr2_h{h}": d[h] for h in X.H},
                     "A": float(np.mean([d[h] for h in FH])), "mean_rank": float(rk.loc[list(FH), v].mean()),
                     "goal_ok": bool(all(d[h] >= THR["goal"] for h in FH))})
    return pd.DataFrame(rows), {h: best[h] for h in X.H}


# ------------------------------------------------------------------------------------------------ explainability
@lru_cache(maxsize=1)
def covariates():
    """Rain of the last 7 days (ERA5, sum) and TP of the last 7 days (mean of measured days) at each station and day."""
    era = pd.read_csv(N.NAT / "era5_daily.csv", parse_dates=["date"], dtype={"station": str}, usecols=["station", "date", "era5_tp_mm"])
    obs = pd.read_csv(N.NAT / "daily_observed.csv", parse_dates=["date"], dtype={"station": str}, usecols=["station", "date", "total_phosphorus"])
    d = era.merge(obs, on=["station", "date"], how="outer").sort_values(["station", "date"])
    g = d.groupby("station")
    d["precip7"] = g.era5_tp_mm.transform(lambda v: v.rolling(7, min_periods=7).sum())
    d["tp7"] = g.total_phosphorus.transform(lambda v: v.rolling(7, min_periods=1).mean())
    d["anchor_date"] = d.date.dt.strftime("%Y-%m-%d")
    return d[["station", "anchor_date", "precip7", "tp7"]]


def _within(P, col, cov, sign, stations):
    rho = []
    for st in stations:
        g = P[P.station == st][[col, cov]].dropna()
        if len(g) >= 30 and g[col].std() > 0 and g[cov].std() > 0:
            rho.append(stats.spearmanr(g[col], g[cov])[0])
    rho = np.array([r for r in rho if np.isfinite(r)])
    if not len(rho):
        return None
    med, agree = float(np.median(rho)), float((np.sign(rho) == sign).mean())
    return {"median_rho": med, "agree": agree, "n": len(rho), "pass": bool(sign * med > 0 and agree >= THR["agree"])}


def explain(v, net, split, rec, seeds, stations, T_real=None):
    """X1 mechanistic share, X2 seed stability, X3 bound hitting, X4 expected directions, X5 kinetic placebo (if run)."""
    vv = X.Variant.parse(v)
    pars = [pd.read_csv(RUNS / X.tag(v, net, split, rec, s) / "params_eval.csv", dtype={"station": str}) for s in seeds]
    pars = [p[p.station.isin(stations)] for p in pars]; allp = pd.concat(pars, ignore_index=True)
    names = vv.params() + (["KN", "KP", "KI"] if vv.k == "learned" else []) + (["v"] if vv.t == "distance" else [])
    out = {"variant": v, "n_daily": vv.n_daily()}
    if vv.r == "bounded":
        mech, res = allp.d7_mech.abs().mean(), (allp.d7_full - allp.d7_mech).abs().mean(); out["X1"] = float(mech / (mech + res))
    else:
        out["X1"] = 1.0
    sd = []
    for n in names:
        lo, hi = X.RANGE[n]
        meds = pd.concat([p.groupby("station")[n].median().rename(i) for i, p in enumerate(pars)], axis=1).dropna()
        sd += list((meds.std(axis=1, ddof=0) / (hi - lo)).to_numpy())
    out["X2"] = float(np.median(sd)) if sd else 0.0
    hits = {}
    for n in names:
        if n == "share":
            continue
        lo, hi = X.RANGE[n]; z = (allp[n].dropna() - lo) / (hi - lo)
        hits[n] = float(((z < 0.01) | (z > 0.99)).mean()) if len(z) else 0.0
    out["X3"] = max(hits.values()) if hits else 0.0; out["X3_worst"] = max(hits, key=hits.get) if hits else ""
    P = allp.groupby(["station", "anchor_date"]).mean(numeric_only=True).reset_index().merge(covariates(), on=["station", "anchor_date"], how="left")
    checks = {}
    if vv.p != "station":
        checks["loss up with rain"] = _within(P, "loss", "precip7", +1, stations)
        if vv.t != "none":
            checks["exchange up with rain"] = _within(P, "w", "precip7", +1, stations)
        if vv.t == "free":
            checks["travel time down with rain"] = _within(P, "tau1", "precip7", -1, stations)
        if vv.g == "ricker":
            checks["capacity up with TP"] = _within(P, "K", "tp7", +1, stations)
    if vv.t == "free":
        dist = config(net, split, rec)["distance_km"]; idx = {s: i for i, s in enumerate(X.NETWORKS[net][0])}
        links = [(P[P.station == s][col].median(), dist[idx[s]][k]) for s in stations for k, col in enumerate(("tau1", "tau2"))
                 if len(X.NETWORKS[net][1].get(s, [])) > k]
        if len(links) >= 5:
            r = stats.spearmanr([a for a, _ in links], [b for _, b in links])[0]
            checks["travel time up with distance"] = {"median_rho": float(r), "agree": np.nan, "n": len(links), "pass": bool(r > 0)}
    checks = {k: c for k, c in checks.items() if c is not None}
    out["X4"] = float(np.mean([c["pass"] for c in checks.values()])) if checks else np.nan
    out["X4_detail"] = "; ".join(f"{k}: rho {c['median_rho']:+.2f}" + (f", agree {c['agree']:.0%}" if np.isfinite(c["agree"]) else "") + (" ok" if c["pass"] else " fail")
                                 for k, c in checks.items())
    if T_real is not None:
        e_pl = ensemble(v, net, split, rec, seeds, placebo=True)
        if e_pl is not None:
            e = ensemble(v, net, split, rec, seeds); drops = []
            for st in stations:
                a = e[(e.station == st) & (e.horizon == 7)][KEY + ["y_true_raw", "y_pred_raw"]].merge(
                    e_pl[(e_pl.station == st) & (e_pl.horizon == 7)][KEY + ["y_pred_raw"]].rename(columns={"y_pred_raw": "p_pl"}), on=KEY)
                if len(a) >= 10:                                              # real minus kinetic placebo, 7 days
                    y = a.y_true_raw.to_numpy(float); q = np.full_like(y, y.mean())
                    drops.append(S.metrics(y, a.y_pred_raw.to_numpy(float), q)["r2"] - S.metrics(y, a.p_pl.to_numpy(float), q)["r2"])
            out["X5"] = float(np.median(drops)) if drops else np.nan
    out["screen"] = bool(out["X1"] >= THR["X1"] and out["X2"] <= THR["X2"] and out["X3"] <= THR["X3"] and (np.isnan(out["X4"]) or out["X4"] >= THR["X4"] - 1e-9))
    if "X5" in out:
        out["final"] = bool(out["screen"] and np.isfinite(out["X5"]) and out["X5"] >= THR["X5"])
    return out


def leaderboard(variants, net, rec, seeds, placebo=False):
    T = dev_table(variants, net, rec, seeds); acc, best = accuracy(T, list(variants) + ["bb-C1", "bb-C2"])
    st = eligible(net, "dev", rec)
    ex = pd.DataFrame([explain(v, net, "dev", rec, seeds, st, T_real=T if placebo else None) for v in variants if not v.startswith("bb-")])
    lb = acc.merge(ex, on="variant", how="left")
    lb["record"] = rec; lb["n_seeds"] = len(seeds)
    return lb.sort_values("A", ascending=False).reset_index(drop=True), T, best


# ------------------------------------------------------------------------------------------------ test years
def _test_station(a):
    code, variants, net, recs, seeds = a
    per = pd.read_csv(N.OUT / "persistence.csv", dtype={"station": str})
    per = per[(per.station == code) & (pd.to_datetime(per.anchor_date) >= pd.Timestamp(N.SPLITS["test"][0]))]
    per = per[KEY + ["y_true_raw", "y_pred_raw"]].rename(columns={"y_pred_raw": "p_pers"}); per["anchor_date"] = per.anchor_date.astype(str).str[:10]
    cfgs = [(m, mo, r, [S.load(t) for t in tags]) for m, mo, r, tags in S.configs(code) if r in (0,) + tuple(recs)]
    for v in variants:
        for r in recs:
            fr = [load(X.tag(v, net, "test", r, s)) for s in seeds]
            cfgs.append((v, "xpinn", r, [None if f is None else f[f.station == code] for f in fr]))
    frames = {}
    for model, mode, rec, fr in cfgs:
        ok = [f.drop_duplicates(KEY) for f in fr if f is not None]
        if len(ok) == len(fr) and ok:
            frames[(model, mode, rec)] = ok
    rows = []
    for h in sorted(per.horizon.unique()):
        ph = per[per.horizon == h]; nb = len(ph); valid, keys = {}, ph[KEY]
        for cfg, ok in frames.items():
            if min(len(f[f.horizon == h][KEY].merge(ph[KEY], on=KEY)) for f in ok) / nb >= 0.8:
                valid[cfg] = ok
        for ok in valid.values():
            for f in ok:
                keys = keys.merge(f[f.horizon == h][KEY], on=KEY)
        b = keys.merge(ph, on=KEY); y, q = b.y_true_raw.to_numpy(float), b.p_pers.to_numpy(float)
        for (model, mode, rec), ok in valid.items():
            p = np.mean([b[KEY].merge(f[KEY + ["y_pred_raw"]], on=KEY, how="left").y_pred_raw.to_numpy(float) for f in ok], axis=0)
            rows.append({"station": code, "model": model, "mode": mode, "record": rec, "horizon": int(h), **S.metrics(y, p, q)})
        rows.append({"station": code, "model": "persistence", "mode": "baseline", "record": 0, "horizon": int(h), **S.metrics(y, q, q)})
    return pd.DataFrame(rows)


def test_table(variants, net, recs, seeds, workers=12):
    codes = [c for c in X.NETWORKS[net][0] if all((c, r) in S.ELIG for r in recs)]
    with Pool(min(workers, len(codes))) as pool:
        return pd.concat(pool.map(_test_station, [(c, variants, net, recs, seeds) for c in codes]), ignore_index=True)


def test_compare(E, focus, recs):
    """Each focus variant against every other model at the same record (Wilcoxon across stations, Holm within record and
    horizon); medians, mean ranks, and the gap to the best national model."""
    E = E.assign(label=E.model + " (" + E["mode"] + ")")
    cmp_, summ = [], []
    for r in recs:
        for h in X.H:
            q = E[(E.horizon == h) & ((E.record == r) | (E.model == "persistence") | ((E["mode"] == "zero_shot")))]
            W = q.pivot_table(index="station", columns="label", values="r2").dropna(axis=1, thresh=max(6, int(0.8 * q.station.nunique())))
            W = W.dropna()
            rk = W.rank(axis=1, ascending=False).mean()
            nat = [c for c in W.columns if not c.endswith("(xpinn)")]
            best_nat = W[nat].median().idxmax()
            for c in W.columns:
                summ.append({"record": r, "horizon": h, "model": c, "n_stations": len(W), "median_r2": float(W[c].median()), "mean_rank": float(rk[c]),
                             "best_national": best_nat})
            if h == 1:
                continue
            for f in focus:
                lab = f"{f} (xpinn)"
                if lab not in W.columns:
                    continue
                g = []
                for c in W.columns:
                    if c == lab:
                        continue
                    d = W[lab] - W[c]; p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                    g.append({"focus": f, "record": r, "horizon": h, "other": c, "n_stations": len(d), "median_diff": float(d.median()),
                              "share_better": float((d > 0).mean()), "wilcoxon_p": float(p), "is_best_national": c == best_nat})
                if g:
                    adj = multipletests([x["wilcoxon_p"] for x in g], method="holm")[1]
                    for x, a_ in zip(g, adj):
                        x["holm_p"] = float(a_)
                    cmp_ += g
    return pd.DataFrame(cmp_), pd.DataFrame(summ)
