"""Further materials from the runs of stage 8 (no training).

The reading rules of stage 8 stay on its scores; this is
description. Reads the saved runs of stage 8 and of stage 6 (`predictions_eval.csv`, `params_eval.csv`, `config.json`, the
`config.json` of the pre-trainings) and the arrays of `core2.prepare` under the settings of stage 6. Writes to stage8/more/:

  spread.csv             per record and model: summary score (station-median R2, mean of 3, 7 and 14 days) of the single runs
                         (mean, SD, lowest, highest), of the seed-mean forecast, and their difference
  spread_forecast.csv    per record, model and horizon: SD over the five runs of the forecast at each anchor (mg m-3 and
                         ln(1 + C)): median, mean and 90th percentile over the shared forecasts
  terms_runs.csv         per basin, record, model, seed and horizon: the forecast change in ln Chl-a from the last measured value
                         split into additive terms (as explain/terms_runs.csv; for `growth` also the production and loss terms)
  terms.csv              the same as seed mean and SD, by basin and for the four basins together
  terms_check.csv        per run: largest difference between the rebuilt forecast and the saved one, share of clipped forecasts;
                         terms_check.json: largest difference of the `main` rows to explain/terms.csv
  constants_runs.csv     the constants per station, record, model and seed
  constants_outside.csv  `main` and `free-const`: per record and constant, stations (mean over seeds) and station-seed pairs
                         below and above the range of the main model, and stations with at least one constant outside
  constants_free.csv     per station and record: the constants of `main` and `free-const` side by side (mean over seeds), with
                         the half-life of the return at 20 degrees C and the effect of 10 mm/day of rain after 7 days
  growth_constants.csv   `growth`: per record and constant, station median, starting value at the target, median distance from
                         the start as a share of the range width, SD over seeds (median over stations) and SD over stations
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Names: bb-C1 and the other bb-* models are the joint network of the paper (one xLSTM trained on the stations together,
without process terms; 'bb' for black box); core (P-mcf-all-kT) is the process model and hybrid (H-res-all) the hybrid
model.
Usage: newinput_ablate_more.py all | spread | terms | constants
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "xpinn"))
import newinput as NI                                                              # noqa: E402
import newinput_ablate as AB                                                       # noqa: E402
import newinput_explain as EX                                                      # noqa: E402

OUT = AB.OUT / "more"
HS, SEEDS, RECS, KEY = (3, 7, 14), AB.SEEDS, AB.RECS, NI.KEY
MODELS = ["main", "core", "bb-nut", "free-eps", "free-const", "no-pre", "growth"]
SPLIT_MODELS = ["main", "free-eps", "free-const", "no-pre", "growth"]
K5, TERMS, GRO, KIN = EX.K5, EX.TERMS, ["production", "loss"], ["mu", "loss", "KN", "KP", "KI"]
pd.set_option("display.width", 280); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60)


def spread():
    E = pd.read_csv(AB.OUT / "test" / "station_r2.csv", dtype={"station": str, "seed": str}); E = E[E.horizon.isin(HS) & E.model.isin(MODELS)]; rows = []
    for (rec, m), g in E.groupby(["record", "model"]):
        sc = g.groupby(["seed", "horizon"]).r2.median().unstack("horizon")[list(HS)].mean(axis=1); one = sc.drop("ens")
        rows.append({"record": rec, "model": m, "n_seeds": len(one), "single_mean": one.mean(), "single_sd": one.std(ddof=1), "single_min": one.min(), "single_max": one.max(),
                     "seed_mean_forecast": sc["ens"], "gain_from_averaging": sc["ens"] - one.mean()})
    S = pd.DataFrame(rows).sort_values(["record", "single_mean"], ascending=[False, False]); S.to_csv(OUT / "spread.csv", index=False)
    fr = []
    for rec in RECS:
        parts = []
        for net in NI.NETS:
            W = None
            for m in MODELS:
                for s in SEEDS:
                    x = pd.read_csv(AB.run_dir(m, net, rec, s) / "predictions_eval.csv", dtype={"station": str}); x = x[x.observed & x.y_pred_raw.notna()].set_index(KEY)
                    W = x.y_pred_raw.rename(f"{m}|{s}").to_frame() if W is None else W.join(x.y_pred_raw.rename(f"{m}|{s}"), how="inner")
            parts.append(W.reset_index())
        A = pd.concat(parts, ignore_index=True)
        for h, q in A[A.horizon.isin(HS)].groupby("horizon"):
            for m in MODELS:
                x = q[[f"{m}|{s}" for s in SEEDS]].to_numpy(float); sd, sl = x.std(1, ddof=1), np.log1p(np.clip(x, 0, None)).std(1, ddof=1)
                fr.append({"record": rec, "horizon": int(h), "model": m, "n_runs": x.shape[1], "n_forecasts": len(x), "sd_mg_median": np.median(sd), "sd_mg_mean": sd.mean(),
                           "sd_mg_p90": np.quantile(sd, 0.9), "sd_log_median": np.median(sl), "sd_log_mean": sl.mean(), "sd_log_p90": np.quantile(sl, 0.9)})
    F = pd.DataFrame(fr).sort_values(["record", "horizon", "sd_log_median"], ascending=[False, True, True]); F.to_csv(OUT / "spread_forecast.csv", index=False)
    print(S.round(4).to_string(index=False)); print(F.round(4).to_string(index=False))
    return S, F


def terms():
    X2 = EX.core2(); rr, chk = [], []
    for rec in RECS:
        for net in NI.NETS:
            D = X2.prepare(net, "test", rec); ev, S = D["ev"], len(D["stations"]); E = len(ev)
            lb = np.log(np.maximum(D["base"][ev].astype(float), 1e-3)); s7, age, rain, mm = (D[k][ev].astype(float) for k in ("smooth7", "age", "rain", "mmean"))
            m0, mk, drv = D["mclim0_cf"][ev].astype(float), D["mclim_cf"][ev].astype(float), D["drv_hold"][ev].astype(float)
            Tw = np.clip(drv[..., 0], 0, 40); dec = np.exp(-np.arange(1, 15) / X2.RAIN_EFOLD); anc = D["anc"][ev].astype(str)
            for m in SPLIT_MODELS:
                for s in SEEDS:
                    d = AB.run_dir(m, net, rec, s); p, _ = EX.params(d)
                    assert len(p) == E * S and (p.station.to_numpy().reshape(E, S)[0] == np.array(D["stations"])).all() and (p.anchor_date.to_numpy().reshape(E, S)[:, 0] == anc).all()
                    ka, om, th, be, al, eps = (p[k].to_numpy(float).reshape(E, S) for k in K5 + ["eps"])
                    q = {"kap": np.minimum(ka[..., None] * th[..., None] ** (Tw - 20), X2.KAPPA_MAX), "tk": mm[..., None] + om[..., None] * (mk - mm[..., None]),
                         "f": -be[..., None] * rain[..., None] * dec}
                    flush, prod, loss = q["f"].copy(), None, None
                    if m == "growth":                                               # the model's growth term: mu * Monod factor - loss (core2.monod)
                        mu, lo, KN, KP, KI = (p[k].to_numpy(float).reshape(E, S)[..., None] for k in KIN)
                        Nn, Pp, Ii = (np.maximum(drv[..., c], 0) for c in (1, 2, 3))
                        prod = mu * X2.THETA ** (Tw - 20) * Nn / (KN + Nn) * Pp / (KP + Pp) * Ii / (KI + Ii); loss = -lo * np.ones_like(prod); q["f"] = flush + prod + loss
                    t0 = mm + om * (m0 - mm); la0 = al * lb + (1 - al) * s7; q["la"] = t0 + (la0 - t0) * (1 - q["kap"][..., 0]) ** age
                    full, mech = EX.rollout(q, eps), EX.rollout(q, 0.0)
                    pe = pd.read_csv(d / "predictions_eval.csv", dtype={"station": str}); dif = 0.0
                    for h in X2.H:
                        y = pe[pe.horizon == h].y_pred_raw.to_numpy(float).reshape(E, S); dif = max(dif, float(np.max(np.abs(np.exp(full[..., h - 1]) - y) / np.maximum(y, 1e-3))))
                    l1 = np.log1p(D["base"][ev].astype(float)); d7f, d7m = np.log1p(np.exp(full[..., 6])) - l1, np.log1p(np.exp(mech[..., 6])) - l1
                    a, e = np.abs(d7m).mean(), np.abs(d7f - d7m).mean()
                    c = {"record": rec, "net": net, "model": m, "seed": s, "max_rel_diff_forecast": dif, "max_abs_diff_d7_full": float(np.abs(d7f - p.d7_full.to_numpy(float).reshape(E, S)).max()),
                         "process_share_7d_stage6_definition": float(a / (a + e))}
                    names = TERMS + (GRO if m == "growth" else [])
                    for n in HS:
                        W = np.ones((E, S, n))                                         # W[k-1] = product of (1 - kap_j), j = k+1..n
                        for k in range(n - 2, -1, -1):
                            W[..., k] = W[..., k + 1] * (1 - q["kap"][..., k + 1])
                        P0 = W[..., 0] * (1 - q["kap"][..., 0]); kw = W * q["kap"][..., :n]
                        T = {"smoothing": P0 * (la0 - lb), "ageing": P0 * (q["la"] - la0), "return_mean": (1 - P0) * (mm - lb),
                             "return_season": (kw * (om[..., None] * (mk[..., :n] - mm[..., None]))).sum(-1), "flushing": (W * flush[..., :n]).sum(-1), "network": eps * W.sum(-1)}
                        if m == "growth":
                            T["production"], T["loss"] = (W * prod[..., :n]).sum(-1), (W * loss[..., :n]).sum(-1)
                        tot = full[..., n - 1] - lb; lin = sum(T.values()); rem = tot - lin; c[f"clipped_share_{n}d"] = float((np.abs(rem) > 1e-6).mean())
                        c[f"max_abs_remainder_{n}d"] = float(np.abs(rem).max()); T["remainder"] = rem
                        sizes = {k: float(np.abs(v).mean()) for k, v in T.items()}; den = sum(sizes[k] for k in names); var = float(tot.var())
                        proc = lin - T["network"]; pa, ne = float(np.abs(proc).mean()), sizes["network"]
                        extra = {"net_growth": T["production"] + T["loss"]} if m == "growth" else {}
                        for k, v in {**T, **extra}.items():
                            rr.append({"record": rec, "net": net, "model": m, "seed": s, "horizon": n, "term": k, "mean_abs": float(np.abs(v).mean()), "mean": float(v.mean()),
                                       "share_abs": sizes[k] / den if k in names else np.nan, "cov_share": float(np.cov(v.ravel(), tot.ravel())[0, 1] / var) if var > 0 else np.nan})
                        rr.append({"record": rec, "net": net, "model": m, "seed": s, "horizon": n, "term": "process_all", "mean_abs": pa, "mean": float(proc.mean()), "share_abs": pa / (pa + ne),
                                   "cov_share": float(np.cov(proc.ravel(), tot.ravel())[0, 1] / var) if var > 0 else np.nan})
                        rr.append({"record": rec, "net": net, "model": m, "seed": s, "horizon": n, "term": "total_change", "mean_abs": float(np.abs(tot).mean()), "mean": float(tot.mean()),
                                   "share_abs": np.nan, "cov_share": 1.0})
                    chk.append(c)
    R = pd.DataFrame(rr); R.to_csv(OUT / "terms_runs.csv", index=False); Ck = pd.DataFrame(chk); Ck.to_csv(OUT / "terms_check.csv", index=False)
    v = ["mean_abs", "mean", "share_abs", "cov_share"]
    by = R.groupby(["record", "model", "net", "horizon", "term"])[v].agg(["mean", "std"]); by.columns = [f"{a}_{'seed_mean' if b == 'mean' else 'seed_sd'}" for a, b in by.columns]
    al = R.groupby(["record", "model", "seed", "horizon", "term"])[v].mean().groupby(["record", "model", "horizon", "term"]).agg(["mean", "std"])    # the four basins weighted equally
    al.columns = by.columns; al = al.reset_index().assign(net="all")
    Tm = pd.concat([by.reset_index(), al], ignore_index=True); Tm.to_csv(OUT / "terms.csv", index=False)
    ref = pd.read_csv(EX.OUT / "terms.csv"); j = Tm[Tm.model == "main"].merge(ref, on=["record", "net", "horizon", "term"], suffixes=("", "_explain"))
    cols = [c for c in by.columns if c + "_explain" in j]
    ck = {"rows_main": int((Tm.model == "main").sum()), "rows_compared": len(j), "max_abs_diff_to_explain_terms": float(max((j[c] - j[c + "_explain"]).abs().max() for c in cols))}
    (OUT / "terms_check.json").write_text(json.dumps(ck, indent=1)); print(json.dumps(ck, indent=1))
    print(Ck.drop(columns=["seed"]).groupby(["record", "model"]).max(numeric_only=True).round(6).to_string())
    print("process share, stage-6 definition:", Ck.groupby(["record", "model"]).process_share_7d_stage6_definition.mean().round(4).to_dict())
    a = Tm[Tm.net == "all"]
    for col in ("share_abs_seed_mean", "share_abs_seed_sd", "mean_abs_seed_mean", "cov_share_seed_mean", "mean_seed_mean"):
        print("==", col); print(a.pivot_table(index=["record", "model", "horizon"], columns="term", values=col).reindex(columns=TERMS + GRO + ["net_growth", "remainder", "process_all", "total_change"]).round(4).to_string())
    print("== network share of the summed term sizes at 7 days by basin")
    print(Tm[(Tm.horizon == 7) & (Tm.term == "network") & (Tm.net != "all")].pivot_table(index=["record", "model"], columns="net", values="share_abs_seed_mean").round(3).to_string())
    return Tm


def constants():
    rows = []
    for rec in RECS:
        for net in NI.NETS:
            for m in ("main", "free-const", "free-eps", "no-pre", "growth"):
                for s in SEEDS:
                    d = AB.run_dir(m, net, rec, s); p, c = EX.params(d); k = [x for x in K5 + KIN if x in p]
                    g = p.groupby("station")[k].median().reset_index().assign(record=rec, net=net, model=m, seed=s)
                    if m == "growth":                                               # starting values at the target: source median (mu, loss), range centre (K)
                        nat = json.loads((AB.OUT / m / "pretrain" / c["pretrain"] / "config.json").read_text())["national_median"]
                        for x in KIN:
                            g[f"start_{x}"] = nat[x] if x in nat else sum(AB.RANGES[x]) / 2
                    rows.append(g)
    R = pd.concat(rows, ignore_index=True); R.to_csv(OUT / "constants_runs.csv", index=False)
    M = R.groupby(["record", "net", "model", "station"])[K5].mean().reset_index(); out = []
    for (rec, m), g in M[M.model.isin(["main", "free-const"])].groupby(["record", "model"]):
        r1 = R[(R.record == rec) & (R.model == m)]; anyo = pd.Series(False, index=g.index)
        for c in K5:
            lo, hi = AB.RANGES[c]; anyo |= (g[c] < lo) | (g[c] > hi)
            out.append({"record": rec, "model": m, "constant": c, "stations": len(g), "stations_below": int((g[c] < lo).sum()), "stations_above": int((g[c] > hi).sum()),
                        "pairs": len(r1), "pairs_below": int((r1[c] < lo).sum()), "pairs_above": int((r1[c] > hi).sum()), "min": g[c].min(), "median": g[c].median(), "max": g[c].max()})
        out.append({"record": rec, "model": m, "constant": "any", "stations": len(g), "stations_below": int(anyo.sum()), "stations_above": np.nan, "pairs": len(r1),
                    "pairs_below": int(pd.concat([(r1[c] < AB.RANGES[c][0]) | (r1[c] > AB.RANGES[c][1]) for c in K5], axis=1).any(axis=1).sum()), "pairs_above": np.nan})
    O = pd.DataFrame(out); O.to_csv(OUT / "constants_outside.csv", index=False)
    hl = lambda k: np.log(0.5) / np.log1p(-np.clip(k, 1e-9, 0.9))
    def derived(g):
        w = np.stack([(1 - g.kappa.to_numpy()) ** (7 - k) * np.exp(-k / 3.0) for k in range(1, 8)], 1).sum(1)
        return g.assign(halflife_20C=hl(g.kappa), rain10mm_7d_pct=100 * (np.exp(-g.beta.to_numpy() * 10.0 * w) - 1))
    W = derived(M[M.model == "main"]).merge(derived(M[M.model == "free-const"]), on=["record", "net", "station"], suffixes=("_main", "_free")).drop(columns=["model_main", "model_free"])
    W.to_csv(OUT / "constants_free.csv", index=False)
    G = R[R.model == "growth"]; go = []
    for rec, g in G.groupby("record"):
        st = g.groupby("station")
        for c in KIN:
            lo, hi = AB.RANGES[c]; mean, sd = st[c].mean(), st[c].std(ddof=1); start = st[f"start_{c}"].mean()
            go.append({"record": rec, "constant": c, "range_low": lo, "range_high": hi, "start_median": start.median(), "station_median": mean.median(), "station_min": mean.min(), "station_max": mean.max(),
                       "median_move_share_of_range": float(((mean - start).abs() / (hi - lo)).median()), "max_move_share_of_range": float(((mean - start).abs() / (hi - lo)).max()),
                       "sd_over_seeds_median": sd.median(), "sd_over_stations": mean.std(ddof=1), "runs_per_station": int(st.size().iloc[0])})
    GC = pd.DataFrame(go); GC.to_csv(OUT / "growth_constants.csv", index=False)
    print(O.round(4).to_string(index=False)); print(GC.round(4).to_string(index=False))
    print(W.groupby("record")[["halflife_20C_main", "halflife_20C_free", "rain10mm_7d_pct_main", "rain10mm_7d_pct_free", "alpha_main", "alpha_free", "beta_main", "beta_free", "omega_main", "omega_free"]].median().round(4).to_string())
    print("stations with a positive rain effect (free):", W.groupby("record").apply(lambda g: int((g.rain10mm_7d_pct_free > 0).sum())).to_dict())
    return O, GC


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"; OUT.mkdir(parents=True, exist_ok=True)
    if cmd in ("all", "spread"):
        spread()
    if cmd in ("all", "constants"):
        constants()
    if cmd in ("all", "terms"):
        terms()


if __name__ == "__main__":
    main()
