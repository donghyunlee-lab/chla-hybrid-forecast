"""Explanation materials of the main model (hybrid H-res-all of stage 6). No training.

Reads the saved runs of stage 6 (params_eval.csv, predictions_eval.csv, config.json, the config.json of the pre-training),
the arrays of core2.prepare under the settings of stage 6, and the shared forecasts of stage 7. Writes to explain/:

  fit.csv                per basin, record and seed: selected learning rate, best epoch, gate before and after the target fit,
                         largest distance of each station constant from its starting value (median of the source stations)
  constants_runs.csv     the five constants per station, record and seed, with the starting values
  constants_station.csv  per station and record: the constants (mean and SD over seeds), starting values, half-life of the return
                         at 5, 15, 20 and 25 degrees C, rate factor per 10 degrees C, effect of 10 mm/day of rain after 7 days
  terms_runs.csv         per basin, record, seed and horizon (3, 7, 14 days): the forecast change in ln Chl-a from the last measured
                         value split into additive terms; mean absolute size, mean, share of the absolute sizes, covariance share
  terms.csv              the same as seed mean and SD, by basin and for the four basins together
  terms_station.csv      mean absolute size of the terms per station (seed mean)
  terms_check.csv        per run: largest difference between the rebuilt forecast and the saved one, share of clipped forecasts,
                         and the process share of stage 6 recomputed
  alert.csv              exceedance of 15 and 25 mg m-3 at the target day for every model of stage 7 (seed-mean forecasts)
  diagnosis.csv          per station, record, horizon and model: mean error, SD of the error, correlation, MSE and the share of
                         the squared mean error, with the station's level shift
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Names: bb-C1 and the other bb-* models are the joint network of the paper (one xLSTM trained on the stations together,
without process terms; 'bb' for black box); core (P-mcf-all-kT) is the process model and hybrid (H-res-all) the hybrid
model.
Usage: newinput_explain.py all | fit | terms | alert | diagnosis
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "xpinn"))
import newinput as NI                                                              # noqa: E402

OUT = NI.STAGE5 / "explain"
K5, HS, SEEDS, RECS = ["kappa", "omega", "thk", "beta", "alpha"], (3, 7, 14), NI.SEEDS["test"], NI.RECS["test"]
TERMS = ["smoothing", "ageing", "return_mean", "return_season", "flushing", "network"]
THR, DIAG = (15.0, 25.0), ["hybrid", "core", "xLSTM zero-shot", "xLSTM fine-tuned", "ARIMA", "persistence"]


def core2():
    """core2 as a run of stage 6 sees it (station list, data folder, network inputs)."""
    for k in NI.SWITCHES:
        os.environ.pop(k, None)
    os.environ.update(NI.env())
    import core2 as X2
    return X2


def runs(model="hybrid"):
    for rec in RECS:
        for net in NI.NETS:
            for s in SEEDS:
                yield rec, net, s, NI.run_dir(model, net, "test", rec, s)


def params(d):
    """Station constants of a run (they do not change with the anchor) and its configuration."""
    p = pd.read_csv(d / "params_eval.csv", dtype={"station": str}); c = json.loads((d / "config.json").read_text())
    return p, c


def fit():
    rows, cs = [], []
    for rec, net, s, d in runs():
        p, c = params(d); pre = json.loads((NI.OUT / "pretrain" / c["pretrain"] / "config.json").read_text()); nat = pre["national_median"]
        st = p.groupby("station")[K5].median()
        r = {"record": rec, "net": net, "seed": s, "lr": c["lr_selected"], "best_epoch": c["best_epoch"], "gate_pretrained": pre["gate"][0], "gate": c["gate"][0],
             "pretrain_lr": pre["lr_selected"], "pretrain_best_epoch": pre["best_epoch"], "mean_abs_eps": float(p.eps.abs().mean())}
        for k in K5:
            r[f"start_{k}"] = nat[k]; r[f"max_move_{k}"] = float((st[k] - nat[k]).abs().max())
        rows.append(r)
        cs.append(st.reset_index().assign(record=rec, net=net, seed=s, **{f"start_{k}": nat[k] for k in K5}))
    F = pd.DataFrame(rows); F.to_csv(OUT / "fit.csv", index=False); R = pd.concat(cs, ignore_index=True); R.to_csv(OUT / "constants_runs.csv", index=False)
    hl = lambda k: np.log(0.5) / np.log1p(-np.clip(k, 1e-9, 0.9))
    R["halflife_20C"] = hl(R.kappa); R["factor_per_10C"] = R.thk ** 10
    for T in (5, 15, 25):
        R[f"halflife_{T}C"] = hl(R.kappa * R.thk ** (T - 20))
    w = np.stack([(1 - R.kappa.to_numpy()) ** (7 - k) * np.exp(-k / 3.0) for k in range(1, 8)], 1).sum(1)      # at 20 degrees C
    R["rain10mm_7d_pct"] = 100 * (np.exp(-R.beta.to_numpy() * 10.0 * w) - 1)
    cols = K5 + ["halflife_20C", "halflife_5C", "halflife_15C", "halflife_25C", "factor_per_10C", "rain10mm_7d_pct"]
    g = R.groupby(["record", "net", "station"]); S = g[cols].mean().join(g[K5].std().add_suffix("_sd")).join(g[[f"start_{k}" for k in K5]].mean()).reset_index()
    S.to_csv(OUT / "constants_station.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 60)
    print(F.groupby(["record", "net"]).agg(lr=("lr", lambda x: sorted(set(x))), epoch_min=("best_epoch", "min"), epoch_max=("best_epoch", "max"), first_epoch_runs=("best_epoch", lambda x: int((x <= 0).sum())),
                                           gate_pre=("gate_pretrained", lambda x: float(np.abs(x).mean())), gate=("gate", lambda x: float(np.abs(x).mean())), eps=("mean_abs_eps", "mean"),
                                           move_kappa=("max_move_kappa", "max"), move_omega=("max_move_omega", "max"), move_alpha=("max_move_alpha", "max")).round(4).to_string())
    print(S.groupby(["record", "net"])[cols].median().round(4).to_string())
    return F, S


def rollout(q, eps):
    """The model's daily steps in numpy (with its clipping). q: arrays of a basin. Returns ln C after 1..14 days, shape (E, S, 14)."""
    C = np.exp(q["la"]); out = []
    for k in range(14):
        lg = q["f"][..., k] - q["kap"][..., k] * (np.log(np.maximum(C, 1e-3)) - q["tk"][..., k]) + eps
        C = np.clip(C * np.exp(np.clip(lg, -1.0, 1.0)), 1e-3, 1e4); out.append(np.log(C))
    return np.stack(out, -1)


def terms():
    X2 = core2(); rr, st_rows, chk, lev = [], [], [], []
    for rec in RECS:
        for net in NI.NETS:
            D = X2.prepare(net, "test", rec); ev, S = D["ev"], len(D["stations"]); E = len(ev)
            lb = np.log(np.maximum(D["base"][ev].astype(float), 1e-3)); s7, age, rain, mm = (D[k][ev].astype(float) for k in ("smooth7", "age", "rain", "mmean"))
            m0, mk, Tw = D["mclim0_cf"][ev].astype(float), D["mclim_cf"][ev].astype(float), np.clip(D["drv_hold"][ev][..., 0].astype(float), 0, 40)
            dec = np.exp(-np.arange(1, 15) / X2.RAIN_EFOLD); anc = D["anc"][ev].astype(str)
            lev += [{"record": rec, "net": net, "station": s_, "record_mean_ln": float(mm[0, i])} for i, s_ in enumerate(D["stations"])]
            for s in SEEDS:
                d = NI.run_dir("hybrid", net, "test", rec, s); p, _ = params(d)
                assert len(p) == E * S and (p.station.to_numpy().reshape(E, S)[0] == np.array(D["stations"])).all() and (p.anchor_date.to_numpy().reshape(E, S)[:, 0] == anc).all()
                ka, om, th, be, al, eps = (p[k].to_numpy(float).reshape(E, S) for k in K5 + ["eps"])
                q = {"kap": np.minimum(ka[..., None] * th[..., None] ** (Tw - 20), X2.KAPPA_MAX), "tk": mm[..., None] + om[..., None] * (mk - mm[..., None]),
                     "f": -be[..., None] * rain[..., None] * dec}
                t0 = mm + om * (m0 - mm); la0 = al * lb + (1 - al) * s7; q["la"] = t0 + (la0 - t0) * (1 - q["kap"][..., 0]) ** age
                full, mech = rollout(q, eps), rollout(q, 0.0)
                # agreement with the saved forecasts (raw units, four horizons) and with the saved 7-day changes
                pe = pd.read_csv(d / "predictions_eval.csv", dtype={"station": str}); dif, clip_share = 0.0, {}
                for h in X2.H:
                    y = pe[pe.horizon == h].y_pred_raw.to_numpy(float).reshape(E, S); mine = np.exp(full[..., h - 1])
                    dif = max(dif, float(np.max(np.abs(mine - y) / np.maximum(y, 1e-3))))
                l1 = np.log1p(D["base"][ev].astype(float)); d7f, d7m = np.log1p(np.exp(full[..., 6])) - l1, np.log1p(np.exp(mech[..., 6])) - l1
                a, e = np.abs(d7m).mean(), np.abs(d7f - d7m).mean()
                c = {"record": rec, "net": net, "seed": s, "max_rel_diff_forecast": dif, "max_abs_diff_d7_full": float(np.abs(d7f - p.d7_full.to_numpy(float).reshape(E, S)).max()),
                     "max_abs_diff_d7_mech": float(np.abs(d7m - p.d7_mech.to_numpy(float).reshape(E, S)).max()), "process_share_7d_stage6_definition": float(a / (a + e))}
                for n in HS:
                    W = np.ones((E, S, n))                                             # W[k-1] = product of (1 - kap_j), j = k+1..n
                    for k in range(n - 2, -1, -1):
                        W[..., k] = W[..., k + 1] * (1 - q["kap"][..., k + 1])
                    P0 = W[..., 0] * (1 - q["kap"][..., 0]); kw = W * q["kap"][..., :n]
                    T = {"smoothing": P0 * (la0 - lb), "ageing": P0 * (q["la"] - la0), "return_mean": (1 - P0) * (mm - lb),
                         "return_season": (kw * (om[..., None] * (mk[..., :n] - mm[..., None]))).sum(-1), "flushing": (W * q["f"][..., :n]).sum(-1), "network": eps * W.sum(-1)}
                    tot = full[..., n - 1] - lb; lin = sum(T.values()); rem = tot - lin; c[f"clipped_share_{n}d"] = float((np.abs(rem) > 1e-6).mean())
                    c[f"max_abs_remainder_{n}d"] = float(np.abs(rem).max()); T["remainder"] = rem
                    sizes = {k: float(np.abs(v).mean()) for k, v in T.items()}; den = sum(sizes[k] for k in TERMS); var = float(tot.var())
                    proc = lin - T["network"]; pa, ne = float(np.abs(proc).mean()), sizes["network"]
                    for k, v in T.items():
                        rr.append({"record": rec, "net": net, "seed": s, "horizon": n, "term": k, "mean_abs": sizes[k], "mean": float(v.mean()), "share_abs": sizes[k] / den if k in TERMS else np.nan,
                                   "cov_share": float(np.cov(v.ravel(), tot.ravel())[0, 1] / var) if var > 0 else np.nan})
                    rr.append({"record": rec, "net": net, "seed": s, "horizon": n, "term": "process_all", "mean_abs": pa, "mean": float(proc.mean()), "share_abs": pa / (pa + ne),
                               "cov_share": float(np.cov(proc.ravel(), tot.ravel())[0, 1] / var) if var > 0 else np.nan})
                    rr.append({"record": rec, "net": net, "seed": s, "horizon": n, "term": "total_change", "mean_abs": float(np.abs(tot).mean()), "mean": float(tot.mean()), "share_abs": np.nan, "cov_share": 1.0})
                    for i, s_ in enumerate(D["stations"]):
                        st_rows.append({"record": rec, "net": net, "station": s_, "seed": s, "horizon": n, **{k: float(np.abs(v[:, i]).mean()) for k, v in T.items()},
                                        "total_change": float(np.abs(tot[:, i]).mean())})
                chk.append(c)
    R = pd.DataFrame(rr); R.to_csv(OUT / "terms_runs.csv", index=False); Ck = pd.DataFrame(chk); Ck.to_csv(OUT / "terms_check.csv", index=False)
    pd.DataFrame(lev).to_csv(OUT / "levels.csv", index=False)
    v = ["mean_abs", "mean", "share_abs", "cov_share"]
    by = R.groupby(["record", "net", "horizon", "term"])[v].agg(["mean", "std"]); by.columns = [f"{a}_{'seed_mean' if b == 'mean' else 'seed_sd'}" for a, b in by.columns]
    al = R.groupby(["record", "seed", "horizon", "term"])[v].mean().groupby(["record", "horizon", "term"]).agg(["mean", "std"])      # the four basins weighted equally
    al.columns = by.columns; al = al.reset_index().assign(net="all")
    Tm = pd.concat([by.reset_index(), al], ignore_index=True); Tm.to_csv(OUT / "terms.csv", index=False)
    St = pd.DataFrame(st_rows).groupby(["record", "net", "station", "horizon"]).mean(numeric_only=True).drop(columns="seed").reset_index(); St.to_csv(OUT / "terms_station.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 60)
    print(Ck.drop(columns=["seed"]).groupby(["record", "net"]).max().round(6).to_string())
    print("process share, stage-6 definition, mean and SD over runs:", Ck.groupby("record").process_share_7d_stage6_definition.agg(["mean", "std"]).round(4).to_dict())
    a = Tm[Tm.net == "all"]
    for col in ("mean_abs_seed_mean", "share_abs_seed_mean", "cov_share_seed_mean", "mean_seed_mean"):
        print("==", col); print(a.pivot_table(index=["record", "horizon"], columns="term", values=col).reindex(columns=TERMS + ["remainder", "process_all", "total_change"]).round(4).to_string())
    print("== share_abs by basin, 7 days"); print(Tm[(Tm.horizon == 7) & (Tm.net != "all")].pivot_table(index=["record", "net"], columns="term", values="share_abs_seed_mean").reindex(columns=TERMS + ["process_all"]).round(3).to_string())
    return Tm


def frame():
    import newinput_bench_more as BM
    return BM.shared()


def alert(A=None):
    A = frame() if A is None else A; ens = [c for c in A.columns if c.endswith("|ens")]; rows = []
    for (rec, h), q in A[A.horizon.isin([1, *HS])].groupby(["record", "horizon"]):
        for scope, qq in [("all", q)] + [(b, q[q.net == b]) for b in NI.NETS]:
            y = qq.y.to_numpy(float)
            for thr in THR:
                o = y >= thr
                for c in ens:
                    f = qq[c].to_numpy(float) >= thr; hit, miss, fa = int((o & f).sum()), int((o & ~f).sum()), int((~o & f).sum())
                    rows.append({"record": rec, "scope": scope, "horizon": int(h), "threshold": thr, "model": c.split("|")[0], "n": len(qq), "base_rate": float(o.mean()), "hits": hit, "misses": miss,
                                 "false_alarms": fa, "hit_rate": hit / (hit + miss) if hit + miss else np.nan, "false_alarm_ratio": fa / (hit + fa) if hit + fa else np.nan,
                                 "csi": hit / (hit + miss + fa) if hit + miss + fa else np.nan, "frequency_bias": (hit + fa) / (hit + miss) if hit + miss else np.nan})
    R = pd.DataFrame(rows); R.to_csv(OUT / "alert.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 60)
    a = R[R.scope == "all"]
    print(a[a.horizon == 7].groupby(["record", "threshold"]).agg(n=("n", "first"), base_rate=("base_rate", "first")).round(4).to_string())
    for thr in THR:
        for col in ("csi", "hit_rate", "false_alarm_ratio"):
            print(f"== {col}, threshold {thr:g}, 27 stations pooled (rows: model; columns: record, horizon)")
            print(a[a.threshold == thr].pivot_table(index="model", columns=["record", "horizon"], values=col).sort_values((36, 7), ascending=False).round(3).to_string())
    return R


def diagnosis(A=None):
    A = frame() if A is None else A; L = pd.read_csv(OUT / "levels.csv", dtype={"station": str}); rows = []
    for (rec, st), q in A.groupby(["record", "station"]):
        t = q[q.horizon == 1]; lvl = float(np.log(np.maximum(t.y.to_numpy(float), 0.1)).mean())
        base = {"record": rec, "net": q.net.iloc[0], "station": st, "test_mean_ln": lvl, "test_mean": float(t.y.mean())}
        for h in HS:
            g = q[q.horizon == h]; y = g.y.to_numpy(float)
            if len(g) < NI.MIN_N:
                continue
            for m in DIAG:
                p = g[f"{m}|ens"].to_numpy(float); e = p - y; mse = float((e ** 2).mean())
                rows.append({**base, "horizon": h, "model": m, "n": len(g), "mean_error": float(e.mean()), "sd_error": float(e.std()), "corr": float(np.corrcoef(p, y)[0, 1]),
                             "sd_ratio": float(p.std() / y.std()), "mse": mse, "bias_share_of_mse": float(e.mean() ** 2 / mse), "r2": NI.r2(y, p),
                             "r2_without_mean_error": NI.r2(y, p - e.mean())})
    R = pd.DataFrame(rows).merge(L, on=["record", "net", "station"], how="left"); R["level_shift_ln"] = R.test_mean_ln - R.record_mean_ln; R.to_csv(OUT / "diagnosis.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 60)
    print("== level shift (ln Chl-a, test years minus record mean), station median by basin")
    print(R[(R.horizon == 7) & (R.model == "hybrid")].groupby(["record", "net"]).level_shift_ln.agg(["median", "min", "max"]).round(3).to_string())
    for col in ("mean_error", "bias_share_of_mse", "corr", "r2", "r2_without_mean_error"):
        print(f"== {col}, 7 days, station median by basin"); print(R[R.horizon == 7].pivot_table(index=["record", "net"], columns="model", values=col, aggfunc="median")[DIAG].round(3).to_string())
    print("== all 27 stations, station median, mean of 3, 7 and 14 days")
    print(R.groupby(["record", "model", "horizon"])[["r2", "r2_without_mean_error", "corr", "bias_share_of_mse"]].median().groupby(["record", "model"]).mean().round(3).to_string())
    return R


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"; OUT.mkdir(parents=True, exist_ok=True)
    if cmd in ("all", "fit"):
        fit()
    if cmd in ("all", "terms"):
        terms()
    if cmd in ("all", "alert", "diagnosis"):
        A = frame()
        if cmd in ("all", "alert"):
            alert(A)
        if cmd in ("all", "diagnosis"):
            diagnosis(A)


if __name__ == "__main__":
    main()
