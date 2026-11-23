"""Stage 7, secondary scores (no training): RMSE, the R2 of the 27 stations pooled, and the one-day scores, for
every model of stage 7. The reading rules stay on the station-level paired differences of `newinput_bench.py score`.

The shared forecasts are joined exactly as in `newinput_bench.py score` (same files, same order of joins); the station R2
computed here is compared with the saved `station_r2.csv` and the difference is written to `more_check.json`.

Writes to stage7/test/ (no existing file is changed):
  more_station.csv    station R2 and RMSE (mg m-3) per record, horizon, model and seed (`ens` = seed-mean forecast)
  more_pooled.csv     R2 and RMSE of the forecasts pooled, per record, horizon and model (seed-mean forecasts): the 27 stations
                      (`all`) and each basin
  more_summary.csv    per record and model: station-median R2 and RMSE and pooled R2 and RMSE at 1, 3, 7 and 14 days, and the
                      means over 3, 7 and 14 days
  more_paired_1d.csv  one day ahead: the core and the hybrid against every other model, paired over the stations (R2 and RMSE)
  more_check.json     forecasts used and the largest difference from the saved station R2
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Names: bb-C1 and the other bb-* models are the joint network of the paper (one xLSTM trained on the stations together,
without process terms; 'bb' for black box); core (P-mcf-all-kT) is the process model and hybrid (H-res-all) the hybrid
model.
Usage: newinput_bench_more.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "xpinn"))
import newinput as NI                                                              # noqa: E402
import newinput_bench as NB                                                        # noqa: E402

SD, KEY, HS, MIN_N = NB.OUT / "test", NB.KEY, (1, *NB.HS), NB.MIN_N


def rmse(y, p):
    return float(np.sqrt(((y - p) ** 2).mean()))


def shared():
    """The shared forecasts of every record and basin, joined as in newinput_bench.score, with the seed-mean forecasts."""
    variants, s6, cache, parts = list(NB.VARIANTS), [str(s) for s in NI.SEEDS["test"]], {}, []
    models = [(nm, NB.STATV, pat, None) for nm, pat in NB.STAT.items()] + [(NB.name(nm, v), v, pat, NB.SEEDS) for v in variants for nm, pat in NB.LEARNED.items()]
    seeded = {m: s6 for m in NB.STAGE6} | {nm: [str(s) for s in sds] for nm, _, _, sds in models if sds}
    six = {"mean of six": NB.SIX, "mean of six (32)": [NB.name(m, "all") if m in NB.LEARNED else m for m in NB.SIX]}

    def bench(v, tag):
        if (v, tag) not in cache:
            d = NB.OUT / v / "runs" / tag; cache[(v, tag)] = NB.read(d / "predictions_test.csv") if (d / "DONE").exists() else None
        return cache[(v, tag)]

    for rec in NB.RECS:
        for net in NI.NETS:
            W = None
            for m in NB.STAGE6:
                for s in s6:
                    x = pd.read_csv(NI.run_dir(m, net, "test", rec, s) / "predictions_eval.csv", dtype={"station": str}); x = x[x.observed & x.y_pred_raw.notna()].copy()
                    x["anchor_date"] = x.anchor_date.astype(str).str[:10]; x = x.set_index(KEY)
                    if W is None:
                        W = x[["y_true_raw", "base_raw"]].rename(columns={"y_true_raw": "y", "base_raw": "persistence|ens"})
                    W = W.join(x.y_pred_raw.rename(f"{m}|{s}"), how="inner")
            for code in sorted(W.index.get_level_values("station").unique()):
                g = W[W.index.get_level_values("station") == code]
                for nm, v, pat, sds in models:
                    fr = [bench(v, pat.format(c=code, r=rec, s=s)) for s in (sds or [0])]
                    assert all(f is not None for f in fr), (code, rec, nm)
                    b = fr[0][["y_true_raw"]].rename(columns={"y_true_raw": "yb"})
                    for c, f in zip([f"{nm}|{s}" for s in sds] if sds else [f"{nm}|ens"], fr):
                        b = b.join(f.y_pred_raw.rename(c), how="inner")
                    g = g.join(b, how="inner").drop(columns="yb")
                g = g.copy()
                for m, sds in seeded.items():
                    g[f"{m}|ens"] = g[[f"{m}|{s}" for s in sds]].mean(axis=1)
                for k, ms in six.items():
                    g[f"{k}|ens"] = g[[f"{m}|ens" for m in ms]].mean(axis=1)
                parts.append(g.reset_index().assign(net=net, record=rec))
    return pd.concat(parts, ignore_index=True)


def main():
    A = shared(); cols = [c for c in A.columns if "|" in c]; ens = [c for c in cols if c.endswith("|ens")]; rows, pooled = [], []
    for (rec, h), q in A.groupby(["record", "horizon"]):
        y = q.y.to_numpy(float)
        for scope, qq in [("all", q)] + [(b, q[q.net == b]) for b in NI.NETS]:
            yy = qq.y.to_numpy(float)
            for c in ens:
                p = qq[c].to_numpy(float); pooled.append({"record": rec, "scope": scope, "horizon": int(h), "model": c.split("|")[0], "r2": NI.r2(yy, p), "rmse": rmse(yy, p), "n": len(qq)})
        for st, g in q.groupby("station"):
            if len(g) < MIN_N:
                continue
            yy = g.y.to_numpy(float)
            for c in cols:
                m, s = c.split("|"); p = g[c].to_numpy(float)
                rows.append({"net": g.net.iloc[0], "station": st, "record": rec, "horizon": int(h), "model": m, "seed": s, "r2": NI.r2(yy, p), "rmse": rmse(yy, p), "n": len(g)})
    E, P = pd.DataFrame(rows), pd.DataFrame(pooled); E.to_csv(SD / "more_station.csv", index=False); P.to_csv(SD / "more_pooled.csv", index=False)
    # agreement with the saved station R2 of the driver's scorer
    S = pd.read_csv(SD / "station_r2.csv", dtype={"station": str, "seed": str}); k = ["station", "record", "horizon", "model", "seed"]
    M = S.merge(E.assign(seed=E.seed.astype(str)), on=k, how="outer", suffixes=("_saved", "_here"), indicator=True)
    chk = {"forecasts": {str(r): int(n) for r, n in A.groupby("record").size().items()}, "stations": int(A.station.nunique()), "models": len(ens),
           "rows_saved": len(S), "rows_here": len(E), "rows_in_both": int((M._merge == "both").sum()),
           "max_abs_r2_diff": float((M.r2_saved - M.r2_here).abs().max()), "max_abs_n_diff": float((M.n_saved - M.n_here).abs().max())}
    (SD / "more_check.json").write_text(json.dumps(chk, indent=1))
    # summary: seed-mean forecasts, station median and pooled, 1, 3, 7 and 14 days and the mean of 3, 7 and 14
    En = E[(E.seed == "ens") & E.horizon.isin(HS)]; Pa = P[(P.scope == "all") & P.horizon.isin(HS)]; out = []
    for name, src, col in (("median_r2", En, "r2"), ("median_rmse", En, "rmse"), ("pooled_r2", Pa, "r2"), ("pooled_rmse", Pa, "rmse")):
        t = (src.groupby(["record", "model", "horizon"])[col].median() if name.startswith("median") else src.set_index(["record", "model", "horizon"])[col]).unstack("horizon")
        t[f"{name}_mean3_7_14"] = t[list(NB.HS)].mean(axis=1); out.append(t.rename(columns={h: f"{name}_{h}" for h in HS}))
    summ = pd.concat(out, axis=1).reset_index().sort_values(["record", "median_r2_mean3_7_14"], ascending=[False, False]); summ.to_csv(SD / "more_summary.csv", index=False)
    # one day ahead: the core and the hybrid against every other model, paired over the stations
    D = En[En.horizon == 1].pivot_table(index=["record", "station"], columns="model", values=["r2", "rmse"]); pr = []
    for rec in NB.RECS:
        q = D.xs(rec, level="record")
        for a in NB.MINE:
            for b in [m for m in q["r2"].columns if m != a]:
                for col in ("r2", "rmse"):
                    pr.append(NI.paired(q[col][a] - q[col][b], record=rec, score=col, model=a, against=b, horizon=1))
    PR = pd.DataFrame(pr); PR.to_csv(SD / "more_paired_1d.csv", index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60)
    print(json.dumps(chk, indent=1)); print(summ.round(3).to_string(index=False))
    print(PR[PR.score == "r2"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
