"""Case series for a figure of the manuscript (no training): the 7-day forecasts of the main model at two stations
of the test years, with the terms of each forecast change.

Stations, by a rule: among the 20 river stations and among
the 7 reservoir stations, the station with the median 7-day R2 of the hybrid model with 36-month records (seed-mean forecast,
`stage7/test/more_station.csv`); with an even number of stations the lower of the two middle ones. Reads the saved runs of
stage 6 (five seeds) and the arrays of `core2.prepare` under the settings of stage 6, as `newinput_explain.terms`. Writes
explain/case_series.csv: per station and anchor day the last measured value, the measured value and the seed-mean forecast 7
days later, and the six terms of the forecast change in ln Chl-a (seed mean), with their sum and the total change.
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Names: bb-C1 and the other bb-* models are the joint network of the paper (one xLSTM trained on the stations together,
without process terms; 'bb' for black box); core (P-mcf-all-kT) is the process model and hybrid (H-res-all) the hybrid
model.
Usage: newinput_case.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "xpinn"))
import newinput as NI                                                              # noqa: E402
import newinput_explain as EX                                                      # noqa: E402

REC, N = 36, 7


def chosen():
    E = pd.read_csv(NI.STAGE5 / "stage7" / "test" / "more_station.csv", dtype={"station": str, "seed": str})
    typ = pd.read_csv(ROOT / "data" / "national" / "stations.csv", dtype=str).set_index("code")["type"]
    g = E[(E.record == REC) & (E.model == "hybrid") & (E.seed == "ens") & (E.horizon == N)].copy(); g["type"] = g.station.map(typ); out = []
    for k in ("river", "lake"):
        q = g[g.type == k].sort_values("r2").reset_index(drop=True); r = q.iloc[(len(q) - 1) // 2]              # median station (lower middle if even)
        out.append({"type": k, "station": r.station, "net": r.net, "r2_7d": float(r.r2), "stations_of_this_type": len(q)})
    return pd.DataFrame(out)


def main():
    C = chosen(); print(C.to_string(index=False)); X2 = EX.core2(); rows = []
    for c in C.itertuples():
        D = X2.prepare(c.net, "test", REC); ev, S = D["ev"], len(D["stations"]); E = len(ev); i = D["stations"].index(c.station)
        lb = np.log(np.maximum(D["base"][ev].astype(float), 1e-3)); s7, age, rain, mm = (D[k][ev].astype(float) for k in ("smooth7", "age", "rain", "mmean"))
        m0, mk, Tw = D["mclim0_cf"][ev].astype(float), D["mclim_cf"][ev].astype(float), np.clip(D["drv_hold"][ev][..., 0].astype(float), 0, 40)
        dec = np.exp(-np.arange(1, 15) / X2.RAIN_EFOLD); anc = D["anc"][ev].astype(str); acc, fc, meta = {k: 0.0 for k in EX.TERMS + ["total"]}, 0.0, None
        for s in NI.SEEDS["test"]:
            d = NI.run_dir("hybrid", c.net, "test", REC, s); p, _ = EX.params(d)
            assert len(p) == E * S and (p.station.to_numpy().reshape(E, S)[0] == np.array(D["stations"])).all() and (p.anchor_date.to_numpy().reshape(E, S)[:, 0] == anc).all()
            ka, om, th, be, al, eps = (p[k].to_numpy(float).reshape(E, S) for k in EX.K5 + ["eps"])
            q = {"kap": np.minimum(ka[..., None] * th[..., None] ** (Tw - 20), X2.KAPPA_MAX), "tk": mm[..., None] + om[..., None] * (mk - mm[..., None]), "f": -be[..., None] * rain[..., None] * dec}
            t0 = mm + om * (m0 - mm); la0 = al * lb + (1 - al) * s7; q["la"] = t0 + (la0 - t0) * (1 - q["kap"][..., 0]) ** age
            full = EX.rollout(q, eps); W = np.ones((E, S, N))
            for k in range(N - 2, -1, -1):
                W[..., k] = W[..., k + 1] * (1 - q["kap"][..., k + 1])
            P0 = W[..., 0] * (1 - q["kap"][..., 0]); kw = W * q["kap"][..., :N]
            T = {"smoothing": P0 * (la0 - lb), "ageing": P0 * (q["la"] - la0), "return_mean": (1 - P0) * (mm - lb),
                 "return_season": (kw * (om[..., None] * (mk[..., :N] - mm[..., None]))).sum(-1), "flushing": (W * q["f"][..., :N]).sum(-1), "network": eps * W.sum(-1)}
            T["total"] = full[..., N - 1] - lb
            for k in acc:
                acc[k] = acc[k] + T[k][:, i] / len(NI.SEEDS["test"])
            pe = pd.read_csv(d / "predictions_eval.csv", dtype={"station": str}); pe = pe[pe.horizon == N]
            assert (pe.station.to_numpy().reshape(E, S)[0] == np.array(D["stations"])).all() and (pe.anchor_date.to_numpy().reshape(E, S)[:, 0] == anc).all()
            fc = fc + pe.y_pred_raw.to_numpy(float).reshape(E, S)[:, i] / len(NI.SEEDS["test"])
            if meta is None:
                g = pe[pe.station == c.station]; meta = g[["station", "anchor_date", "target_date", "observed", "y_true_raw", "base_raw"]].reset_index(drop=True)
                assert len(meta) == E and (meta.anchor_date.to_numpy() == anc).all()
        out = meta.assign(type=c.type, net=c.net, forecast=fc, age=age[:, i], record_mean_ln=mm[:, i], level_ln=(mm + 0.0)[:, i], **acc)
        out["terms_sum"] = out[EX.TERMS].sum(axis=1); rows.append(out)
        print(c.station, len(out), "largest |total - sum of terms|:", float((out.total - out.terms_sum).abs().max()), "largest rel diff forecast vs exp(ln base + total) of seed means:",
              float((np.abs(np.exp(np.log(np.maximum(out.base_raw, 1e-3)) + out.total) - out.forecast) / out.forecast).max()))
    R = pd.concat(rows, ignore_index=True); R.to_csv(EX.OUT / "case_series.csv", index=False); C.to_csv(EX.OUT / "case_stations.csv", index=False)
    print(R.groupby("station")[EX.TERMS + ["total"]].agg(lambda v: float(np.abs(v).mean())).round(4).to_string())


if __name__ == "__main__":
    main()
