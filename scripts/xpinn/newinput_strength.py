"""Materials on the strengths of the main model and of its constraints (no training).

Reads the station scores and the shared forecasts of stage 7 (`stage7/test/more_station.csv`, `newinput_bench_more.shared`)
and the saved model files and run information of stages 5b, 6 and 7. Nothing is trained and no file of stages 6 to 8 is
changed. Writes to results/input_design_20261114/strength/:
  spread.csv             per record and model with seeds: summary score (station-median R2, mean of 3, 7 and 14 days) of the
                         single runs (mean, SD, lowest, highest), of the seed-mean forecast, and their difference
  spread_check.json      largest difference between these single-run means and SDs and those of `summary.csv` of stage 7
  spread_forecast.csv    per record, model and horizon: SD over the runs of the forecast at each anchor (mg m-3 and ln(1 + C)):
                         median, mean and 90th percentile over the shared forecasts
  single_run_paired.csv  station R2 averaged over the single runs, the hybrid minus every other model with seeds, paired over
                         the stations at 3, 7 and 14 days and for their mean
  fitted.csv             number of values adjusted on the target record, per model
  by_type.csv, by_type_paired.csv, by_type_terms.csv, by_type_constants.csv
                         scores of every model
                         of stage 7, paired differences of the hybrid, term sizes and constants, by station type (river, lake)
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Names: bb-C1 and the other bb-* models are the joint network of the paper (one xLSTM trained on the stations together,
without process terms; 'bb' for black box); core (P-mcf-all-kT) is the process model and hybrid (H-res-all) the hybrid
model.
Usage: newinput_strength.py all | spread | fitted | by_type
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

OUT, SD, HS = NI.STAGE5 / "strength", NB.OUT / "test", NB.HS


def spread():
    OUT.mkdir(parents=True, exist_ok=True)
    E = pd.read_csv(SD / "more_station.csv", dtype={"station": str, "seed": str}); E = E[E.horizon.isin(HS)]
    seeded = sorted(set(E[E.seed != "ens"].model)); E = E[E.model.isin(seeded)]
    # 1. summary score of the single runs and of the seed-mean forecast
    rows = []
    for (rec, m), g in E.groupby(["record", "model"]):
        sc = g.groupby(["seed", "horizon"]).r2.median().unstack("horizon")[list(HS)].mean(axis=1); one = sc.drop("ens")
        rows.append({"record": rec, "model": m, "n_seeds": len(one), "single_mean": one.mean(), "single_sd": one.std(ddof=1), "single_min": one.min(),
                     "single_max": one.max(), "seed_mean_forecast": sc["ens"], "gain_from_averaging": sc["ens"] - one.mean()})
    S = pd.DataFrame(rows).sort_values(["record", "single_mean"], ascending=[False, False]); S.to_csv(OUT / "spread.csv", index=False)
    ref = pd.read_csv(SD / "summary.csv").merge(S, on=["record", "model"])
    chk = {"models_with_seeds": len(seeded), "rows": len(S), "rows_in_summary": len(ref),
           "max_abs_diff_single_mean": float((ref.seed_mean - ref.single_mean).abs().max()), "max_abs_diff_single_sd": float((ref.seed_sd - ref.single_sd).abs().max()),
           "max_abs_diff_seed_mean_forecast": float((ref.mean3_7_14 - ref.seed_mean_forecast).abs().max())}
    # 3. single runs paired over the stations: the hybrid minus every other model with seeds
    P = E[E.seed != "ens"].groupby(["record", "horizon", "station", "model"]).r2.mean().unstack("model"); pr = []
    for rec in NB.RECS:
        q = P.xs(rec, level="record"); allh = q.groupby("station").mean()
        for b in [m for m in seeded if m != "hybrid"]:
            for h in HS:
                x = q.xs(h, level="horizon"); pr.append(NI.paired(x["hybrid"] - x[b], record=rec, model="hybrid", against=b, horizon=h))
            pr.append(NI.paired(allh["hybrid"] - allh[b], record=rec, model="hybrid", against=b, horizon="mean3_7_14"))
    PR = pd.DataFrame(pr); PR.to_csv(OUT / "single_run_paired.csv", index=False)
    # 2. SD over the runs of the forecast at each anchor
    import newinput_bench_more as NM
    A = NM.shared(); A = A[A.horizon.isin(HS)]; fr = []
    for (rec, h), q in A.groupby(["record", "horizon"]):
        for m in seeded:
            x = q[[c for c in q.columns if c.startswith(m + "|") and not c.endswith("|ens")]].to_numpy(float)
            sd, sl = x.std(1, ddof=1), np.log1p(np.clip(x, 0, None)).std(1, ddof=1)
            fr.append({"record": rec, "horizon": int(h), "model": m, "n_runs": x.shape[1], "n_forecasts": len(x), "sd_mg_median": np.median(sd), "sd_mg_mean": sd.mean(),
                       "sd_mg_p90": np.quantile(sd, 0.9), "sd_log_median": np.median(sl), "sd_log_mean": sl.mean(), "sd_log_p90": np.quantile(sl, 0.9),
                       "mean_forecast_mg": x.mean()})
    F = pd.DataFrame(fr).sort_values(["record", "horizon", "sd_log_median"], ascending=[False, True, True]); F.to_csv(OUT / "spread_forecast.csv", index=False)
    chk["forecasts"] = {str(r): int(n) for r, n in A[A.horizon == 7].groupby("record").size().items()}
    (OUT / "spread_check.json").write_text(json.dumps(chk, indent=1))
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60)
    print(json.dumps(chk, indent=1)); print(S.round(4).to_string(index=False)); print(F[F.horizon == 7].round(4).to_string(index=False))
    print(PR[PR.horizon != 3].round(4).to_string(index=False))


def fitted():
    """Values adjusted on the target record, read from the saved model files and run information (36-month record, seed 42)."""
    import torch
    OUT.mkdir(parents=True, exist_ok=True); rec, seed, rows = 36, NI.SEEDS["test"][0], []
    for net in NI.NETS:
        nst = None
        for m in ("hybrid", "core", *NI.BB):
            d = NI.run_dir(m, net, "test", rec, seed); sd = torch.load(d / "model.pt", map_location="cpu"); cfg = json.load(open(d / "config.json"))
            n = {k: int(v.numel()) for k, v in sd.items() if hasattr(v, "numel")}; bb = sum(v for k, v in n.items() if k.startswith("bb."))
            con, gate = sum(v for k, v in n.items() if k in ("zs", "zk", "zv")), n.get("gate", 0); nst = len(cfg["stations"])
            if m in NI.BB:
                rows.append({"basin": net, "model": m, "unit": "basin", "stations": nst, "fitted_on_target": bb, "station_constants": 0, "gate": 0, "network_weights_carried_over": 0,
                             "note": "all weights of the joint network are trained on the basin's record"})
            else:
                rows.append({"basin": net, "model": m, "unit": "basin", "stations": nst, "fitted_on_target": con + gate, "station_constants": con, "gate": gate,
                             "network_weights_carried_over": bb, "note": "network trained on the other three basins and frozen" if bb else "no network"})
        codes = sorted(p.name.split("_")[1] for p in (NB.OUT / "n17" / "runs").glob(f"tgt_*_xLSTM_full_tune_r{rec}_s{seed}"))
        mine = [c for c in codes if c in set(json.load(open(NI.run_dir("core", net, "test", rec, seed) / "config.json"))["stations"])]
        for v in NB.VARIANTS:
            for bbname in NB.NETS:
                ps = [json.load(open(NB.OUT / v / "runs" / f"tgt_{c}_{bbname}_full_tune_r{rec}_s{seed}" / "model_info.json"))["params_total"] for c in mine]
                rows.append({"basin": net, "model": NB.name(f"{bbname} fine-tuned", v), "unit": "station", "stations": len(mine), "fitted_on_target": int(np.sum(ps)),
                             "station_constants": 0, "gate": 0, "network_weights_carried_over": 0, "note": f"one model per station, all {ps[0]} weights updated"})
                rows.append({"basin": net, "model": NB.name(f"{bbname} zero-shot", v), "unit": "station", "stations": len(mine), "fitted_on_target": 0, "station_constants": 0,
                             "gate": 0, "network_weights_carried_over": int(ps[0]), "note": "trained on the other three basins, nothing fitted at the target"})
    T = pd.DataFrame(rows); T["fitted_per_station"] = T.fitted_on_target / T.stations; T.to_csv(OUT / "fitted.csv", index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 80)
    print(T.to_string(index=False))


def by_type():
    """Scores, paired differences of the hybrid, term sizes and constants by station type (river or lake)."""
    OUT.mkdir(parents=True, exist_ok=True)
    typ = pd.read_csv(ROOT / "data" / "national" / "stations.csv", dtype=str).set_index("code")["type"]
    E = pd.read_csv(SD / "more_station.csv", dtype={"station": str, "seed": str}); E = E[(E.seed == "ens") & E.horizon.isin(HS)].copy()
    E["type"] = E.station.map(typ); assert E.type.notna().all()
    M = E.groupby(["record", "type", "model", "horizon"]).r2.median().unstack("horizon"); M["mean3_7_14"] = M[list(HS)].mean(axis=1)
    M["stations"] = E.groupby(["record", "type", "model"]).station.nunique()
    M = M.reset_index().sort_values(["record", "type", "mean3_7_14"], ascending=[False, True, False]); M.to_csv(OUT / "by_type.csv", index=False)
    P = E.pivot_table(index=["record", "type", "station", "horizon"], columns="model", values="r2"); pr = []
    for (rec, ty), q in P.groupby(level=["record", "type"]):
        q = q.droplevel(["record", "type"]); allh = q.groupby("station").mean()
        for b in [m for m in q.columns if m != "hybrid"]:
            for h in HS:
                x = q.xs(h, level="horizon"); pr.append(NI.paired(x["hybrid"] - x[b], record=rec, type=ty, model="hybrid", against=b, horizon=h))
            pr.append(NI.paired(allh["hybrid"] - allh[b], record=rec, type=ty, model="hybrid", against=b, horizon="mean3_7_14"))
    PR = pd.DataFrame(pr); PR.to_csv(OUT / "by_type_paired.csv", index=False)
    EX = NI.STAGE5 / "explain"; terms = ["smoothing", "ageing", "return_mean", "return_season", "flushing", "network"]
    T = pd.read_csv(EX / "terms_station.csv", dtype={"station": str}); T["type"] = T.station.map(typ)
    G = T.groupby(["record", "type", "horizon"])[terms].mean(); G = G.join(G.div(G.sum(axis=1), axis=0).add_prefix("share_"))
    G["stations"] = T.groupby(["record", "type", "horizon"]).station.nunique(); G.reset_index().to_csv(OUT / "by_type_terms.csv", index=False)
    K = pd.read_csv(EX / "constants_station.csv", dtype={"station": str}); K["type"] = K.station.map(typ)
    cols = ["kappa", "omega", "thk", "beta", "alpha", "halflife_20C", "halflife_5C", "halflife_25C", "factor_per_10C", "rain10mm_7d_pct"]
    Kc = K.groupby(["record", "type"])[cols].agg(["median", "min", "max"]); Kc.columns = [f"{a}_{b}" for a, b in Kc.columns]
    Kc["stations"] = K.groupby(["record", "type"]).station.nunique(); Kc.reset_index().to_csv(OUT / "by_type_constants.csv", index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 60)
    print(M.round(3).to_string(index=False)); print(PR[PR.horizon == "mean3_7_14"].round(4).to_string(index=False))
    print(G.round(3).to_string()); print(Kc.round(4).T.to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "spread"):
        spread()
    if what in ("all", "fitted"):
        fitted()
    if what == "by_type":
        by_type()
