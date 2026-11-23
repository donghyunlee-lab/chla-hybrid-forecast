"""Stage 6 (learned selection skipped, the 17-channel input fixed, the process core and the PINN candidate compared with the
black box again): the two final models of stage 4 on the input design
of stage 5 (27 stations with the whole period, 30 input days, station identifier, network inputs Chl-a, water temperature,
weather, TN and TP), against the black box of stage 5b on the same forecasts.

  core    P-mcf-all-kT: process model without a network (stage 3). The network inputs do not enter it.
  hybrid  H-res-all: the core plus a bounded daily forcing from a network trained on the other three basins and frozen at the
          target (stage 4), here with the 17-channel network input.
  bb-nut, bb-chl, bb-all: the joint black box bb-C1 with 17 channels (the chosen input), Chl-a only and all 32 channels
          (runs of stage 5b, not repeated).

Every run is its own process; the settings reach it
through the environment (XPINN_STATIONS, PINN_NATIONAL_DIR, XPINN_INPUTS, XPINN2_OUT).
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Names: bb-C1 and the other bb-* models are the joint network of the paper (one xLSTM trained on the stations together,
without process terms; 'bb' for black box); core (P-mcf-all-kT) is the process model and hybrid (H-res-all) the hybrid
model.
Usage: newinput.py all [--workers 16] | run --split dev|test | score --split dev|test | repro
"""
import argparse, datetime as dt, hashlib, json, os, platform, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE5 = ROOT / "results" / "input_design_20261114"
OUT = Path(os.environ["NEWINPUT_OUT"]) if os.environ.get("NEWINPUT_OUT") else STAGE5 / "stage6"
S5B = Path(os.environ["NEWINPUT_BB"]) if os.environ.get("NEWINPUT_BB") else STAGE5 / "stage5b"
PY, STATIONS, DATA = str(ROOT / ".venv" / "bin" / "python"), STAGE5 / "stations.txt", os.environ.get("NEWINPUT_DATA", "data/national_v2")
INPUTS = ["chlorophyll_a", "water_temperature", "era5_ssrd_mj", "era5_t2m", "era5_tp_mm", "era5_wind_max", "total_nitrogen", "total_phosphorus"]   # `nut` of stage 5b
CORE = "anc-smooth.age-yes.tgt-mcf.gro-none.flush-yes.par-station.tr-none.res-none.drv-hold.kin-fixed.kt-yes"                                      # P-mcf-all-kT
HYB = "anc-smooth.age-yes.tgt-mcf.gro-none.flush-yes.par-station.tr-none.res-bounded.drv-hold.kin-fixed.kt-yes.gate-yes.pre-nat"                 # H-res-all
BB = {"bb-nut": "nut-id", "bb-chl": "chl-id", "bb-all": "all-id"}
MINE, REF = ("core", "hybrid"), "bb-nut"
NETS = tuple(os.environ.get("NEWINPUT_NETS", "nakdong,han,geum,yeongsan").split(","))
SEEDS = {"dev": (42, 123, 456), "test": tuple(int(x) for x in os.environ.get("NEWINPUT_TEST_SEEDS", "42,123,456,1,2").split(","))}
RECS, SRC_REC, HS, MIN_N, MATCH = {"dev": (24, 12), "test": (36, 12)}, {"dev": 24, "test": 36}, (3, 7, 14), 60, -0.02
KEY, SWITCHES = ["station", "anchor_date", "horizon"], ("XPINN_SEQ_LEN", "XPINN_INPUTS", "XPINN_STATION_ID")
PAIRS = [("hybrid", "core"), ("core", "bb-nut"), ("hybrid", "bb-nut"), ("core", "bb-chl"), ("hybrid", "bb-chl"), ("core", "bb-all"), ("hybrid", "bb-all"),
         ("core", "persistence"), ("hybrid", "persistence"), ("bb-nut", "persistence")]


def log(event, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kw}, ensure_ascii=False) + "\n")


def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def env():
    e = {k: x for k, x in os.environ.items() if k not in SWITCHES}
    e.update({"XPINN2_OUT": str(OUT), "XPINN_STATIONS": str(STATIONS), "PINN_NATIONAL_DIR": DATA, "XPINN_INPUTS": ",".join(INPUTS)})
    return e


def tag(spec, net, split, rec, seed):
    return f"{spec}__{net}_{split}_r{rec}_s{seed}"


def pre_tag(leave, split, seed):
    return f"{HYB}__lo-{leave}_{split}_r{SRC_REC[split]}_s{seed}"


def run_dir(model, net, split, rec, seed):
    if model in BB:
        return S5B / BB[model] / "runs" / tag("bb-C1", net, split, rec, seed)
    return OUT / "runs" / tag(CORE if model == "core" else HYB, net, split, rec, seed)


def pool(cmds, workers):
    """cmds: (DONE path, script, arguments). Unfinished ones run `workers` at a time, with one retry."""
    def one(c):
        lf = OUT / "logs" / f"{c[0].parent.name}.log"; lf.parent.mkdir(parents=True, exist_ok=True)
        with open(lf, "a") as f:
            return subprocess.run([PY, str(ROOT / "scripts" / "xpinn" / c[1])] + c[2], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env()).returncode
    todo = [c for c in cmds if not c[0].exists()]
    for attempt in (1, 2):
        if not todo:
            break
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
        todo = [c for c in todo if not c[0].exists()]
        if todo:
            log("failed", attempt=attempt, runs=[c[0].parent.name for c in todo])
    return [c[0].parent.name for c in todo]


def core_cmd(net, split, rec, seed):
    return (run_dir("core", net, split, rec, seed) / "DONE", "core2.py", ["--spec", CORE, "--net", net, "--split", split, "--record", str(rec), "--seed", str(seed)])


def pre_cmd(leave, split, seed, suffix=""):
    return (OUT / "pretrain" / (pre_tag(leave, split, seed) + suffix) / "DONE", "hybrid.py",
            ["pretrain", "--spec", HYB, "--leave", leave, "--split", split, "--seed", str(seed)] + (["--suffix", suffix] if suffix else []))


def fit_cmd(net, split, rec, seed, suffix=""):
    return (OUT / "runs" / (tag(HYB, net, split, rec, seed) + suffix) / "DONE", "hybrid.py",
            ["fit", "--spec", HYB, "--net", net, "--split", split, "--record", str(rec), "--seed", str(seed)] + (["--suffix", suffix] if suffix else []))


def run(split, workers):
    """The core runs and the pre-training on the other basins together, then the target runs of the hybrid."""
    sd = SEEDS[split]
    failed = pool([pre_cmd(n, split, s) for n in NETS for s in sd] + [core_cmd(n, split, r, s) for n in NETS for r in RECS[split] for s in sd], workers)
    return failed + pool([fit_cmd(n, split, r, s) for n in NETS for r in RECS[split] for s in sd], workers)


def r2(y, p):
    sst = float(((y - y.mean()) ** 2).sum())
    return float(1 - ((y - p) ** 2).sum() / sst) if sst > 0 else np.nan


def paired(dd, **kw):
    """Paired differences over stations: median, mean, share of stations better, Wilcoxon signed-rank p (unadjusted)."""
    from scipy import stats
    dd = dd.dropna()
    return {**kw, "n_stations": len(dd), "median_diff": dd.median(), "mean_diff": dd.mean(), "share_better": float((dd > 0).mean()),
            "wilcoxon_p": float(stats.wilcoxon(dd).pvalue) if (dd != 0).any() else 1.0}


def score(split):
    """Station R2 (original units, measured days) on the forecasts shared by every model and seed; summaries, paired
    differences, the readings by the rules of stage 4, and what the hybrid's network and the constants do."""
    sd = OUT / split; sd.mkdir(parents=True, exist_ok=True); rows, counts, seeds = [], [], [str(s) for s in SEEDS[split]]; models = [*MINE, *BB]
    for rec in RECS[split]:
        for net in NETS:
            W = None
            for m in models:
                for s in seeds:
                    d = run_dir(m, net, split, rec, s)
                    if not (d / "DONE").exists():
                        continue
                    x = pd.read_csv(d / "predictions_eval.csv", dtype={"station": str}); x = x[x.observed & x.y_pred_raw.notna()]
                    counts.append({"record": rec, "net": net, "model": m, "seed": s, "forecasts": len(x)}); x = x.set_index(KEY)
                    if W is None:
                        W = x[["y_true_raw", "base_raw"]].rename(columns={"y_true_raw": "y", "base_raw": "persistence"})
                    W = W.join(x.y_pred_raw.rename(f"{m}|{s}"), how="inner")
            if W is None:
                continue
            for m in models:
                cols = [f"{m}|{s}" for s in seeds if f"{m}|{s}" in W]
                if cols:
                    W[f"{m}|ens"] = W[cols].mean(axis=1)
            counts.append({"record": rec, "net": net, "model": "shared", "seed": "", "forecasts": len(W)})
            for (st, h), g in W.reset_index().groupby(["station", "horizon"]):
                if len(g) < MIN_N:
                    continue
                y = g.y.to_numpy(float)
                for col in [c for c in g.columns if "|" in c or c == "persistence"]:
                    m, s = col.split("|") if col != "persistence" else ("persistence", "")
                    rows.append({"net": net, "station": st, "record": rec, "horizon": int(h), "model": m, "seed": s, "r2": r2(y, g[col].to_numpy(float)), "n": len(g)})
    E = pd.DataFrame(rows); E.to_csv(sd / "station_r2.csv", index=False); pd.DataFrame(counts).to_csv(sd / "forecast_counts.csv", index=False)
    E = E[E.horizon.isin([1, *HS])]
    med = E.groupby(["record", "model", "seed", "horizon"]).r2.median().unstack("horizon"); med["mean3_7_14"] = med[list(HS)].mean(axis=1)
    sl = med.index.get_level_values("seed"); seedrows = med[sl.isin(seeds)]
    summ = med[sl.isin(["ens", ""])].reset_index().drop(columns="seed")
    sm = seedrows.groupby(["record", "model"]).agg(["mean", "std"]); sm.columns = [f"{a}_seed_{b}" for a, b in sm.columns]
    summ.merge(sm.reset_index(), on=["record", "model"], how="left").to_csv(sd / "summary.csv", index=False)
    # by basin: the basin's station median (mean of 3, 7 and 14 days), seed-mean forecast and mean and SD over seeds
    bs = E[E.horizon.isin(HS)].groupby(["record", "net", "model", "seed", "horizon"]).r2.median().unstack("horizon").mean(axis=1).rename("score").reset_index()
    b1 = bs[bs.seed.isin(["ens", ""])].set_index(["record", "net", "model"]).score.rename("seed_mean_forecast")
    b2 = bs[bs.seed.isin(seeds)].groupby(["record", "net", "model"]).score.agg(["mean", "std"]).rename(columns={"mean": "seed_mean", "std": "seed_sd"})
    pd.concat([b1, b2], axis=1).reset_index().to_csv(sd / "basin_summary.csv", index=False)
    # primary score: mean over the records of the seed mean of mean3_7_14 (persistence: one value)
    ps = seedrows.reset_index().groupby(["model", "seed"]).mean3_7_14.mean().unstack("seed")
    prim = pd.DataFrame({"primary": ps.mean(axis=1), "seed_sd": ps.std(axis=1), **{f"seed_{c}": ps[c] for c in ps.columns}}).reset_index()
    pers = med[sl == ""].reset_index(); prim = pd.concat([prim, pd.DataFrame([{"model": "persistence", "primary": pers.mean3_7_14.mean()}])], ignore_index=True)
    for rec in RECS[split]:
        v = {**seedrows.xs(rec, level="record").groupby("model").mean3_7_14.mean().to_dict(), "persistence": float(pers[pers.record == rec].mean3_7_14.iloc[0])}
        prim[f"record_{rec}"] = prim.model.map(v)
    prim = prim.sort_values("primary", ascending=False); prim.to_csv(sd / "primary.csv", index=False)
    # paired over stations (seed-mean forecasts) and over seeds (station-median score); persistence has one forecast
    P = E[E.seed.isin(["ens", ""])].pivot_table(index=["record", "station", "horizon"], columns="model", values="r2"); P = P[P.index.get_level_values("horizon").isin(HS)]
    sr = seedrows.mean3_7_14.unstack("model"); pr = []
    for rec in RECS[split]:
        q = P.xs(rec, level="record"); t = sr.xs(rec, level="record")
        for a, b in [p for p in PAIRS if p[0] in q and p[1] in q]:
            d = q[a] - q[b]; ds = t[a] - (t[b] if b in t else float(pers[pers.record == rec].mean3_7_14.iloc[0]))
            extra = {"seed_mean_diff": ds.mean(), "seed_sd": ds.std(), "seeds_better": int((ds > 0).sum()), "n_seeds": len(ds)}
            pr += [paired(d.xs(h, level="horizon"), record=rec, model=a, against=b, horizon=h) for h in HS]
            pr.append({**paired(d.groupby("station").mean(), record=rec, model=a, against=b, horizon="mean3_7_14"), **extra})
    PR = pd.DataFrame(pr); PR.to_csv(sd / "paired.csv", index=False)
    # readings by the reading rules of stage 4, by record length
    verdict = {}
    for rec in RECS[split]:
        v = {}
        for a, b in [("hybrid", "core"), ("core", REF), ("hybrid", REF)]:
            q = PR[(PR.record == rec) & (PR.model == a) & (PR.against == b) & PR.horizon.isin(HS)]
            if len(q) == len(HS):
                v[f"{a} improves on {b}"] = bool(((q.median_diff > 0) & (q.wilcoxon_p < 0.05)).sum() >= 2)
                if b == REF:
                    v[f"{a} matches {b}"] = bool((q.median_diff >= MATCH).all())
        verdict[str(rec)] = v
    (sd / "verdicts.json").write_text(json.dumps(verdict, indent=1))
    # what the network of the hybrid does (share of the 7-day change from the process terms, gate, size of the forcing) and the constants
    ex, cs = [], []
    for rec in RECS[split]:
        for net in NETS:
            for m in MINE:
                for s in seeds:
                    d = run_dir(m, net, split, rec, s)
                    if not (d / "params_eval.csv").exists():
                        continue
                    p = pd.read_csv(d / "params_eval.csv", dtype={"station": str}); c = json.loads((d / "config.json").read_text())
                    a, e = p.d7_mech.abs().mean(), (p.d7_full - p.d7_mech).abs().mean(); g = c.get("gate")
                    ex.append({"record": rec, "net": net, "model": m, "seed": s, "process_share_7d": a / (a + e) if a + e > 0 else np.nan,
                               "gate": g[0] if g else np.nan, "mean_abs_eps": p.eps.abs().mean() if "eps" in p else 0.0, "lr": c["lr_selected"], "best_epoch": c["best_epoch"], "best_val": c["best_val"]})
                    k = [x for x in ("kappa", "omega", "thk", "beta", "alpha") if x in p]
                    cs.append(p.groupby("station")[k].median().reset_index().assign(record=rec, net=net, model=m, seed=s))
    X = pd.DataFrame(ex); X.to_csv(sd / "explain_runs.csv", index=False)
    if len(X):
        X.groupby(["record", "model"])[["process_share_7d", "gate", "mean_abs_eps"]].agg(["mean", "std"]).to_csv(sd / "explain.csv")
        Cn = pd.concat(cs, ignore_index=True); k = [x for x in ("kappa", "omega", "thk", "beta", "alpha") if x in Cn]
        Cn.groupby(["record", "net", "model", "station"])[k].mean().reset_index().to_csv(sd / "constants_station.csv", index=False)       # mean over seeds
        Cn.groupby(["record", "net", "model", "seed"])[k].median().groupby(["record", "net", "model"]).agg(["mean", "std"]).to_csv(sd / "constants.csv")
    log("scored", split=split, primary=prim.round(4).to_dict("records"), verdicts=verdict)
    return prim, verdict


def repro(workers):
    """One pre-training and one target run of the hybrid repeated under another name and compared with the first."""
    failed = pool([pre_cmd(NETS[0], "dev", 42, "__repro")], workers) + pool([fit_cmd(NETS[0], "dev", 24, 42, "__repro")], workers)
    a, b = OUT / "pretrain" / pre_tag(NETS[0], "dev", 42), OUT / "pretrain" / (pre_tag(NETS[0], "dev", 42) + "__repro")
    c, d = run_dir("hybrid", NETS[0], "dev", 24, 42), OUT / "runs" / (tag(HYB, NETS[0], "dev", 24, 42) + "__repro")
    res = {"failed": failed}
    if not failed:
        ja, jb = json.loads((a / "config.json").read_text()), json.loads((b / "config.json").read_text())
        x, y = (pd.read_csv(f / "predictions_eval.csv", dtype={"station": str}).set_index(KEY).y_pred_raw for f in (c, d))
        res.update({"pretrain_same_state": ja["state_sha256"] == jb["state_sha256"], "pretrain_val": [ja["best_val"], jb["best_val"]],
                    "forecasts": len(x), "max_abs_diff": float(np.nanmax(np.abs(x.to_numpy(float) - y.reindex(x.index).to_numpy(float)))),
                    "same_file": (c / "predictions_eval.csv").read_bytes() == (d / "predictions_eval.csv").read_bytes()})
    (OUT / "repro_check.json").write_text(json.dumps(res, indent=1)); log("repro", **res)
    return res


def manifest(workers):
    files = [f"scripts/xpinn/{x}" for x in ("newinput.py", "hybrid.py", "core.py", "core2.py", "explore.py", "final2.py", "seasonal.py", "score.py")] + \
            ["scripts/clean_revision/common.py", "scripts/clean_revision/models.py", "scripts/national/nat.py", f"{DATA}/daily_observed.csv", f"{DATA}/era5_daily.csv", f"{DATA}/stations.csv"]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    m = {"created": dt.datetime.now().isoformat(timespec="seconds"), "data": DATA, "host": platform.node(), "python": sys.version.split()[0], "gpu": gpu, "git_head": head,
         "settings": {"core": CORE, "hybrid": HYB, "network_inputs": INPUTS, "black_box_rows": {k: str(S5B / v) for k, v in BB.items()}, "input_days": 30, "basins": NETS,
                      "seeds": SEEDS, "records": RECS, "source_record_months": SRC_REC, "workers": workers, "match_margin": MATCH, "min_forecasts_per_station_horizon": MIN_N,
                      "learning_rates": [1e-3, 3e-4, 1e-4], "epochs": int(os.environ.get("XPINN_EPOCHS", 100))},
         "stations": STATIONS.read_text().split(), "sha256": {f: sha(ROOT / f) for f in files if (ROOT / f).exists()}, "stations_sha256": sha(STATIONS)}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "MANIFEST.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
    cd = OUT / "code"; cd.mkdir(exist_ok=True)
    for f in files[:11]:
        (cd / Path(f).name).write_bytes((ROOT / f).read_bytes())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["all", "run", "score", "repro"]); ap.add_argument("--split", default="dev", choices=list(RECS))
    ap.add_argument("--workers", type=int, default=16); a = ap.parse_args()
    assert STATIONS.exists(), f"station list missing: {STATIONS}"
    if a.cmd == "all":
        manifest(a.workers); log("start", core=CORE, hybrid=HYB, inputs=INPUTS, basins=NETS, seeds=SEEDS, records=RECS, stations=len(STATIONS.read_text().split()), data=DATA)
        for split in ("dev", "test"):
            failed = run(split, a.workers); log("runs done", split=split, failed=failed)
            score(split); (OUT / f"DONE_{split}").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        repro(a.workers); (OUT / "DONE_all").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    elif a.cmd == "run":
        failed = run(a.split, a.workers); log("runs done", split=a.split, failed=failed)
    elif a.cmd == "score":
        prim, verdict = score(a.split); print(prim.round(4).to_string(index=False)); print(json.dumps(verdict, indent=1))
    else:
        print(repro(a.workers))


if __name__ == "__main__":
    main()
