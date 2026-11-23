"""Stage 5b: which variables enter the joint black-box xLSTM (bb-C1), and whether the network is told which station a window
comes from (station coordinates, or too many variables at once, may lower the accuracy). Four variable sets x station identifier
on or off; everything else as in stage 5a (30 input days, the 27 stations of results/input_design_20261114/stations.txt, the
second build of the daily file).
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Names: bb-C1 and the other bb-* models are the joint network of the paper (one xLSTM trained on the stations together,
without process terms; 'bb' for black box); core (P-mcf-all-kT) is the process model and hybrid (H-res-all) the hybrid
model.
Usage: inputvars.py all [--workers 16]   (development year: runs, scores, choice)
       inputvars.py run|score --split dev|test [--configs all-id nut-id] [--seeds 42 123 456 1 2]"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, platform, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE5 = ROOT / "results" / "input_design_20261114"
OUT = Path(os.environ["INPUTVARS_OUT"]) if os.environ.get("INPUTVARS_OUT") else STAGE5 / "stage5b"
PY, STATIONS, DATA = str(ROOT / ".venv" / "bin" / "python"), STAGE5 / "stations.txt", os.environ.get("INPUTVARS_DATA", "data/national_v2")
CHL, MET, NUT = ["chlorophyll_a"], ["water_temperature", "era5_ssrd_mj", "era5_t2m", "era5_tp_mm", "era5_wind_max"], ["total_nitrogen", "total_phosphorus"]
SETS = {"chl": CHL, "met": CHL + MET, "nut": CHL + MET + NUT, "all": None}        # None: the 32 channels of the paper, no switch set
IDS = {"id": None, "noid": "0"}                                                   # None: station embedding as in the paper, no switch set
CONFIGS, BASE = tuple(f"{v}-{i}" for v in SETS for i in IDS), "all-id"
NETS = tuple(os.environ.get("INPUTVARS_NETS", "nakdong,han,geum,yeongsan").split(","))
SEEDS = tuple(int(x) for x in os.environ.get("INPUTVARS_SEEDS", "42,123,456").split(","))
RECS, SPEC, HS, MARGIN, MIN_N = {"dev": (24, 12), "test": (36, 12)}, "bb-C1", (3, 7, 14), 0.005, 60
KEY, SWITCHES = ["station", "anchor_date", "horizon"], ("XPINN_SEQ_LEN", "XPINN_INPUTS", "XPINN_STATION_ID")


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


def run_dir(cfg, net, split, rec, seed):
    return OUT / cfg / "runs" / f"{SPEC}__{net}_{split}_r{rec}_s{seed}"


def env_of(cfg):
    """Environment of one run. The current input (all-id) sets no switch, so it takes the code path of every earlier run."""
    v, i = cfg.split("-"); e = {k: x for k, x in os.environ.items() if k not in SWITCHES}
    e.update({"XPINN2_OUT": str(OUT / cfg), "XPINN_STATIONS": str(STATIONS), "PINN_NATIONAL_DIR": DATA})
    if SETS[v] is not None:
        e["XPINN_INPUTS"] = ",".join(SETS[v])
    if IDS[i] is not None:
        e["XPINN_STATION_ID"] = IDS[i]
    return e


def run(split, configs, seeds, workers):
    """Every unfinished run, `workers` at a time, with one retry. Each run is its own process, so the switches are set by the environment."""
    jobs = [(c, n, r, s) for c in configs for n in NETS for r in RECS[split] for s in seeds]
    def one(j):
        c, n, r, s = j; lf = OUT / "logs" / f"{c}_{run_dir(c, n, split, r, s).name}.log"; lf.parent.mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(ROOT / "scripts" / "xpinn" / "core2.py"), "--spec", SPEC, "--net", n, "--split", split, "--record", str(r), "--seed", str(s)]
        with open(lf, "a") as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env_of(c)).returncode
    done = lambda j: (run_dir(j[0], j[1], split, j[2], j[3]) / "DONE").exists()
    todo = [j for j in jobs if not done(j)]
    for attempt in (1, 2):
        if not todo:
            break
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
        todo = [j for j in todo if not done(j)]
        if todo:
            log("failed", attempt=attempt, split=split, runs=[f"{j[0]} {j[1]} r{j[2]} s{j[3]}" for j in todo])
    return [f"{j[0]} {j[1]} r{j[2]} s{j[3]}" for j in todo]


def r2(y, p):
    sst = float(((y - y.mean()) ** 2).sum())
    return float(1 - ((y - p) ** 2).sum() / sst) if sst > 0 else np.nan


def paired(dd, **kw):
    """Paired differences over stations: median, mean, share of stations better, Wilcoxon signed-rank p (unadjusted)."""
    from scipy import stats
    dd = dd.dropna()
    return {**kw, "n_stations": len(dd), "median_diff": dd.median(), "mean_diff": dd.mean(), "share_better": float((dd > 0).mean()),
            "wilcoxon_p": float(stats.wilcoxon(dd).pvalue) if (dd != 0).any() else 1.0}


def contrasts(t):
    """The two questions as contrasts of a table with one column per configuration: station identifier off - on (mean of the four
    variable sets) and each smaller variable set - all variables (mean of the two identifier levels)."""
    c = {"station identifier: off - on": t[[f"{v}-noid" for v in SETS]].mean(axis=1) - t[[f"{v}-id" for v in SETS]].mean(axis=1)}
    c.update({f"variables: {v} - all": t[[f"{v}-{i}" for i in IDS]].mean(axis=1) - t[[f"all-{i}" for i in IDS]].mean(axis=1) for v in SETS if v != "all"})
    return c


def score(split, configs, seeds):
    """Station R2 (original units, measured days) on the forecasts shared by every configuration and seed; summaries, the two
    main effects and the choice."""
    sd = OUT / split; sd.mkdir(parents=True, exist_ok=True); rows, counts = [], []
    for rec in RECS[split]:
        for net in NETS:
            W = None
            for c in configs:
                for s in seeds:
                    d = run_dir(c, net, split, rec, s)
                    if not (d / "DONE").exists():
                        continue
                    x = pd.read_csv(d / "predictions_eval.csv", dtype={"station": str}); x = x[x.observed & x.y_pred_raw.notna()]
                    counts.append({"record": rec, "net": net, "config": c, "seed": s, "forecasts": len(x)})
                    x = x.set_index(KEY)
                    if W is None:
                        W = x[["y_true_raw", "base_raw"]].rename(columns={"y_true_raw": "y", "base_raw": "persistence"})
                    W = W.join(x.y_pred_raw.rename(f"{c}|{s}"), how="inner")
            if W is None:
                continue
            for c in configs:
                cols = [f"{c}|{s}" for s in seeds if f"{c}|{s}" in W]
                if cols:
                    W[f"{c}|ens"] = W[cols].mean(axis=1)
            counts.append({"record": rec, "net": net, "config": "shared", "seed": "", "forecasts": len(W)})
            for (st, h), g in W.reset_index().groupby(["station", "horizon"]):
                if len(g) < MIN_N:
                    continue
                y = g.y.to_numpy(float)
                for col in [c for c in g.columns if "|" in c or c == "persistence"]:
                    cfg, seed = col.split("|") if col != "persistence" else ("persistence", "")
                    rows.append({"net": net, "station": st, "record": rec, "horizon": int(h), "config": cfg, "seed": seed, "r2": r2(y, g[col].to_numpy(float)), "n": len(g)})
    E = pd.DataFrame(rows); E.to_csv(sd / "station_r2.csv", index=False); pd.DataFrame(counts).to_csv(sd / "forecast_counts.csv", index=False)
    E = E[E.horizon.isin([1, *HS])]
    # station medians: seed-mean forecast, each seed, persistence
    med = E.groupby(["record", "config", "seed", "horizon"]).r2.median().unstack("horizon"); med["mean3_7_14"] = med[list(HS)].mean(axis=1)
    sl = med.index.get_level_values("seed"); seedrows = med[sl.isin([str(s) for s in seeds])]
    summ = med[sl.isin(["ens", ""])].reset_index().drop(columns="seed")
    sm = seedrows.groupby(["record", "config"]).agg(["mean", "std"]); sm.columns = [f"{a}_seed_{b}" for a, b in sm.columns]
    summ.merge(sm.reset_index(), on=["record", "config"], how="left").to_csv(sd / "summary.csv", index=False)
    bas = E[E.seed.isin(["ens", ""])].groupby(["record", "net", "config", "horizon"]).r2.median().unstack("horizon"); bas["mean3_7_14"] = bas[list(HS)].mean(axis=1)
    bas.reset_index().to_csv(sd / "basin_summary.csv", index=False)
    # paired with the current input (seed-mean forecasts; each horizon and the station mean of 3, 7 and 14 days)
    P = E[E.seed == "ens"].pivot_table(index=["record", "station", "horizon"], columns="config", values="r2"); P = P[P.index.get_level_values("horizon").isin(HS)]
    pr = []
    for rec in RECS[split]:
        q = P.xs(rec, level="record")
        for c in [x for x in configs if x != BASE and x in q and BASE in q]:
            d = q[c] - q[BASE]
            pr += [paired(d.xs(h, level="horizon"), record=rec, config=c, against=BASE, horizon=h) for h in HS]
            pr.append(paired(d.groupby("station").mean(), record=rec, config=c, against=BASE, horizon="mean3_7_14"))
    pd.DataFrame(pr).to_csv(sd / "paired_vs_base.csv", index=False)
    # main effects, over stations (seed-mean forecasts) and over seeds (station-median score)
    if all(c in configs for c in CONFIGS):
        me, sr = [], seedrows.mean3_7_14.unstack("config")
        for rec in [*RECS[split], "both"]:
            st = (P.xs(rec, level="record") if rec != "both" else P.groupby(["station", "horizon"]).mean()).groupby("station").mean()
            t = sr.xs(rec, level="record") if rec != "both" else sr.groupby("seed").mean()
            for (name, dd), ds in zip(contrasts(st).items(), contrasts(t).values()):
                me.append({**paired(dd, contrast=name, record=rec), "seed_mean_diff": ds.mean(), "seed_sd": ds.std(), "seeds_better": int((ds > 0).sum()), "n_seeds": len(ds)})
        pd.DataFrame(me).to_csv(sd / "main_effects.csv", index=False)
    # primary score: mean over the records of the seed mean of mean3_7_14, with the SD over seeds of the record mean
    ps = seedrows.reset_index().groupby(["config", "seed"]).mean3_7_14.mean().unstack("seed")
    prim = pd.DataFrame({"primary": ps.mean(axis=1), "seed_sd": ps.std(axis=1), **{f"seed_{c}": ps[c] for c in ps.columns}}).reset_index()
    prim = prim.sort_values("primary", ascending=False); prim.to_csv(sd / "primary.csv", index=False)
    choice = None
    if split == "dev" and BASE in prim.config.tolist():
        b = float(prim.loc[prim.config == BASE, "primary"].iloc[0]); pp = pd.DataFrame(pr); passed = []
        for _, r in prim[prim.config != BASE].iterrows():                     # highest primary score first
            q = pp[(pp.config == r.config) & (pp.horizon == "mean3_7_14")]
            if r.primary - b >= MARGIN and len(q) == len(RECS[split]) and (q.share_better > 0.5).all():
                passed.append(r.config)
        top = prim.iloc[0]
        choice = {"best_by_primary": top.config, "primary_best": float(top.primary), "primary_current": b, "gain_over_current": float(top.primary - b),
                  "rule_margin": MARGIN, "passed_rule": passed, "chosen": passed[0] if passed else BASE, "replaces_current": bool(passed)}
        (OUT / "chosen.json").write_text(json.dumps(choice, indent=1))
    rp = repro(split, seeds)
    log("scored", split=split, primary=prim.round(4).to_dict("records"), choice=choice, repro_vs_5a=rp)
    return prim, choice, rp


def repro(split, seeds):
    """The current input sets no switch, so its runs repeat the 30-day runs of stage 5a (same code path, seeds and data for the
    scored years). Largest difference between the two sets of forecasts, run by run."""
    rows = []
    for rec in RECS[split]:
        for net in NETS:
            for s in seeds:
                a, b = run_dir(BASE, net, split, rec, s) / "predictions_eval.csv", STAGE5 / "L30" / "runs" / f"{SPEC}__{net}_{split}_r{rec}_s{s}" / "predictions_eval.csv"
                if a.exists() and b.exists():
                    x, y = pd.read_csv(a, dtype={"station": str}).set_index(KEY).y_pred_raw, pd.read_csv(b, dtype={"station": str}).set_index(KEY).y_pred_raw
                    j = x.index.intersection(y.index)
                    rows.append({"record": rec, "net": net, "seed": s, "forecasts": len(x), "forecasts_5a": len(y), "shared": len(j),
                                 "max_abs_diff": float(np.nanmax(np.abs(x.reindex(j).to_numpy(float) - y.reindex(j).to_numpy(float)))) if len(j) else np.nan,
                                 "same_file": a.read_bytes() == b.read_bytes()})
    if not rows:
        return None
    R = pd.DataFrame(rows); R.to_csv(OUT / split / "repro_vs_5a.csv", index=False)
    return {"runs": len(R), "same_file": int(R.same_file.sum()), "max_abs_diff": float(R.max_abs_diff.max())}


def manifest(workers, name=""):
    files = ["scripts/xpinn/inputvars.py", "scripts/xpinn/core.py", "scripts/xpinn/core2.py", "scripts/clean_revision/common.py", "scripts/clean_revision/models.py",
             "scripts/national/nat.py", f"{DATA}/daily_observed.csv", f"{DATA}/era5_daily.csv", f"{DATA}/stations.csv"]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    m = {"created": dt.datetime.now().isoformat(timespec="seconds"), "data": DATA, "host": platform.node(), "python": sys.version.split()[0], "gpu": gpu, "git_head": head,
         "settings": {"spec": SPEC, "variable_sets": {k: (v or "all 32 channels") for k, v in SETS.items()}, "station_identifier": {"id": "on", "noid": "off"},
                      "configurations": CONFIGS, "current": BASE, "input_days": 30, "basins": NETS, "seeds": SEEDS, "records": RECS, "workers": workers, "margin": MARGIN,
                      "min_forecasts_per_station_horizon": MIN_N, "learning_rates": [1e-3, 3e-4, 1e-4], "epochs": int(os.environ.get("XPINN_EPOCHS", 100))},
         "stations": STATIONS.read_text().split(), "sha256": {f: sha(ROOT / f) for f in files if (ROOT / f).exists()}, "stations_sha256": sha(STATIONS)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (f"MANIFEST_{name}.json" if name else "MANIFEST.json")).write_text(json.dumps(m, indent=1, ensure_ascii=False))
    cd = OUT / (f"code_{name}" if name else "code"); cd.mkdir(exist_ok=True)
    for f in files[:6]:
        (cd / Path(f).name).write_bytes((ROOT / f).read_bytes())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["all", "run", "score"]); ap.add_argument("--split", default="dev", choices=list(RECS))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS)); ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=16); a = ap.parse_args()
    assert STATIONS.exists(), f"station list missing: {STATIONS}"
    if a.cmd == "all":
        manifest(a.workers); log("start", configurations=CONFIGS, basins=NETS, seeds=SEEDS, records=RECS["dev"], stations=len(STATIONS.read_text().split()), data=DATA)
        failed = run("dev", CONFIGS, SEEDS, a.workers); log("runs done", split="dev", failed=failed)
        score("dev", CONFIGS, SEEDS); (OUT / "DONE_dev").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    elif a.cmd == "run":
        manifest(a.workers, a.split); log("start", split=a.split, configurations=a.configs, seeds=a.seeds, data=DATA)
        failed = run(a.split, a.configs, a.seeds, a.workers); log("runs done", split=a.split, configurations=a.configs, seeds=a.seeds, failed=failed)
    else:
        prim, choice, rp = score(a.split, a.configs, a.seeds); print(prim.round(4).to_string(index=False)); print(choice); print(rp)


if __name__ == "__main__":
    main()
