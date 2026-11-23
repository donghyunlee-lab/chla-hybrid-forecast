"""Stage 5a: input length of the joint black-box xLSTM (bb-C1); a 30-day input may be too long for the networks.
The same network, data and training as in the exploration, with 7, 14, 21 or 30 input days. Stations: results/input_design_20261114/
stations.txt (only stations whose data cover the whole period; station_rule.py).
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Names: bb-C1 and the other bb-* models are the joint network of the paper (one xLSTM trained on the stations together,
without process terms; 'bb' for black box); core (P-mcf-all-kT) is the process model and hybrid (H-res-all) the hybrid
model.
Usage: inputlen.py all [--workers 16]   (development year: runs, scores, choice)
       inputlen.py run|score --split dev|test [--lens 14 30] [--seeds 42 123 456 1 2]"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, platform, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["INPUTLEN_OUT"]) if os.environ.get("INPUTLEN_OUT") else ROOT / "results" / "input_design_20261114"
PY, STATIONS = str(ROOT / ".venv" / "bin" / "python"), OUT / "stations.txt"
# Data folder: the second build adds the 2025 values of the eight Yeongsan-Seomjin stations. It equals the first build up to
# 2024-12-31, so the development-year runs made with the first build are unchanged.
DATA = os.environ.get("INPUTLEN_DATA", "data/national_v2")
LENS = tuple(int(x) for x in os.environ.get("INPUTLEN_LENS", "7,14,21,30").split(","))
NETS = tuple(os.environ.get("INPUTLEN_NETS", "nakdong,han,geum,yeongsan").split(","))
SEEDS = tuple(int(x) for x in os.environ.get("INPUTLEN_SEEDS", "42,123,456").split(","))
RECS, SPEC, HS, BASE, MARGIN, MIN_N = {"dev": (24, 12), "test": (36, 12)}, "bb-C1", (3, 7, 14), 30, 0.005, 60
KEY = ["station", "anchor_date", "horizon"]


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


def run_dir(L, net, split, rec, seed):
    return OUT / f"L{L:02d}" / "runs" / f"{SPEC}__{net}_{split}_r{rec}_s{seed}"


def run(split, lens, seeds, workers):
    """Every unfinished run, `workers` at a time, with one retry. Each run is its own process, so the input length is set by the environment."""
    jobs = [(L, n, r, s) for L in lens for n in NETS for r in RECS[split] for s in seeds]
    def one(j):
        L, n, r, s = j; lf = OUT / "logs" / f"L{L:02d}_{run_dir(L, n, split, r, s).name}.log"; lf.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "XPINN2_OUT": str(OUT / f"L{L:02d}"), "XPINN_SEQ_LEN": str(L), "XPINN_STATIONS": str(STATIONS), "PINN_NATIONAL_DIR": DATA}
        cmd = [PY, str(ROOT / "scripts" / "xpinn" / "core2.py"), "--spec", SPEC, "--net", n, "--split", split, "--record", str(r), "--seed", str(s)]
        with open(lf, "a") as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env).returncode
    todo = [j for j in jobs if not (run_dir(j[0], j[1], split, j[2], j[3]) / "DONE").exists()]
    for attempt in (1, 2):
        if not todo:
            break
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
        todo = [j for j in todo if not (run_dir(j[0], j[1], split, j[2], j[3]) / "DONE").exists()]
        if todo:
            log("failed", attempt=attempt, split=split, runs=[f"L{j[0]} {j[1]} r{j[2]} s{j[3]}" for j in todo])
    return [f"L{j[0]} {j[1]} r{j[2]} s{j[3]}" for j in todo]


def r2(y, p):
    sst = float(((y - y.mean()) ** 2).sum())
    return float(1 - ((y - p) ** 2).sum() / sst) if sst > 0 else np.nan


def score(split, lens, seeds):
    """Station R2 (original units, measured days) on the forecasts shared by every input length and seed; summaries and the choice."""
    from scipy import stats
    sd = OUT / split; sd.mkdir(parents=True, exist_ok=True); rows, counts = [], []
    for rec in RECS[split]:
        for net in NETS:
            W = None
            for L in lens:
                for s in seeds:
                    d = run_dir(L, net, split, rec, s)
                    if not (d / "DONE").exists():
                        continue
                    x = pd.read_csv(d / "predictions_eval.csv", dtype={"station": str}); x = x[x.observed & x.y_pred_raw.notna()]
                    counts.append({"record": rec, "net": net, "L": L, "seed": s, "forecasts": len(x)})
                    x = x.set_index(KEY)
                    if W is None:
                        W = x[["y_true_raw", "base_raw"]].rename(columns={"y_true_raw": "y", "base_raw": "persistence"})
                    W = W.join(x.y_pred_raw.rename(f"L{L}_s{s}"), how="inner")
            if W is None:
                continue
            for L in lens:
                c = [f"L{L}_s{s}" for s in seeds if f"L{L}_s{s}" in W]
                if c:
                    W[f"L{L}_ens"] = W[c].mean(axis=1)
            counts.append({"record": rec, "net": net, "L": "shared", "seed": "", "forecasts": len(W)})
            for (st, h), g in W.reset_index().groupby(["station", "horizon"]):
                if len(g) < MIN_N:
                    continue
                for col in [c for c in g.columns if c.startswith("L") or c == "persistence"]:
                    L, seed = (col.split("_")[0][1:], col.split("_")[1].lstrip("s")) if col != "persistence" else ("persistence", "")
                    rows.append({"net": net, "station": st, "record": rec, "horizon": int(h), "L": L, "seed": seed, "r2": r2(g.y.to_numpy(float), g[col].to_numpy(float)), "n": len(g)})
    E = pd.DataFrame(rows); E.to_csv(sd / "station_r2.csv", index=False); pd.DataFrame(counts).to_csv(sd / "forecast_counts.csv", index=False)
    E = E[E.horizon.isin([1, *HS])]; lab = [str(L) for L in lens]
    # station medians: seed-mean forecast, each seed, persistence
    med = E.groupby(["record", "L", "seed", "horizon"]).r2.median().unstack("horizon"); med["mean3_7_14"] = med[list(HS)].mean(axis=1)
    seedrows = med[med.index.get_level_values("seed").isin([str(s) for s in seeds])]
    summ = med[med.index.get_level_values("seed").isin(["ens", ""])].reset_index().drop(columns="seed")
    sm = seedrows.groupby(["record", "L"]).agg(["mean", "std"]); sm.columns = [f"{a}_seed_{b}" for a, b in sm.columns]
    summ = summ.merge(sm.reset_index(), on=["record", "L"], how="left"); summ.to_csv(sd / "summary.csv", index=False)
    bas = E[E.seed.isin(["ens", ""])].groupby(["record", "net", "L", "horizon"]).r2.median().unstack("horizon"); bas["mean3_7_14"] = bas[list(HS)].mean(axis=1)
    bas.reset_index().to_csv(sd / "basin_summary.csv", index=False)
    # paired with the 30-day input (seed-mean forecasts; each horizon and the station mean of 3, 7, 14 days)
    P = E[E.seed == "ens"].pivot_table(index=["record", "station", "horizon"], columns="L", values="r2"); pr = []
    for rec in RECS[split]:
        for L in [x for x in lab if x != str(BASE) and x in P and str(BASE) in P]:
            for h in [*HS, "mean3_7_14"]:
                q = P.xs(rec, level="record"); q = q[q.index.get_level_values("horizon").isin(HS)]
                dd = (q[L] - q[str(BASE)]).groupby("station").mean() if h == "mean3_7_14" else (q[L] - q[str(BASE)]).xs(h, level="horizon")
                dd = dd.dropna()
                pr.append({"record": rec, "L": L, "against": BASE, "horizon": h, "n_stations": len(dd), "median_diff": dd.median(), "share_better": float((dd > 0).mean()),
                           "wilcoxon_p": float(stats.wilcoxon(dd).pvalue) if (dd != 0).any() else 1.0})
    pd.DataFrame(pr).to_csv(sd / "paired_vs_30.csv", index=False)
    # primary score: mean over the records of the seed mean of mean3_7_14, with the SD over seeds of the record mean
    ps = seedrows.reset_index().groupby(["L", "seed"]).mean3_7_14.mean().unstack("seed")
    prim = pd.DataFrame({"primary": ps.mean(axis=1), "seed_sd": ps.std(axis=1), **{f"seed_{c}": ps[c] for c in ps.columns}}).reset_index()
    prim["L"] = prim.L.astype(int); prim = prim.sort_values("L"); prim.to_csv(sd / "primary.csv", index=False)
    choice = None
    if split == "dev" and BASE in prim.L.tolist():
        b = float(prim.loc[prim.L == BASE, "primary"].iloc[0]); top = prim.sort_values("primary", ascending=False).iloc[0]; L = int(top.L); ok = False
        if L != BASE:
            pp = pd.DataFrame(pr); pp = pp[(pp.L == str(L)) & (pp.horizon == "mean3_7_14")]
            ok = bool(top.primary - b >= MARGIN and (pp.share_better > 0.5).all())
        choice = {"best_by_primary": L, "primary_best": float(top.primary), "primary_30": b, "gain_over_30": float(top.primary - b), "rule_margin": MARGIN,
                  "chosen": L if ok else BASE, "replaces_30": ok}
        (OUT / "chosen.json").write_text(json.dumps(choice, indent=1))
    log("scored", split=split, primary=prim.round(4).to_dict("records"), choice=choice)
    return prim, choice


def manifest(workers, name=""):
    files = ["scripts/xpinn/inputlen.py", "scripts/xpinn/core.py", "scripts/xpinn/core2.py", "scripts/clean_revision/common.py", "scripts/clean_revision/models.py",
             "scripts/national/nat.py", f"{DATA}/daily_observed.csv", f"{DATA}/era5_daily.csv", f"{DATA}/stations.csv"]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    m = {"created": dt.datetime.now().isoformat(timespec="seconds"), "data": DATA, "host": platform.node(), "python": sys.version.split()[0], "gpu": gpu, "git_head": head,
         "settings": {"spec": SPEC, "lengths": LENS, "basins": NETS, "seeds": SEEDS, "records": RECS, "workers": workers, "baseline_length": BASE, "margin": MARGIN,
                      "min_forecasts_per_station_horizon": MIN_N, "learning_rates": [1e-3, 3e-4, 1e-4], "epochs": int(os.environ.get("XPINN_EPOCHS", 100))},
         "stations": STATIONS.read_text().split(), "sha256": {f: sha(ROOT / f) for f in files if (ROOT / f).exists()}, "stations_sha256": sha(STATIONS)}
    (OUT / (f"MANIFEST_{name}.json" if name else "MANIFEST.json")).write_text(json.dumps(m, indent=1, ensure_ascii=False))
    cd = OUT / (f"code_{name}" if name else "code"); cd.mkdir(exist_ok=True)
    for f in files[:6]:
        (cd / Path(f).name).write_bytes((ROOT / f).read_bytes())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["all", "run", "score"]); ap.add_argument("--split", default="dev", choices=list(RECS))
    ap.add_argument("--lens", type=int, nargs="*", default=list(LENS)); ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=16); a = ap.parse_args()
    assert STATIONS.exists(), f"station list missing: {STATIONS}"
    if a.cmd == "all":
        manifest(a.workers); log("start", lengths=LENS, basins=NETS, seeds=SEEDS, records=RECS["dev"], stations=len(STATIONS.read_text().split()))
        failed = run("dev", LENS, SEEDS, a.workers); log("runs done", split="dev", failed=failed)
        score("dev", LENS, SEEDS); (OUT / "DONE_dev").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    elif a.cmd == "run":
        manifest(a.workers, a.split); log("start", split=a.split, lengths=a.lens, seeds=a.seeds, data=DATA)
        failed = run(a.split, a.lens, a.seeds, a.workers); log("runs done", split=a.split, lengths=a.lens, seeds=a.seeds, failed=failed)
    else:
        prim, choice = score(a.split, a.lens, a.seeds); print(prim.round(4).to_string(index=False)); print(choice)


if __name__ == "__main__":
    main()
