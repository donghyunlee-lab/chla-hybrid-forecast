"""Stage 8: what each part of the main model (the hybrid: process core plus constrained network) does.

The main model is the hybrid of stage 6 (H-res-all: process core, station constants inside sigmoid ranges, a bounded daily forcing
from an xLSTM trained on the other three basins and frozen at the target, a gate). It is not changed here. Each variant changes one
part and is run on the same data, records and seeds as the main model (test years 2024-2025, records of 36 and 12 months):

  free-eps    the network's forcing without its bound and without its penalty (|eps| <= 1 per day is the numerical limit of a step)
  free-const  the station constants without the sigmoid ranges (a linear map with the same value and slope at the start)
  no-pre      no pre-training on the other basins: network, constants and gate are trained on the target basin's record
  growth      the Monod growth term of the original submission added, its kinetic constants inside the original sigmoid ranges

Already available and read from stage 6: without the network (`core`) and the network alone (`bb-nut`).
The training code is unchanged (hybrid.py, core2.py); the two `free` variants replace two module settings of core2 in their own
processes and write to their own folders.
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Names: bb-C1 and the other bb-* models are the joint network of the paper (one xLSTM trained on the stations together,
without process terms; 'bb' for black box); core (P-mcf-all-kT) is the process model and hybrid (H-res-all) the hybrid
model.
Usage: newinput_ablate.py all [--workers 16] | run | score | child VARIANT pretrain|fit|core --net/--leave N --record R --seed S
Environment, code tests only: NEWINPUT_ABLATE_OUT, NEWINPUT_ABLATE_SEEDS, NEWINPUT_ABLATE_NETS, XPINN_EPOCHS.
"""
import argparse, datetime as dt, json, os, platform, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import newinput as NI                                                              # noqa: E402

OUT = Path(os.environ["NEWINPUT_ABLATE_OUT"]) if os.environ.get("NEWINPUT_ABLATE_OUT") else NI.STAGE5 / "stage8"
MAIN = NI.HYB
VARIANTS = {"free-eps": (MAIN, "eps"), "free-const": (MAIN, "const"), "no-pre": (MAIN.replace(".pre-nat", ""), ""),
            "growth": (MAIN.replace("gro-none", "gro-monod").replace("kin-fixed", "kin-learned"), "")}
if os.environ.get("NEWINPUT_ABLATE_SAME"):                                         # code test: the main model itself through this driver
    VARIANTS = {"same": (MAIN, ""), **VARIANTS}
SEEDS = tuple(int(x) for x in os.environ.get("NEWINPUT_ABLATE_SEEDS", "42,123,456,1,2").split(","))
NETS = tuple(os.environ.get("NEWINPUT_ABLATE_NETS", ",".join(NI.NETS)).split(","))
RECS, SRC_REC, SPLIT, HS, MIN_N, MATCH, KEY = (36, 12), 36, "test", NI.HS, NI.MIN_N, NI.MATCH, NI.KEY
STAGE6 = {"main": "hybrid", "core": "core", "bb-nut": "bb-nut"}
RANGES = {"kappa": (0.0, 0.5), "omega": (0.0, 1.0), "thk": (0.9, 1.1), "beta": (0.0, 0.05), "alpha": (0.0, 1.0),
          "mu": (0.1, 1.5), "loss": (0.015, 0.3), "KN": (0.01, 0.5), "KP": (0.001, 0.05), "KI": (50.0, 1000.0)}


def log(event, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kw}, ensure_ascii=False) + "\n")


def tag(spec, net, rec, seed):
    return f"{spec}__{net}_{SPLIT}_r{rec}_s{seed}"


def pre_tag(spec, leave, seed):
    return f"{spec}__lo-{leave}_{SPLIT}_r{SRC_REC}_s{seed}"


def child(variant, mode, argv):
    """One run of the unchanged training code with the variant's change."""
    spec, patch = VARIANTS[variant]; a = dict(zip(argv[::2], argv[1::2]))
    os.environ["XPINN2_OUT"] = str(OUT / variant)                                # read by core2 when it is imported
    for p in ("clean_revision", "national", "xpinn"):
        sys.path.insert(0, str(ROOT / "scripts" / p))
    import core2 as X2
    assert X2.Spec.parse(spec).name == spec, (spec, X2.Spec.parse(spec).name)
    if patch == "eps":
        X2.EPS_MAX, X2.EPS_PEN = 1.0, 0.0
    elif patch == "const":
        def free(z, name):                                                         # value and slope of the sigmoid map at z = 0, no range
            lo, hi = X2.RANGE[name]; p = (lo + hi) / 2 + (hi - lo) * z / 4
            return p.clamp_min(0.5) if name == "thk" else p                        # a power needs a positive base (the range is 0.9-1.1)
        X2.bound = free
    if mode == "core":
        sys.argv = ["core2.py", "--spec", spec, "--net", a["--net"], "--split", SPLIT, "--record", a["--record"], "--seed", a["--seed"]]
        return X2.main()
    import hybrid as HY
    ep = int(os.environ.get("XPINN_EPOCHS", 100))
    if mode == "pretrain":
        HY.pretrain(spec, a["--leave"], SPLIT, int(a["--seed"]), ep)
    else:
        HY.fit(spec, a["--net"], SPLIT, int(a["--record"]), int(a["--seed"]), ep)


def cmds(variant):
    """(DONE path, arguments): the pre-trainings of a variant, then its target runs."""
    spec, _ = VARIANTS[variant]; o = OUT / variant
    if ".pre-nat" not in spec:
        return [], [(o / "runs" / tag(spec, n, r, s) / "DONE", [variant, "core", "--net", n, "--record", str(r), "--seed", str(s)]) for n in NETS for r in RECS for s in SEEDS]
    return ([(o / "pretrain" / pre_tag(spec, n, s) / "DONE", [variant, "pretrain", "--leave", n, "--seed", str(s)]) for n in NETS for s in SEEDS],
            [(o / "runs" / tag(spec, n, r, s) / "DONE", [variant, "fit", "--net", n, "--record", str(r), "--seed", str(s)]) for n in NETS for r in RECS for s in SEEDS])


def pool(cs, workers):
    """Unfinished runs, `workers` at a time, with one retry. Returns the names of the runs that failed."""
    env = {**NI.env(), "XPINN2_OUT": ""}

    def one(c):
        lf = OUT / "logs" / c[1][0] / f"{c[0].parent.name}.log"; lf.parent.mkdir(parents=True, exist_ok=True)
        with open(lf, "a") as f:
            return subprocess.run([NI.PY, str(Path(__file__).resolve()), "child"] + c[1], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env).returncode
    todo = [c for c in cs if not c[0].exists()]
    for attempt in (1, 2):
        if not todo:
            break
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
        todo = [c for c in todo if not c[0].exists()]
        if todo:
            log("failed", attempt=attempt, runs=[f"{c[1][0]}/{c[0].parent.name}" for c in todo])
    return [f"{c[1][0]}/{c[0].parent.name}" for c in todo]


def run(workers):
    """Pre-trainings of the three variants that have them and the runs of `no-pre` together, then the other target runs."""
    C = {v: cmds(v) for v in VARIANTS}
    failed = pool([c for v in VARIANTS for c in C[v][0]] + [c for v in VARIANTS if not C[v][0] for c in C[v][1]], workers)
    return failed + pool([c for v in VARIANTS if C[v][0] for c in C[v][1]], workers)


def run_dir(model, net, rec, seed):
    return NI.run_dir(STAGE6[model], net, SPLIT, rec, seed) if model in STAGE6 else OUT / model / "runs" / tag(VARIANTS[model][0], net, rec, seed)


def score():
    """Station R2 (original units, measured days) on the forecasts shared by the main model, the core, the black box and the four
    variants; pooled R2 and RMSE; each variant against the main model by the reading rules; what the parts do in each variant."""
    sd = OUT / "test"; sd.mkdir(parents=True, exist_ok=True); models = [*STAGE6, *VARIANTS]; seeds = [str(s) for s in SEEDS]; rows, pooled, counts, ex, cs = [], [], [], [], []
    for rec in RECS:
        parts = []
        for net in NETS:
            W = None
            for m in models:
                for s in seeds:
                    x = pd.read_csv(run_dir(m, net, rec, s) / "predictions_eval.csv", dtype={"station": str}); x = x[x.observed & x.y_pred_raw.notna()].set_index(KEY)
                    if W is None:
                        W = x[["y_true_raw", "base_raw"]].rename(columns={"y_true_raw": "y", "base_raw": "persistence|ens"})
                    W = W.join(x.y_pred_raw.rename(f"{m}|{s}"), how="inner")
                W[f"{m}|ens"] = W[[f"{m}|{s}" for s in seeds]].mean(axis=1)
            counts.append({"record": rec, "net": net, "shared_forecasts": len(W)}); parts.append(W.reset_index().assign(net=net))
            for m in [x for x in models if x != "bb-nut"]:                          # what the parts do, run by run
                for s in seeds:
                    d = run_dir(m, net, rec, s); p = pd.read_csv(d / "params_eval.csv", dtype={"station": str}); c = json.loads((d / "config.json").read_text())
                    a, e, g = p.d7_mech.abs().mean(), (p.d7_full - p.d7_mech).abs().mean(), c.get("gate")
                    ex.append({"record": rec, "net": net, "model": m, "seed": s, "process_share_7d": a / (a + e) if a + e > 0 else np.nan, "mean_abs_eps": p.eps.abs().mean() if "eps" in p else 0.0,
                               "max_abs_eps": p.eps.abs().max() if "eps" in p else 0.0, "growth_part_7d": (p.d7_full - p.d7_nogro).abs().mean() if "d7_nogro" in p else 0.0,
                               "gate": g[0] if g else np.nan, "lr": c["lr_selected"], "best_epoch": c["best_epoch"]})
                    k = [x for x in RANGES if x in p]; cs.append(p.groupby("station")[k].median().reset_index().assign(record=rec, net=net, model=m, seed=s))
        A = pd.concat(parts, ignore_index=True); cols = [c for c in A.columns if "|" in c]
        for h, q in A.groupby("horizon"):
            y = q.y.to_numpy(float)
            for col in [c for c in cols if c.endswith("|ens")]:
                p = q[col].to_numpy(float); pooled.append({"record": rec, "horizon": int(h), "model": col.split("|")[0], "r2": NI.r2(y, p), "rmse": float(np.sqrt(((y - p) ** 2).mean())), "n": len(q)})
            for st, g in q.groupby("station"):
                if len(g) < MIN_N:
                    continue
                yy = g.y.to_numpy(float)
                for col in cols:
                    m, s = col.split("|"); p = g[col].to_numpy(float)
                    rows.append({"net": g.net.iloc[0], "station": st, "record": rec, "horizon": int(h), "model": m, "seed": s, "r2": NI.r2(yy, p), "rmse": float(np.sqrt(((yy - p) ** 2).mean())), "n": len(g)})
    E = pd.DataFrame(rows); E.to_csv(sd / "station_r2.csv", index=False); pd.DataFrame(counts).to_csv(sd / "counts.csv", index=False); pd.DataFrame(pooled).to_csv(sd / "pooled.csv", index=False)
    E = E[E.horizon.isin([1, *HS])]
    med = E.groupby(["record", "model", "seed", "horizon"]).r2.median().unstack("horizon"); med["mean3_7_14"] = med[list(HS)].mean(axis=1)
    sl = med.index.get_level_values("seed"); summ = med[sl == "ens"].reset_index().drop(columns="seed")
    sm = med[sl != "ens"].groupby(["record", "model"]).mean3_7_14.agg(["mean", "std"]).rename(columns={"mean": "seed_mean", "std": "seed_sd"}).reset_index()
    rm = E[E.seed == "ens"].groupby(["record", "model", "horizon"]).rmse.median().unstack("horizon").add_prefix("rmse_").reset_index()
    summ = summ.merge(sm, on=["record", "model"], how="left").merge(rm, on=["record", "model"], how="left").sort_values(["record", "mean3_7_14"], ascending=[False, False])
    summ.to_csv(sd / "summary.csv", index=False)
    Hh = E[E.horizon.isin(HS)]; En = Hh[Hh.seed == "ens"]
    En.groupby(["record", "net", "model", "horizon"]).r2.median().unstack("horizon").mean(axis=1).rename("score").reset_index().pivot_table(index=["record", "net"], columns="model", values="score").to_csv(sd / "basin.csv")
    P = En.pivot_table(index=["record", "station", "horizon"], columns="model", values="r2"); pr, verdict = [], {}
    for rec in RECS:
        q = P.xs(rec, level="record"); v = {}
        for a, b in [(x, "main") for x in models if x not in ("main", "persistence")] + [("main", "core"), ("main", "bb-nut"), ("main", "persistence")]:
            d = q[a] - q[b]; per = [NI.paired(d.xs(h, level="horizon"), record=rec, model=a, against=b, horizon=h) for h in HS]
            S1 = Hh[(Hh.record == rec) & (Hh.model == a) & (Hh.seed != "ens")].pivot_table(index=["station", "horizon"], columns="seed", values="r2")
            ps = [float(S1[s].sub(q[b]).groupby("station").mean().median()) for s in S1.columns]
            pr += per + [{**NI.paired(d.groupby("station").mean(), record=rec, model=a, against=b, horizon="mean3_7_14"), "seed_mean_of_median_diff": float(np.mean(ps)),
                          "seed_sd_of_median_diff": float(np.std(ps, ddof=1)) if len(ps) > 1 else np.nan, "seeds_positive": int(np.sum(np.array(ps) > 0)), "n_seeds": len(ps)}]
            md, pv = np.array([x["median_diff"] for x in per]), np.array([x["wilcoxon_p"] for x in per])
            v[f"{a} matches {b}"] = bool((md >= MATCH).all()); v[f"{a} improves on {b}"] = bool(((md > 0) & (pv < 0.05)).sum() >= 2); v[f"{b} improves on {a}"] = bool(((md < 0) & (pv < 0.05)).sum() >= 2)
        verdict[str(rec)] = v
    PR = pd.DataFrame(pr); PR.to_csv(sd / "paired.csv", index=False); (sd / "verdicts.json").write_text(json.dumps(verdict, indent=1))
    X = pd.DataFrame(ex); X.to_csv(sd / "explain_runs.csv", index=False)
    X.groupby(["record", "model"])[["process_share_7d", "mean_abs_eps", "max_abs_eps", "growth_part_7d", "gate"]].agg(["mean", "std"]).to_csv(sd / "explain.csv")
    Cn = pd.concat(cs, ignore_index=True); k = [x for x in RANGES if x in Cn]; Cs = Cn.groupby(["record", "net", "model", "station"])[k].mean().reset_index(); Cs.to_csv(sd / "constants_station.csv", index=False)
    out = []                                                                        # station constants (mean over seeds): median, and the share outside or at the edge of the range
    for (rec, m), g in Cs.groupby(["record", "model"]):
        for c in [x for x in k if g[x].notna().any()]:
            lo, hi = RANGES[c]; x = g[c].dropna(); w = (hi - lo) * 0.01
            out.append({"record": rec, "model": m, "constant": c, "stations": len(x), "median": x.median(), "min": x.min(), "max": x.max(), "share_outside_range": float(((x < lo) | (x > hi)).mean()),
                        "share_within_1pct_of_a_bound": float(((x <= lo + w) | (x >= hi - w)).mean())})
    pd.DataFrame(out).to_csv(sd / "constants.csv", index=False)
    log("scored", summary=summ[["record", "model", "mean3_7_14"]].round(4).to_dict("records"), verdicts=verdict)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(pd.DataFrame(counts).groupby("record").shared_forecasts.sum().to_dict())
    print(summ.round(4).to_string(index=False)); print(PR[PR.horizon == "mean3_7_14"].round(4).to_string(index=False)); print(json.dumps(verdict, indent=1))
    return summ, verdict


def manifest(workers):
    files = [f"scripts/xpinn/{x}" for x in ("newinput_ablate.py", "newinput.py", "hybrid.py", "core.py", "core2.py", "explore.py", "final2.py", "seasonal.py", "score.py")] + \
            ["scripts/clean_revision/common.py", "scripts/clean_revision/models.py", "scripts/national/nat.py", f"{NI.DATA}/daily_observed.csv", f"{NI.DATA}/era5_daily.csv", f"{NI.DATA}/stations.csv"]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    m = {"created": dt.datetime.now().isoformat(timespec="seconds"), "data": NI.DATA, "host": platform.node(), "python": sys.version.split()[0], "gpu": gpu, "git_head": head,
         "settings": {"main_model": MAIN, "variants": {v: {"spec": s, "change": c or "specification only"} for v, (s, c) in VARIANTS.items()}, "free_eps": {"EPS_MAX": 1.0, "EPS_PEN": 0.0},
                      "free_const": "station constants = range centre + range width * z / 4 (theta_kappa >= 0.5)", "network_inputs": NI.INPUTS, "basins": NETS, "seeds": SEEDS, "records": RECS,
                      "source_record_months": SRC_REC, "split": SPLIT, "workers": workers, "match_margin": MATCH, "min_forecasts_per_station_horizon": MIN_N,
                      "epochs": int(os.environ.get("XPINN_EPOCHS", 100)), "stage6": str(NI.OUT)},
         "stations": NI.STATIONS.read_text().split(), "sha256": {f: NI.sha(ROOT / f) for f in files if (ROOT / f).exists()}, "stations_sha256": NI.sha(NI.STATIONS)}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "MANIFEST.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
    cd = OUT / "code"; cd.mkdir(exist_ok=True)
    for f in files[:12]:
        (cd / Path(f).name).write_bytes((ROOT / f).read_bytes())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        return child(sys.argv[2], sys.argv[3], sys.argv[4:])
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["all", "run", "score"]); ap.add_argument("--workers", type=int, default=16); a = ap.parse_args()
    if a.cmd in ("all", "run"):
        if a.cmd == "all":
            manifest(a.workers); log("start", main=MAIN, variants={v: s for v, (s, _) in VARIANTS.items()}, basins=NETS, seeds=SEEDS, records=RECS)
        failed = run(a.workers); log("runs done", failed=failed); print("failed:", failed)
        if a.cmd == "all":
            (OUT / "DONE_runs").write_text(time.strftime("%Y-%m-%d %H:%M:%S")); score(); (OUT / "DONE_all").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        score()


if __name__ == "__main__":
    main()
