"""Stage 7 (the model search stopped, the comparison models repeated on the 27 stations and the second data build): the
comparison models of the national benchmark on the input design of stage 5,
scored next to the core and the hybrid of stage 6 on the same forecasts.

The training code is that of the national benchmark, unchanged (scripts/national/train_nat.py, baselines_nat.py and the modules
they import). This driver changes three things around that code and nothing else:
  stations   the 27 stations with the whole period (stations.txt); the source stations of a fold are the stations of the other
             three basins among the 27;
  data       the second build of the daily file (data/national_v2), so the Yeongsan-Seomjin stations have their 2025 values;
  inputs     the input channels of the learned models: variant `n17` = the 17 channels of stage 5b (Chl-a, water temperature,
             weather, TN, TP; primary), variant `all` = the 32 channels of the paper (sensitivity check).
The baselines that read Chl-a only (climatology, damped anomaly, ARIMA, harmonic ARIMA) do not depend on the inputs and are
written once, in the folder of `n17`.

Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Names: bb-C1 and the other bb-* models are the joint network of the paper (one xLSTM trained on the stations together,
without process terms; 'bb' for black box); core (P-mcf-all-kT) is the process model and hybrid (H-res-all) the hybrid
model.
Usage: newinput_bench.py all [--slots 16] [--workers 8] | nets | baselines --variant n17|all | score [--variants n17,all] [--sub test]
       | repro | child VARIANT SCRIPT ARGS...   (child: one run of a benchmark script under the three changes)
Environment, code tests only: NEWINPUT_BENCH_OUT, NEWINPUT_BENCH_SEEDS, NEWINPUT_BENCH_EPOCHS, NEWINPUT_BENCH_STATIONS, NEWINPUT_BENCH_DATA.
"""
import argparse, datetime as dt, hashlib, json, os, platform, runpy, subprocess, sys, threading, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE5 = ROOT / "results" / "input_design_20261114"
OUT = Path(os.environ["NEWINPUT_BENCH_OUT"]) if os.environ.get("NEWINPUT_BENCH_OUT") else STAGE5 / "stage7"
PY = str(ROOT / ".venv" / "bin" / "python")
STATIONS = Path(os.environ["NEWINPUT_BENCH_STATIONS"]) if os.environ.get("NEWINPUT_BENCH_STATIONS") else STAGE5 / "stations.txt"
DATA = os.environ.get("NEWINPUT_BENCH_DATA", "data/national_v2")
NUT = ["chlorophyll_a", "water_temperature", "era5_ssrd_mj", "era5_t2m", "era5_tp_mm", "era5_wind_max", "total_nitrogen", "total_phosphorus"]   # `nut` of stage 5b
VARIANTS = {"n17": NUT, "all": None}                    # input channels of the learned models (None: all 32)
STATV = "n17"                                           # folder of the baselines that read Chl-a only
NETS, RECS, SRC_REC = ("xLSTM", "PINN-sigmoid"), (36, 12), 36
SEEDS = tuple(int(x) for x in os.environ.get("NEWINPUT_BENCH_SEEDS", "42,123,456,1,2").split(","))
EPOCHS = os.environ.get("NEWINPUT_BENCH_EPOCHS")        # code tests only
HS, MIN_N, MATCH, KEY = (3, 7, 14), 60, -0.02, ["station", "anchor_date", "horizon"]
STAGE6 = ["core", "hybrid", "bb-nut", "bb-chl", "bb-all"]
STAT = {"climatology": "bl_climatology_{c}_r{r}", "damped anomaly": "bl_damped_{c}_r{r}", "ARIMA": "bl_arima_{c}_r{r}", "harmonic ARIMA": "bl_harmonic_{c}_r{r}"}
LEARNED = {"XGBoost": "bl_xgboost_{c}_r{r}_s{s}", "XGBoost zero-shot": "bl_xgbzs_{c}_s{s}",
           "xLSTM zero-shot": "tgt_{c}_xLSTM_zero_shot_s{s}", "xLSTM fine-tuned": "tgt_{c}_xLSTM_full_tune_r{r}_s{s}",
           "PINN-sigmoid zero-shot": "tgt_{c}_PINN-sigmoid_zero_shot_s{s}", "PINN-sigmoid fine-tuned": "tgt_{c}_PINN-sigmoid_full_tune_r{r}_s{s}"}
SIX = ["damped anomaly", "harmonic ARIMA", "XGBoost", "core", "XGBoost zero-shot", "xLSTM fine-tuned"]      # the six of stage 4
MINE = ("core", "hybrid")
_SET = {}


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


def setup(variant):
    """Point the benchmark modules at the station list, the data folder and the variant's input channels (once per process)."""
    if _SET:
        assert _SET["variant"] == variant, "one variant per process"
        return _SET["N"], _SET["C"]
    os.environ["PINN_NATIONAL_DIR"], os.environ["PINN_RESULTS_DIR"] = DATA, str(OUT / variant)
    sys.path.insert(0, str(ROOT / "scripts" / "clean_revision")); sys.path.insert(0, str(ROOT / "scripts" / "national"))
    import nat as N
    keep = set(STATIONS.read_text().split())
    N.STATIONS = N.STATIONS[N.STATIONS.code.isin(keep)].reset_index(drop=True); N.CODES = N.STATIONS.code.tolist()
    assert set(N.CODES) == keep, sorted(keep - set(N.CODES))
    N.FOLDS = {b: [c for c in N.CODES if N.BASIN_OF[c] == b] for b in N.BASINS}
    C = N.patch()
    if VARIANTS[variant]:
        v, full = set(VARIANTS[variant]), C.build_arrays
        assert v <= set(C.VARS + C.ERA5), sorted(v - set(C.VARS + C.ERA5))

        def build_arrays(*a, **k):                      # the rule of stage 5b: a variable brings its value, measured flag and age (Chl-a also its last value)
            arr = full(*a, **k); fn = arr["feat_names"]; keep_i = [i for i, c in enumerate(fn) if c.split("__")[0] in v]
            arr["X"], arr["feat_names"] = np.ascontiguousarray(arr["X"][..., keep_i]), [fn[i] for i in keep_i]
            return arr
        C.build_arrays = build_arrays
    _SET.update(variant=variant, N=N, C=C)
    return N, C


def child(variant, script, argv):
    """One run of a benchmark script (train_nat.py), unchanged, under the three changes."""
    setup(variant); path, here = ROOT / "scripts" / "national" / script, Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != here]          # as if the script were started directly
    sys.argv = [str(path)] + argv
    runpy.run_path(str(path), run_name="__main__")


# ---------------------------------------------------------------------------------------------------------------- runs
def pre_tag(fold, net, s):
    return f"pre_{fold}_{net}_s{s}"


def done(v, tag):
    return (OUT / v / "runs" / tag / "DONE").exists()


def jobs(N, variants):
    """(variant, tag, arguments, runs it needs): variant by variant and seed by seed; the source models of a seed, then its target runs."""
    ok, rec, ep, J = N.eligible_set(), ["--record_end", N.RECORD_END, "--val_frac_in_record", str(N.VAL_FRAC)], (["--epochs", EPOCHS] if EPOCHS else []), []
    for v in variants:
        for s in SEEDS:
            for fold in N.BASINS:
                for net in NETS:
                    J.append((v, pre_tag(fold, net, s), N.NETS[net] + ["--seed", str(s), "--mode", "pretrain", "--fold", fold, "--record_in_pretrain", "1",
                                                                       "--record_months", str(SRC_REC)] + rec + ep, []))
            for code in N.CODES:
                fold = N.BASIN_OF[code]
                for net in NETS:
                    base, pre = N.NETS[net] + ["--seed", str(s), "--fold", fold, "--target_station", code], pre_tag(fold, net, s)
                    ck = ["--init_ckpt", str(OUT / v / "runs" / pre / "best_model.pt")]
                    J.append((v, f"tgt_{code}_{net}_zero_shot_s{s}", base + ["--mode", "zero_shot"] + ck, [pre]))
                    J += [(v, f"tgt_{code}_{net}_full_tune_r{m}_s{s}", base + ["--mode", "full_tune", "--record_months", str(m)] + rec + ck + ep, [pre])
                          for m in RECS if (code, m) in ok]
    return J


def schedule(J, slots):
    """`slots` runs at a time; a run starts when the runs it needs are finished. Finished runs are skipped. Returns the failed ones."""
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    pending = [j for j in J if not done(j[0], j[1])]; running, failed = {}, set(); n_ok, last, t0 = 0, 0, time.time()
    while pending or running:
        for k, (p, fh) in list(running.items()):
            if p.poll() is not None:
                fh.close(); del running[k]
                if p.returncode == 0 and done(*k):
                    n_ok += 1
                else:
                    failed.add(k); log("failed", variant=k[0], run=k[1], exit=p.returncode)
        i = 0
        while len(running) < slots and i < len(pending):
            v, tag, args, deps = pending[i]
            if done(v, tag):
                pending.pop(i); continue
            if any((v, d) in failed for d in deps):
                failed.add((v, tag)); pending.pop(i); log("skipped", variant=v, run=tag, reason="a run it needs failed"); continue
            if all(done(v, d) for d in deps):
                lf = OUT / "logs" / v / f"{tag}.log"; lf.parent.mkdir(parents=True, exist_ok=True); fh = open(lf, "a")
                cmd = [PY, str(Path(__file__).resolve()), "child", v, "train_nat.py", "--tag", tag] + args
                running[(v, tag)] = (subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT)), fh); pending.pop(i); continue
            i += 1
        if n_ok - last >= 100:
            last = n_ok; log("progress", done=n_ok, running=len(running), pending=len(pending), failed=len(failed), hours=round((time.time() - t0) / 3600, 2))
        time.sleep(2)
    return sorted(failed)


def baselines(variant, workers):
    """The CPU models of the benchmark (functions of baselines_nat.py): records of 36 and 12 months, the seeds of the networks."""
    import multiprocessing as mp
    N, C = setup(variant)
    import baselines_nat as BN
    per = C.RESULTS / "persistence.csv"
    if not per.exists():
        per.parent.mkdir(parents=True, exist_ok=True); __import__("persistence_v4").persistence_frame().to_csv(per, index=False)
    tasks = [("xgbzs", f, s) for s in SEEDS for f in N.BASINS] + ([("stat", c, m) for c in N.CODES for m in RECS] if variant == STATV else [])
    tasks += [("xgb", c, m, s) for s in SEEDS for c in N.CODES for m in RECS]; failed = []
    with mp.get_context("fork").Pool(workers, initializer=BN.init) as pool:
        for i, r in enumerate(pool.imap_unordered(BN.run, tasks), 1):
            if isinstance(r, tuple) and r and r[0] == "FAILED":
                failed.append([str(x) for x in r[1]])
            if i % 50 == 0:
                print(time.strftime("%H:%M:%S"), i, "/", len(tasks), flush=True)
    log("baselines done", variant=variant, tasks=len(tasks), failed=failed)
    return failed


def baselines_chain(workers):
    for v in VARIANTS:
        lf = OUT / "logs" / f"baselines_{v}.log"; lf.parent.mkdir(parents=True, exist_ok=True)
        with open(lf, "a") as f:
            rc = subprocess.run([PY, str(Path(__file__).resolve()), "baselines", "--variant", v, "--workers", str(workers)], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
        log("baselines exit", variant=v, exit=rc)


# --------------------------------------------------------------------------------------------------------------- scores
def name(model, v):
    return model if v == "n17" else f"{model} (32)"


def read(path):
    x = pd.read_csv(path, dtype={"station": str}); x = x[x.observed & x.y_pred_raw.notna()][KEY + ["y_true_raw", "y_pred_raw"]].copy()
    x["anchor_date"] = x.anchor_date.astype(str).str[:10]
    return x.drop_duplicates(KEY).set_index(KEY)


def score(variants, sub):
    """Station R2 (original units, measured days) of the test years on the forecasts shared by every model and seed: the models of
    stage 6 (their own five seeds) and the comparison models of this stage. Summaries, ranks, paired differences and the readings
    by the rules of stage 4."""
    from scipy import stats
    import newinput as NI
    sd = OUT / sub; sd.mkdir(parents=True, exist_ok=True); rows, counts, missing, cache = [], [], [], {}; s6 = [str(s) for s in NI.SEEDS["test"]]
    models = [(nm, STATV, pat, None) for nm, pat in STAT.items()] + [(name(nm, v), v, pat, SEEDS) for v in variants for nm, pat in LEARNED.items()]
    seeded = {m: s6 for m in STAGE6} | {nm: [str(s) for s in sds] for nm, _, _, sds in models if sds}
    six = {"mean of six": SIX} | ({"mean of six (32)": [name(m, "all") if m in LEARNED else m for m in SIX]} if "all" in variants else {})
    six = {k: v for k, v in six.items() if all(m in seeded or m in STAT for m in v)}

    def bench(v, tag):
        if (v, tag) not in cache:
            d = OUT / v / "runs" / tag; cache[(v, tag)] = read(d / "predictions_test.csv") if (d / "DONE").exists() else None
        return cache[(v, tag)]

    for rec in RECS:
        for net in NI.NETS:
            W = None
            for m in STAGE6:
                for s in s6:
                    x = pd.read_csv(NI.run_dir(m, net, "test", rec, s) / "predictions_eval.csv", dtype={"station": str}); x = x[x.observed & x.y_pred_raw.notna()].copy()
                    x["anchor_date"] = x.anchor_date.astype(str).str[:10]; x = x.set_index(KEY)
                    if W is None:
                        W = x[["y_true_raw", "base_raw"]].rename(columns={"y_true_raw": "y", "base_raw": "persistence|ens"})
                    W = W.join(x.y_pred_raw.rename(f"{m}|{s}"), how="inner")
            for code in sorted(W.index.get_level_values("station").unique()):
                g = W[W.index.get_level_values("station") == code]; n6, miss, dy = len(g), [], 0.0
                for nm, v, pat, sds in models:
                    fr = [bench(v, pat.format(c=code, r=rec, s=s)) for s in (sds or [0])]
                    if any(f is None for f in fr):
                        miss.append(nm); continue
                    b = fr[0][["y_true_raw"]].rename(columns={"y_true_raw": "yb"})
                    for c, f in zip([f"{nm}|{s}" for s in sds] if sds else [f"{nm}|ens"], fr):
                        b = b.join(f.y_pred_raw.rename(c), how="inner")
                    g = g.join(b, how="inner"); dy = max(dy, float((g.y - g.yb).abs().max())) if len(g) else dy; g = g.drop(columns="yb")
                if miss:
                    missing.append({"station": code, "net": net, "record": rec, "missing": ";".join(miss)}); continue
                g = g.copy()
                for m, sds in seeded.items():
                    g[f"{m}|ens"] = g[[f"{m}|{s}" for s in sds]].mean(axis=1)
                for k, ms in six.items():
                    g[f"{k}|ens"] = g[[f"{m}|ens" for m in ms]].mean(axis=1)
                a = g.index.get_level_values("anchor_date")
                counts.append({"station": code, "net": net, "record": rec, "stage6_forecasts": n6, "shared_forecasts": len(g), "first_anchor": a.min() if len(g) else "",
                               "last_anchor": a.max() if len(g) else "", "max_abs_truth_diff": dy})
                for h, q in g.groupby(level="horizon"):
                    if len(q) < MIN_N:
                        continue
                    y = q.y.to_numpy(float)
                    for col in [c for c in q.columns if "|" in c]:
                        m, s = col.split("|")
                        rows.append({"net": net, "station": code, "record": rec, "horizon": int(h), "model": m, "seed": s, "r2": NI.r2(y, q[col].to_numpy(float)), "n": len(q)})
    E = pd.DataFrame(rows); E.to_csv(sd / "station_r2.csv", index=False); Cn = pd.DataFrame(counts); Cn.to_csv(sd / "counts.csv", index=False)
    pd.DataFrame(missing, columns=["station", "net", "record", "missing"]).to_csv(sd / "missing.csv", index=False)
    E = E[E.horizon.isin([1, *HS])]
    med = E.groupby(["record", "model", "seed", "horizon"]).r2.median().unstack("horizon"); med["mean3_7_14"] = med[list(HS)].mean(axis=1)
    sl = med.index.get_level_values("seed"); summ = med[sl == "ens"].reset_index().drop(columns="seed")
    sm = med[sl != "ens"].groupby(["record", "model"]).mean3_7_14.agg(["mean", "std", "count"]).rename(columns={"mean": "seed_mean", "std": "seed_sd", "count": "n_seeds"}).reset_index()
    nst = E[(E.seed == "ens") & (E.horizon == 7)].groupby(["record", "model"]).station.nunique().rename("stations_7d").reset_index()
    summ = summ.merge(sm, on=["record", "model"], how="left").merge(nst, on=["record", "model"], how="left").sort_values(["record", "mean3_7_14"], ascending=[False, False])
    summ.to_csv(sd / "summary.csv", index=False)
    Hh = E[E.horizon.isin(HS)]; En = Hh[Hh.seed == "ens"]
    bs = En.groupby(["record", "net", "model", "horizon"]).r2.median().unstack("horizon").mean(axis=1).rename("score").reset_index()
    bs.pivot_table(index=["record", "net"], columns="model", values="score").to_csv(sd / "basin.csv")
    # ranks over the stations (station mean of 3, 7 and 14 days, seed-mean forecasts; rank 1 = highest R2) and the Friedman test
    P = En.pivot_table(index=["record", "station", "horizon"], columns="model", values="r2"); rk, fr = [], {}
    for rec in RECS:
        q = P.xs(rec, level="record").groupby("station").mean().dropna()
        r = q.rank(axis=1, ascending=False); rk.append(pd.DataFrame({"record": rec, "model": r.columns, "mean_rank": r.mean().to_numpy(), "stations": len(q)}))
        fr[str(rec)] = {"stations": len(q), "models": q.shape[1], "friedman_p": float(stats.friedmanchisquare(*[q[c].to_numpy() for c in q.columns]).pvalue) if len(q) > 2 else None}
    pd.concat(rk).sort_values(["record", "mean_rank"], ascending=[False, True]).to_csv(sd / "ranks.csv", index=False)
    # paired over the stations (seed-mean forecasts), every ordered pair; for the core and the hybrid also run by run
    pr, verdict = [], {}
    for rec in RECS:
        q = P.xs(rec, level="record"); v = {}
        for a in q.columns:
            S1 = Hh[(Hh.record == rec) & (Hh.model == a) & (Hh.seed != "ens")].pivot_table(index=["station", "horizon"], columns="seed", values="r2") if a in MINE else None
            for b in [m for m in q.columns if m != a]:
                d = q[a] - q[b]; per = [NI.paired(d.xs(h, level="horizon"), record=rec, model=a, against=b, horizon=h) for h in HS]
                row = NI.paired(d.groupby("station").mean(), record=rec, model=a, against=b, horizon="mean3_7_14")
                if S1 is not None:
                    ps = [float(S1[s].sub(q[b]).groupby("station").mean().median()) for s in S1.columns]
                    row |= {"seed_mean_of_median_diff": float(np.mean(ps)), "seed_sd_of_median_diff": float(np.std(ps, ddof=1)), "seeds_positive": int(np.sum(np.array(ps) > 0)), "n_seeds": len(ps)}
                pr += per + [row]
                if a in MINE or (a, b) in [(name("PINN-sigmoid fine-tuned", x), name("xLSTM fine-tuned", x)) for x in variants]:
                    md, pv = np.array([x["median_diff"] for x in per]), np.array([x["wilcoxon_p"] for x in per])
                    v[f"{a} matches {b}"] = bool((md >= MATCH).all()); v[f"{a} improves on {b}"] = bool(((md > 0) & (pv < 0.05)).sum() >= 2)
                    v[f"{b} improves on {a}"] = bool(((md < 0) & (pv < 0.05)).sum() >= 2)
        verdict[str(rec)] = v
    PR = pd.DataFrame(pr); PR.to_csv(sd / "paired.csv", index=False)
    (sd / "verdicts.json").write_text(json.dumps({"friedman": fr, "readings": verdict}, indent=1, ensure_ascii=False))
    log("scored", sub=sub, variants=variants, missing=len(missing), summary=summ[["record", "model", "mean3_7_14"]].round(4).to_dict("records"))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print("missing:", missing)
    print(Cn.groupby(["record", "net"]).agg(stations=("station", "count"), stage6=("stage6_forecasts", "sum"), shared=("shared_forecasts", "sum"), first=("first_anchor", "min"),
                                            last=("last_anchor", "max"), truth_diff=("max_abs_truth_diff", "max")).to_string())
    print(summ.round(4).to_string(index=False))
    print(PR[(PR.horizon == "mean3_7_14") & PR.model.isin(MINE)].round(4).to_string(index=False))
    return summ, verdict


def repro(slots):
    """One source model and one fine-tuned run of the xLSTM (variant n17) repeated under another name and compared with the first."""
    N, _ = setup("n17"); J = {j[1]: j for j in jobs(N, ["n17"])}; fold, s = N.BASINS[0], SEEDS[0]
    pre = J[pre_tag(fold, "xLSTM", s)]; tgt = next(j for t, j in J.items() if t.startswith("tgt_") and "_xLSTM_full_tune_r12_" in t and t.endswith(f"_s{s}") and pre[1] in j[3])
    ck = [str(OUT / "n17" / "runs" / (pre[1] + x) / "best_model.pt") for x in ("", "__repro")]
    failed = schedule([("n17", pre[1] + "__repro", pre[2], []), ("n17", tgt[1] + "__repro", [ck[1] if x == ck[0] else x for x in tgt[2]], [pre[1] + "__repro"])], slots)
    res = {"failed": [list(f) for f in failed], "runs": [pre[1], tgt[1]]}
    if not failed:
        for k, t in (("source_model", pre[1]), ("fine_tuned", tgt[1])):
            a, b = (OUT / "n17" / "runs" / x / "predictions_test.csv" for x in (t, t + "__repro"))
            x, y = (pd.read_csv(f, dtype={"station": str}).set_index(KEY).y_pred_raw for f in (a, b))
            res[k] = {"forecasts": len(x), "max_abs_diff": float(np.nanmax(np.abs(x.to_numpy(float) - y.reindex(x.index).to_numpy(float)))), "same_file": a.read_bytes() == b.read_bytes()}
    (OUT / "repro_check.json").write_text(json.dumps(res, indent=1)); log("repro", **res)
    return res


def manifest(slots, workers):
    from importlib.metadata import version
    code = ["scripts/xpinn/newinput_bench.py", "scripts/xpinn/newinput.py", "scripts/national/nat.py", "scripts/national/train_nat.py", "scripts/national/baselines_nat.py"] + \
           [f"scripts/clean_revision/{x}.py" for x in ("common", "models", "physics", "baselines", "bench_baselines_v7", "baselines_v4", "v5_additions", "persistence_v4", "stats_v4", "identifiability")] + \
           sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src" / "pinn_hab" / "models").glob("*.py"))
    data = [f"{DATA}/{x}" for x in ("daily_observed.csv", "era5_daily.csv", "stations.csv")]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    N, _ = setup("n17"); J = jobs(N, list(VARIANTS))
    m = {"created": dt.datetime.now().isoformat(timespec="seconds"), "data": DATA, "host": platform.node(), "python": sys.version.split()[0], "gpu": gpu, "git_head": head,
         "libraries": {x: version(x) for x in ("torch", "xgboost", "statsmodels", "numpy", "pandas", "scipy")},
         "settings": {"variants": {k: (v or "all 32 channels") for k, v in VARIANTS.items()}, "networks": NETS, "records": RECS, "source_record_months": SRC_REC, "seeds": SEEDS,
                      "record_end": N.RECORD_END, "val_frac_in_record": N.VAL_FRAC, "test_period": N.SPLITS["test"], "folds": N.FOLDS, "network_runs": len(J), "slots": slots,
                      "cpu_workers": workers, "epochs": EPOCHS or "100 (default of train_nat.py)", "match_margin": MATCH, "min_forecasts_per_station_horizon": MIN_N,
                      "stage6": str(STAGE5 / "stage6")},
         "stations": N.CODES, "stations_sha256": sha(STATIONS), "sha256": {f: sha(ROOT / f) for f in code + data if (ROOT / f).exists()}}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "MANIFEST.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
    cd = OUT / "code"; cd.mkdir(exist_ok=True)
    for f in code:
        if (ROOT / f).exists():
            (cd / ("src_" + Path(f).name if f.startswith("src/") else Path(f).name)).write_bytes((ROOT / f).read_bytes())
    return N, J


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        return child(sys.argv[2], sys.argv[3], sys.argv[4:])
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["all", "nets", "baselines", "score", "repro"]); ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--variant", default="n17", choices=list(VARIANTS)); ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--sub", default="test"); a = ap.parse_args()
    assert STATIONS.exists(), f"station list missing: {STATIONS}"
    if a.cmd == "all":
        N, J = manifest(a.slots, a.workers)
        log("start", variants=list(VARIANTS), networks=NETS, records=RECS, seeds=SEEDS, stations=len(N.CODES), folds={k: len(v) for k, v in N.FOLDS.items()}, data=DATA, network_runs=len(J))
        th = threading.Thread(target=baselines_chain, args=(a.workers,)); th.start()
        for attempt in (1, 2):
            failed = schedule(J, a.slots); log("network runs done", attempt=attempt, failed=[list(f) for f in failed])
            if not failed:
                break
        th.join(); (OUT / "DONE_runs").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        score(list(VARIANTS), "test"); (OUT / "DONE_test").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        repro(a.slots); (OUT / "DONE_all").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    elif a.cmd == "nets":
        N, _ = setup("n17"); failed = schedule(jobs(N, a.variants.split(",")), a.slots); log("network runs done", failed=[list(f) for f in failed]); print("failed:", failed)
    elif a.cmd == "baselines":
        print("failed:", baselines(a.variant, a.workers))
    elif a.cmd == "score":
        score(a.variants.split(","), a.sub)
    else:
        print(repro(a.slots))


if __name__ == "__main__":
    main()
