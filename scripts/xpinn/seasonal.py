"""Stage 3 of the design of the process equation (designs of core2.py): a relaxation target that moves from the record
mean toward the seasonal climatology as the record holds more years that agree on the season (core2.py, targets msy, meb, mcf).
Rules: RULE below.
Step 1, development year 2023 in all four basins (records 24 and 12 months ending 2022-12-31, seed 42): the P-mean-all backbone
with the targets clim, msy, meb and mcf, the earlier designs P-mean-all, P-all-kT and Z-mlp-kT, and each basin's references
(persistence, damped anomaly, ARIMA, harmonic ARIMA, black box C1). Step 2, on the best target of step 1: the temperature-
dependent rate, the gated driver network and the growth anomaly. Finalists: the best design of step 2 without a network and the
network design; test years 2024-2025 in all four basins (records 36 and 12 months, seeds 42, 123, 456) against the national
benchmark and P-mean-all; term contributions on both periods. Outputs in OUT/seasonal/.
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Names: bb-C1 and the other bb-* models are the joint network of the paper (one xLSTM trained on the stations together,
without process terms; 'bb' for black box); core (P-mcf-all-kT) is the process model and hybrid (H-res-all) the hybrid
model.
Usage: seasonal.py all|step1|step2|test|explain [--workers 16]"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "xpinn"))
import core2 as X2                                                               # noqa: E402
import explore as E                                                              # noqa: E402
import final2 as F                                                               # noqa: E402  (its import lets the shared scorers read this family's runs)
import score as SC                                                               # noqa: E402

ST = X2.OUT / "seasonal"; NETS = tuple(os.environ.get("SEASONAL_NETS", "nakdong,han,geum,yeongsan").split(","))   # fewer basins in short code tests
RECS, SEED, SEEDS, TEST_RECS, NEWT = E.RECS, E.SEED, F.SEEDS, F.TEST_RECS, ("clim", "msy", "meb", "mcf")
ALL = dict(anc="smooth", age="yes", flush="yes")
RULE = {"gain": 0.005, "floor": -0.010}       # step-1 verdict: mean target gain at 24 months >= gain and no basin-record cell below floor
WHY1 = ("Step 1, development year 2023 in all four basins: relaxation targets that move from the record mean toward "
        "the record's seasonal climatology by a weight that is 0 with one year of record: fixed by the number of "
        "years (msy), by the years' agreement on the season (meb), learned with the climatology of the other years "
        "(mcf), or the full climatology (clim).")
WHY2 = ("Step 2 adds to the target {t} of step 1 the temperature-dependent relaxation rate, the gated driver network "
        "and the growth anomaly.")


def sp(**kw) -> str:
    return X2.Spec(**kw).normal().name


def step2_specs(t):
    return {f"P-{t}-all-kT": sp(tgt=t, par="station", kt="yes", **ALL), f"Z-mlp-kT-{t}": sp(tgt=t, par="mlp", kt="yes", gate="yes", **ALL),
            f"S-all-anom-climdrv-{t}": sp(tgt=t, gro="anom", par="station", drv="clim", **ALL)}


REF = {k: F.spec_of(k) for k in ("P-mean-all", "P-all-kT", "Z-mlp-kT")}
STEP1 = {f"P-{t}-all": sp(tgt=t, par="station", **ALL) for t in NEWT}
LABELS = {**REF, **STEP1, **{k: v for t in NEWT for k, v in step2_specs(t).items()}}
NAME = {**{v: k for k, v in LABELS.items()}, "bb-C1": "black box C1"}


# ------------------------------------------------------------------------------------------------ development year
def dev_jobs(specs, nets=NETS):
    return [(s, "dev", r, SEED, False, n) for s in specs for n in nets for r in RECS]


def dev_refs(net, rec, workers=12):
    """Damped anomaly, ARIMA and harmonic ARIMA of the development year, as score.dev_refs: the Nakdong ones are the first
    search's; for the other basins they are fitted here on the keys of the P-mean-all run and cached in OUT/seasonal/refs."""
    f = E.FIRST / "refs" / f"{net}_dev_r{rec}.csv" if net == E.NET else ST / "refs" / f"{net}_dev_r{rec}.csv"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        base = F.load_any(X2.tag(REF["P-mean-all"], net, "dev", rec, SEED))
        tasks = [(st, rec, base[base.station == st][["station", "anchor_date", "horizon", "target_date", "y_true_raw"]]) for st in sorted(base.station.unique())]
        with Pool(min(workers, len(tasks))) as pool:
            r = pd.concat(pool.map(SC._ref_task, tasks), ignore_index=True)
        r["anchor_date"] = r.anchor_date.astype(str).str[:10]
        r[r.observed.astype(bool)][SC.KEY + ["y_pred_raw", "model"]].to_csv(f, index=False, float_format="%.6g")
    r = pd.read_csv(f, dtype={"station": str}); r["anchor_date"] = r.anchor_date.astype(str).str[:10]
    return r


def eligible(net, rec):
    c = json.loads((X2.OUT / "runs" / X2.tag(REF["P-mean-all"], net, "dev", rec, SEED) / "config.json").read_text())["counts"]
    return [s for s, k in c.items() if k["chl_train"] >= 30 and k["chl_val"] >= 10 and min(k["eval_targets"]) >= SC.MIN_EVAL]


def dev_table(specs, net, rec):
    frames = {s: f for s in specs if (f := F.load_any(X2.tag(s, net, "dev", rec, SEED))) is not None}
    truth = F.load_any(X2.tag("bb-C1", net, "dev", rec, SEED)); frames["bb-C1"] = truth; frames["persistence"] = truth.assign(y_pred_raw=truth.base_raw)
    for name, g in dev_refs(net, rec).groupby("model"):
        frames[name] = g
    return SC.r2_table(frames, eligible(net, rec), truth)


def explain(spec, net, stations):
    """Record stability X2r and bound hitting X3 as in explore.explain (these designs have no residual and no daily loss or
    exchange), for any basin. The learned seasonal weight ω of mcf is left out of X2r: with one year of record it has nothing
    to learn from. Station medians of the parameters with the longer record."""
    s = X2.Spec.parse(spec); names = s.params() + (["KN", "KP", "KI"] if s.kin == "learned" else [])
    P = {}
    for r in RECS:
        f = X2.OUT / "runs" / X2.tag(spec, net, "dev", r, SEED) / "params_eval.csv"
        if not f.exists():
            return {}
        p = pd.read_csv(f, dtype={"station": str}); P[r] = p[p.station.isin(stations)]
    allp, diffs, hits, out = pd.concat(P.values(), ignore_index=True), [], {}, {}
    for n in names:
        lo, hi = X2.RANGE[n]
        if not (n == "omega" and s.tgt == "mcf"):
            a, b = (P[r].groupby("station")[n].median() for r in RECS); diffs += list(((a - b).abs() / (hi - lo)).dropna().to_numpy())
        z = (allp[n].dropna() - lo) / (hi - lo); hits[n] = float(((z < 0.01) | (z > 0.99)).mean()) if len(z) else 0.0
        out[f"med_{n}"] = float(P[RECS[0]].groupby("station")[n].median().median())
    out["X2r"] = float(np.median(diffs)) if diffs else 0.0; out["X3"] = max(hits.values()) if hits else 0.0
    out["X3_worst"] = max(hits, key=hits.get) if hits else ""
    out["screen"] = bool(out["X2r"] <= E.THR["X2r"] and out["X3"] <= E.THR["X3"])
    return out


def leaderboard(specs, step):
    """A per basin and record against the best reference of that basin, record and horizon; the means over the eight cells, the
    four 24-month and the four 12-month cells; the screen in every basin; gains against the same design with the record-mean
    target (tgt_gain), without growth (kin_gain) and without the network (net_gain), per cell."""
    d = ST / "dev"; d.mkdir(parents=True, exist_ok=True); rows, best = {}, {}
    for s in specs:
        rows[s] = {"label": NAME.get(s, s), "spec": s, "step": "ref" if s in REF.values() else ("1" if s in STEP1.values() else "2")}
    rows["bb-C1"] = {"label": NAME["bb-C1"], "spec": "bb-C1", "step": "ref"}
    for net in NETS:
        st = eligible(net, RECS[0])
        for r in RECS:
            T = dev_table(specs, net, r); T.to_csv(d / f"station_r2_{net}_r{r}.csv", index=False, float_format="%.5g")
            acc, b = SC.accuracy(T, specs + ["bb-C1"], refs=E.REFS); best[f"{net}_{r}"] = b
            for a in acc.itertuples():
                rows[a.variant].update({f"A_{net}_{r}": a.A, f"goal_{net}_{r}": a.goal_ok, **{f"d{h}_{net}_{r}": getattr(a, f"dr2_h{h}") for h in SC.FH}})
        for s in specs:
            rows[s].update({f"{k}_{net}": v for k, v in explain(s, net, st).items()})
    lb = pd.DataFrame(rows.values()); cells = [f"A_{n}_{r}" for n in NETS for r in RECS]
    lb["A"] = lb[cells].mean(axis=1)
    for r in RECS:
        lb[f"A_{r}"] = lb[[f"A_{n}_{r}" for n in NETS]].mean(axis=1)
    lb["goal_cells"] = lb[[f"goal_{n}_{r}" for n in NETS for r in RECS]].fillna(False).astype(bool).sum(axis=1)
    lb["screen"] = lb[[f"screen_{n}" for n in NETS]].fillna(False).astype(bool).all(axis=1)
    C = lb.set_index("spec")[cells]
    sibs = {"tgt_gain": lambda x: replace(x, tgt="mean").normal().name if x.tgt in NEWT else None,
            "kin_gain": lambda x: replace(x, gro="none").normal().name if x.gro != "none" else None,
            "net_gain": lambda x: F.sibling(x.name) if x.n_daily() else None}
    for col, sib in sibs.items():
        g = {s: C.loc[s] - C.loc[o] for s in specs if (o := sib(X2.Spec.parse(s))) and o in C.index}
        for r in RECS:
            lb[f"{col}_{r}"] = lb.spec.map(lambda s: g[s][[f"A_{n}_{r}" for n in NETS]].mean() if s in g else np.nan)
        lb[f"{col}_min"] = lb.spec.map(lambda s: g[s].min() if s in g else np.nan)
    lb = lb.sort_values("A", ascending=False).reset_index(drop=True)
    lb.to_csv(ST / f"leaderboard_step{step}.csv", index=False, float_format="%.5g")
    return lb, best


def choose_target(lb):
    """Rule 1: the step-1 target with the highest mean A over the eight cells among the designs passing the screen in every
    basin (all of them if none passes). Verdict: mean target gain at 24 months >= RULE gain and no cell below RULE floor."""
    c = lb[lb.label.isin(list(STEP1))]; ok = c[c.screen]
    b = (ok if len(ok) else c).iloc[0]
    return X2.Spec.parse(b.spec).tgt, b, bool(b.tgt_gain_24 >= RULE["gain"] and b.tgt_gain_min >= RULE["floor"]), bool(len(ok))


def choose_finalists(lb, t):
    """Rule 2: F1 = the highest mean A among the step-1 design with target t and its step-2 designs without a network that pass
    the screen (all of them if none passes); F2 = the network design on t."""
    s2 = step2_specs(t); cand = [STEP1[f"P-{t}-all"]] + [v for k, v in s2.items() if not k.startswith("Z-")]
    c = lb[lb.spec.isin(cand)]; ok = c[c.screen]
    return [(ok if len(ok) else c).iloc[0].spec, s2[f"Z-mlp-kT-{t}"]]


# ------------------------------------------------------------------------------------------------ test years
def test(fins, workers):
    d = ST / "test"; d.mkdir(parents=True, exist_ok=True); refs = [REF["P-mean-all"], REF["Z-mlp-kT"]]; focus = fins + refs
    failed = E.run_jobs([(s, "test", r, sd, False, n) for s in focus for n in NETS for r in TEST_RECS for sd in SEEDS], workers)
    out, pm = {}, f"{REF['P-mean-all']} (xpinn)"
    for n in NETS:
        Et = SC.test_table(focus + ["bb-C1"], n, TEST_RECS, SEEDS); Et.to_csv(d / f"station_metrics_{n}.csv", index=False, float_format="%.5g")
        cmp_, summ = SC.test_compare(Et, focus, TEST_RECS)
        cmp_.to_csv(d / f"comparisons_{n}.csv", index=False, float_format="%.5g"); summ.to_csv(d / f"summary_r2_{n}.csv", index=False, float_format="%.5g")
        out[n] = {NAME[s]: {f"{int(x.record)}m_h{int(x.horizon)}": round(float(x.median_diff), 4) for x in cmp_[(cmp_.focus == s) & cmp_.is_best_national].itertuples()}
                  for s in focus}
        out[n]["vs P-mean-all"] = {NAME[s]: {f"{int(x.record)}m_h{int(x.horizon)}": [round(float(x.median_diff), 4), round(float(x.share_better), 2)]
                                             for x in cmp_[(cmp_.focus == s) & (cmp_.other == pm)].itertuples()} for s in fins}
    E.log("seasonal test", finalists=[NAME[s] for s in fins], failed=failed, gap_to_best_national=out)
    return out


# ------------------------------------------------------------------------------------------------ explanation
def explain_terms(fins, workers=8):
    """explain_terms.py for each finalist, basin and period (development year with seed 42, test years with three seeds)."""
    cmds = [[E.PY, str(ROOT / "scripts" / "xpinn" / "explain_terms.py"), "--label", NAME[s], "--net", n, "--split", split,
             "--records", *map(str, recs), "--seeds", *map(str, seeds)]
            for s in fins for n in NETS for split, recs, seeds in (("dev", RECS, (SEED,)), ("test", TEST_RECS, SEEDS))]
    def one(c):
        with open(ST / "explain.log", "a") as f:
            return subprocess.run(c, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    with ThreadPoolExecutor(workers) as ex:
        codes = list(ex.map(one, cmds))
    return [" ".join(c[2:9]) for c, k in zip(cmds, codes) if k]


def weights(fins):
    """Seasonal weights per basin, period, record and station (they do not depend on the design), and the learned ω of mcf."""
    rows = []
    for d in sorted((X2.OUT / "runs").glob("*")):
        c = d / "config.json"
        if (d / "DONE").exists() and c.exists() and d.name.split("__")[0] in list(STEP1.values()) + fins:
            j = json.loads(c.read_text())
            rows += [{"net": j["net"], "split": j["split"], "record": j["record_months"], "station": st, "years": len(w["years"]),
                      "w_sy": w["w_sy"], "w_eb": w["w_eb"]} for st, w in j.get("seasonal_weights", {}).items()]
    if not rows:
        return None
    w = pd.DataFrame(rows).drop_duplicates(["net", "split", "record", "station"])
    om = []
    for n in NETS:
        for r in RECS:
            f = X2.OUT / "runs" / X2.tag(STEP1["P-mcf-all"], n, "dev", r, SEED) / "params_eval.csv"
            if f.exists():
                p = pd.read_csv(f, dtype={"station": str}); om += [{"net": n, "split": "dev", "record": r, "station": st, "omega_mcf": float(v)}
                                                                    for st, v in p.groupby("station").omega.median().items()]
    if om:
        w = w.merge(pd.DataFrame(om), on=["net", "split", "record", "station"], how="left")
    w.sort_values(["split", "net", "record", "station"]).to_csv(ST / "weights.csv", index=False, float_format="%.4g")
    return w


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["all", "step1", "step2", "test", "explain"]); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(); ST.mkdir(parents=True, exist_ok=True); s1 = list(STEP1.values()) + list(REF.values())
    (ST / "labels.json").write_text(json.dumps(NAME, indent=1))                    # spec -> label, for the workbook
    if a.stage in ("all", "step1"):
        E.log("seasonal step1 start", why=WHY1, specs=STEP1, references=REF, rule=RULE, nets=NETS)
        jobs = dev_jobs(s1) + [("bb-C1", "dev", r, SEED, False, n) for n in NETS if n != E.NET for r in RECS]
        failed = E.run_jobs(jobs, a.workers); lb, best = leaderboard(s1, 1)
        t, b, verdict, screened = choose_target(lb)
        (ST / "target.json").write_text(json.dumps({"target": t, "design": b.label, "verdict": verdict, "screen_passed": screened}, indent=1))
        E.log("seasonal step1 done", failed=failed, target=t, design=b.label, verdict=verdict, screen_passed=screened, best_reference=best,
              summary=lb[["label", "A", "A_24", "A_12", "goal_cells", "screen", "tgt_gain_24", "tgt_gain_12", "tgt_gain_min"]].round(4).to_dict("records"))
    if a.stage in ("all", "step2"):
        t = json.loads((ST / "target.json").read_text())["target"]; s2 = step2_specs(t)
        E.log("seasonal step2 start", why=WHY2.format(t=t), specs=s2)
        failed = E.run_jobs(dev_jobs(list(s2.values())), a.workers); lb, _ = leaderboard(s1 + list(s2.values()), 2)
        fins = choose_finalists(lb, t); (ST / "finalists.json").write_text(json.dumps([NAME[s] for s in fins]))
        E.log("seasonal step2 done", failed=failed, finalists=[NAME[s] for s in fins],
              summary=lb[["label", "A", "A_24", "A_12", "goal_cells", "screen", "tgt_gain_24", "kin_gain_24", "net_gain_24"]].round(4).to_dict("records"))
    fins = [LABELS[x] for x in json.loads((ST / "finalists.json").read_text())] if (ST / "finalists.json").exists() else []
    if a.stage in ("all", "test"):
        test(fins, a.workers)
    if a.stage in ("all", "explain"):
        bad = explain_terms(fins); weights(fins)
        E.log("seasonal explain", finalists=[NAME[s] for s in fins], failed=bad)
    if a.stage == "all":
        (X2.OUT / "DONE_seasonal").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
