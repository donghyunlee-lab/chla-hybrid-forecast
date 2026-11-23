"""Exploration loop of the designs of core2.py.
Batches of diverse designs, one seed (42), development year 2023 with records of 24 and 12 months ending 2022-12-31.
After each batch every design run so far is scored (accuracy against the references, explainability checks, kinetic
ablation against the same design without growth, network gain against the same design without the network), the
leaderboard is written, and the next batch is designed from it (rationale in BATCHES).
Usage: explore.py run --batch b1 [--workers 12] | explore.py score --batch <last batch run> (rewrites that batch's snapshot)"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "xpinn"))
import core2 as X2                                                               # noqa: E402
import score as SC                                                               # noqa: E402  (shared metrics of the first search)

OUT = X2.OUT; LEDGER = OUT / "ledger.jsonl"; PY = str(ROOT / ".venv" / "bin" / "python")
FIRST = ROOT / "results" / "xpinn_20261027"                                       # references and black box of the first search (same data)
NET, RECS, SEED = "nakdong", (24, 12), 42
REFS = ["persistence", "damped anomaly", "ARIMA", "harmonic ARIMA", "bb-C1"]
THR = {"X1": 0.8, "X2r": 0.15, "X3": 0.10, "X4": 2 / 3, "goal": -0.02}


def sp(**kw) -> str:
    return X2.Spec(**kw).normal().name


BATCHES = {
    "b1": {"why": "batch 1: persistence-first designs without kinetics (relaxation to the seasonal climatology, to "
                  "the recent 30-day mean or to a blend; anchor smoothing; ageing; rain flushing), the process "
                  "designs of the first family, alternative kinetics (temperature only, growth anomaly, learned "
                  "constants), station-constant designs, residual and transport; siblings without growth for the "
                  "kinetic ablation",
           "specs": {
               "P-clim": sp(tgt="clim", par="station"), "P-recent": sp(tgt="recent", par="station"), "P-blend": sp(tgt="blend", par="station"),
               "P-clim-nn": sp(tgt="clim", par="daily"), "P-blend-nn": sp(tgt="blend", par="daily"), "P-blend-age": sp(tgt="blend", par="station", age="yes"),
               "P-blend-smooth": sp(tgt="blend", par="station", anc="smooth"), "P-blend-flush": sp(tgt="blend", par="station", flush="yes"),
               "R-winner": sp(tgt="clim", gro="monod", par="daily", tr="dist"), "R-winner-nogro": sp(tgt="clim", par="daily", tr="dist"),
               "R-runnerup": sp(tgt="clim", gro="monod", par="daily"), "R-recent": sp(tgt="recent", gro="monod", par="daily"),
               "R-recent-nogro": sp(tgt="recent", par="daily"), "R-blend": sp(tgt="blend", gro="monod", par="daily"),
               "R-blend-mixed": sp(tgt="blend", gro="monod", par="mixed"), "R-blend-dist": sp(tgt="blend", gro="monod", par="daily", tr="dist"),
               "K-temp": sp(tgt="blend", gro="temp", par="mixed"), "K-anom": sp(tgt="blend", gro="anom", par="mixed"),
               "K-anom-climdrv": sp(tgt="blend", gro="anom", par="mixed", drv="clim"), "K-monod-learned": sp(tgt="blend", gro="monod", par="mixed", kin="learned"),
               "K-anom-learned": sp(tgt="blend", gro="anom", par="mixed", kin="learned"),
               "F-monod": sp(tgt="blend", gro="monod", par="mixed", flush="yes"), "F-anom": sp(tgt="blend", gro="anom", par="mixed", flush="yes"),
               "F-temp": sp(tgt="blend", gro="temp", par="mixed", flush="yes"), "F-winner": sp(tgt="clim", gro="monod", par="daily", flush="yes"),
               "A-smooth": sp(tgt="blend", gro="monod", par="mixed", anc="smooth"), "A-age": sp(tgt="blend", gro="monod", par="mixed", age="yes"),
               "A-anom-all": sp(tgt="blend", gro="anom", par="mixed", flush="yes", anc="smooth", age="yes"),
               "A-persist-all": sp(tgt="blend", par="station", anc="smooth", age="yes", flush="yes"),
               "S-monod": sp(tgt="blend", gro="monod", par="station", flush="yes"), "S-anom": sp(tgt="blend", gro="anom", par="station", flush="yes"),
               "S-temp": sp(tgt="blend", gro="temp", par="station", flush="yes"),
               "S-anom-recent-all": sp(tgt="recent", gro="anom", par="station", flush="yes", anc="smooth", age="yes"),
               "D-nogro": sp(tgt="blend", par="daily", flush="yes"), "D-anom": sp(tgt="blend", gro="anom", par="daily", flush="yes"),
               "D-monod-all": sp(tgt="blend", gro="monod", par="daily", flush="yes", anc="smooth", age="yes"),
               "D-recent": sp(tgt="recent", gro="monod", par="daily", flush="yes"), "D-anom-climdrv": sp(tgt="blend", gro="anom", par="daily", flush="yes", drv="clim"),
               "X-resid": sp(tgt="blend", gro="anom", par="mixed", flush="yes", res="bounded"), "X-dist": sp(tgt="blend", gro="anom", par="mixed", flush="yes", tr="dist"),
               "X-dist-nn": sp(tgt="blend", gro="anom", par="daily", flush="yes", tr="dist")}},
    "b2": {"why": "batch 2: designs relaxing to the record mean of ln Chl-a and to its blend with the recent mean: "
                  "ageing, anchor smoothing and rain flushing, daily rates from the network, kinetics that can switch"
                  " off (mua >= 0) and give the seasonal departure from the mean through the drivers (sea: Monod "
                  "factor relative to its mean over the record; seaT: temperature only), the growth anomaly and Monod"
                  " kinetics, station-constant physics, learned constants, transport and a bounded residual; siblings"
                  " without growth for the kinetic ablation",
           "specs": {
               "P-mean": sp(tgt="mean", par="station"), "P-mean-age": sp(tgt="mean", par="station", age="yes"),
               "P-mean-smooth": sp(tgt="mean", par="station", anc="smooth"), "P-mean-flush": sp(tgt="mean", par="station", flush="yes"),
               "P-mean-flush-age": sp(tgt="mean", par="station", flush="yes", age="yes"),
               "P-mean-all": sp(tgt="mean", par="station", flush="yes", age="yes", anc="smooth"),
               "P-mrec": sp(tgt="mrec", par="station"), "P-mrec-age": sp(tgt="mrec", par="station", age="yes"), "P-mrec-flush": sp(tgt="mrec", par="station", flush="yes"),
               "D-mean-nogro": sp(tgt="mean", par="daily", flush="yes"), "D-mean-age": sp(tgt="mean", par="daily", flush="yes", age="yes"),
               "D-mrec-nogro": sp(tgt="mrec", par="daily", flush="yes"),
               "D-mean-anom-climdrv": sp(tgt="mean", gro="anom", par="daily", flush="yes", drv="clim"), "D-mean-sea": sp(tgt="mean", gro="sea", par="daily", flush="yes"),
               "D-mean-sea-climdrv": sp(tgt="mean", gro="sea", par="daily", flush="yes", drv="clim"),
               "D-mean-seaT-climdrv": sp(tgt="mean", gro="seaT", par="daily", flush="yes", drv="clim"), "D-mean-monod": sp(tgt="mean", gro="monod", par="daily", flush="yes"),
               "D-mean-sea-age": sp(tgt="mean", gro="sea", par="daily", flush="yes", drv="clim", age="yes"),
               "D-mrec-sea-climdrv": sp(tgt="mrec", gro="sea", par="daily", flush="yes", drv="clim"),
               "M-mean-anom-climdrv": sp(tgt="mean", gro="anom", par="mixed", flush="yes", drv="clim"), "M-mean-sea": sp(tgt="mean", gro="sea", par="mixed", flush="yes"),
               "M-mean-sea-climdrv": sp(tgt="mean", gro="sea", par="mixed", flush="yes", drv="clim"),
               "M-mean-seaT-climdrv": sp(tgt="mean", gro="seaT", par="mixed", flush="yes", drv="clim"), "M-mean-monod": sp(tgt="mean", gro="monod", par="mixed", flush="yes"),
               "M-mean-temp": sp(tgt="mean", gro="temp", par="mixed", flush="yes"),
               "M-mean-sea-age": sp(tgt="mean", gro="sea", par="mixed", flush="yes", drv="clim", age="yes"),
               "M-mean-sea-learned": sp(tgt="mean", gro="sea", par="mixed", flush="yes", drv="clim", kin="learned"),
               "M-mrec-sea-climdrv": sp(tgt="mrec", gro="sea", par="mixed", flush="yes", drv="clim"),
               "S-mean-sea-climdrv": sp(tgt="mean", gro="sea", par="station", flush="yes", drv="clim"), "S-mean-sea": sp(tgt="mean", gro="sea", par="station", flush="yes"),
               "S-mean-anom-climdrv": sp(tgt="mean", gro="anom", par="station", flush="yes", drv="clim"),
               "S-mean-seaT-climdrv": sp(tgt="mean", gro="seaT", par="station", flush="yes", drv="clim"),
               "S-mean-sea-age": sp(tgt="mean", gro="sea", par="station", flush="yes", drv="clim", age="yes"),
               "S-mean-sea-learned": sp(tgt="mean", gro="sea", par="station", flush="yes", drv="clim", kin="learned"),
               "X-mean-dist": sp(tgt="mean", par="station", flush="yes", tr="dist"),
               "X-mean-sea-dist": sp(tgt="mean", gro="sea", par="mixed", flush="yes", drv="clim", tr="dist"),
               "X-mean-resid": sp(tgt="mean", gro="sea", par="mixed", flush="yes", drv="clim", res="bounded")}},
    "b3": {"why": "batch 3: networks on the P-mean-all backbone within physical limits: moving the relaxation rate "
                  "(kmod) or every station constant (mod) within +-1 on the logit scale, a target shift (mnet, +-0.75"
                  " ln units), a bounded residual or all rates daily; the term ablation of the backbone (age with "
                  "smoothing, flushing with smoothing), the temperature-dependent relaxation rate (kt), transport, "
                  "and kinetics on the backbone (growth anomaly, sea and seaT as station constants, the growth "
                  "anomaly as mixed and mod)",
           "specs": {
               "P-mean-age-smooth": sp(tgt="mean", par="station", age="yes", anc="smooth"),
               "P-mean-flush-smooth": sp(tgt="mean", par="station", flush="yes", anc="smooth"),
               "P-all-kT": sp(tgt="mean", par="station", flush="yes", age="yes", anc="smooth", kt="yes"),
               "P-all-dist": sp(tgt="mean", par="station", flush="yes", age="yes", anc="smooth", tr="dist"),
               "S-all-anom-climdrv": sp(tgt="mean", gro="anom", par="station", flush="yes", age="yes", anc="smooth", drv="clim"),
               "S-all-seaT-climdrv": sp(tgt="mean", gro="seaT", par="station", flush="yes", age="yes", anc="smooth", drv="clim"),
               "S-all-sea-climdrv": sp(tgt="mean", gro="sea", par="station", flush="yes", age="yes", anc="smooth", drv="clim"),
               "M-all-anom-climdrv": sp(tgt="mean", gro="anom", par="mixed", flush="yes", age="yes", anc="smooth", drv="clim"),
               "N-kmod": sp(tgt="mean", par="kmod", flush="yes", age="yes", anc="smooth"),
               "N-mod": sp(tgt="mean", par="mod", flush="yes", age="yes", anc="smooth"),
               "N-resid": sp(tgt="mean", par="station", flush="yes", age="yes", anc="smooth", res="bounded"),
               "N-mnet": sp(tgt="mnet", par="station", flush="yes", age="yes", anc="smooth"),
               "N-mnet-kmod": sp(tgt="mnet", par="kmod", flush="yes", age="yes", anc="smooth"),
               "N-daily-all": sp(tgt="mean", par="daily", flush="yes", age="yes", anc="smooth"),
               "N-mod-anom-climdrv": sp(tgt="mean", gro="anom", par="mod", flush="yes", age="yes", anc="smooth", drv="clim")}},
    "b4": {"why": "batch 4: a small network of the drivers alone (water temperature, N, P, light and rain on the "
                  "forecast day) that moves the relaxation rate (kmlp) or every station constant (mlp) or gives the "
                  "target shift (emlp); fitting criteria on the scale and horizons of the score (flat, raw, rawflat) "
                  "for the backbone, the temperature-dependent rate, the growth anomaly and the driver network; one "
                  "set of constants for all river stations (pooled); the temperature-dependent rate with ageing "
                  "alone, with the drivers' climatology trajectory over the horizon and with the growth anomaly",
           "specs": {
               "G-kmlp": sp(tgt="mean", par="kmlp", flush="yes", age="yes", anc="smooth"),
               "G-mlp": sp(tgt="mean", par="mlp", flush="yes", age="yes", anc="smooth"),
               "G-emlp": sp(tgt="mnet", par="emlp", flush="yes", age="yes", anc="smooth"),
               "F-all-flat": sp(tgt="mean", par="station", flush="yes", age="yes", anc="smooth", fit="flat"),
               "F-all-raw": sp(tgt="mean", par="station", flush="yes", age="yes", anc="smooth", fit="raw"),
               "F-all-rawflat": sp(tgt="mean", par="station", flush="yes", age="yes", anc="smooth", fit="rawflat"),
               "F-age-rawflat": sp(tgt="mean", par="station", age="yes", fit="rawflat"),
               "F-anom-rawflat": sp(tgt="mean", gro="anom", par="station", flush="yes", age="yes", anc="smooth", drv="clim", fit="rawflat"),
               "F-kmlp-rawflat": sp(tgt="mean", par="kmlp", flush="yes", age="yes", anc="smooth", fit="rawflat"),
               "F-kT-rawflat": sp(tgt="mean", par="station", flush="yes", age="yes", anc="smooth", kt="yes", fit="rawflat"),
               "P-all-pooled": sp(tgt="mean", par="pooled", flush="yes", age="yes", anc="smooth"),
               "P-all-kT-pooled": sp(tgt="mean", par="pooled", flush="yes", age="yes", anc="smooth", kt="yes"),
               "P-age-kT": sp(tgt="mean", par="station", age="yes", kt="yes"),
               "P-all-kT-climdrv": sp(tgt="mean", par="station", flush="yes", age="yes", anc="smooth", kt="yes", drv="clim"),
               "S-kT-anom-climdrv": sp(tgt="mean", gro="anom", par="station", flush="yes", age="yes", anc="smooth", kt="yes", drv="clim")}},
    "b5": {"why": "batch 5: every network output passes a learned gate that starts at 0 (gate), for the driver "
                  "network (mlp, kmlp) and the xLSTM (mod, kmod) on the P-mean-all backbone, on the "
                  "temperature-dependent backbone (P-all-kT) and on the simplest backbone (ageing only), and with the"
                  " growth anomaly",
           "specs": {
               "Z-mlp": sp(tgt="mean", par="mlp", flush="yes", age="yes", anc="smooth", gate="yes"),
               "Z-kmlp": sp(tgt="mean", par="kmlp", flush="yes", age="yes", anc="smooth", gate="yes"),
               "Z-mod": sp(tgt="mean", par="mod", flush="yes", age="yes", anc="smooth", gate="yes"),
               "Z-kmod": sp(tgt="mean", par="kmod", flush="yes", age="yes", anc="smooth", gate="yes"),
               "Z-mlp-kT": sp(tgt="mean", par="mlp", flush="yes", age="yes", anc="smooth", kt="yes", gate="yes"),
               "Z-kmlp-kT": sp(tgt="mean", par="kmlp", flush="yes", age="yes", anc="smooth", kt="yes", gate="yes"),
               "Z-mlp-age": sp(tgt="mean", par="mlp", age="yes", gate="yes"),
               "Z-mlp-anom": sp(tgt="mean", gro="anom", par="mlp", flush="yes", age="yes", anc="smooth", drv="clim", gate="yes"),
               "Z-mod-anom": sp(tgt="mean", gro="anom", par="mod", flush="yes", age="yes", anc="smooth", drv="clim", gate="yes")}},
}


def log(event, **kw):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kw}, ensure_ascii=False, default=str) + "\n")


def done(spec, rec, seed=SEED, placebo=False, split="dev", net=NET):
    return (OUT / "runs" / X2.tag(spec, net, split, rec, seed, placebo) / "DONE").exists()


def run_jobs(jobs, workers):
    """jobs: (spec, split, record, seed, placebo, net). Unfinished ones run `workers` at a time, with one retry."""
    def one(j):
        s, split, r, seed, pl, net = j
        lf = OUT / "logs" / f"{X2.tag(s, net, split, r, seed, pl)}.log"; lf.parent.mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(ROOT / "scripts" / "xpinn" / "core2.py"), "--spec", s, "--net", net, "--split", split, "--record", str(r), "--seed", str(seed)]
        with open(lf, "a") as f:
            return subprocess.run(cmd + (["--placebo"] if pl else []), stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    todo = [j for j in jobs if not done(j[0], j[2], j[3], j[4], j[1], j[5])]
    for attempt in (1, 2):
        if not todo:
            break
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
        todo = [j for j in todo if not done(j[0], j[2], j[3], j[4], j[1], j[5])]
        if todo:
            log("failed", attempt=attempt, runs=[X2.tag(j[0], j[5], j[1], j[2], j[3], j[4]) for j in todo])
    return todo


# ------------------------------------------------------------------------------------------------ scoring
def load(path):
    if not (path / "DONE").exists():
        return None
    x = pd.read_csv(path / "predictions_eval.csv", dtype={"station": str})
    x = x[x.observed & x.y_pred_raw.notna()].copy(); x["anchor_date"] = x.anchor_date.astype(str).str[:10]
    return x[SC.KEY + ["target_date", "y_true_raw", "y_pred_raw", "base_raw"]]


def eligible(rec):
    for d in sorted((OUT / "runs").glob(f"*__{NET}_dev_r{rec}_s{SEED}")):
        if (d / "DONE").exists():
            c = json.loads((d / "config.json").read_text())["counts"]
            return [s for s, k in c.items() if k["chl_train"] >= 30 and k["chl_val"] >= 10 and min(k["eval_targets"]) >= SC.MIN_EVAL]
    return []


def table(specs, rec):
    frames = {s: f for s in specs if (f := load(OUT / "runs" / X2.tag(s, NET, "dev", rec, SEED))) is not None}
    frames["bb-C1"] = load(FIRST / "runs" / f"bb-C1__{NET}_dev_r{rec}_s{SEED}")
    truth = frames["bb-C1"]; frames["persistence"] = truth.assign(y_pred_raw=truth.base_raw)
    r = pd.read_csv(FIRST / "refs" / f"{NET}_dev_r{rec}.csv", dtype={"station": str}); r["anchor_date"] = r.anchor_date.astype(str).str[:10]
    for name, g in r.groupby("model"):
        frames[name] = g
    return SC.r2_table(frames, eligible(rec), truth)


def params(spec, rec):
    f = OUT / "runs" / X2.tag(spec, NET, "dev", rec, SEED) / "params_eval.csv"
    return pd.read_csv(f, dtype={"station": str}) if f.exists() else None


def explain(spec, stations):
    s = X2.Spec.parse(spec); names = s.params() + (["KN", "KP", "KI"] if s.kin == "learned" else []) + (["v"] if s.tr == "dist" else [])
    P = {r: params(spec, r) for r in RECS}
    if any(p is None for p in P.values()):
        return {}
    P = {r: p[p.station.isin(stations)] for r, p in P.items()}; allp = pd.concat(P.values(), ignore_index=True)
    out = {"n_daily": s.n_daily()}
    if s.res == "bounded":
        m, e = allp.d7_mech.abs().mean(), (allp.d7_full - allp.d7_mech).abs().mean(); out["X1"] = float(m / (m + e))
    else:
        out["X1"] = 1.0
    diffs, hits = [], {}
    for n in names:
        lo, hi = X2.RANGE[n]
        a, b = (P[r].groupby("station")[n].median() for r in RECS)
        diffs += list(((a - b).abs() / (hi - lo)).dropna().to_numpy())
        if n != "share":
            z = (allp[n].dropna() - lo) / (hi - lo); hits[n] = float(((z < 0.01) | (z > 0.99)).mean()) if len(z) else 0.0
    out["X2r"] = float(np.median(diffs)) if diffs else 0.0
    out["X3"] = max(hits.values()) if hits else 0.0; out["X3_worst"] = max(hits, key=hits.get) if hits else ""
    checks = []
    for col, need in (("loss", "loss"), ("w", "w")):
        if col in s.daily():
            for r in RECS:
                c = SC._within(P[r], col, "rain", +1, stations)
                if c:
                    checks.append((f"{col} up with rain ({r}m)", c))
    out["X4"] = float(np.mean([c["pass"] for _, c in checks])) if checks else np.nan
    out["X4_detail"] = "; ".join(f"{k}: rho {c['median_rho']:+.2f}, agree {c['agree']:.0%}" for k, c in checks)
    k = (allp.d7_full - allp.d7_nogro).abs().mean(); out["growth_share"] = float(k / (k + allp.d7_nogro.abs().mean())) if s.gro != "none" else 0.0
    t = (allp.d7_full - allp.d7_local).abs().mean(); out["transport_share"] = float(t / (t + allp.d7_local.abs().mean())) if s.tr != "none" else 0.0
    for n in names:                                                          # station medians, for reading the explanation
        out[f"med_{n}"] = float(allp.groupby("station")[n].median().median())
    out["explainable"] = bool(out["X1"] >= THR["X1"] and out["X2r"] <= THR["X2r"] and out["X3"] <= THR["X3"] and (np.isnan(out["X4"]) or out["X4"] >= THR["X4"] - 1e-9))
    return out


def score():
    names = {}
    for b, d in BATCHES.items():
        for lab, s in d["specs"].items():
            names.setdefault(s, (lab, b))
    specs = [s for s in names if all(done(s, r) for r in RECS)]
    rows, best_refs = {s: {"spec": s, "label": names[s][0], "batch": names[s][1]} for s in specs}, {}
    for r in RECS:
        T = table(specs, r); T.to_csv(OUT / f"station_r2_r{r}.csv", index=False, float_format="%.5g")
        acc, best = SC.accuracy(T, specs + ["bb-C1"], refs=REFS); best_refs[r] = best
        for _, a in acc.iterrows():
            if a.variant in rows:
                rows[a.variant].update({f"A_{r}": a.A, **{f"{c}_{r}": a[c] for c in ("r2_h1", "r2_h3", "r2_h7", "r2_h14", "dr2_h3", "dr2_h7", "dr2_h14", "mean_rank", "goal_ok")}})
        c1 = acc[acc.variant == "bb-C1"].iloc[0]; rows.setdefault("bb-C1", {"spec": "bb-C1", "label": "black box C1", "batch": "ref"}).update({f"A_{r}": c1.A, **{f"{c}_{r}": c1[c] for c in ("r2_h3", "r2_h7", "r2_h14", "mean_rank")}})
    st = eligible(RECS[0])
    for s in specs:
        rows[s].update(explain(s, st))
    lb = pd.DataFrame(rows.values()); lb["A"] = lb[[f"A_{r}" for r in RECS]].mean(1)
    lb["goal_both"] = lb[[f"goal_ok_{r}" for r in RECS]].fillna(False).astype(bool).all(1) if f"goal_ok_{RECS[0]}" in lb else False
    sib = {s: X2.Spec.parse(s) for s in specs}                                # kinetic ablation: the same design without growth
    A = dict(zip(lb.spec, lb.A))
    lb["kin_gain"] = [A[s] - A.get(replace(sib[s], gro="none").normal().name, np.nan) if s in sib and sib[s].gro != "none" else np.nan for s in lb.spec]
    nonet = lambda x: replace(x, par="station", res="none", tgt="mean" if x.tgt == "mnet" else x.tgt).normal().name   # network-free sibling
    lb["net_gain"] = [A[s] - A.get(nonet(sib[s]), np.nan) if s in sib and sib[s].n_daily() else np.nan for s in lb.spec]
    lb = lb.sort_values("A", ascending=False).reset_index(drop=True)
    lb.to_csv(OUT / "leaderboard.csv", index=False, float_format="%.5g")
    return lb, best_refs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "score"]); ap.add_argument("--batch", default="b1"); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "run":
        specs = list(dict.fromkeys(BATCHES[a.batch]["specs"].values()))
        log("batch start", batch=a.batch, why=BATCHES[a.batch]["why"], n_specs=len(specs), specs=BATCHES[a.batch]["specs"])
        failed = run_jobs([(s, "dev", r, SEED, False, NET) for s in specs for r in RECS], a.workers)
        log("batch runs done", batch=a.batch, failed=failed)
    lb, best = score()
    (OUT / "batches").mkdir(parents=True, exist_ok=True); lb.to_csv(OUT / "batches" / f"leaderboard_after_{a.batch}.csv", index=False, float_format="%.5g")
    top = lb[lb.spec != "bb-C1"].head(10)
    log("scored", after=a.batch, best_reference=best, top10=top[["label", "A", "explainable"]].to_dict("records"))
    (OUT / f"DONE_{a.batch}").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
