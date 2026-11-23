"""Finalists of the designs of core2.py. For at most three designs
chosen on the development year: seeds 42, 123 and 456 at 24 and 12 months and the kinetic placebo (driver trajectories
shuffled across days within each station, retrained), then the test years 2024-2025 once (Nakdong, records 36 and 12
months, against the national benchmark and bb-C1 of core.py, the joint network of the paper), and the other three basins
for every finalist without transport. Network gain: A of a finalist with a network minus A of its network-free sibling
when that sibling is also a finalist (seed ensembles). Outputs in OUT/final/<stage>/. Usage:
final2.py run|dev|test|spatial --finalists LABEL [LABEL ...] [--workers 12]"""
from __future__ import annotations
import argparse, json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "xpinn"))
import core2 as X2                                                               # noqa: E402
import explore as E                                                              # noqa: E402
import score as SC                                                               # noqa: E402

SEEDS, TEST_RECS, OTHER = (42, 123, 456), (36, 12), ("han", "geum", "yeongsan")
ST = X2.OUT / "final"; THR = {**E.THR, "X2": 0.10, "X5": 0.005}


def load_any(t):
    """A run of this family, else of the first search (the black box bb-C1)."""
    for base in (X2.OUT / "runs", SC.RUNS):
        if (base / t / "DONE").exists():
            x = pd.read_csv(base / t / "predictions_eval.csv", dtype={"station": str})
            x = x[x.observed & x.y_pred_raw.notna()].copy(); x["anchor_date"] = x.anchor_date.astype(str).str[:10]
            return x[SC.KEY + ["target_date", "y_true_raw", "y_pred_raw", "base_raw"]]
    return None


SC.load = load_any                                                               # the shared scorers read both run folders


def spec_of(label):
    for d in E.BATCHES.values():
        if label in d["specs"]:
            return d["specs"][label]
    raise KeyError(label)


def kinetic(spec):
    """Designs with a driver-dependent physical term (growth kinetics or the temperature-dependent relaxation rate)."""
    s = X2.Spec.parse(spec)
    return s.gro != "none" or s.kt == "yes"


def sibling(spec):
    """The same design without the network (station constants, no residual, the record-mean target for mnet)."""
    s = X2.Spec.parse(spec)
    return replace(s, par="station", res="none", tgt="mean" if s.tgt == "mnet" else s.tgt).normal().name


# ------------------------------------------------------------------------------------------------ development year
def dev_frames(specs, rec, placebo=False):
    frames = {s: SC.ensemble(s, E.NET, "dev", rec, SEEDS, placebo=placebo) for s in specs}
    frames = {s: f for s, f in frames.items() if f is not None}
    truth = SC.ensemble("bb-C1", E.NET, "dev", rec, SEEDS); frames["bb-C1"] = truth; frames["persistence"] = truth.assign(y_pred_raw=truth.base_raw)
    r = pd.read_csv(E.FIRST / "refs" / f"{E.NET}_dev_r{rec}.csv", dtype={"station": str}); r["anchor_date"] = r.anchor_date.astype(str).str[:10]
    for name, g in r.groupby("model"):
        frames[name] = g
    return frames, truth


def placebo_drop(spec, rec, stations, h=7):
    """Median over stations of R² (seed mean) minus R² of the placebo retraining (seed mean) at horizon h."""
    e, p = SC.ensemble(spec, E.NET, "dev", rec, SEEDS), SC.ensemble(spec, E.NET, "dev", rec, SEEDS, placebo=True)
    if e is None or p is None:
        return np.nan
    a = e[e.horizon == h][SC.KEY + ["y_true_raw", "y_pred_raw"]].merge(p[p.horizon == h][SC.KEY + ["y_pred_raw"]].rename(columns={"y_pred_raw": "p_pl"}), on=SC.KEY)
    d = []
    for st in stations:
        g = a[a.station == st]
        if len(g) >= 10:
            y = g.y_true_raw.to_numpy(float); q = np.full_like(y, y.mean())
            d.append(SC.S.metrics(y, g.y_pred_raw.to_numpy(float), q)["r2"] - SC.S.metrics(y, g.p_pl.to_numpy(float), q)["r2"])
    return float(np.median(d)) if d else np.nan


def explain3(spec, stations):
    """X1-X4 and record stability as in explore.explain, plus X2 seed stability over the three seeds."""
    s = X2.Spec.parse(spec); names = s.params() + (["KN", "KP", "KI"] if s.kin == "learned" else []) + (["v"] if s.tr == "dist" else [])
    P = {}
    for r in E.RECS:
        for sd in SEEDS:
            f = X2.OUT / "runs" / X2.tag(spec, E.NET, "dev", r, sd) / "params_eval.csv"
            if not f.exists():
                return {}
            p = pd.read_csv(f, dtype={"station": str}); P[(r, sd)] = p[p.station.isin(stations)]
    allp = pd.concat(P.values(), ignore_index=True); out = {"n_daily": s.n_daily()}
    if s.res == "bounded":
        m, e = allp.d7_mech.abs().mean(), (allp.d7_full - allp.d7_mech).abs().mean(); out["X1"] = float(m / (m + e))
    else:
        out["X1"] = 1.0
    sd_, rr, hits = [], [], {}
    for n in names:
        lo, hi = X2.RANGE[n]
        for r in E.RECS:
            meds = pd.concat([P[(r, sd)].groupby("station")[n].median().rename(sd) for sd in SEEDS], axis=1).dropna()
            sd_ += list((meds.std(axis=1, ddof=0) / (hi - lo)).to_numpy())
        a, b = (pd.concat([P[(r, sd)] for sd in SEEDS]).groupby("station")[n].median() for r in E.RECS)
        rr += list(((a - b).abs() / (hi - lo)).dropna().to_numpy())
        if n != "share":
            z = (allp[n].dropna() - lo) / (hi - lo); hits[n] = float(((z < 0.01) | (z > 0.99)).mean()) if len(z) else 0.0
    out["X2"] = float(np.median(sd_)) if sd_ else 0.0; out["X2r"] = float(np.median(rr)) if rr else 0.0
    out["X3"] = max(hits.values()) if hits else 0.0; out["X3_worst"] = max(hits, key=hits.get) if hits else ""
    checks = []
    for col in ("loss", "w"):
        if col in s.daily():
            for r in E.RECS:
                Pm = pd.concat([P[(r, sd)] for sd in SEEDS]).groupby(["station", "anchor_date"]).mean(numeric_only=True).reset_index()
                c = SC._within(Pm, col, "rain", +1, stations)
                if c:
                    checks.append((f"{col} up with rain ({r}m)", c))
    out["X4"] = float(np.mean([c["pass"] for _, c in checks])) if checks else np.nan
    out["X4_detail"] = "; ".join(f"{k}: rho {c['median_rho']:+.2f}, agree {c['agree']:.0%}" for k, c in checks)
    k = (allp.d7_full - allp.d7_nogro).abs().mean(); out["growth_share"] = float(k / (k + allp.d7_nogro.abs().mean())) if s.gro != "none" else 0.0
    t = (allp.d7_full - allp.d7_local).abs().mean(); out["transport_share"] = float(t / (t + allp.d7_local.abs().mean())) if s.tr != "none" else 0.0
    for n in names:
        out[f"med_{n}"] = float(allp.groupby("station")[n].median().median())
    out["screen"] = bool(out["X1"] >= THR["X1"] and out["X2"] <= THR["X2"] and out["X2r"] <= THR["X2r"] and out["X3"] <= THR["X3"]
                         and (np.isnan(out["X4"]) or out["X4"] >= THR["X4"] - 1e-9))
    return out


def dev(fin, labels):
    d = ST / "dev"; d.mkdir(parents=True, exist_ok=True); rows = {s: {"label": labels[s], "spec": s} for s in fin}
    stations = E.eligible(E.RECS[0])
    for r in E.RECS:
        frames, truth = dev_frames(fin, r); T = SC.r2_table(frames, E.eligible(r), truth); T.to_csv(d / f"station_r2_r{r}.csv", index=False, float_format="%.5g")
        acc, best = SC.accuracy(T, fin + ["bb-C1"], refs=E.REFS)
        for a in acc.itertuples():
            if a.variant in rows:
                rows[a.variant].update({f"A_{r}": a.A, f"goal_ok_{r}": a.goal_ok, **{f"dr2_h{h}_{r}": getattr(a, f"dr2_h{h}") for h in SC.FH},
                                        **{f"r2_h{h}_{r}": getattr(a, f"r2_h{h}") for h in (1,) + SC.FH}})
        c1 = acc[acc.variant == "bb-C1"].iloc[0]; rows.setdefault("bb-C1", {"label": "black box C1", "spec": "bb-C1"}).update({f"A_{r}": c1.A})
        for s in fin:
            if kinetic(s):
                rows[s].update({f"X5_{r}": placebo_drop(s, r, E.eligible(r)), **{f"X5_h{h}_{r}": placebo_drop(s, r, E.eligible(r), h) for h in (3, 14)}})
    for s in fin:
        rows[s].update(explain3(s, stations))
        x5 = [rows[s].get(f"X5_{r}", np.nan) for r in E.RECS]
        rows[s]["final"] = bool(rows[s].get("screen") and rows[s].get("goal_ok_24") and rows[s].get("goal_ok_12")
                                and (not kinetic(s) or all(np.isfinite(x) and x >= THR["X5"] for x in x5)))
    lb = pd.DataFrame(rows.values()); lb["A"] = lb[[f"A_{r}" for r in E.RECS]].mean(1)
    A = dict(zip(lb.spec, lb.A))
    lb["net_gain"] = [A[s] - A[sibling(s)] if s in fin and X2.Spec.parse(s).n_daily() and sibling(s) in A else np.nan for s in lb.spec]
    lb["goal_both"] = lb[[f"goal_ok_{r}" for r in E.RECS]].fillna(False).astype(bool).all(1)
    lb.sort_values("A", ascending=False).to_csv(d / "leaderboard.csv", index=False, float_format="%.5g")
    E.log("final dev", finalists=[labels[s] for s in fin], seeds=SEEDS, summary=lb[["label", "A", "goal_both"] + [c for c in ("screen", "final") if c in lb]].to_dict("records"))
    return lb


# ------------------------------------------------------------------------------------------------ test years and other basins
def test(fin, labels, workers):
    d = ST / "test"; d.mkdir(parents=True, exist_ok=True)
    failed = E.run_jobs([(s, "test", r, sd, False, E.NET) for s in fin for r in TEST_RECS for sd in SEEDS], workers)
    Et = SC.test_table(fin + ["bb-C1"], E.NET, TEST_RECS, SEEDS); Et.to_csv(d / "station_metrics.csv", index=False, float_format="%.5g")
    cmp_, summ = SC.test_compare(Et, fin, TEST_RECS)
    cmp_.to_csv(d / "comparisons.csv", index=False, float_format="%.5g"); summ.to_csv(d / "summary_r2.csv", index=False, float_format="%.5g")
    gap = {labels[s]: {f"{int(x.record)}m_h{int(x.horizon)}": round(float(x.median_diff), 4) for x in cmp_[(cmp_.focus == s) & cmp_.is_best_national].itertuples()} for s in fin}
    ok = {k: bool(all(v.get(f"36m_h{h}", -1) >= THR["goal"] for h in SC.FH)) for k, v in gap.items()}
    E.log("final test", finalists=[labels[s] for s in fin], failed=failed, gap_to_best_national=gap, success_36m=ok)
    return gap, ok


def spatial(spec, label, workers):
    d = ST / "spatial" / label; d.mkdir(parents=True, exist_ok=True); w = replace(X2.Spec.parse(spec), tr="none").normal().name; out = {"design": label, "spec": w}
    for net in OTHER:
        failed = E.run_jobs([(w, "test", r, sd, False, net) for r in TEST_RECS for sd in SEEDS], workers)
        Et = SC.test_table([w, "bb-C1"], net, TEST_RECS, SEEDS); Et.to_csv(d / f"station_metrics_{net}.csv", index=False, float_format="%.5g")
        cmp_, summ = SC.test_compare(Et, [w], TEST_RECS)
        cmp_.to_csv(d / f"comparisons_{net}.csv", index=False, float_format="%.5g"); summ.to_csv(d / f"summary_r2_{net}.csv", index=False, float_format="%.5g")
        out[net] = {"failed": failed, **{f"{int(x.record)}m_h{int(x.horizon)}": round(float(x.median_diff), 4) for x in cmp_[cmp_.is_best_national].itertuples()}}
    E.log("final spatial", **out)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["run", "dev", "test", "spatial"]); ap.add_argument("--finalists", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=12); a = ap.parse_args()
    fin = [spec_of(x) for x in a.finalists]; labels = dict(zip(fin, a.finalists))
    if a.stage == "run":
        jobs = [(s, "dev", r, sd, pl, E.NET) for s in fin for r in E.RECS for sd in SEEDS for pl in ((False, True) if kinetic(s) else (False,))]
        E.log("final start", finalists=a.finalists, seeds=SEEDS, n_runs=len(jobs))
        E.log("final runs done", failed=E.run_jobs(jobs, a.workers))
        print(dev(fin, labels).to_string())
    elif a.stage == "dev":
        print(dev(fin, labels).to_string())
    elif a.stage == "test":
        print(json.dumps(test(fin, labels, a.workers), indent=1))
    else:
        print(json.dumps([spatial(s, labels[s], a.workers) for s in fin], indent=1))


if __name__ == "__main__":
    main()
