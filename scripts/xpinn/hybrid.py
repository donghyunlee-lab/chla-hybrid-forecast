"""Stage 4 of the design of the process equation (designs of core2.py): the hybrid model of the paper, one model redesigned from the strengths of the three best model types, a hybrid of the
seasonal-relaxation process core (the strength of the damped anomaly), a state filter (the error memory of ARIMA) and a network
trained on the other basins (the cross-station information of XGBoost and the transfer networks). Runs are repeated over seeds
for standard deviations and checked for reproducibility. Rules: RULE below.

Leave one basin out: for a target basin the network and the starting values of the station constants come from joint training on
the other three basins (records ending before the evaluation period: 24 months for the development year, 36 for the test years).
At the target the network is frozen; the station constants and the gate are fitted on the target's record (36 or 12 months).
Step 1 (development year 2023, seed 42): cores K0 (stage 3 P-mcf-all-kT), K1 (state filter), K2 and K3 (the same two with
starting values from the other basins). Step 2: on the chosen core, a network that adds a bounded daily forcing (H-res), moves the
station constants (H-mod) or both (H-modres). Step 3: test years 2024-2025 with every seed for K0, the chosen core and the chosen
hybrid; per-seed and seed-ensemble metrics against the national benchmark. Step 4: term contributions and a reproducibility check.
Outputs in OUT/hybrid/, pre-training in OUT/pretrain/, runs in OUT/runs/.
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Names: bb-C1 and the other bb-* models are the joint network of the paper (one xLSTM trained on the stations together,
without process terms; 'bb' for black box); core (P-mcf-all-kT) is the process model and hybrid (H-res-all) the hybrid
model.
Usage: hybrid.py all [--workers 16] | pretrain --spec S --leave NET --split dev|test --seed N | fit --spec S --net NET --split
dev|test --record R --seed N [--suffix X] | score"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, platform, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
for _p in ("clean_revision", "national", "xpinn"):
    sys.path.insert(0, str(ROOT / "scripts" / _p))
import core as K1                                                                # noqa: E402
import core2 as X2                                                               # noqa: E402
import explore as E                                                              # noqa: E402
import final2 as F                                                               # noqa: E402  (lets the shared scorers read this family's runs)
import score as SC                                                               # noqa: E402
import seasonal as SE                                                            # noqa: E402
import nat as N                                                                  # noqa: E402
import stats_nat as S                                                            # noqa: E402

HY, PRE, NETS = X2.OUT / "hybrid", X2.OUT / "pretrain", SE.NETS
SEEDS = tuple(int(x) for x in os.environ.get("HYBRID_SEEDS", "42,123,456,1,2").split(","))   # the seeds of the national benchmark
DEV_SEED, RECS, TEST_RECS, SRC_REC, LRS = E.SEED, E.RECS, F.TEST_RECS, {"dev": 24, "test": 36}, (1e-3, 3e-4, 1e-4)
RULE = {"core_margin": 0.003, "net_gain": 0.005}
KEY = ["station", "anchor_date", "horizon"]
BASE = dict(tgt="mcf", par="station", flush="yes", kt="yes")


def sp(**kw) -> str:
    return X2.Spec(**kw).normal().name


CORES = {"P-mcf-all-kT": sp(anc="smooth", age="yes", **BASE), "P-mcf-filter-kT": sp(anc="filter", **BASE),
         "P-mcf-all-kT-pre": sp(anc="smooth", age="yes", pre="nat", **BASE), "P-mcf-filter-kT-pre": sp(anc="filter", pre="nat", **BASE)}


def hybrids(anchor: str) -> dict:
    """Network designs on the core with this anchor (all = smoothing and ageing, filter = state filter)."""
    a = dict(anc="smooth", age="yes") if anchor == "all" else dict(anc="filter")
    b = {**BASE, **a, "pre": "nat", "gate": "yes"}
    return {f"H-res-{anchor}": sp(**{**b, "res": "bounded"}), f"H-mod-{anchor}": sp(**{**b, "par": "mod"}),
            f"H-modres-{anchor}": sp(**{**b, "par": "mod", "res": "bounded"})}


LABELS = {**CORES, **hybrids("all"), **hybrids("filter")}
NAME = {**{v: k for k, v in LABELS.items()}, "bb-C1": "black box C1"}
WHY = ("Stage 4 builds one model from three parts: the seasonal-relaxation core, a state filter in place of smoothing"
       " and ageing, and a network trained on the other three basins that adds a bounded forcing or moves the station"
       " constants, with the constants starting from the other basins' medians.")


def pre_tag(spec, leave, split, seed):
    return f"{spec}__lo-{leave}_{split}_r{SRC_REC[split]}_s{seed}"


def device():
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(n), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ training
def step_loss(net, D, j, s, dev, hw, csd):
    p, aux = net(X2.batch(D, j, s, dev))
    if s.fit == "log":
        loss = K1.wmse(p, torch.as_tensor(D["ylog"][j], device=dev), torch.as_tensor(D["m_tr"][j], device=dev), hw)
    else:
        a, b_ = X2.criterion(p, D, j, D["m_tr"], s, dev, hw, csd); loss = a / b_.clamp_min(1.0)
    if aux is not None and aux["eps"] is not None:
        loss = loss + X2.EPS_PEN * (aux["eps"] ** 2).mean()
    return loss


def joint_train(s, Ds, lr, seed, dev, epochs, patience=15, bs=32):
    """One model per source basin with its own station constants; the network and the gate are shared. Early stopping on the
    mean validation loss of the basins."""
    K1.seed_all(seed); nets = {}
    for n, D in Ds.items():
        net = X2.XNet2(s, D["X"].shape[-1], D["stations"], D["up"], D["dist"], D["drec"]).to(dev)
        if nets:
            f0 = next(iter(nets.values())); net.bb, net.gate = f0.bb, f0.gate
        nets[n] = net
    f0 = next(iter(nets.values()))
    shared = (list(f0.bb.parameters()) if f0.bb is not None else []) + ([f0.gate] if f0.gate is not None else [])
    consts = [p for net in nets.values() for p in net.constants()]
    opt = torch.optim.Adam(([{"params": shared, "lr": lr, "weight_decay": 1e-4}] if shared else []) +
                           [{"params": consts, "lr": min(10 * lr, 1e-2), "weight_decay": 0.0}])
    Wt = {n: X2.weights(D, s, dev) for n, D in Ds.items()}; rng = np.random.RandomState(seed)
    tr = {n: D["tr"][D["m_tr"][D["tr"]].sum((1, 2)) > 0] for n, D in Ds.items()}

    def snap():
        return {"bb": copy.deepcopy(f0.bb.state_dict()) if f0.bb is not None else None,
                "gate": f0.gate.detach().clone() if f0.gate is not None else None, "zs": {n: net.zs.detach().clone() for n, net in nets.items()}}
    best, state, bad, ep_best = np.inf, snap(), 0, -1
    for ep in range(epochs):
        for net in nets.values():
            net.train()
        batches = []
        for n in Ds:
            perm = rng.permutation(tr[n]); batches += [(n, perm[i:i + bs]) for i in range(0, len(perm), bs)]
        for bi in rng.permutation(len(batches)):
            n, j = batches[bi]; loss = step_loss(nets[n], Ds[n], j, s, dev, *Wt[n])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(shared + consts, 5.0); opt.step()
        vl = float(np.mean([X2.val_loss(nets[n], Ds[n], Ds[n]["va"], Ds[n]["m_va"], s, dev, *Wt[n]) for n in Ds]))
        if vl < best - 1e-6:
            best, state, bad, ep_best = vl, snap(), 0, ep
        else:
            bad += 1
            if bad >= patience:
                break
    return state, best, ep_best, f0.cn, {n: net.sid.tolist() for n, net in nets.items()}


def pretrain(spec, leave, split, seed, epochs, suffix=""):
    s = X2.Spec.parse(spec); d = PRE / (pre_tag(s.name, leave, split, seed) + suffix); d.mkdir(parents=True, exist_ok=True); dev = device(); t0 = time.time()
    if (d / "DONE").exists():
        print("skip", d.name); return
    Ds = {n: X2.prepare(n, split, SRC_REC[split]) for n in NETS if n != leave}; runs = []
    for lr in LRS:
        state, best, ep, cn, sids = joint_train(s, Ds, lr, seed, dev, epochs); runs.append((lr, state, best, ep))
        print(f"[pre {d.name}] lr {lr:g} val {best:.4f} epoch {ep}", flush=True)
    lr, state, best, ep = min(runs, key=lambda r: r[2])
    z = torch.cat(list(state["zs"].values()), 0)                                # every source station's constants (logit scale)
    art = {**state, "cn": cn, "z_nat": z.median(0).values, "source_ids": sorted(i for v in sids.values() for i in v),
           "stations": {n: Ds[n]["stations"] for n in Ds}}
    torch.save(art, d / "state.pt")
    nat = {c: float(X2.bound(art["z_nat"][k], c)) for k, c in enumerate(cn)}
    json.dump({"spec": s.name, "leave_out": leave, "split": split, "source_record_months": SRC_REC[split], "seed": seed, "lr_selected": lr, "best_val": best,
               "best_epoch": ep, "lr_table": [{"lr": r[0], "best_val": r[2], "best_epoch": r[3]} for r in runs], "constants": cn, "national_median": nat,
               "gate": state["gate"].tolist() if state["gate"] is not None else None, "sources": art["stations"], "state_sha256": sha(d / "state.pt"),
               "time_s": time.time() - t0, "torch": torch.__version__}, open(d / "config.json", "w"), indent=1, ensure_ascii=False)
    (d / "DONE").write_text("ok"); print(f"[pre {d.name}] done {(time.time() - t0) / 60:.1f} min, lr {lr:g}", flush=True)


def target_train(s, D, art, lr, seed, dev, epochs, patience=15, bs=32):
    """Station constants (starting from the other basins' medians) and the gate are fitted on the target's record; the network
    is frozen, with the mean station embedding of the source stations."""
    K1.seed_all(seed); net = X2.XNet2(s, D["X"].shape[-1], D["stations"], D["up"], D["dist"], D["drec"]).to(dev); hw, csd = X2.weights(D, s, dev)
    assert net.cn == art["cn"], (net.cn, art["cn"])
    with torch.no_grad():
        if net.bb is not None:
            net.bb.load_state_dict(art["bb"]); emb = getattr(net.bb, "station_embedding", None)
            if emb is not None:
                emb.weight[net.sid] = emb.weight[torch.as_tensor(art["source_ids"], device=dev)].mean(0)
            for p in net.bb.parameters():
                p.requires_grad = False
        if net.gate is not None:
            net.gate.copy_(art["gate"].to(dev))
        net.zs.copy_(art["z_nat"].to(dev).expand_as(net.zs))
    opt = torch.optim.Adam(([{"params": [net.gate], "lr": lr, "weight_decay": 1e-4}] if net.gate is not None else []) +
                           [{"params": net.constants(), "lr": min(10 * lr, 1e-2), "weight_decay": 0.0}])
    rng = np.random.RandomState(seed); tr = D["tr"][D["m_tr"][D["tr"]].sum((1, 2)) > 0]
    best, best_state, bad, ep_best = np.inf, copy.deepcopy(net.state_dict()), 0, -1
    for ep in range(epochs):
        net.train()
        if net.bb is not None:
            net.bb.eval()                                                      # frozen network: no dropout
        perm = rng.permutation(tr)
        for i in range(0, len(perm), bs):
            loss = step_loss(net, D, perm[i:i + bs], s, dev, hw, csd)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 5.0); opt.step()
        vl = X2.val_loss(net, D, D["va"], D["m_va"], s, dev, hw, csd)
        if vl < best - 1e-6:
            best, best_state, bad, ep_best = vl, copy.deepcopy(net.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, best, ep_best


def fit(spec, net_name, split, rec, seed, epochs, suffix=""):
    s = X2.Spec.parse(spec); assert s.pre == "nat", "designs without national pre-training run through core2.py"
    rd = X2.OUT / "runs" / (X2.tag(s.name, net_name, split, rec, seed) + suffix); rd.mkdir(parents=True, exist_ok=True); dev = device(); t0 = time.time()
    if (rd / "DONE").exists():
        print("skip", rd.name); return
    pd_ = PRE / pre_tag(s.name, net_name, split, seed); art = torch.load(pd_ / "state.pt", map_location=dev)
    D = X2.prepare(net_name, split, rec); runs = []
    for lr in LRS:
        net, best, ep = target_train(s, D, art, lr, seed, dev, epochs); runs.append((lr, net, best, ep))
        print(f"[{rd.name}] lr {lr:g} val {best:.4f} epoch {ep}", flush=True)
    lr, net, best, ep = min(runs, key=lambda r: r[2])
    torch.save(net.state_dict(), rd / "model.pt")
    pred, par = X2.frames(D, s, X2.evaluate(net, D, s, dev))
    pred.to_csv(rd / "predictions_eval.csv", index=False, float_format="%.6g")
    if par is not None:
        par.to_csv(rd / "params_eval.csv", index=False, float_format="%.6g")
    json.dump({"spec": s.name, "placebo": False, "net": net_name, "split": split, "record_months": rec, "seed": seed, "lr_selected": lr, "best_val": best,
               "best_epoch": ep, "lr_table": [{"lr": r[0], "best_val": r[2], "best_epoch": r[3]} for r in runs], "record": D["record"], "counts": D["counts"],
               "seasonal_weights": D["sea"], "pretrain": pd_.name, "pretrain_state_sha256": sha(pd_ / "state.pt"),
               "gate": net.gate.tolist() if net.gate is not None else None, "time_s": time.time() - t0, "stations": D["stations"], "torch": torch.__version__},
              open(rd / "config.json", "w"), indent=1, ensure_ascii=False)
    (rd / "DONE").write_text("ok"); print(f"[{rd.name}] done {(time.time() - t0) / 60:.1f} min, lr {lr:g}", flush=True)


# ------------------------------------------------------------------------------------------------ jobs
def run_cmds(cmds, workers):
    """cmds: (DONE path, arguments of this script). Unfinished ones run `workers` at a time, with one retry."""
    def one(c):
        lf = X2.OUT / "logs" / f"hybrid_{c[0].parent.name}.log"; lf.parent.mkdir(parents=True, exist_ok=True)
        with open(lf, "a") as f:
            return subprocess.run([E.PY, str(Path(__file__).resolve())] + c[1], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    todo = [c for c in cmds if not c[0].exists()]
    for attempt in (1, 2):
        if not todo:
            break
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
        todo = [c for c in todo if not c[0].exists()]
        if todo:
            E.log("failed", attempt=attempt, runs=[c[0].parent.name for c in todo])
    return [c[0].parent.name for c in todo]


def pre_cmd(spec, leave, split, seed, suffix=""):
    return (PRE / (pre_tag(spec, leave, split, seed) + suffix) / "DONE",
            ["pretrain", "--spec", spec, "--leave", leave, "--split", split, "--seed", str(seed)] + (["--suffix", suffix] if suffix else []))


def fit_cmd(spec, net, split, rec, seed, suffix=""):
    return (X2.OUT / "runs" / (X2.tag(spec, net, split, rec, seed) + suffix) / "DONE",
            ["fit", "--spec", spec, "--net", net, "--split", split, "--record", str(rec), "--seed", str(seed)] + (["--suffix", suffix] if suffix else []))


def run_designs(specs, split, recs, seeds, workers, extra=()):
    """Every run of these designs: pre-training on the other basins first for the designs that need it, then the target runs."""
    pre = [s for s in specs if X2.Spec.parse(s).pre == "nat"]; plain = [s for s in specs if s not in pre]
    failed = run_cmds([pre_cmd(s, n, split, sd) for s in pre for n in NETS for sd in seeds] + list(extra), workers)
    failed += run_cmds([fit_cmd(s, n, split, r, sd) for s in pre for n in NETS for r in recs for sd in seeds], workers)
    failed += [X2.tag(*j[:1], j[5], j[1], j[2], j[3]) for j in E.run_jobs([(s, split, r, sd, False, n) for s in plain for n in NETS for r in recs for sd in seeds], workers)]
    return failed


# ------------------------------------------------------------------------------------------------ development year
def eligible(net, rec):
    for d in sorted((X2.OUT / "runs").glob(f"*__{net}_dev_r{rec}_s{DEV_SEED}")):
        if (d / "DONE").exists() and (d / "config.json").exists():
            c = json.loads((d / "config.json").read_text())["counts"]
            return [s for s, k in c.items() if k["chl_train"] >= 30 and k["chl_val"] >= 10 and min(k["eval_targets"]) >= SC.MIN_EVAL]
    return []


def dev_refs(net, rec, base, workers=12):
    """Damped anomaly, ARIMA and harmonic ARIMA of the development year (as in stage 3, same cache)."""
    f = E.FIRST / "refs" / f"{net}_dev_r{rec}.csv" if net == E.NET else SE.ST / "refs" / f"{net}_dev_r{rec}.csv"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        tasks = [(st, rec, base[base.station == st][["station", "anchor_date", "horizon", "target_date", "y_true_raw"]]) for st in sorted(base.station.unique())]
        with Pool(min(workers, len(tasks))) as pool:
            r = pd.concat(pool.map(SC._ref_task, tasks), ignore_index=True)
        r["anchor_date"] = r.anchor_date.astype(str).str[:10]
        r[r.observed.astype(bool)][SC.KEY + ["y_pred_raw", "model"]].to_csv(f, index=False, float_format="%.6g")
    r = pd.read_csv(f, dtype={"station": str}); r["anchor_date"] = r.anchor_date.astype(str).str[:10]
    return r


def dev_table(specs, net, rec):
    frames = {s: f for s in specs if (f := F.load_any(X2.tag(s, net, "dev", rec, DEV_SEED))) is not None}
    truth = F.load_any(X2.tag("bb-C1", net, "dev", rec, DEV_SEED)); frames["bb-C1"] = truth; frames["persistence"] = truth.assign(y_pred_raw=truth.base_raw)
    for name, g in dev_refs(net, rec, truth).groupby("model"):
        frames[name] = g
    return SC.r2_table(frames, eligible(net, rec), truth)


def x1(spec, net, stations):
    """Share of the 7-day change that comes from the process terms (1 without a residual)."""
    if X2.Spec.parse(spec).res != "bounded":
        return 1.0
    p = pd.concat([pd.read_csv(X2.OUT / "runs" / X2.tag(spec, net, "dev", r, DEV_SEED) / "params_eval.csv", dtype={"station": str}) for r in RECS])
    p = p[p.station.isin(stations)]; m, e = p.d7_mech.abs().mean(), (p.d7_full - p.d7_mech).abs().mean()
    return float(m / (m + e))


def dev_board(specs, name):
    """A per basin and record against the best reference of that basin, record and horizon (as in stages 2-3), the mean over the
    eight cells, the goal, the screen (X2r, X3) in every basin and the smallest X1."""
    d = HY / "dev"; d.mkdir(parents=True, exist_ok=True)
    rows = {s: {"label": NAME.get(s, s), "spec": s} for s in specs + ["bb-C1"]}; best = {}
    for net in NETS:
        st = eligible(net, RECS[0])
        for r in RECS:
            T = dev_table(specs, net, r); T.to_csv(d / f"station_r2_{name}_{net}_r{r}.csv", index=False, float_format="%.5g")
            acc, b = SC.accuracy(T, specs + ["bb-C1"], refs=E.REFS); best[f"{net}_{r}"] = b
            for a in acc.itertuples():
                rows[a.variant].update({f"A_{net}_{r}": a.A, f"goal_{net}_{r}": a.goal_ok, **{f"d{h}_{net}_{r}": getattr(a, f"dr2_h{h}") for h in SC.FH}})
        for s in specs:
            rows[s].update({f"{k}_{net}": v for k, v in SE.explain(s, net, st).items()}); rows[s][f"X1_{net}"] = x1(s, net, st)
    lb = pd.DataFrame(rows.values()); cells = [f"A_{n}_{r}" for n in NETS for r in RECS]
    lb["A"] = lb[cells].mean(axis=1)
    for r in RECS:
        lb[f"A_{r}"] = lb[[f"A_{n}_{r}" for n in NETS]].mean(axis=1)
    lb["goal_cells"] = lb[[f"goal_{n}_{r}" for n in NETS for r in RECS]].fillna(False).astype(bool).sum(axis=1)
    lb["screen"] = lb[[f"screen_{n}" for n in NETS]].fillna(False).astype(bool).all(axis=1)
    lb["X1_min"] = lb[[f"X1_{n}" for n in NETS]].min(axis=1)
    lb = lb.sort_values("A", ascending=False).reset_index(drop=True); lb.to_csv(HY / f"leaderboard_{name}.csv", index=False, float_format="%.5g")
    return lb, best


def choose_core(lb):
    """Rule 1: the core with the highest mean A over the eight development cells; a core other than K0 (stage 3) must exceed K0 by
    at least RULE core_margin, otherwise K0 stays."""
    c = lb[lb.spec.isin(list(CORES.values()))].set_index("label"); k0 = "P-mcf-all-kT"; top = c.A.idxmax()
    return top if top != k0 and c.A[top] - c.A[k0] >= RULE["core_margin"] else k0


def choose_hybrid(lb, cands, core):
    """Rule 2: the network design with the highest mean A. Verdict 'the network helps': its A is at least RULE net_gain above
    the chosen core."""
    c = lb[lb.label.isin(list(cands))].set_index("label"); top = c.A.idxmax()
    return top, bool(c.A[top] - lb.set_index("label").A[core] >= RULE["net_gain"])


# ------------------------------------------------------------------------------------------------ test years
def ens(frames):
    """Mean forecast of the given frames on their common forecasts (None unless all are present)."""
    fr = [f.drop_duplicates(KEY) for f in frames if f is not None and len(f)]
    if not fr or len(fr) != len(frames):
        return None
    e = fr[0][KEY + ["y_true_raw"]].copy()
    for i, f in enumerate(fr):
        e = e.merge(f[KEY + ["y_pred_raw"]].rename(columns={"y_pred_raw": f"p{i}"}), on=KEY)
    e["y_pred_raw"] = e[[f"p{i}" for i in range(len(fr))]].mean(axis=1)
    return e[KEY + ["y_true_raw", "y_pred_raw"]]


NATIONAL = {"damped anomaly": lambda c, r: [f"bl_damped_{c}_r{r}"], "harmonic ARIMA": lambda c, r: [f"bl_harmonic_{c}_r{r}"], "ARIMA": lambda c, r: [f"bl_arima_{c}_r{r}"],
            "XGBoost": lambda c, r: [f"bl_xgboost_{c}_r{r}_s{s}" for s in S.SEEDS], "XGBoost zero-shot": lambda c, r: [f"bl_xgbzs_{c}_s{s}" for s in S.SEEDS],
            "xLSTM fine-tuned": lambda c, r: [f"tgt_{c}_xLSTM_full_tune_r{r}_s{s}" for s in S.SEEDS],
            "PINN-sigmoid fine-tuned": lambda c, r: [f"tgt_{c}_PINN-sigmoid_full_tune_r{r}_s{s}" for s in S.SEEDS]}
SIX = ["damped anomaly", "harmonic ARIMA", "XGBoost", "P-mcf-all-kT", "XGBoost zero-shot", "xLSTM fine-tuned"]


def _test_station(a):
    net, code, designs, seeds = a
    per = pd.read_csv(N.OUT / "persistence.csv", dtype={"station": str}); per = per[per.station == code].copy()
    per["anchor_date"] = per.anchor_date.astype(str).str[:10]; rows = []
    for rec in TEST_RECS:
        fr = {("persistence", "ens"): per[KEY + ["y_true_raw", "y_pred_raw"]]}
        for name, tags in NATIONAL.items():
            fr[(name, "ens")] = ens([S.load(t) for t in tags(code, rec)])
        for lab, spec in designs.items():
            one = {sd: F.load_any(X2.tag(spec, net, "test", rec, sd)) for sd in seeds}
            one = {sd: (None if f is None else f[f.station == code][KEY + ["y_true_raw", "y_pred_raw"]]) for sd, f in one.items()}
            for sd, f in one.items():
                fr[(lab, str(sd))] = f
            fr[(lab, "ens")] = ens(list(one.values()))
        if any(f is None or not len(f) for f in fr.values()):
            return pd.DataFrame([{"station": code, "net": net, "record": rec, "missing": ";".join(f"{k[0]}:{k[1]}" for k, f in fr.items() if f is None or not len(f))}])
        fr[("mean of six", "ens")] = ens([fr[(k, "ens")] for k in SIX])
        fr[("mean of core and xLSTM", "ens")] = ens([fr[("P-mcf-all-kT", "ens")], fr[("xLSTM fine-tuned", "ens")]])
        for h in X2.H:
            keys = per[per.horizon == h][KEY + ["y_true_raw"]]
            for f in fr.values():
                keys = keys.merge(f[f.horizon == h][KEY], on=KEY)
            if len(keys) < 60:
                continue
            y = keys.y_true_raw.to_numpy(float); q = keys[KEY].merge(per[KEY + ["y_pred_raw"]], on=KEY, how="left").y_pred_raw.to_numpy(float)
            for (model, sd), f in fr.items():
                p = keys[KEY].merge(f[KEY + ["y_pred_raw"]], on=KEY, how="left").y_pred_raw.to_numpy(float)
                rows.append({"station": code, "net": net, "record": rec, "horizon": int(h), "model": model, "seed": sd, **S.metrics(y, p, q)})
    return pd.DataFrame(rows)


def score_test(designs, seeds, workers=12):
    """Per station, record and horizon on the forecasts every model shares: metrics of each seed and of the seed-mean forecast
    of the designs, of the national benchmark models, of the mean of six models and of the mean of the stage 3 core and the
    fine-tuned xLSTM. Summaries: station-median R2 (mean and SD over seeds), other metrics, paired differences."""
    d = HY / "test"; d.mkdir(parents=True, exist_ok=True)
    tasks = [(n, c, designs, seeds) for n in NETS for c in K1.NETWORKS[n][0] if all((c, r) in S.ELIG for r in TEST_RECS)]
    with Pool(min(workers, len(tasks))) as pool:
        M = pd.concat(pool.map(_test_station, tasks), ignore_index=True)
    if "missing" in M:
        M[M.missing.notna()].to_csv(d / "missing.csv", index=False); M = M[M.missing.isna()].drop(columns="missing")
    M.to_csv(d / "station_metrics.csv", index=False, float_format="%.5g")
    mets = ["r2", "rmse", "mae", "skill_vs_persistence", "csi15", "csi25"]
    med = M.groupby(["model", "seed", "record", "horizon"])[mets].median().reset_index()
    e = med[med.seed == "ens"].drop(columns="seed"); p = med[med.seed != "ens"]
    sd = p.groupby(["model", "record", "horizon"])[mets].agg(["mean", "std", "count"]); sd.columns = [f"{a}_seed_{b}" for a, b in sd.columns]
    summ = e.merge(sd.reset_index(), on=["model", "record", "horizon"], how="left"); summ.to_csv(d / "summary.csv", index=False, float_format="%.5g")
    refs = ["persistence", "damped anomaly", "harmonic ARIMA", "ARIMA", "XGBoost", "XGBoost zero-shot", "xLSTM fine-tuned", "PINN-sigmoid fine-tuned",
            "mean of six", "mean of core and xLSTM"] + list(designs)
    W = M.pivot_table(index=["station", "record", "horizon"], columns=["model", "seed"], values="r2"); rows = []
    for lab in designs:
        for ref in [r for r in refs if r != lab]:
            for (rec, h), g in W.groupby(level=["record", "horizon"]):
                dd = (g[(lab, "ens")] - g[(ref, "ens")]).dropna()
                per_seed = [float((g[(lab, str(s_))] - g[(ref, "ens")]).median()) for s_ in seeds if (lab, str(s_)) in g]
                rows.append({"design": lab, "against": ref, "record": rec, "horizon": h, "n_stations": len(dd), "median_diff": dd.median(), "share_better": (dd > 0).mean(),
                             "wilcoxon_p": stats.wilcoxon(dd).pvalue if (dd != 0).any() else 1.0, "seed_mean_of_median_diff": np.mean(per_seed) if per_seed else np.nan,
                             "seed_sd_of_median_diff": np.std(per_seed, ddof=1) if len(per_seed) > 1 else np.nan, "seeds_positive": int(np.sum(np.array(per_seed) > 0))})
    C = pd.DataFrame(rows); C.to_csv(d / "paired.csv", index=False, float_format="%.5g")
    bas = M[M.seed == "ens"].groupby(["net", "model", "record", "horizon"]).r2.median().reset_index(); bas.to_csv(d / "basin_median_r2.csv", index=False, float_format="%.5g")
    return summ, C


# ------------------------------------------------------------------------------------------------ reproducibility
def manifest(workers):
    """What is needed to repeat the stage: code hashes and a copy of the code, environment, data hashes, settings."""
    HY.mkdir(parents=True, exist_ok=True); code = HY / "code"; code.mkdir(exist_ok=True)
    files = sorted((ROOT / "scripts" / "xpinn").glob("*.py")) + [ROOT / "scripts" / "national" / f for f in ("nat.py", "stats_nat.py")] + \
            [ROOT / "scripts" / "clean_revision" / f for f in ("common.py", "models.py")]
    for f in files:
        if f.parent.name == "xpinn":
            shutil.copy2(f, code / f.name)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    data = {str(f.relative_to(ROOT)): sha(f) for f in sorted((ROOT / "data" / "national").glob("*.csv"))}
    m = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "command": "hybrid.py all", "workers": workers, "seeds": SEEDS, "development_seed": DEV_SEED,
         "learning_rates": LRS, "epochs": int(os.environ.get("XPINN_EPOCHS", 100)), "patience": 15, "batch_size": 32, "rule": RULE, "source_record_months": SRC_REC,
         "development_records": RECS, "test_records": TEST_RECS, "basins": NETS, "labels": LABELS,
         "code_sha256": {str(f.relative_to(ROOT)): sha(f) for f in files}, "data_sha256": data, "git_head": git("rev-parse", "HEAD"),
         "git_uncommitted_in_scripts_xpinn": git("status", "--porcelain", "scripts/xpinn").splitlines(),
         "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
         "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() or None,
         "numpy": np.__version__, "pandas": pd.__version__, "cudnn_deterministic": True,
         "note": "Seeds fix the initial weights and the batch order; GPU arithmetic may still differ in the last digits between runs (see repro_check.csv)."}
    (HY / "MANIFEST.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
    (HY / "labels.json").write_text(json.dumps(NAME, indent=1))


def repro_cmds(specs):
    """The repeated pre-training of the reproducibility check (run together with the pre-training of the test years)."""
    return [pre_cmd(s, NETS[min(1, len(NETS) - 1)], "test", SEEDS[0], "__repro") for s in specs if X2.Spec.parse(s).pre == "nat"]


def repro_check(specs, workers):
    """One pre-training and one target run per design repeated under another name (test years, 36 months, first seed, first
    basin after the Nakdong); largest difference between the two sets of weights and of forecasts."""
    net, rows = NETS[min(1, len(NETS) - 1)], []
    pre = [s for s in specs if X2.Spec.parse(s).pre == "nat"]
    for s in pre:
        a = PRE / pre_tag(s, net, "test", SEEDS[0]); b = a.with_name(a.name + "__repro")
        if (a / "DONE").exists() and (b / "DONE").exists():
            x, y = (torch.load(p / "state.pt", map_location="cpu") for p in (a, b)); dd = [float((x["z_nat"] - y["z_nat"]).abs().max())]
            dd += [float((x["bb"][k].float() - y["bb"][k].float()).abs().max()) for k in (x["bb"] or {})]
            cx, cy = (json.loads((p / "config.json").read_text()) for p in (a, b))
            rows.append({"design": NAME[s], "kind": "pre-training", "run": a.name, "n": len(dd), "max_abs_diff": max(dd), "mean_abs_diff": float(np.mean(dd)),
                         "identical": bool(max(dd) == 0 and cx["best_val"] == cy["best_val"] and cx["lr_selected"] == cy["lr_selected"])})
    run_cmds([fit_cmd(s, net, "test", TEST_RECS[0], SEEDS[0], "__repro") for s in pre], workers)
    for s in pre:
        a = X2.OUT / "runs" / X2.tag(s, net, "test", TEST_RECS[0], SEEDS[0]); b = a.with_name(a.name + "__repro")
        if (a / "DONE").exists() and (b / "DONE").exists():
            x, y = (pd.read_csv(p / "predictions_eval.csv") for p in (a, b))
            dd = (x.y_pred_raw - y.y_pred_raw).abs()
            rows.append({"design": NAME[s], "kind": "target run (forecasts, mg/m3)", "run": a.name, "n": len(x), "max_abs_diff": float(dd.max()),
                         "mean_abs_diff": float(dd.mean()), "identical": bool((dd.fillna(0) == 0).all())})
    pd.DataFrame(rows).to_csv(HY / "repro_check.csv", index=False)
    return rows


def explain_all(labels, workers=8):
    cmds = [[E.PY, str(ROOT / "scripts" / "xpinn" / "explain_terms.py"), "--label", lab, "--net", n, "--split", split, "--records", *map(str, recs), "--seeds", *map(str, seeds)]
            for lab in labels for n in NETS for split, recs, seeds in (("dev", RECS, (DEV_SEED,)), ("test", TEST_RECS, SEEDS))]
    def one(c):
        with open(HY / "explain.log", "a") as f:
            return subprocess.run(c, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    with ThreadPoolExecutor(workers) as ex:
        codes = list(ex.map(one, cmds))
    return [" ".join(c[2:9]) for c, k in zip(cmds, codes) if k]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["all", "pretrain", "fit", "score"]); ap.add_argument("--spec"); ap.add_argument("--leave")
    ap.add_argument("--net"); ap.add_argument("--split", default="dev"); ap.add_argument("--record", type=int); ap.add_argument("--seed", type=int, default=DEV_SEED)
    ap.add_argument("--suffix", default=""); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("XPINN_EPOCHS", 100))); a = ap.parse_args()
    if a.cmd == "pretrain":
        return pretrain(a.spec, a.leave, a.split, a.seed, a.epochs, a.suffix)
    if a.cmd == "fit":
        return fit(a.spec, a.net, a.split, a.record, a.seed, a.epochs, a.suffix)
    HY.mkdir(parents=True, exist_ok=True); cols = ["label", "A", "A_24", "A_12", "goal_cells", "screen", "X1_min"]
    if a.cmd == "all":
        manifest(a.workers); E.log("hybrid start", why=WHY, cores=CORES, rule=RULE, seeds=SEEDS, nets=NETS, source_record=SRC_REC)
        # step 1: cores on the development year
        failed = E.run_jobs([("bb-C1", "dev", r, DEV_SEED, False, n) for n in NETS if n != E.NET for r in RECS], a.workers)
        failed += run_designs(list(CORES.values()), "dev", RECS, (DEV_SEED,), a.workers)
        lb, best = dev_board(list(CORES.values()), "step1"); core = choose_core(lb)
        anchor = "filter" if "filter" in core else "all"; hy = hybrids(anchor)
        E.log("hybrid step1 done", failed=failed, core=core, best_reference=best, summary=lb[cols].round(4).to_dict("records"))
        # step 2: network designs on the chosen core's anchor
        failed = run_designs(list(hy.values()), "dev", RECS, (DEV_SEED,), a.workers)
        lb, _ = dev_board(list(CORES.values()) + list(hy.values()), "step2"); top, helps = choose_hybrid(lb, hy, core)
        (HY / "chosen.json").write_text(json.dumps({"core": core, "hybrid": top, "network_helps_on_development_year": helps}, indent=1))
        E.log("hybrid step2 done", failed=failed, hybrid=top, network_helps=helps, summary=lb[cols].round(4).to_dict("records"))
        # step 3: test years with every seed
        fin = list(dict.fromkeys(["P-mcf-all-kT", core, top]))
        failed = run_designs([LABELS[x] for x in fin], "test", TEST_RECS, SEEDS, a.workers, extra=repro_cmds([LABELS[x] for x in fin]))
        E.log("hybrid test runs done", designs=fin, failed=failed)
    ch = json.loads((HY / "chosen.json").read_text()); fin = list(dict.fromkeys(["P-mcf-all-kT", ch["core"], ch["hybrid"]]))
    summ, C = score_test({x: LABELS[x] for x in fin}, SEEDS, min(12, a.workers))
    key = C[C.against.isin(["damped anomaly", "xLSTM fine-tuned", "mean of six", "P-mcf-all-kT"]) & C.horizon.isin([3, 7, 14])]
    E.log("hybrid test scored", designs=fin, paired=key[["design", "against", "record", "horizon", "median_diff", "share_better", "seed_sd_of_median_diff"]].round(4).to_dict("records"))
    if a.cmd == "all":
        rep = repro_check([LABELS[x] for x in fin], a.workers); bad = explain_all([x for x in fin if x != "P-mcf-all-kT"])
        E.log("hybrid checks", reproducibility=rep, explain_failed=bad)
        (X2.OUT / "DONE_hybrid").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
