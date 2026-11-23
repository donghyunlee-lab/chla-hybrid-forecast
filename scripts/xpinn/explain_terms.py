"""What each named term of a trained design contributes (designs of core2.py).
For each saved run (record × seed) the model is reloaded and the evaluation forecasts are recomputed with one term switched
off at a time: relaxation (κ = 0), rain flushing, ageing of the last measurement, anchor smoothing, the temperature
dependence of κ, growth, transport, and the network (station constants unmoved, record-mean target). Contribution = median
over stations of R²(full) − R²(term off) on the raw scale at 3, 7 and 14 days (the seeds' forecasts averaged first). Stage 3 adds the seasonal target (back to the record mean).
For parameters that change from day to day: Spearman correlation with the drivers on the forecast day, per station, median.
For designs with the driver network (mlp, kmlp, emlp): the response of each network-given parameter to one driver at a time
(5th-95th percentile of the training days, the other drivers at their medians), station median.
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Outputs in OUT/explain/<label>/ (other basins or the test years in a subfolder <net>_<split>). Usage: explain_terms.py --label P-mean-all [--seeds 42 123 456] [--split dev] [--net nakdong]"""
from __future__ import annotations
import argparse, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "xpinn"))
import core2 as X2                                                               # noqa: E402
import explore as E                                                              # noqa: E402

FEATS = ("water temperature", "log N", "log P", "log light", "log rain")


def spec_of(label):
    for d in E.BATCHES.values():
        if label in d["specs"]:
            return d["specs"][label]
    import seasonal as SE                                                        # stage 3 designs
    if label in SE.LABELS:
        return SE.LABELS[label]
    import hybrid as HB                                                          # stage 4 designs
    return HB.LABELS[label]


def switches(s):
    """Named terms of the design and the spec (or network change) that switches each off."""
    out = {}
    if s.tgt != "none":
        out["relaxation"] = "kappa0"
    if s.flush == "yes":
        out["rain flushing"] = replace(s, flush="no")
    if s.age == "yes":
        out["ageing"] = replace(s, age="no")
    if s.anc == "smooth":
        out["anchor smoothing"] = replace(s, anc="last")
    if s.anc == "filter":                                                    # stage 4: back to the last value aged since its measurement
        out["state filter"] = replace(s, anc="last", age="yes")
    if s.kt == "yes":
        out["temperature dependence of κ"] = replace(s, kt="no")
    if s.tgt in ("clim", "msy", "meb", "mcf"):                               # stage 3: back to the record mean
        out["seasonal target"] = replace(s, tgt="mean")
    if s.gro != "none":
        out["growth"] = "nogrowth"
    if s.tr != "none":
        out["transport"] = "notransport"
    if s.modulated() or s.tgt == "mnet" or s.res == "bounded":             # designs whose network moves station constants
        out["network"] = "nonet"
    return out


def forecast(net, D, s, dev, how=None, bs=128):
    """Log1p forecasts at the evaluation anchors (n, S, 4) with one term switched off (how) or none, and the parameters."""
    ev = D["ev"]; out, par = [], []
    keep_s, keep_mn = net.s, net.mn
    if isinstance(how, X2.Spec):
        net.s = how
    elif how == "nonet":
        net.mn = []
        if s.tgt == "mnet":
            net.s = replace(s, tgt="mean")
    net.eval()
    with torch.no_grad():
        for i in range(0, len(ev), bs):
            b = X2.batch(D, ev[i:i + bs], s, dev)
            if how == "kappa0":
                orig = net.physical
                net.physical = lambda b_, f=orig: (lambda P, e, k: ({**P, "kappa": torch.zeros_like(P["kappa"])}, e, k))(*f(b_))
            p, aux = net(b, transport=how != "notransport", growth=how != "nogrowth", resid=how != "nonet")
            if how == "kappa0":
                net.physical = orig
            out.append(p.cpu().numpy())
            par.append({n: t.cpu().numpy() for n, t in aux["P"].items() if t.dim() == 2})
    net.s, net.mn = keep_s, keep_mn
    return np.concatenate(out), {n: np.concatenate([q[n] for q in par]) for n in par[0]}


def r2(D, F):
    """Raw-scale R² per station and horizon on the observed evaluation targets (S, 4)."""
    ev = D["ev"]; y, m = D["yraw"][ev], D["m_ev"][ev] > 0; p = np.expm1(F); out = np.full(y.shape[1:], np.nan)
    for i in range(y.shape[1]):
        for k in range(y.shape[2]):
            mm = m[:, i, k]
            if mm.sum() >= 10:
                yy, pp = y[mm, i, k], p[mm, i, k]; out[i, k] = 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--label", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--split", default="dev"); ap.add_argument("--net", default=E.NET); ap.add_argument("--records", type=int, nargs="+", default=list(E.RECS))
    a = ap.parse_args(); spec = spec_of(a.label); s = X2.Spec.parse(spec); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    od = X2.OUT / "explain" / a.label / ("" if (a.net, a.split) == (E.NET, "dev") else f"{a.net}_{a.split}"); od.mkdir(parents=True, exist_ok=True); sw = switches(s); rows, cors, curves, pars = [], [], [], []
    for rec in a.records:
        D = X2.prepare(a.net, a.split, rec); st = D["stations"]; F = {k: [] for k in ["full", *sw]}; P0 = []
        for seed in a.seeds:
            rd = X2.OUT / "runs" / X2.tag(spec, a.net, a.split, rec, seed)
            net = X2.XNet2(s, D["X"].shape[-1], st, D["up"], D["dist"], D["drec"]).to(dev)
            missing, unexpected = net.load_state_dict(torch.load(rd / "model.pt", map_location=dev), strict=False)
            assert set(missing) <= {"fmu", "fsd"} and not unexpected, (missing, unexpected)   # runs saved before the driver network
            f, P = forecast(net, D, s, dev); F["full"].append(f); P0.append(P)
            for name, how in sw.items():
                F[name].append(forecast(net, D, s, dev, how)[0])
            if net.mlp is not None:                                           # one driver at a time, the others at their medians
                bt = X2.batch(D, D["tr"], s, dev)
                with torch.no_grad():
                    f0 = X2.XNet2.drivers(bt).reshape(-1, 5); med = f0.median(0).values; q = torch.quantile(f0, torch.tensor([0.05, 0.95], device=dev), 0)
                    for c, fname in enumerate(FEATS):
                        g = torch.linspace(float(q[0, c]), float(q[1, c]), 21, device=dev); x = med.repeat(21, 1); x[:, c] = g
                        z = net.mlp((x - net.fmu) / net.fsd)
                        z = z * net.gate if net.gate is not None else z           # as in XNet2.physical
                        for j, n in enumerate(net.dn + net.mn):
                            for si, stn in enumerate(st):
                                if n in net.mn:
                                    v = X2.bound(net.zs[si, net.cn.index(n)] + X2.MOD_C * torch.tanh(z[:, len(net.dn) + net.mn.index(n)]), n)
                                else:
                                    v = X2.bound(z[:, j], n)
                                curves += [{"record": rec, "seed": seed, "station": stn, "param": n, "driver": fname, "x": float(xx), "value": float(vv)}
                                           for xx, vv in zip(g.cpu().numpy(), v.cpu().numpy())]
        Fm = {k: np.mean(v, 0) for k, v in F.items()}; base = r2(D, Fm["full"])
        for name in sw:
            d = base - r2(D, Fm[name])
            rows += [{"record": rec, "term": name, "horizon": h, "median_r2_loss": float(np.nanmedian(d[:, k])),
                      "share_stations_lose": float(np.nanmean(d[:, k] > 0)), "median_r2_full": float(np.nanmedian(base[:, k]))}
                     for k, h in enumerate(X2.H) if h != 1]
        Pm = {n: np.mean([p[n] for p in P0], 0) for n in P0[0]}
        drv = X2.XNet2.drivers(X2.batch(D, D["ev"], s, dev)).cpu().numpy()
        for n, v in Pm.items():
            pars += [{"record": rec, "station": stn, "param": n, "median": float(np.median(v[:, i])), "p10": float(np.quantile(v[:, i], 0.1)),
                      "p90": float(np.quantile(v[:, i], 0.9))} for i, stn in enumerate(st)]
            if np.nanmedian([np.std(v[:, i]) for i in range(len(st))]) > 1e-6:      # varies from day to day
                for c, fname in enumerate(FEATS):
                    rho = [stats.spearmanr(v[:, i], drv[:, i, c])[0] for i in range(len(st)) if np.std(drv[:, i, c]) > 0 and np.std(v[:, i]) > 0]
                    cors.append({"record": rec, "param": n, "driver": fname, "median_rho": float(np.nanmedian(rho)), "share_positive": float(np.nanmean(np.array(rho) > 0))})
    pd.DataFrame(rows).to_csv(od / "terms.csv", index=False, float_format="%.5g"); pd.DataFrame(pars).to_csv(od / "params_by_station.csv", index=False, float_format="%.5g")
    if cors:
        pd.DataFrame(cors).to_csv(od / "param_driver_correlation.csv", index=False, float_format="%.4g")
    if curves:
        c = pd.DataFrame(curves); c.to_csv(od / "driver_response_raw.csv", index=False, float_format="%.5g")
        c.groupby(["record", "param", "driver", "x"]).value.median().reset_index().to_csv(od / "driver_response.csv", index=False, float_format="%.5g")
    E.log("explain terms", label=a.label, seeds=a.seeds, split=a.split, output=str(od.relative_to(X2.OUT)))
    print(pd.DataFrame(rows).pivot_table(index="term", columns=["record", "horizon"], values="median_r2_loss").round(3).to_string())
    if cors:
        print(pd.DataFrame(cors).pivot_table(index=["param", "driver"], columns="record", values="median_rho").round(2).to_string())


if __name__ == "__main__":
    main()
