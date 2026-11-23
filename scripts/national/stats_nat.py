"""Analysis of the national benchmark. No model is fitted here.

Per station: every model, mode, record length, horizon and seed (and the seed ensemble = mean forecast), compared on the
forecasts all models share at that station and horizon. Metrics: R^2, RMSE, MAE, MSE skill against persistence, and CSI,
POD and FAR at 15 and 25 mg m^-3. Across stations (unit of inference): paired differences of the seed-ensemble R^2 with
two-sided Wilcoxon signed-rank tests, Holm adjustment within each family, median difference, share of stations improved and
a 95 % bootstrap interval of the median (H1-H4); Friedman tests, mean ranks and Wilcoxon tests against the sigmoid PINN
(H5); Page trend tests across record lengths (H6). Nationally, for each basin and for lakes and rivers.
Outputs: stations/<basin>/<code>_<name>.csv, national/, by_basin/<basin>/, by_type/<type>/, kinetic_parameters.csv."""
from __future__ import annotations
import hashlib, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "clean_revision")); sys.path.insert(0, str(ROOT / "scripts" / "national"))
import nat as N                                                                  # noqa: E402
C = N.patch()
OUT = N.OUT; RUNS = OUT / "runs"
NETS = list(N.NETS); FH = (3, 7, 14)
BASE_REC = ["climatology", "damped anomaly", "ARIMA", "harmonic ARIMA", "XGBoost"]
KEY = ["station", "anchor_date", "horizon"]
SEEDS = list(N.SEEDS)


def load(tag):
    d = RUNS / tag
    if not (d / "DONE").exists():
        return None
    x = pd.read_csv(d / "predictions_test.csv", parse_dates=["target_date"], dtype={"station": str})
    x = x[x.observed & x.y_pred_raw.notna()][KEY + ["target_date", "y_true_raw", "y_pred_raw"]]
    x["anchor_date"] = x.anchor_date.astype(str).str[:10]
    return x


ELIG = N.eligible_set()


def configs(code):
    """(model, mode, record, [tags per seed or one deterministic tag]); record-based configs only for eligible station-records."""
    cf = []
    for net in NETS:
        cf.append((net, "zero_shot", 0, [f"tgt_{code}_{net}_zero_shot_s{s}" for s in SEEDS]))
        for m in N.RECORDS:
            if (code, m) in ELIG:
                cf.append((net, "full_tune", m, [f"tgt_{code}_{net}_full_tune_r{m}_s{s}" for s in SEEDS]))
                cf.append((net, "scratch", m, [f"tgt_{code}_{net}_scratch_r{m}_s{s}" for s in SEEDS]))
    for m in [m for m in N.RECORDS if (code, m) in ELIG]:
        for name, key in (("climatology", "climatology"), ("damped anomaly", "damped"), ("ARIMA", "arima"), ("harmonic ARIMA", "harmonic")):
            cf.append((name, "baseline", m, [f"bl_{key}_{code}_r{m}"]))
        cf.append(("XGBoost", "baseline", m, [f"bl_xgboost_{code}_r{m}_s{s}" for s in SEEDS]))
    cf.append(("XGBoost", "zero_shot", 0, [f"bl_xgbzs_{code}_s{s}" for s in SEEDS]))
    return cf


def metrics(y, p, q_pers):
    e = y - p; sst = ((y - y.mean()) ** 2).sum()
    mse = float((e ** 2).mean()); mse_p = float(((y - q_pers) ** 2).mean())
    out = {"r2": float(1 - (e ** 2).sum() / sst) if sst > 0 else np.nan, "rmse": mse ** 0.5, "mae": float(np.abs(e).mean()),
           "skill_vs_persistence": 1 - mse / mse_p if mse_p > 0 else np.nan, "n": int(len(y))}
    for t in (15.0, 25.0):
        o, f = y >= t, p >= t; hit = int((o & f).sum()); miss = int((o & ~f).sum()); fa = int((~o & f).sum())
        out[f"csi{int(t)}"] = hit / (hit + miss + fa) if hit + miss + fa else np.nan
        out[f"pod{int(t)}"] = hit / (hit + miss) if hit + miss else np.nan
        out[f"far{int(t)}"] = fa / (hit + fa) if hit + fa else np.nan
    return out


def station_metrics(code, min_cover=0.8):
    """Per horizon, models that forecast at least `min_cover` of the persistence anchors are compared on the anchors they all
    share; models below the threshold are listed as missing for that horizon."""
    per = pd.read_csv(OUT / "persistence.csv", dtype={"station": str})
    per = per[(per.station == code) & (pd.to_datetime(per.anchor_date) >= pd.Timestamp(N.SPLITS["test"][0]))]          # test anchors only
    per = per[KEY + ["y_true_raw", "y_pred_raw"]].rename(columns={"y_pred_raw": "p_pers"}); per["anchor_date"] = per.anchor_date.astype(str).str[:10]
    frames, missing = {}, []
    for model, mode, rec, tags in configs(code):
        fr = [load(t) for t in tags]
        ok = [(s, f.drop_duplicates(KEY)) for s, f in zip(SEEDS if len(tags) > 1 else ["det"], fr) if f is not None]
        if len(ok) < len(tags):
            missing.append(f"{model}|{mode}|{rec}: {len(tags) - len(ok)} of {len(tags)} runs missing")
        if ok:
            frames[(model, mode, rec)] = ok
    rows = []
    for h in sorted(per.horizon.unique()):
        ph = per[per.horizon == h]; nb = len(ph)
        if nb == 0:
            continue
        valid, keys = {}, ph[KEY]
        for cfg, ok in frames.items():
            cover = min(len(f[f.horizon == h][KEY].merge(ph[KEY], on=KEY)) for _, f in ok) / nb
            if cover >= min_cover:
                valid[cfg] = ok
            else:
                missing.append(f"{cfg[0]}|{cfg[1]}|{cfg[2]} h{h}: forecasts for {cover:.0%} of anchors")
        for ok in valid.values():
            for _, f in ok:
                keys = keys.merge(f[f.horizon == h][KEY], on=KEY)
        base = keys.merge(ph, on=KEY)
        y, q = base.y_true_raw.to_numpy(float), base.p_pers.to_numpy(float)
        for (model, mode, rec), ok in valid.items():
            preds = []
            for s, f in ok:
                pr = base[KEY].merge(f[KEY + ["y_pred_raw"]], on=KEY).y_pred_raw.to_numpy(float); preds.append(pr)
                if len(ok) > 1:
                    rows.append({"station": code, "model": model, "mode": mode, "record": rec, "horizon": int(h), "seed": str(s), **metrics(y, pr, q)})
            rows.append({"station": code, "model": model, "mode": mode, "record": rec, "horizon": int(h), "seed": "ens" if len(ok) > 1 else "det",
                         "n_seeds": len(ok), **metrics(y, np.mean(preds, axis=0), q)})
        rows.append({"station": code, "model": "persistence", "mode": "baseline", "record": 0, "horizon": int(h), "seed": "det", "n_seeds": 1, **metrics(y, q, q)})
    df = pd.DataFrame(rows)
    if not len(df):
        return code, df, missing
    df.insert(1, "basin", N.BASIN_OF[code]); df.insert(2, "type", N.TYPE_OF[code]); df.insert(3, "name", N.NAME_OF[code])
    d = OUT / "stations" / N.BASIN_OF[code]; d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / f"{code}_{N.NAME_OF[code]}.csv", index=False, float_format="%.5g")
    return code, df, missing


def kinetic(code):
    rows = []
    for net in ("PINN-sigmoid", "PINN-soft", "PINN-none", "control"):
        for mode in ("full_tune", "scratch"):
            for m in N.RECORDS:
                for s in SEEDS:
                    f = RUNS / f"tgt_{code}_{net}_{mode}_r{m}_s{s}" / "physics_report.json"
                    if not f.exists():
                        continue
                    rep = json.loads(f.read_text()); conv = rep["converged"]["params"]; pos = rep["converged"].get("position_in_range", {})
                    rows.append({"station": code, "model": net, "mode": mode, "record": m, "seed": s, **{f"{k}_conv": conv.get(k) for k in ("K_N", "K_P", "K_I", "r", "m")},
                                 **{f"{k}_pos": pos.get(k) for k in ("K_N", "K_P", "K_I", "r", "m")}, "kinetic_r2_test": rep["converged"].get("kinetic_r2_test", {}).get("r2")})
    return rows


def paired(E, a, b, mode_a, mode_b, rec_a, rec_b, h):
    x = E[(E.model == a) & (E["mode"] == mode_a) & (E.record == rec_a) & (E.horizon == h)].set_index("station").r2
    y = E[(E.model == b) & (E["mode"] == mode_b) & (E.record == rec_b) & (E.horizon == h)].set_index("station").r2
    d = (x - y).dropna()
    if len(d) < 6:
        return None
    p = stats.wilcoxon(d.to_numpy(), alternative="two-sided").pvalue if np.any(d != 0) else 1.0
    rng = np.random.RandomState(7); bs = [np.median(rng.choice(d.to_numpy(), len(d))) for _ in range(2000)]
    return {"n_stations": len(d), "median_diff": float(d.median()), "mean_diff": float(d.mean()), "share_improved": float((d > 0).mean()),
            "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)), "wilcoxon_p": float(p)}


def comparisons(E):
    rows = []
    def add(fam, setting, comp, a, b, ma, mb, ra, rb):
        for h in FH:
            r = paired(E, a, b, ma, mb, ra, rb, h)
            if r:
                rows.append({"family": fam, "setting": setting, "comparison": comp, "horizon": h, **r})
    for mode in ("full_tune", "scratch"):
        for m in N.RECORDS:
            st = f"{mode} {m}m"
            add("H1", st, "PINN-sigmoid - xLSTM", "PINN-sigmoid", "xLSTM", mode, mode, m, m)
            add("H2", st, "PINN-sigmoid - PINN-none", "PINN-sigmoid", "PINN-none", mode, mode, m, m)
            add("H2", st, "PINN-sigmoid - PINN-soft", "PINN-sigmoid", "PINN-soft", mode, mode, m, m)
            add("H3", st, "PINN-sigmoid - control", "PINN-sigmoid", "control", mode, mode, m, m)
    add("H1", "zero-shot", "PINN-sigmoid - xLSTM", "PINN-sigmoid", "xLSTM", "zero_shot", "zero_shot", 0, 0)
    for net in ("xLSTM", "PINN-sigmoid"):
        for m in N.RECORDS:
            add("H4", f"{net} {m}m", "fine-tune - scratch", net, net, "full_tune", "scratch", m, m)
            add("H4", f"{net} {m}m", "zero-shot - scratch", net, net, "zero_shot", "scratch", 0, m)
    df = pd.DataFrame(rows)
    if len(df):
        df["holm_p"] = np.nan
        for fam, g in df.groupby("family"):
            df.loc[g.index, "holm_p"] = multipletests(g.wilcoxon_p.to_numpy(), method="holm")[1]
    return df


def benchmark(E):
    """H5: 16 models at each record length (networks fine-tuned, or trained from scratch), Friedman test and mean ranks."""
    fr, rk, ph = [], [], []
    for net_mode in ("full_tune", "scratch"):
        for m in N.RECORDS:
            for h in FH:
                sel = [(n, net_mode, m) for n in NETS] + [(b, "baseline", m) for b in BASE_REC] + [("XGBoost", "zero_shot", 0), ("persistence", "baseline", 0)]
                cols = {}
                for model, mode, rec in sel:
                    s = E[(E.model == model) & (E["mode"] == mode) & (E.record == rec) & (E.horizon == h)].set_index("station").r2
                    cols[f"{model}|{mode}"] = s
                T = pd.DataFrame(cols).dropna()
                if len(T) < 6:
                    continue
                f = stats.friedmanchisquare(*[T[c].to_numpy() for c in T.columns])
                fr.append({"networks": net_mode, "record": m, "horizon": h, "n_stations": len(T), "friedman_chi2": float(f.statistic), "friedman_p": float(f.pvalue)})
                ranks = T.rank(axis=1, ascending=False).mean()
                ref = f"PINN-sigmoid|{net_mode}"; ps = []
                for c in T.columns:
                    d = T[ref] - T[c]
                    p = stats.wilcoxon(d.to_numpy()).pvalue if c != ref and np.any(d != 0) else np.nan
                    rk.append({"networks": net_mode, "record": m, "horizon": h, "model": c, "mean_rank": float(ranks[c]), "median_r2": float(T[c].median()),
                               "pinn_minus_model_median": float(d.median()) if c != ref else np.nan, "share_pinn_better": float((d > 0).mean()) if c != ref else np.nan,
                               "wilcoxon_p": p})
                g = [r for r in rk if r["networks"] == net_mode and r["record"] == m and r["horizon"] == h and not np.isnan(r["wilcoxon_p"])]
                adj = multipletests([r["wilcoxon_p"] for r in g], method="holm")[1]
                for r, a in zip(g, adj):
                    r["holm_p"] = float(a)
    for mode in ("full_tune", "scratch"):                                         # H6: trend of the H1 difference with record length
        for h in FH:
            D = pd.DataFrame({m: E[(E.model == "PINN-sigmoid") & (E["mode"] == mode) & (E.record == m) & (E.horizon == h)].set_index("station").r2
                              - E[(E.model == "xLSTM") & (E["mode"] == mode) & (E.record == m) & (E.horizon == h)].set_index("station").r2 for m in N.RECORDS}).dropna()
            if len(D) >= 6:
                t = stats.page_trend_test(D.to_numpy(), predicted_ranks=list(range(len(N.RECORDS), 0, -1)))      # larger gain with less data
                ph.append({"mode": mode, "horizon": h, "n_stations": len(D), "page_L": float(t.statistic), "page_p": float(t.pvalue),
                           **{f"median_diff_{m}m": float(D[m].median()) for m in N.RECORDS}})
    return pd.DataFrame(fr), pd.DataFrame(rk), pd.DataFrame(ph)


def write(E, folder, title):
    folder.mkdir(parents=True, exist_ok=True)
    E.to_csv(folder / "metrics_ensemble.csv", index=False, float_format="%.5g")
    cmp_ = comparisons(E); cmp_.to_csv(folder / "comparisons.csv", index=False, float_format="%.5g")
    fr, rk, ph = benchmark(E)
    fr.to_csv(folder / "friedman.csv", index=False, float_format="%.5g"); rk.to_csv(folder / "ranks.csv", index=False, float_format="%.5g")
    ph.to_csv(folder / "page_trend.csv", index=False, float_format="%.5g")
    lines = [f"# {title}: {E.station.nunique()} stations", ""]
    if len(cmp_):
        lines += ["## Paired comparisons (seed-ensemble R^2; Wilcoxon across stations, Holm within family)", "",
                  "| family | setting | comparison | h | stations | median diff [95% CI] | share improved | Holm p |", "|---|---|---|---|---|---|---|---|"]
        for _, r in cmp_.iterrows():
            lines.append(f"| {r.family} | {r.setting} | {r.comparison} | {r.horizon} | {r.n_stations} | {r.median_diff:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | "
                         f"{r.share_improved:.2f} | {r.holm_p:.3g}{' *' if r.holm_p < 0.05 else ''} |")
    if len(rk):
        for h in FH:
            q = rk[(rk.networks == "full_tune") & (rk.horizon == h)]
            if not len(q):
                continue
            lines += ["", f"## Mean rank, networks fine-tuned, {h} days (1 = best)", "", "| model | " + " | ".join(f"{m} m" for m in N.RECORDS) + " |", "|---|" + "---|" * len(N.RECORDS)]
            for model in q.model.unique():
                vals = [q[(q.model == model) & (q.record == m)].mean_rank for m in N.RECORDS]
                lines.append(f"| {model} | " + " | ".join(f"{v.iloc[0]:.1f}" if len(v) else "" for v in vals) + " |")
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    t0 = time.time()
    with Pool(int(os.environ.get("NAT_CPU_WORKERS", "12"))) as pool:
        res = pool.map(station_metrics, N.CODES)
        kin = pool.map(kinetic, N.CODES)
    M = pd.concat([d for _, d, _ in res if len(d)], ignore_index=True)
    missing = {c: m for c, _, m in res if m}
    (OUT / "national").mkdir(parents=True, exist_ok=True)
    (OUT / "national" / "missing.json").write_text(json.dumps(missing, ensure_ascii=False, indent=1), encoding="utf-8")
    M.to_csv(OUT / "national" / "metrics_all.csv.gz", index=False, float_format="%.5g")
    pd.DataFrame([r for k in kin for r in k]).to_csv(OUT / "kinetic_parameters.csv", index=False, float_format="%.5g")
    E = M[M.seed.isin(["ens", "det"])].copy()
    write(E, OUT / "national", "National")
    for b in N.BASINS:
        write(E[E.basin == b], OUT / "by_basin" / b, f"{b} basin")
    for t in ("lake", "river"):
        write(E[E.type == t], OUT / "by_type" / t, f"{t.capitalize()}s")
    files = sorted(p for p in (OUT / "national").glob("*") if p.is_file())
    pd.DataFrame([{"file": str(p.relative_to(OUT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
                 + [{"file": "scripts/national/stats_nat.py", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}]).to_csv(OUT / "national" / "MANIFEST.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s; stations {M.station.nunique()}; missing configurations at {len(missing)} stations", flush=True)


if __name__ == "__main__":
    main()
