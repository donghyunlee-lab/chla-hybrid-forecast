"""Accuracy tables and paired comparisons with a moving-block bootstrap over dates.

Models are seed ensembles (mean prediction of seeds 42, 123, 456) or single deterministic runs (ARIMA, harmonic ARIMA),
read from results/clean_revision_v3 (pure networks, XGBoost, ARIMA) and results/clean_revision_v4 (physics runs, PINN
transfer, short records, new baselines). Test years 2024-2025, observed targets only.
Reported per horizon: pooled R2, R2 of each station, the mean of station R2 (pooled R2 also rewards getting the level
differences between stations right), R2 per test year, and the seed mean and SD of pooled R2.
Paired comparisons use a moving-block bootstrap over dates: blocks of 30 days (14 and 60 as sensitivity), the same days
for all stations, 2,000 draws, circular blocks; the difference in pooled R2 and in station-mean R2 with percentile CIs.
Per-station differences (30-day blocks) are reported for the source stations.
Writes models.csv, comparisons.csv, station_comparisons.csv and summary.md to the folder OUT below (not part of this package)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

V3, V4 = C.ROOT / "results" / "clean_revision_v3", C.ROOT / "results" / "clean_revision_v4"
OUT = V4 / "stats"; SEEDS = (42, 123, 456); KEY = ["station", "anchor_date", "horizon"]
N_BOOT, BLOCKS, MAIN_BLOCK = 2000, (14, 30, 60), 30


def fam(d, base):
    return [(d, f"{base}_s{s}") for s in SEEDS]


def one(d, tag):
    return [(d, tag)]


def source_models():
    return {"xLSTM": fam(V3, "src_xLSTM_pure"), "LSTM": fam(V3, "src_LSTM_pure"), "iTransformer": fam(V3, "src_iTransformer_pure"),
            "ModernTCN": fam(V3, "src_ModernTCN_pure"), "Mamba": fam(V3, "src_Mamba_pure"), "xLSTM direct head": fam(V3, "src_xLSTM_pure_level"),
            "PINN sigmoid w0.3": fam(V4, "src_xLSTM_pinn-sigmoid"), "PINN none w0.3": fam(V4, "src_xLSTM_pinn-none"), "PINN soft w0.3": fam(V4, "src_xLSTM_pinn-soft"),
            "PINN sigmoid w0.1": fam(V4, "src_xLSTM_pinn-sigmoid_w0.1"), "PINN sigmoid w1.0": fam(V4, "src_xLSTM_pinn-sigmoid_w1.0"),
            "PINN placebo w0.3": fam(V4, "src_xLSTM_pinn-null"),                                  # placebo: growth term zero
            "XGBoost residual": fam(V3, "baseline_xgboost_residual"), "XGBoost residual, no calendar": fam(V4, "baseline_xgboost_residual_nodoy"),
            "XGBoost level": fam(V3, "baseline_xgboost"), "ARIMA": one(V3, "baseline_sarima_source"), "Harmonic ARIMA": one(V4, "baseline_harmonic_source")}


SOURCE_PAIRS = [("xLSTM", "XGBoost residual"), ("xLSTM", "XGBoost residual, no calendar"), ("XGBoost residual", "XGBoost residual, no calendar"),
                ("xLSTM", "ARIMA"), ("xLSTM", "Harmonic ARIMA"), ("Harmonic ARIMA", "ARIMA"), ("XGBoost residual", "Harmonic ARIMA"),
                ("xLSTM direct head", "XGBoost residual"), ("PINN sigmoid w0.3", "xLSTM"), ("PINN sigmoid w0.1", "xLSTM"), ("PINN sigmoid w1.0", "xLSTM"),
                ("PINN none w0.3", "PINN sigmoid w0.3"), ("PINN soft w0.3", "PINN sigmoid w0.3"),
                ("PINN placebo w0.3", "xLSTM"), ("PINN sigmoid w0.3", "PINN placebo w0.3")]


def transfer_models(tg):
    m = {}
    for label, v, d in (("pure", "pure", V3), ("PINN", "sigmoid", V4)):
        for mode, name in (("zero_shot", "zero-shot"), ("scratch", "scratch"), ("linear_probe", "linear probe"), ("full_tune", "fine-tune")):
            m[f"{name} {label}"] = fam(d, f"tgt_{tg}_{mode}_{v}_m24")
    m.update({"scratch placebo": fam(V4, f"tgt_{tg}_scratch_null_m24"), "fine-tune placebo": fam(V4, f"tgt_{tg}_full_tune_null_m24")})   # placebo runs
    m.update({"XGBoost target-only": fam(V3, f"baseline_xgboost_target_{tg}_m24"), "ARIMA": one(V3, f"baseline_sarima_{tg}"), "Harmonic ARIMA": one(V4, f"baseline_harmonic_{tg}")})
    return m


TRANSFER_PAIRS = [("fine-tune pure", "XGBoost target-only"), ("fine-tune PINN", "XGBoost target-only"), ("zero-shot pure", "XGBoost target-only"),
                  ("fine-tune PINN", "fine-tune pure"), ("scratch PINN", "scratch pure"), ("fine-tune pure", "scratch pure"), ("fine-tune pure", "Harmonic ARIMA"),
                  ("fine-tune PINN", "Harmonic ARIMA"), ("scratch PINN", "Harmonic ARIMA"),
                  ("fine-tune PINN", "fine-tune placebo"), ("scratch PINN", "scratch placebo"), ("fine-tune placebo", "fine-tune pure"),
                  ("scratch placebo", "scratch pure"), ("fine-tune placebo", "Harmonic ARIMA")]   # harmonic ARIMA = strongest target-only baseline


def short_models(tg, months):
    return {"fine-tune pure": fam(V4, f"tgt_{tg}_full_tune_pure_end12m{months}"), "scratch pure": fam(V4, f"tgt_{tg}_scratch_pure_end12m{months}"),
            "fine-tune PINN": fam(V4, f"tgt_{tg}_full_tune_sigmoid_end12m{months}"), "scratch PINN": fam(V4, f"tgt_{tg}_scratch_sigmoid_end12m{months}"),
            "XGBoost target-only": fam(V4, f"baseline_xgboost_target_{tg}_end12m{months}"), "zero-shot pure": fam(V3, f"tgt_{tg}_zero_shot_pure_m24")}


SHORT_PAIRS = [("fine-tune pure", "XGBoost target-only"), ("fine-tune PINN", "XGBoost target-only"), ("zero-shot pure", "XGBoost target-only"),
               ("fine-tune pure", "scratch pure"), ("fine-tune PINN", "fine-tune pure")]


def load_pred(d, tag):
    f = d / "runs" / tag / "predictions_test.csv"
    if not (d / "runs" / tag / "DONE").exists() or not f.exists():
        return None
    x = pd.read_csv(f, parse_dates=["target_date"])
    x = x[x.observed & x.y_pred_raw.notna()][KEY + ["target_date", "y_true_raw", "y_pred_raw"]]
    x["anchor_date"] = x.anchor_date.astype(str).str[:10]
    return x


def ensemble(members):
    fr = [f for f in (load_pred(d, t) for d, t in members) if f is not None]
    if len(fr) < len(members):
        return None, []
    e = fr[0].drop(columns="y_pred_raw").copy()
    for f in fr[1:]:
        e = e.merge(f[KEY], on=KEY)                    # anchors present in every member
    e["y_pred_raw"] = np.mean([f.set_index(KEY).loc[pd.MultiIndex.from_frame(e[KEY])]["y_pred_raw"].to_numpy() for f in fr], axis=0)
    return e, fr


def r2(y, p):
    sst = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / sst) if sst > 0 else np.nan


def model_rows(group, target, record, name, e, members):
    rows = []
    for h, g in e.groupby("horizon"):
        st = {s: r2(gs.y_true_raw.values, gs.y_pred_raw.values) for s, gs in g.groupby("station")}
        seeds = [r2(f[f.horizon == h].y_true_raw.values, f[f.horizon == h].y_pred_raw.values) for f in members] if len(members) > 1 else []
        rows.append({"group": group, "target": target, "record": record, "model": name, "horizon": int(h), "n": len(g), "r2_pooled": r2(g.y_true_raw.values, g.y_pred_raw.values),
                     "r2_station_mean": float(np.nanmean(list(st.values()))), **{f"r2_{y}": r2(g[g.target_date.dt.year == y].y_true_raw.values, g[g.target_date.dt.year == y].y_pred_raw.values) for y in (2024, 2025)},
                     "seed_mean": float(np.mean(seeds)) if seeds else np.nan, "seed_sd": float(np.std(seeds, ddof=1)) if seeds else np.nan, **{f"r2_{s}": v for s, v in st.items()}})
    return rows


def block_weights(n_days, block, rng):
    nb = int(np.ceil(n_days / block))
    days = ((rng.randint(0, n_days, (N_BOOT, nb))[:, :, None] + np.arange(block)[None, None, :]).reshape(N_BOOT, -1)[:, :n_days]) % n_days
    flat = days + (np.arange(N_BOOT)[:, None] * n_days)
    return np.bincount(flat.ravel(), minlength=N_BOOT * n_days).reshape(N_BOOT, n_days).astype(float)


def wr2(W, y, p):
    sw = W.sum(1); ym = (W @ y) / sw
    return 1 - (W @ (y - p) ** 2) / ((W @ y ** 2) - sw * ym ** 2)


def compare(a, b, group, target, record, na, nb):
    j = a.merge(b[KEY + ["y_pred_raw"]], on=KEY, suffixes=("", "_ref")); rows, srows = [], []
    for h, g in j.groupby("horizon"):
        y, p, q = g.y_true_raw.to_numpy(float), g.y_pred_raw.to_numpy(float), g.y_pred_raw_ref.to_numpy(float)
        days = np.sort(g.anchor_date.unique()); pos = {d: i for i, d in enumerate(days)}; di = g.anchor_date.map(pos).to_numpy()
        stations = g.station.to_numpy(); us = np.unique(stations)
        def sm(yy, pp, mask=None):
            return float(np.nanmean([r2(yy[stations == s], pp[stations == s]) for s in us]))
        row = {"group": group, "target": target, "record": record, "comparison": f"{na} - {nb}", "model": na, "reference": nb, "horizon": int(h), "n": len(g), "n_days": len(days),
               "r2_model": r2(y, p), "r2_reference": r2(y, q), "diff_pooled": r2(y, p) - r2(y, q), "diff_station_mean": sm(y, p) - sm(y, q)}
        for block in BLOCKS:
            rng = np.random.RandomState(1000 + block); Wd = block_weights(len(days), block, rng); W = Wd[:, di]
            dp = wr2(W, y, p) - wr2(W, y, q)
            dst = [wr2(W[:, stations == s], y[stations == s], p[stations == s]) - wr2(W[:, stations == s], y[stations == s], q[stations == s]) for s in us]
            ds = np.nanmean(dst, axis=0)
            for nm, d in (("pooled", dp), ("station_mean", ds)):
                lo, hi = np.nanpercentile(d, [2.5, 97.5]); row[f"ci{block}_{nm}_lo"], row[f"ci{block}_{nm}_hi"] = float(lo), float(hi)
            if block == MAIN_BLOCK and len(us) > 1:                    # per-station differences (station means hinge on Hyeondo)
                for s, d in zip(us, dst):
                    m = stations == s; lo, hi = np.nanpercentile(d, [2.5, 97.5])
                    srows.append({"group": group, "target": target, "record": record, "comparison": f"{na} - {nb}", "horizon": int(h), "station": s, "n": int(m.sum()),
                                  "r2_model": r2(y[m], p[m]), "r2_reference": r2(y[m], q[m]), "diff": r2(y[m], p[m]) - r2(y[m], q[m]), "ci_lo": float(lo), "ci_hi": float(hi),
                                  "excl0": bool(lo > 0 or hi < 0)})
        for nm in ("pooled", "station_mean"):
            row[f"excl0_{nm}"] = bool(row[f"ci{MAIN_BLOCK}_{nm}_lo"] > 0 or row[f"ci{MAIN_BLOCK}_{nm}_hi"] < 0)
        rows.append(row)
    return rows, srows


def main():
    OUT.mkdir(parents=True, exist_ok=True); mrows, crows, srows, missing = [], [], [], []
    groups = [("source", "", "24m", source_models(), SOURCE_PAIRS)]
    groups += [("transfer", tg, "24m", transfer_models(tg), TRANSFER_PAIRS) for tg in C.TARGETS]
    groups += [("short record", tg, f"{m}m to 2022-12", short_models(tg, m), SHORT_PAIRS) for tg in C.TARGETS for m in (6, 12)]
    for group, target, record, models, pairs in groups:
        ens = {}
        for name, members in models.items():
            e, fr = ensemble(members)
            if e is None:
                missing.append(f"{group} {target} {record}: {name}"); continue
            ens[name] = e; mrows += model_rows(group, target, record, name, e, fr)
        for na, nb in pairs:
            if na in ens and nb in ens:
                c, s = compare(ens[na], ens[nb], group, target, record, na, nb); crows += c; srows += s
        print("done", group, target, record, flush=True)
    mt, ct, st = pd.DataFrame(mrows), pd.DataFrame(crows), pd.DataFrame(srows)
    mt.to_csv(OUT / "models.csv", index=False); ct.to_csv(OUT / "comparisons.csv", index=False); st.to_csv(OUT / "station_comparisons.csv", index=False)

    f3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    ci = lambda r, nm: f"{r[f'diff_{nm}']:+.3f} [{r[f'ci{MAIN_BLOCK}_{nm}_lo']:+.3f}, {r[f'ci{MAIN_BLOCK}_{nm}_hi']:+.3f}]" + (" *" if r[f"excl0_{nm}"] else "")
    lines = ["# Accuracy and paired comparisons (test 2024-2025; seed ensembles)", "",
             f"Differences in R2 with {MAIN_BLOCK}-day date-block bootstrap 95% CI (* = excludes 0). Station mean = mean of per-station R2.", ""]
    for (group, target, record), g in mt.groupby(["group", "target", "record"], sort=False):
        lines += [f"## {group} {target} {record}".strip(), "", "| model | h1 | h3 | h7 | h14 | h7 station mean | h14 station mean |", "|---|---|---|---|---|---|---|"]
        for name, gm in g.groupby("model", sort=False):
            v = gm.set_index("horizon")
            lines.append(f"| {name} | " + " | ".join(f3(v.r2_pooled.get(h)) for h in C.HORIZONS) + f" | {f3(v.r2_station_mean.get(7))} | {f3(v.r2_station_mean.get(14))} |")
        cg = ct[(ct.group == group) & (ct.target == target) & (ct.record == record)] if len(ct) else ct
        if len(cg):
            lines += ["", "| comparison | h | pooled R2 difference | station-mean difference |", "|---|---|---|---|"]
            for _, r in cg.iterrows():
                lines.append(f"| {r.comparison} | {r.horizon} | {ci(r, 'pooled')} | {ci(r, 'station_mean')} |")
        lines.append("")
    if len(st):
        lines += [f"## Per-station R2 differences, source ({MAIN_BLOCK}-day blocks)", "", "| comparison | h | " + " | ".join(C.SOURCE) + " |", "|---|---|" + "---|" * len(C.SOURCE)]
        for (cmp, h), g in st[st.group == "source"].groupby(["comparison", "horizon"], sort=False):
            v = g.set_index("station")
            lines.append(f"| {cmp} | {h} | " + " | ".join(f"{v.loc[s, 'diff']:+.3f} [{v.loc[s, 'ci_lo']:+.3f}, {v.loc[s, 'ci_hi']:+.3f}]" + (" *" if v.loc[s, "excl0"] else "") if s in v.index else ""
                                                       for s in C.SOURCE) + " |")
        lines.append("")
    if missing:
        lines += ["Missing (not finished or not run): " + "; ".join(missing), ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8"); print("\n".join(lines))


if __name__ == "__main__":
    main()
