"""All metrics of the original paper, ranks and paired block-bootstrap differences for one rerun folder
(1-day forecasts, test 2024), from the predictions written by baselines_bootstrap.py.

Metrics  : R2, RMSE, MAE, MAPE and KGE on the original normalized scale (MinMax of log1p Chl-a fitted by the original
           code on all rows), formulas as in src/pinn_hab/evaluation/metrics.py (NSE equals R2).
Subsets  : measured target days (main) and all target days (gap days carry the last measured value, which equals the
           persistence forecast).
Ranks    : R2, MAE, MAPE, KGE. RMSE ranks exactly like R2 on the same days, so it is not counted twice.
Bootstrap: the same 30-day circular moving blocks over target dates as baselines_bootstrap.py (seed 20261013, 2,000 draws).
Usage: python all_metrics_bootstrap.py <rerun folder>
       (writes <folder>/experiments/all_metrics_{table,comparisons}.csv and all_metrics_summary.md; the summary also
       has the original-paper format: networks as mean ± SD over seeds from the phase-1 CSVs)"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

RUN = Path(sys.argv[1]).resolve()
P = pd.read_csv(RUN / "experiments" / "baselines_bootstrap_predictions.csv")
LO, HI = 0.09531017980432487, 5.470167623474696                                # log1p(0.1), log1p(236.5): the original scaler range
_scale = RUN / "experiments" / "target_scale.json"                               # written by baselines_bootstrap.py (scaler of that run)
if _scale.exists():
    import json
    LO, HI = (json.loads(_scale.read_text())[k] for k in ("lo", "hi"))
MODELS = [c for c in P.columns if c not in ("station", "target_date", "measured", "y_log1p")]
y = P.y_log1p.to_numpy(); yn = (y - LO) / (HI - LO)
measured = P.measured.to_numpy().astype(bool)
dates_u, d_idx = np.unique(P.target_date, return_inverse=True); D = len(dates_u)
rng = np.random.default_rng(20261013); B, L = 2000, 30
starts = rng.integers(0, D, size=(B, int(np.ceil(D / L))))
counts = np.zeros((B, D))
for b in range(B):
    idx = ((starts[b][:, None] + np.arange(L)[None, :]) % D).ravel()[:D]
    np.add.at(counts[b], idx, 1.0)
METRICS = ["R2", "RMSE", "MAE", "MAPE", "KGE"]
HIGHER = {"R2": True, "RMSE": False, "MAE": False, "MAPE": False, "KGE": True}
RANKED = ["R2", "MAE", "MAPE", "KGE"]
FMT = {"R2": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE": "{:.2f}", "KGE": "{:.3f}"}
DFMT = {"R2": "{:+.4f}", "RMSE": "{:+.4f}", "MAE": "{:+.4f}", "MAPE": "{:+.2f}", "KGE": "{:+.3f}"}


def per_date(v, mask):
    out = np.zeros(D); np.add.at(out, d_idx[mask], v[mask]); return out


def metrics(p, mask, boot):
    """point values (boot=False) or arrays over the bootstrap draws (boot=True), from per-date sums"""
    pn = (p - LO) / (HI - LO); e = yn - pn; nz = mask & (yn != 0)
    sums = {"n": (np.ones_like(y), mask), "sy": (yn, mask), "sp": (pn, mask), "syy": (yn * yn, mask), "spp": (pn * pn, mask),
            "syp": (yn * pn, mask), "sae": (np.abs(e), mask), "se": (e * e, mask),
            "nz": (np.ones_like(y), nz), "sape": (np.abs(e) / np.where(yn != 0, np.abs(yn), 1), nz)}
    S = {k: per_date(v, m) for k, (v, m) in sums.items()}
    S = {k: (counts @ v if boot else v.sum()) for k, v in S.items()}
    W = S["n"]; my, mp = S["sy"] / W, S["sp"] / W
    vy, vp, cov = S["syy"] / W - my ** 2, S["spp"] / W - mp ** 2, S["syp"] / W - my * mp
    r, alpha, beta = cov / np.sqrt(vy * vp), np.sqrt(vp / vy), mp / my
    return {"R2": 1 - S["se"] / (S["syy"] - S["sy"] ** 2 / W), "RMSE": np.sqrt(S["se"] / W), "MAE": S["sae"] / W,
            "MAPE": 100 * S["sape"] / S["nz"], "KGE": 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2),
            "KGE_r": r, "KGE_alpha": alpha, "KGE_beta": beta}


NETS = ["ModernTCN", "iTransformer", "Mamba"]
pairs = [("xLSTM (pure)", o) for o in ("Harmonic ARIMA", "ARIMA", "XGBoost", "Persistence")] + \
        [("xLSTM (pure)", f"{n} (pure)") for n in NETS] + \
        [(f"{n} (PINN)", f"{n} (pure)") for n in ["xLSTM"] + NETS] + \
        [("xLSTM (PINN)", o) for o in ("Harmonic ARIMA", "ARIMA", "XGBoost", "Persistence")]
tables, rows, md = [], [], []
for subset, mask in (("measured days", measured), ("all days", np.ones_like(measured))):
    tab = pd.DataFrame({k: metrics(P[k].to_numpy(), mask, False) for k in MODELS}).T
    ranks = pd.DataFrame({c: tab[c].rank(ascending=not HIGHER[c], method="min") for c in RANKED})
    tab = tab.join(ranks.add_prefix("rank_")).assign(mean_rank=ranks.mean(axis=1), subset=subset, n=int(mask.sum())).sort_values("mean_rank")
    tables.append(tab)
    boot = {k: metrics(P[k].to_numpy(), mask, True) for k in {m for pr in pairs for m in pr}}
    point = {k: metrics(P[k].to_numpy(), mask, False) for k in boot}
    md += [f"\n## {subset} (n = {int(mask.sum())})\n", "| model | " + " | ".join(METRICS) + " | mean rank |", "|---" * (len(METRICS) + 2) + "|"]
    md += [f"| {k} | " + " | ".join(FMT[c].format(tab.loc[k, c]) for c in METRICS) + f" | {tab.loc[k, 'mean_rank']:.2f} |" for k in tab.index]
    md += ["", "Paired differences (first minus second), 95% block-bootstrap interval; * = interval excludes 0.", "",
           "| comparison | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for a, b in pairs:
        cells = []
        for c in METRICS:
            diff = boot[a][c] - boot[b][c]; lo, hi = np.percentile(diff, [2.5, 97.5]); d = point[a][c] - point[b][c]
            sig = bool(lo > 0 or hi < 0)
            rows.append({"subset": subset, "comparison": f"{a} - {b}", "metric": c, "diff": d, "ci_low": lo, "ci_high": hi,
                         "significant": sig, "better": (a if (d > 0) == HIGHER[c] else b) if sig else ""})
            cells.append(f"{DFMT[c].format(d)} [{DFMT[c].format(lo)}, {DFMT[c].format(hi)}]{' *' if sig else ''}")
        md.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
table = pd.concat(tables); table.index.name = "model"
table.to_csv(RUN / "experiments" / "all_metrics_table.csv")
pd.DataFrame(rows).to_csv(RUN / "experiments" / "all_metrics_comparisons.csv", index=False)
head = [f"# All metrics, past-only rerun ({RUN.name}): 1-day forecasts, test 2024",
        "", "Scale: log1p(Chl-a) min-max normalized as in the original code. Networks and XGBoost: mean forecast of three seeds.",
        "Mean rank over R2, MAE, MAPE and KGE (RMSE ranks like R2). Written by reruns/all_metrics_bootstrap.py."]
# original-paper format (all days): networks as mean ± SD over the three seeds from the phase-1 CSVs, baselines as single
# values (XGBoost: forecast averaged over its three seeds, as above)
s1 = pd.read_csv(RUN / "experiments" / "phase1" / "phase1_step1_results.csv")
s2 = pd.read_csv(RUN / "experiments" / "phase1" / "phase1_step2_results.csv")
s2 = s2.rename(columns={f"PINN_{c}": c for c in ("R²", "RMSE", "MAE", "KGE")})
orig = []
for kind, d in (("pure", s1), ("PINN", s2)):
    for net, g in d.groupby("Backbone"):
        orig.append((f"{net} ({kind})", g["R²"].mean(), *(f"{g[c].mean():.4f} ± {g[c].std():.4f}" for c in ("R²", "RMSE", "MAE", "KGE"))))
base = table[table.subset == "all days"]
for k in ("Harmonic ARIMA", "ARIMA", "XGBoost", "Persistence"):
    orig.append((k, base.loc[k, "R2"], *(f"{base.loc[k, c]:.4f}" for c in ("R2", "RMSE", "MAE", "KGE"))))
md += ["\n## Original-paper format, all days (networks: mean ± SD over three seeds)\n", "| model | R2 | RMSE | MAE | KGE |", "|---|---|---|---|---|"]
md += [f"| {r[0]} | " + " | ".join(r[2:]) + " |" for r in sorted(orig, key=lambda r: -r[1])]
(RUN / "experiments" / "all_metrics_summary.md").write_text("\n".join(head + md) + "\n", encoding="utf-8")
print("\n".join(md))
