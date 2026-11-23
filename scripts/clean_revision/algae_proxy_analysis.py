"""E9: is chlorophyll-a a usable proxy for harmful cyanobacteria in the study reaches?

Inputs: data/clean/algae_alert_weekly.csv (fetch_algae_alert.py) and data/clean/geum_daily_observed.csv (sensor Chl-a,
the model target). Three checks, 2021-2025:
  1. same water sample at the algae-alert sites: laboratory Chl-a vs harmful cyanobacteria cells (Spearman, by season),
     and how well Chl-a separates samples at or above the alert level of 1,000 cells/mL (AUC);
  2. sensor Chl-a at the paired automatic station vs laboratory Chl-a on the sampling day (do the two Chl-a agree?);
  3. sensor Chl-a vs cyanobacteria cells (the chain the forecasts rely on).
Writes output/results/e9_algae_proxy/{summary.csv, summary.md, fig_e9_algae_proxy.png}."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

OUT = C.ROOT / "output" / "results" / "e9_algae_proxy"
PERIOD = ("2021-01-01", "2025-12-31")
BLOOM_MONTHS = (6, 7, 8, 9)
ALERT_CELLS = 1000.0          # first alert level ("attention") for drinking-water sections, cells/mL
MAX_LAG_DAYS = 3              # sensor value taken on the sampling day, else the nearest observed day within 3 days
N_BOOT = 2000


def boot_ci(x, y, fn, n=N_BOOT, seed=0, groups=None):
    """Percentile CI; resamples whole weeks (groups) when given, so same-week samples of a lake move together."""
    rng = np.random.RandomState(seed); vals = []
    if groups is None:
        groups = np.arange(len(x))
    ug = np.unique(groups); pos = {g: np.where(groups == g)[0] for g in ug}
    for _ in range(n):
        ix = np.concatenate([pos[g] for g in rng.choice(ug, len(ug), replace=True)])
        if len(np.unique(y[ix])) > 1:
            vals.append(fn(x[ix], y[ix]))
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def rho(x, y):
    return stats.spearmanr(x, y).statistic


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg))


def describe(name, x, y, weeks, label=None):
    ok = ~(np.isnan(x) | np.isnan(y)); x, y, weeks = x[ok], y[ok], weeks[ok]
    row = {"comparison": name, "n": int(ok.sum())}
    if row["n"] < 8:
        return row
    row["spearman"] = rho(x, y); row["spearman_lo"], row["spearman_hi"] = boot_ci(x, y, rho, groups=weeks)
    if label is not None:
        lab = label[ok]; row["n_alert"] = int(lab.sum())
        if 0 < lab.sum() < len(lab):
            row["auc_alert"] = auc(x, lab)
            row["auc_lo"], row["auc_hi"] = boot_ci(x, lab, lambda a, b: auc(a, b.astype(bool)), groups=weeks)
    return row


def sensor_on(sensor: pd.Series, dates: pd.Series) -> np.ndarray:
    """Sensor Chl-a on each sampling date, else nearest observed day within MAX_LAG_DAYS (earlier day on ties)."""
    s = sensor.dropna(); out = []
    for d in dates:
        if d in s.index:
            out.append(s[d]); continue
        best = np.nan
        for lag in range(1, MAX_LAG_DAYS + 1):
            for dd in (d - pd.Timedelta(days=lag), d + pd.Timedelta(days=lag)):
                if dd in s.index:
                    best = s[dd]; break
            if not np.isnan(best):
                break
        out.append(best)
    return np.asarray(out, float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    alg = pd.read_csv(C.CLEAN / "algae_alert_weekly.csv", parse_dates=["date"])
    alg = alg[(alg.date >= PERIOD[0]) & (alg.date <= PERIOD[1])].copy()
    alg["week"] = alg.date.dt.to_period("W").astype(str); alg["bloom"] = alg.date.dt.month.isin(BLOOM_MONTHS)
    obs = pd.read_csv(C.CLEAN / "geum_daily_observed.csv", parse_dates=["date"])
    sensor = {st: g.set_index("date")["chlorophyll_a"] for st, g in obs.groupby("station")}
    alg["sensor_chla"] = np.nan
    for st, g in alg.groupby("paired_station"):
        alg.loc[g.index, "sensor_chla"] = sensor_on(sensor[st], g.date)
    alg.to_csv(OUT / "matched_samples.csv", index=False)

    rows = []
    units = [("all sites", alg)] + [(f"{st} ({', '.join(sorted(g.site_name.unique()))})", g) for st, g in alg.groupby("paired_station")]
    for unit, g in units:
        for season, gs in (("all months", g), ("Jun-Sep", g[g.bloom]), ("Oct-May", g[~g.bloom])):
            w = gs.week.to_numpy(); lab = (gs.cyano_cells >= ALERT_CELLS).to_numpy()
            for comp, x, y, label in (("lab Chl-a vs cells (same sample)", gs.chla, gs.cyano_cells, lab),
                                      ("sensor Chl-a vs lab Chl-a", gs.sensor_chla, gs.chla, None),
                                      ("sensor Chl-a vs cells", gs.sensor_chla, gs.cyano_cells, lab)):
                r = describe(comp, x.to_numpy(float), y.to_numpy(float), w, label)
                rows.append({"unit": unit, "season": season, **r})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "summary.csv", index=False)

    fmt = lambda r, a, b, c: f"{r[a]:.2f} [{r[b]:.2f}, {r[c]:.2f}]" if pd.notna(r.get(a)) else ""
    lines = ["# E9: chlorophyll-a as a proxy for harmful cyanobacteria (2021-2025)", "",
             f"Spearman rho with 95% CI (weeks resampled, {N_BOOT} draws); AUC = how well Chl-a separates samples with >= {ALERT_CELLS:.0f} cells/mL. "
             f"Sensor value on the sampling day or nearest observed day within {MAX_LAG_DAYS} days.", "",
             "| unit | season | comparison | n | Spearman | n >= 1,000 | AUC |", "|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        lines.append(f"| {r.unit} | {r.season} | {r.comparison} | {r.n} | {fmt(r, 'spearman', 'spearman_lo', 'spearman_hi')} | "
                     f"{'' if pd.isna(r.get('n_alert')) else int(r.n_alert)} | {fmt(r, 'auc_alert', 'auc_lo', 'auc_hi')} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    st_order = ["남면", "공주", "부여", "대청호", "용담호"]; en = {"남면": "Sejong weir", "공주": "Gongju weir", "부여": "Baekje weir", "대청호": "Daecheong Reservoir", "용담호": "Yongdam Reservoir"}
    fig, axes = plt.subplots(1, len(st_order), figsize=(3.2 * len(st_order), 3.3), sharey=True)
    for ax, st in zip(axes, st_order):
        g = alg[alg.paired_station == st]
        for flag, col, lab in ((False, "#8a8a8a", "Oct-May"), (True, "#1b7a3a", "Jun-Sep")):
            h = g[g.bloom == flag]
            ax.scatter(h.chla.clip(lower=0.1), h.cyano_cells + 1, s=7, alpha=0.55, color=col, label=lab, linewidths=0)
        ax.axhline(ALERT_CELLS + 1, color="#b03030", lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(en[st], fontsize=9); ax.set_xlabel("Chl-a (mg m$^{-3}$, laboratory)", fontsize=8)
        r = res[(res.unit.str.startswith(st)) & (res.season == "all months") & (res.comparison == "lab Chl-a vs cells (same sample)")].iloc[0]
        ax.text(0.03, 0.96, f"rho = {r.spearman:.2f}\nn = {r.n}", transform=ax.transAxes, va="top", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Harmful cyanobacteria (cells mL$^{-1}$ + 1)", fontsize=8); axes[-1].legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_e9_algae_proxy.png", dpi=200); plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
