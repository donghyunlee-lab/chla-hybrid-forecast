"""Additional analyses of the redesigned study. No network is trained. Inputs are the frozen daily data,
the frozen test forecasts and the fitted kinetic parameters of the physics runs; FREEZE_20261010 is not touched.
Writes its tables and a MANIFEST.csv with SHA-256 hashes to the folder OUT below (not part of this package).

1 screen        Pre-training screen: elasticity K/(K+X) of each Monod factor to its half-saturation constant over the observed
                drivers, and the temporal variation of each factor, per station.
2 recovery      Synthetic recovery: Eq. 1 with known parameters on the real drivers and the real observation pattern, plus noise that
                brings the variance of the simulated changes to the observed variance (factor 1.0) or half of that noise (0.5).
                The kinetic regression of the paper is refitted (free and sigmoid) and profiled on 24-point grids
                (approximate 95% interval n (L / L_min - 1) <= 3.84). Scenarios: rivers (pooled), Daecheong, Yongdam x two truths x
                record 2021-2022 or 2021-2025 x daily changes or weekly samples (7-day changes) x two noise levels x three draws;
                each river station alone (2021-2022, daily, realistic noise). The same fit and profile on the real changes.
3 ceiling       Share of the observed day-to-day change that the growth term of Eq. 1 would explain (variance ratio), for
                mu_max 0.2-1.0 per day and the two sets of half-saturation constants.
4 ladder        Baseline ladder on the keys of the frozen comparisons: persistence, climatology (two annual Fourier pairs,
                2021-2023), damped anomaly (climatology plus the last anomaly times the autocorrelation at the forecast lag,
                2021-2023), and the frozen models; 30-day date-block bootstrap differences (stats_v4.compare). Station memory.
5 equifinality  Net growth rates implied by the fitted parameter sets (rivers: sigmoid, soft, none; reservoirs: fine-tuned PINN),
                test years, compared with a reference set.
6 alerts        Weekly cyanobacteria samples (>= 1,000 cells/mL) in the reservoirs, 7 and 14 days ahead: AUC of the calendar
                (climatology), the current Chl-a and the forecasts; logistic models with week-grouped 5-fold cross-validation
                (calendar; + current Chl-a; + forecast) and week-block bootstrap of AUC differences.
Run: CUDA_VISIBLE_DEVICES="" python v5_additions.py [screen recovery ceiling ladder equifinality alerts]"""
from __future__ import annotations
import hashlib, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import stats_v4 as S
import identifiability as I

torch.set_num_threads(1)                                                         # many small fits run in parallel processes
OUT = C.ROOT / "results" / "clean_revision_v4" / "v5_additions"; OUT.mkdir(parents=True, exist_ok=True)
V3, V4 = S.V3, S.V4; H = C.HORIZONS; SEEDS = (42, 123, 456)
EN = {"현도": "Hyeondo", "남면": "Nammyeon", "공주": "Gongju", "부여": "Buyeo", "용담호": "Yongdam", "대청호": "Daecheong"}
LIT = {"K_N": (0.01, 0.5), "K_P": (0.001, 0.05), "K_I": (50.0, 1000.0)}
TRUTH = {"mid": {"mu": 0.8, "K_N": 0.1, "K_P": 0.01, "K_I": 300.0}, "high": {"mu": 1.0, "K_N": 0.3, "K_P": 0.03, "K_I": 700.0}}
PGRID = {"K_N": np.logspace(-2, 1, 24), "K_P": np.logspace(-3, 0, 24), "K_I": np.logspace(1, 4, 24)}
RECORDS = {"2y": ("2021-01-01", "2022-12-31"), "5y": ("2021-01-01", "2025-12-31")}
GROUPS = {"rivers": C.SOURCE, "Daecheong": ["대청호"], "Yongdam": ["용담호"]}
LOG = []


def note(s):
    print(s, flush=True); LOG.append(s); (OUT / "summary.md").write_text("\n".join(LOG), encoding="utf-8")


def md(d: pd.DataFrame, digits=3) -> str:
    d = d.reset_index() if not isinstance(d.index, pd.RangeIndex) else d
    f = lambda v: "" if isinstance(v, float) and np.isnan(v) else (f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v))
    return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)] + ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)])


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); sst = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / sst) if sst > 0 else np.nan


# ------------------------------------------------------------------------------------------------ 1 screen
def screen(df):
    rows = []
    for st in C.STATION_IDS:
        d = df[df.station == st]
        for key, col in (("K_N", "total_nitrogen"), ("K_P", "total_phosphorus"), ("K_I", "par_umol")):
            x = d[col].dropna().clip(lower=0).to_numpy(); lo, hi = LIT[key]
            for lab, K in (("low", lo), ("centre", (lo + hi) / 2), ("high", hi)):
                el, f = K / (K + x), x / (K + x)
                rows.append({"station": EN[st], "param": key, "K": lab, "K_value": K, "n_days": len(x), "median_driver": float(np.median(x)),
                             "median_elasticity": float(np.median(el)), "share_elasticity_gt_0.2": float((el > 0.2).mean()), "factor_cv": float(f.std() / max(f.mean(), 1e-9))})
    b = pd.DataFrame(rows); b.to_csv(OUT / "1_screen.csv", index=False)
    piv = b[b.K == "centre"].pivot_table(index="station", columns="param", values=["median_elasticity", "share_elasticity_gt_0.2", "factor_cv"])
    piv.columns = [f"{a}:{p}" for a, p in piv.columns]
    note("## 1. Pre-training screen (constant at the centre of its literature range)\n\n" + md(piv) + "\n")


# ------------------------------------------------------------------------------------------------ 2 recovery
def series(df, st):
    d = df[df.station == st].sort_values("date").reset_index(drop=True)
    T, _, _ = C.causal_fill(d.water_temperature.to_numpy(float), 3); N, _, _ = C.causal_fill(d.total_nitrogen.to_numpy(float), 3)
    P, _, _ = C.causal_fill(d.total_phosphorus.to_numpy(float), 3)
    return d.date.to_numpy(), T, N, P, d.par_umol.to_numpy(float), d.chlorophyll_a.to_numpy(float)


def make_pairs(df, stations, record, sampling):
    """Observed changes of ln C: daily = consecutive measured days; weekly = Monday-to-Monday 7-day changes (both measured)."""
    s0, s1 = (np.datetime64(x) for x in RECORDS[record]); k = 1 if sampling == "daily" else 7; rows = []
    for st in stations:
        dates, T, N, P, Ir, Cc = series(df, st)
        for i in range(len(dates) - k):
            if not (s0 <= dates[i] and dates[i + k] <= s1) or np.isnan(Cc[i]) or np.isnan(Cc[i + k]):
                continue
            if sampling == "weekly" and pd.Timestamp(dates[i]).weekday() != 0:
                continue
            seg = slice(i, i + k)
            if np.isnan(T[seg]).any() or np.isnan(N[seg]).any() or np.isnan(P[seg]).any() or np.isnan(Ir[seg]).any():
                continue
            rows.append((C.STATION_IDS[st], np.log(Cc[i + k] + 1e-3) - np.log(Cc[i] + 1e-3), T[seg], N[seg], P[seg], Ir[seg]))
    sid = torch.tensor([r[0] for r in rows]); y = torch.tensor([r[1] for r in rows], dtype=torch.float32)
    return sid, y, [torch.tensor(np.stack([r[j] for r in rows]), dtype=torch.float32) for j in (2, 3, 4, 5)]


def growth(T, N, P, Ir, mu, KN, KP, KI):
    return mu * 1.047 ** (torch.clamp(T, 0, 40) - 20) * F.relu(N) / (KN + F.relu(N)) * F.relu(P) / (KP + F.relu(P)) * F.relu(Ir) / (KI + F.relu(Ir))


class Kin(torch.nn.Module):
    """Same kinetic regression as identifiability.Kin, on the CPU of the worker."""
    def __init__(self, mode):
        super().__init__(); self.mode = mode
        self.raw = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.zeros(())) for k in I.KKEYS}); self.mu = torch.nn.Parameter(torch.zeros(6))

    def v(self, k, raw):
        if self.mode == "sigmoid":
            lo, hi = (I.LIT["r"][0] + I.LIT["m"][0], I.LIT["r"][1] + I.LIT["m"][1]) if k == "rm" else (I.LIT | {"mu_max": (0.1, 1.5)})[k]
            return lo + torch.sigmoid(raw) * (hi - lo)
        return I.CENTER[k] * torch.exp(raw)

    def cum_net(self, sid, T, N, P, Ir):
        mu = self.v("mu_max", self.mu)[sid][:, None]; N, P, Ir = F.relu(N), F.relu(P), F.relu(Ir)
        f = 1.047 ** (torch.clamp(T, 0, 40) - 20) * N / (self.v("K_N", self.raw["K_N"]) + N) * P / (self.v("K_P", self.raw["K_P"]) + P) * Ir / (self.v("K_I", self.raw["K_I"]) + Ir)
        return (mu * f - self.v("rm", self.raw["rm"])).sum(1)


def fit(mode, sid, y, X, fixed=None, steps=3000, init=None, lbfgs=300):
    """Adam, then an L-BFGS polish (as converge_physics), so that profile values are not limited by convergence."""
    torch.manual_seed(0); kin = Kin(mode)
    if init is not None:
        kin.load_state_dict(init)
    params = []
    for n_, p in kin.named_parameters():
        if fixed and n_ == f"raw.{fixed[0]}":
            with torch.no_grad():
                p.fill_(math.log(fixed[1] / I.CENTER[fixed[0]]))
            p.requires_grad_(False)
        else:
            params.append(p)
    opt = torch.optim.Adam(params, lr=0.05)
    for _ in range(steps):
        l = F.smooth_l1_loss(kin.cum_net(sid, *X), y); opt.zero_grad(); l.backward(); opt.step()
    if lbfgs:
        lb = torch.optim.LBFGS(params, lr=0.5, max_iter=lbfgs, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad(); l_ = F.smooth_l1_loss(kin.cum_net(sid, *X), y); l_.backward(); return l_
        try:
            lb.step(closure)
        except RuntimeError:
            pass
    with torch.no_grad():
        return kin, float(F.smooth_l1_loss(kin.cum_net(sid, *X), y))


def profile(sid, y, X, lfree, best):
    n = len(y); out = []
    for key, grid in PGRID.items():                                              # two starts per grid value: the best fit and the range centre
        out.append((key, grid, np.array([min(fit("free", sid, y, X, fixed=(key, float(v)), steps=800, init=best)[1],
                                             fit("free", sid, y, X, fixed=(key, float(v)), steps=800, init=None)[1]) for v in grid])))
    lmin = min([lfree] + [float(l.min()) for _, _, l in out]); rows = []
    for key, grid, losses in out:
        inside = grid[n * (losses / lmin - 1) <= I.CHI2_95]; lo, hi = LIT[key]
        rows.append({"param": key, "ci_lo": float(inside.min()) if len(inside) else np.nan, "ci_hi": float(inside.max()) if len(inside) else np.nan,
                     "log_width_ratio": float(np.log(inside.max() / inside.min()) / np.log(hi / lo)) if len(inside) else np.nan,
                     "at_grid_edge": bool(len(inside) and (inside.min() <= grid[0] * 1.001 or inside.max() >= grid[-1] * 0.999)),
                     "argmin": float(grid[int(np.argmin(losses))])})
    return rows


PAIRS = {}


def recovery_job(spec):
    torch.set_num_threads(1)
    sid, y_real, X = PAIRS[(spec["group"], spec["record"], spec["sampling"])]; n = len(y_real); vy = float(y_real.var())
    base = {k: spec[k] for k in ("group", "record", "sampling")} | {"n_pairs": n}
    if spec["truth"] is None:                                                     # real changes
        kin, l = fit("free", sid, y_real, X); best = {k: v.detach().clone() for k, v in kin.state_dict().items()}
        return [], [base | {"data": "real"} | r for r in profile(sid, y_real, X, l, best)]
    tp = TRUTH[spec["truth"]]
    with torch.no_grad():
        mu_t = growth(X[0], X[1], X[2], X[3], tp["mu"], tp["K_N"], tp["K_P"], tp["K_I"]).sum(1); net_t = mu_t - float(mu_t.mean())   # balanced: loss rate = mean growth
        vs = float(net_t.var())
    sig = math.sqrt(max(vy - vs, 1e-12)); gen = torch.Generator().manual_seed(9000 + 101 * spec["rep"] + int(100 * spec["noise"]))
    y = net_t + spec["noise"] * sig * torch.randn(n, generator=gen)
    base |= {"truth": spec["truth"], "noise": spec["noise"], "rep": spec["rep"], "ceiling": vs / vy}
    fits = []
    for mode in ("free", "sigmoid"):
        kin, l = fit(mode, sid, y, X)
        with torch.no_grad():
            est = {k: float(kin.v(k, kin.raw[k])) for k in ("K_N", "K_P", "K_I")}; rate = kin.cum_net(sid, *X)
        fits.append(base | {"mode": mode, "r2_rate": r2(net_t, rate), **{f"est_{k}": v for k, v in est.items()}, **{f"relerr_{k}": est[k] / tp[k] - 1 for k in est}})
        if mode == "free":
            best = {k: v.detach().clone() for k, v in kin.state_dict().items()}; lfree = l
    prof = [base | {"data": "synthetic"} | r | {"true": tp[r["param"]], "covers_truth": bool(not np.isnan(r["ci_lo"]) and r["ci_lo"] <= tp[r["param"]] <= r["ci_hi"])}
            for r in profile(sid, y, X, lfree, best)]
    return fits, prof


def recovery(df):
    specs = []
    for g in GROUPS:
        for rec in RECORDS:
            for smp in ("daily", "weekly"):
                specs.append({"group": g, "record": rec, "sampling": smp, "truth": None})
                specs += [{"group": g, "record": rec, "sampling": smp, "truth": t, "noise": nz, "rep": rp} for t in TRUTH for nz in (1.0, 0.5) for rp in (1, 2, 3)]
    for st in C.SOURCE:
        GROUPS_ST[EN[st]] = [st]
        specs.append({"group": EN[st], "record": "2y", "sampling": "daily", "truth": None})
        specs += [{"group": EN[st], "record": "2y", "sampling": "daily", "truth": t, "noise": 1.0, "rep": rp} for t in TRUTH for rp in (1, 2, 3)]
    for s_ in specs:
        key = (s_["group"], s_["record"], s_["sampling"])
        if key not in PAIRS:
            PAIRS[key] = make_pairs(df, (GROUPS | GROUPS_ST)[s_["group"]], s_["record"], s_["sampling"])
    note(f"recovery: {len(specs)} jobs, pair counts " + ", ".join(f"{k[0]}/{k[1]}/{k[2]}={len(v[1])}" for k, v in PAIRS.items()))
    fits, prof = [], []
    with ProcessPoolExecutor(max_workers=min(48, os.cpu_count() or 8)) as ex:
        for f_, p_ in ex.map(recovery_job, specs, chunksize=1):
            fits += f_; prof += p_
    fr, pr = pd.DataFrame(fits), pd.DataFrame(prof); fr.to_csv(OUT / "2_recovery_fits.csv", index=False); pr.to_csv(OUT / "2_recovery_profiles.csv", index=False)
    syn = pr[pr.data == "synthetic"].groupby(["group", "record", "sampling", "truth", "noise", "param"]).agg(
        n_pairs=("n_pairs", "first"), ceiling=("ceiling", "median"), log_width_ratio=("log_width_ratio", "median"), covers_truth=("covers_truth", "mean"),
        at_edge=("at_grid_edge", "mean")).reset_index()
    real = pr[pr.data == "real"][["group", "record", "sampling", "n_pairs", "param", "ci_lo", "ci_hi", "log_width_ratio", "at_grid_edge", "argmin"]]
    syn.to_csv(OUT / "2_recovery_summary.csv", index=False)
    piv = syn[syn.noise == 1.0].pivot_table(index=["group", "record", "sampling", "truth"], columns="param", values="log_width_ratio")
    note("## 2. Synthetic recovery (median log-width of the approximate 95% profile interval relative to the literature range; < 1 = narrower; realistic noise)\n\n"
         + md(piv) + "\n\nReal changes, same fit:\n\n" + md(real) + "\n")


GROUPS_ST: dict = {}


# ------------------------------------------------------------------------------------------------ 3 ceiling
def ceiling(df):
    rows = []
    for g, stations in (GROUPS | {EN[s]: [s] for s in C.SOURCE}).items():
        sid, y, X = make_pairs(df, stations, "2y", "daily"); vy = float(y.var())
        for tname, tp in TRUTH.items():
            for mu in (0.2, 0.4, 0.6, 0.8, 1.0):
                with torch.no_grad():
                    g_ = growth(X[0], X[1], X[2], X[3], mu, tp["K_N"], tp["K_P"], tp["K_I"]).sum(1)
                rows.append({"group": g, "constants": tname, "mu_max": mu, "n_pairs": len(y), "sd_observed_change": math.sqrt(vy), "sd_growth": float(g_.std()), "ceiling": float(g_.var()) / vy})
    c = pd.DataFrame(rows); c.to_csv(OUT / "3_ceiling.csv", index=False)
    note("## 3. Kinetic R^2 ceiling (share of the observed daily change in ln C that the growth term would explain), 2021-2022\n\n"
         + md(c.pivot_table(index=["group", "constants"], columns="mu_max", values="ceiling")) + "\n")


# ------------------------------------------------------------------------------------------------ 4 ladder
def fourier(doy, K=2):
    cols = [np.ones_like(doy, dtype=float)]
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    return np.stack(cols, 1)


def clim_tables(df, fit_end="2023-12-31"):
    out, acf = {}, {}
    for st in C.STATION_IDS:
        d = df[df.station == st].set_index("date").sort_index(); full = pd.date_range(d.index.min(), d.index.max(), freq="D"); d = d.reindex(full)
        y = np.log1p(d.chlorophyll_a.clip(lower=0)).to_numpy(); X = fourier(full.dayofyear.to_numpy())
        m = (full >= "2021-01-01") & (full <= fit_end) & ~np.isnan(y); beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        t = pd.DataFrame({"y": y, "clim": X @ beta, "a": y - X @ beta}, index=full); out[st] = t
        s = t.a[(t.index >= "2021-01-01") & (t.index <= fit_end)]
        acf[st] = {L: float(pd.concat([s, s.shift(-L)], axis=1).dropna().corr().iloc[0, 1]) for L in range(1, 91)}
    return out, acf


def baseline_frame(keys, cl, acf):
    rows = []
    for st, g in keys.groupby("station"):
        t = cl[st]
        for r in g.itertuples(index=False):
            a0 = pd.Timestamp(r.anchor_date); h = int(r.horizon); tgt = a0 + pd.Timedelta(days=h)
            win = t.loc[a0 - pd.Timedelta(days=C.SEQ_LEN - 1):a0]; obs = win[win.y.notna()]; c = t.clim.get(tgt, np.nan)
            damped = c + (acf[st].get(min(h + (a0 - obs.index[-1]).days, 90), 0.0) * obs.a.iloc[-1] if len(obs) else 0.0)
            rows.append({"station": st, "anchor_date": r.anchor_date, "horizon": h, "target_date": r.target_date, "y_true_raw": r.y_true_raw,
                         "clim": float(np.expm1(c)), "damped": float(np.expm1(damped))})
    b = pd.DataFrame(rows)
    return b.assign(y_pred_raw=b.clim)[S.KEY + ["target_date", "y_true_raw", "y_pred_raw"]], b.assign(y_pred_raw=b.damped)[S.KEY + ["target_date", "y_true_raw", "y_pred_raw"]]


def ladder(df):
    cl, acf = clim_tables(df)
    mem = []
    for st, t in cl.items():
        ef = next((L for L in range(1, 91) if acf[st][L] < 1 / math.e), np.nan)
        mem.append({"station": EN[st], "efold_days": ef, **{f"rho_{L}": acf[st][L] for L in (1, 3, 7, 14, 30)}, "sd_anomaly": float(t.a.std())})
    pd.DataFrame(mem).to_csv(OUT / "4_station_memory.csv", index=False)
    per = __import__("persistence_v4").persistence_frame()
    mrows, crows, srows = [], [], []
    sets = [("source", "", S.source_models(), "xLSTM", [("Harmonic ARIMA", "Damped anomaly"), ("XGBoost residual", "Damped anomaly"), ("xLSTM", "Damped anomaly"),
                                                         ("PINN sigmoid w0.3", "Damped anomaly"), ("Damped anomaly", "Persistence"), ("Climatology", "Persistence"), ("Damped anomaly", "Climatology")])]
    sets += [("transfer", tg, S.transfer_models(tg), "fine-tune PINN", [("fine-tune PINN", "Damped anomaly"), ("fine-tune pure", "Damped anomaly"), ("fine-tune placebo", "Damped anomaly"),
                                                                         ("Harmonic ARIMA", "Damped anomaly"), ("XGBoost target-only", "Damped anomaly"), ("Damped anomaly", "Persistence")]) for tg in C.TARGETS]
    for group, target, models, ref, pairs in sets:
        ens = {}
        for name in {a for a, _ in pairs} | {b for _, b in pairs} | {ref}:
            if name in models:
                ens[name] = S.ensemble(models[name])[0]
        keys = ens[ref]
        ens["Climatology"], ens["Damped anomaly"] = baseline_frame(keys, cl, acf)
        p = keys[S.KEY + ["y_true_raw"]].merge(per, on=S.KEY, how="left", suffixes=("_r", "")).drop(columns=["y_true_raw_r", "base_valid"]); ens["Persistence"] = p
        for nm in ("Climatology", "Damped anomaly", "Persistence"):
            mrows += S.model_rows(group, target, "24m", nm, ens[nm], [])
        for a_, b_ in pairs:
            c, s_ = S.compare(ens[a_], ens[b_], group, target, "24m", a_, b_); crows += c; srows += s_
        note(f"ladder: {group} {target} done ({time.strftime('%H:%M:%S')})")
    m, c, s_ = pd.DataFrame(mrows), pd.DataFrame(crows), pd.DataFrame(srows)
    m.to_csv(OUT / "4_ladder_models.csv", index=False); c.to_csv(OUT / "4_ladder_comparisons.csv", index=False); s_.to_csv(OUT / "4_ladder_station_comparisons.csv", index=False)
    c["txt"] = c.apply(lambda r: f"{r.diff_pooled:+.3f} [{r.ci30_pooled_lo:+.3f}, {r.ci30_pooled_hi:+.3f}]" + (" *" if r.excl0_pooled else ""), axis=1)
    note("## 4. Baseline ladder\n\nStation memory (climatology and autocorrelation fitted on 2021-2023):\n\n" + md(pd.DataFrame(mem))
         + "\n\nPooled R^2 of the new baselines:\n\n" + md(m.pivot_table(index=["group", "target", "model"], columns="horizon", values="r2_pooled"))
         + "\n\nPaired differences (30-day blocks):\n\n" + md(c.pivot_table(index=["group", "target", "comparison"], columns="horizon", values="txt", aggfunc="first")) + "\n")


# ------------------------------------------------------------------------------------------------ 5 equifinality
def equifinality(df):
    def sets_for(tags):
        out = []
        for tag in tags:
            rep = json.load(open(V4 / "runs" / tag / "physics_report.json"))
            for est in ("joint", "converged"):
                p = rep[est]["params"]; out.append({"set": f"{tag}|{est}", "tag": tag, "estimate": est, **{k: p[k] for k in ("K_N", "K_P", "K_I", "r", "m")}, "mu": p["mu_max_sites"]})
        return out
    groups = [("rivers", C.SOURCE, sets_for([f"src_xLSTM_pinn-{p}_s{s}" for p in ("sigmoid", "soft", "none") for s in SEEDS]), "src_xLSTM_pinn-sigmoid_s42|converged")]
    groups += [(EN[tg], [tg], sets_for([f"tgt_{tg}_full_tune_sigmoid_m24_s{s}" for s in SEEDS]), f"tgt_{tg}_full_tune_sigmoid_m24_s42|converged") for tg in C.TARGETS]
    rows = []
    for gname, stations, sets, ref in groups:
        for st in stations:
            dates, T, N, P, Ir, _ = series(df, st); ok = (dates >= np.datetime64("2024-01-01")) & ~np.isnan(T) & ~np.isnan(N) & ~np.isnan(P) & ~np.isnan(Ir)
            Tt, Nt, Pt, It = (torch.tensor(v[ok], dtype=torch.float64) for v in (T, N, P, Ir)); rates = {}
            for s_ in sets:
                g = growth(Tt, Nt, Pt, It, s_["mu"][C.STATION_IDS[st]], s_["K_N"], s_["K_P"], s_["K_I"]).numpy(); rates[s_["set"]] = (g, g - s_["r"] - s_["m"])
            n0 = rates[ref][1]
            for s_ in sets:
                g, n_ = rates[s_["set"]]
                rows.append({"group": gname, "station": EN[st], "set": s_["set"], "estimate": s_["estimate"], "K_N": s_["K_N"], "K_P": s_["K_P"], "K_I": s_["K_I"],
                             "r_plus_m": s_["r"] + s_["m"], "mu_station": s_["mu"][C.STATION_IDS[st]], "growth_mean": float(g.mean()), "net_sd": float(n_.std()),
                             "corr_vs_ref": float(np.corrcoef(n_, n0)[0, 1]), "mean_abs_diff_vs_ref": float(np.abs(n_ - n0).mean())})
    e = pd.DataFrame(rows); e.to_csv(OUT / "5_equifinality.csv", index=False)
    e["family"] = e.set.str.split("|").str[0].str.replace(r"_s\d+$", "", regex=True) + "|" + e.estimate
    agg = e.groupby(["group", "station", "family"]).agg(K_N=("K_N", "median"), K_I=("K_I", "median"), mu=("mu_station", "median"),
                                                         corr_vs_ref=("corr_vs_ref", "min"), mad_vs_ref=("mean_abs_diff_vs_ref", "max")).reset_index()
    note("## 5. Equifinality (test years; reference = sigmoid PINN, converged, seed 42; corr = lowest, mad = largest over seeds)\n\n" + md(agg) + "\n")


# ------------------------------------------------------------------------------------------------ 6 alerts
def alerts(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    cl, _ = clim_tables(df)
    alg = pd.read_csv(C.CLEAN / "algae_alert_weekly.csv", parse_dates=["date"])
    alg = alg[alg.paired_station.isin(C.TARGETS) & (alg.date >= "2024-01-01") & alg.cyano_cells.notna()].copy(); alg["alert"] = alg.cyano_cells >= 1000
    daily = {st: g.set_index("date").chlorophyll_a.sort_index() for st, g in df[df.station.isin(C.TARGETS)].groupby("station")}

    def raw(d, tag):
        f = d / "runs" / tag / "predictions_test.csv"; x = pd.read_csv(f); x["anchor_date"] = x.anchor_date.astype(str).str[:10]
        return x[x.y_pred_raw.notna()].set_index(["anchor_date", "horizon"]).y_pred_raw

    rows = []
    for tg in C.TARGETS:
        pinn = pd.concat([raw(V4, f"tgt_{tg}_full_tune_sigmoid_m24_s{s}") for s in SEEDS], axis=1).dropna().mean(1)
        harm = raw(V4, f"baseline_harmonic_{tg}")
        for r in alg[alg.paired_station == tg].itertuples(index=False):
            for h in (7, 14):
                anc = (r.date - pd.Timedelta(days=h)).strftime("%Y-%m-%d"); w = daily[tg].loc[pd.Timestamp(anc) - pd.Timedelta(days=C.SEQ_LEN - 1):pd.Timestamp(anc)].dropna()
                rows.append({"reservoir": EN[tg], "site": r.site_name, "date": r.date, "week": r.date.to_period("W").start_time, "horizon": h, "alert": bool(r.alert),
                             "summer": r.date.month in (6, 7, 8, 9), "calendar": float(cl[tg].clim.get(r.date, np.nan)),
                             "current": float(np.log1p(w.iloc[-1])) if len(w) else np.nan,
                             "pinn": float(np.log1p(pinn.get((anc, h), np.nan))) if (anc, h) in pinn.index else np.nan,
                             "harmonic": float(np.log1p(harm.get((anc, h), np.nan))) if (anc, h) in harm.index else np.nan})
    e = pd.DataFrame(rows); e.to_csv(OUT / "6_alert_samples.csv", index=False)

    def auc(y, s):
        y, s = np.asarray(y, bool), np.asarray(s, float); m = ~np.isnan(s); y, s = y[m], s[m]
        if y.sum() == 0 or (~y).sum() == 0:
            return np.nan
        rk = pd.Series(s).rank().to_numpy(); return float((rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))

    feats = {"calendar": ["calendar"], "calendar + current": ["calendar", "current"], "calendar + current + PINN": ["calendar", "current", "pinn"],
             "calendar + current + harmonic ARIMA": ["calendar", "current", "harmonic"]}
    res = []; rng = np.random.RandomState(11)
    groups = [((rv, h, s), gg) for (rv, h), gg0 in e.groupby(["reservoir", "horizon"]) for s, gg in (("all", gg0), ("Jun-Sep", gg0[gg0.summer]))]
    for (rv, h, season), g in groups:
        g = g.dropna(subset=["calendar", "current", "pinn", "harmonic"]).reset_index(drop=True)
        if g.alert.sum() < 3 or (~g.alert).sum() < 3:
            continue
        oof = {}
        for name, cols in feats.items():
            p = np.full(len(g), np.nan); gkf = GroupKFold(n_splits=min(5, g.week.nunique()))
            for tr, te in gkf.split(g, g.alert, g.week):
                if g.alert.iloc[tr].nunique() < 2:
                    continue
                sc = StandardScaler().fit(g.loc[tr, cols]); lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(g.loc[tr, cols]), g.alert.iloc[tr])
                p[te] = lr.predict_proba(sc.transform(g.loc[te, cols]))[:, 1]
            oof[name] = p
        singles = {k: g[k].to_numpy(float) for k in ("calendar", "current", "pinn", "harmonic")}
        allp = {**{f"single: {k}": v for k, v in singles.items()}, **{f"logistic: {k}": v for k, v in oof.items()}}
        weeks = g.week.unique(); wi = {w: np.where(g.week.values == w)[0] for w in weeks}; yv = g.alert.to_numpy()
        boot = {k: [] for k in allp}
        for _ in range(2000):
            ii = np.concatenate([wi[w] for w in rng.choice(weeks, len(weeks), replace=True)])
            for k, v in allp.items():
                boot[k].append(auc(yv[ii], v[ii]))
        refk = "logistic: calendar + current"
        for k, v in allp.items():
            d = np.array(boot[k]) - np.array(boot[refk])
            res.append({"reservoir": rv, "horizon": h, "season": season, "predictor": k, "n": len(g), "n_alert": int(yv.sum()), "auc": auc(yv, v),
                        "auc_lo": float(np.nanpercentile(boot[k], 2.5)), "auc_hi": float(np.nanpercentile(boot[k], 97.5)),
                        "diff_vs_calendar_current": auc(yv, v) - auc(yv, allp[refk]), "diff_lo": float(np.nanpercentile(d, 2.5)), "diff_hi": float(np.nanpercentile(d, 97.5))})
    r = pd.DataFrame(res); r.to_csv(OUT / "6_alert_auc.csv", index=False)
    note("## 6. Cyanobacteria alerts 7 and 14 days ahead (AUC; logistic = week-grouped 5-fold out-of-fold; diff vs calendar + current, week-block bootstrap)\n\n" + md(r) + "\n")


def manifest():
    rows = [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in sorted(OUT.glob("*")) if p.is_file() and p.name != "MANIFEST.csv"]
    me = Path(__file__); rows.append({"file": f"code:{me.name}", "sha256": hashlib.sha256(me.read_bytes()).hexdigest(), "bytes": me.stat().st_size})
    pd.DataFrame(rows).to_csv(OUT / "MANIFEST.csv", index=False)


PARTS = {"screen": screen, "recovery": recovery, "ceiling": ceiling, "ladder": ladder, "equifinality": equifinality, "alerts": alerts}

if __name__ == "__main__":
    df = C.load_daily(); todo = sys.argv[1:] or list(PARTS)
    note(f"# v5 additions (started {time.strftime('%Y-%m-%d %H:%M')}; parts {', '.join(todo)})\n")
    for k in todo:
        t0 = time.time()
        try:
            PARTS[k](df); note(f"_{k} finished in {time.time() - t0:.0f} s_\n")
        except Exception as ex:
            import traceback; note(f"_{k} FAILED: {ex!r}_\n```\n{traceback.format_exc()}\n```\n")
    manifest(); note(f"Finished {time.strftime('%Y-%m-%d %H:%M')}")
