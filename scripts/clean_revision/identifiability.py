"""Identifiability module for the physics runs. No network is retrained.

1. Kinetic parameters of every physics run: jointly trained values and values fitted to convergence with the network
   fixed (physics_report.json), position in the literature range (0 = lower bound, 1 = upper bound), kinetic explanatory
   power on the test years.
2. Profiles (Raue et al. 2009): each kinetic parameter fixed on a grid while all others are re-fitted, free positive
   parameterisation, network of each sigmoid run at physics weight 0.3. Two objectives: the kinetic regression alone
   (sequence residuals, which do not involve the network) and the physics part of the training loss (sequence + one-day
   prediction residuals). For the kinetic regression an approximate 95% profile interval uses the Gaussian approximation
   n * (loss / loss_min - 1) <= 3.84 with n the number of distinct consecutive observed pairs. Autocorrelated residuals
   make the real interval wider, so an interval that already spans the literature range is a conservative finding.
3. Kinetic explanatory power at 1, 7 and 14 days, rivers and reservoirs: ln C(t+k) - ln C(t) = sum of daily net rates,
   fitted on the training years, R2 on the test years.
Writes its tables to the folder OUT below (not part of this package)."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from physics import PhysicsParams, converge_physics, physics_losses
from models import build_backbone, Net
from train_dl import predict

V4 = C.ROOT / "results" / "clean_revision_v4"; OUT = V4 / "identifiability"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (42, 123, 456); KEYS = ("K_N", "K_P", "r", "m", "K_I")
LIT = {"K_N": (0.01, 0.5), "K_P": (0.001, 0.05), "r": (0.01, 0.2), "m": (0.005, 0.1), "K_I": (50.0, 1000.0)}
GRIDS = {"K_N": np.logspace(-2, 1, 16), "K_P": np.logspace(-3, 0, 16), "K_I": np.logspace(1, 4, 16),
         "r": np.logspace(-3, math.log10(0.5), 12), "m": np.logspace(-3, math.log10(0.5), 12)}
CHI2_95 = 3.841


def load_net(tag):
    cfg = json.load(open(V4 / "runs" / tag / "config.json"))
    bb = build_backbone(cfg["backbone"], cfg["n_features"], len(C.STATION_IDS), C.SEQ_LEN, len(cfg["horizons"]), cfg["hidden"], cfg["layers"], cfg["dropout"])
    ph = PhysicsParams(cfg["physics"], len(C.STATION_IDS), cfg["kn_max"], cfg["kp_max"], bool(cfg["light"]), param=cfg.get("phys_param", "softplus"), null=bool(cfg.get("phys_null", 0)))
    net = Net(bb, ph).to(dev); net.load_state_dict(torch.load(V4 / "runs" / tag / "best_model.pt", map_location=dev)); net.eval()
    return net, cfg


def run_table():
    rows = []
    for rd in sorted((V4 / "runs").glob("*")):
        f = rd / "physics_report.json"
        if not f.exists():
            continue
        rep, cfg = json.load(open(f)), json.load(open(rd / "config.json"))
        base = {"tag": rd.name, "physics": cfg["physics"], "mode": cfg["mode"], "target": cfg.get("target_station") or "", "record_months": cfg.get("record_months"),
                "record_end": cfg.get("record_end", ""), "kn_max": cfg["kn_max"], "w_max": cfg["w_max"], "seed": cfg["seed"], "w_phys_best": cfg.get("w_phys_best")}
        for est in ("joint", "converged"):
            p = rep[est]["params"]
            r = base | {"estimate": est, **{k: p[k] for k in KEYS}, "r_plus_m": p["r"] + p["m"],
                        "kin_r2_train": rep[est]["kinetic_r2_train"].get("r2"), "kin_r2_test": rep[est]["kinetic_r2_test"].get("r2")}
            for k in KEYS:
                lo, hi = LIT[k] if not (k == "K_N") else (0.01, cfg["kn_max"])
                r[f"pos_{k}"] = (p[k] - lo) / (hi - lo)
            if est == "converged":
                r.update({"fit_loss": rep[est]["fit"]["loss_final"], "fit_grad_norm": rep[est]["fit"]["grad_norm_final"]})
            rows.append(r)
    return pd.DataFrame(rows)


def distinct_pairs(arr, idx):
    """Number of distinct (station, day) consecutive observed pairs with valid drivers among the windows in idx."""
    pv = arr["phys_valid"][idx]; ok = pv[:, :-1, 3] & pv[:, 1:, 3] & pv[:, :-1, 0] & pv[:, :-1, 1] & pv[:, :-1, 2] & pv[:, :-1, 4]
    days = arr["anchor"][idx][:, None] - np.arange(C.SEQ_LEN - 1, 0, -1)[None, :].astype("timedelta64[D]")
    sid = np.repeat(arr["sid"][idx][:, None], C.SEQ_LEN - 1, 1)
    return len(set(zip(sid[ok].tolist(), days[ok].astype("datetime64[D]").astype(str).tolist())))


def profiles(arr, tr, scaler):
    pr, pv, sid = (torch.as_tensor(arr[k][tr], device=dev) for k in ("phys_raw", "phys_valid", "sid"))
    n_pairs = distinct_pairs(arr, tr); rows = []
    for s in SEEDS:
        tag = f"src_xLSTM_pinn-sigmoid_s{s}"
        if not (V4 / "runs" / tag / "DONE").exists():
            continue
        net, cfg = load_net(tag)
        p1 = torch.as_tensor((predict(net, arr, tr, dev, cfg["residual"])[:, 0] * (scaler.y_hi - scaler.y_lo) + scaler.y_lo).astype(np.float32), device=dev)
        for objective, pp in (("kinetic_regression", None), ("training_physics_loss", p1)):
            free = PhysicsParams("none", len(C.STATION_IDS), 0.5, 0.05, True, param="exp").to(dev)
            loss_min = converge_physics(free, pr, pv, sid, pp)["loss_final"]
            best = {k: v.detach().clone() for k, v in free.state_dict().items()}
            for key, grid in GRIDS.items():
                prof = PhysicsParams("none", len(C.STATION_IDS), 0.5, 0.05, True, param="exp").to(dev); prof.load_state_dict(best)
                lo, hi = prof.ranges[key]; centre = (lo + hi) / 2
                for v in grid:
                    with torch.no_grad():
                        prof.raw[key].fill_(math.log(v / centre))
                    info = converge_physics(prof, pr, pv, sid, pp, steps=800, fixed=(key,))
                    rows.append({"seed": s, "objective": objective, "param": key, "value": float(v), "loss": info["loss_final"], "loss_min_free": loss_min,
                                 "n_distinct_pairs": n_pairs, **{f"opt_{k}": x for k, x in prof.snapshot().items() if k != "mu_max_sites"}})
            print("profiles", tag, objective, "done", flush=True)
    return pd.DataFrame(rows)


def profile_summary(pf):
    out = []
    for (s, obj, key), g in pf.groupby(["seed", "objective", "param"]):
        lmin = min(g.loss.min(), g.loss_min_free.iloc[0]); lo, hi = LIT[key]
        inb = g[(g.value >= lo) & (g.value <= hi)]
        row = {"seed": s, "objective": obj, "param": key, "argmin": float(g.value.iloc[int(np.argmin(g.loss.values))]),
               "rel_range_in_literature": float(inb.loss.max() / inb.loss.min() - 1) if len(inb) else np.nan,
               "rel_range_grid": float(g.loss.max() / g.loss.min() - 1)}
        if obj == "kinetic_regression":
            inside = g[g.n_distinct_pairs * (g.loss / lmin - 1) <= CHI2_95]; step = float(g.value.iloc[1] / g.value.iloc[0])   # grid ratio
            row.update({"ci95_lo": float(inside.value.min()) if len(inside) else np.nan, "ci95_hi": float(inside.value.max()) if len(inside) else np.nan,
                        "ci_spans_literature_range": bool(len(inside) and inside.value.min() <= lo * step and inside.value.max() >= hi / step)})
        out.append(row)
    return pd.DataFrame(out)


def md_table(d: pd.DataFrame) -> str:
    d = d.reset_index() if not isinstance(d.index, pd.RangeIndex) else d
    cols = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in d.columns]
    fmt = lambda v: "" if isinstance(v, float) and np.isnan(v) else (f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v))
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)])


# ---- 3. multi-day kinetic explanatory power ----
CENTER = {k: (lo + hi) / 2 for k, (lo, hi) in (LIT | {"mu_max": (0.1, 1.5)}).items()}; CENTER["rm"] = CENTER["r"] + CENTER["m"]
KKEYS = ["K_N", "K_P", "rm", "K_I"]


def station_series(df, st):
    d = df[df.station == st].sort_values("date").reset_index(drop=True)
    T, _, _ = C.causal_fill(d.water_temperature.to_numpy(float), 3); N, _, _ = C.causal_fill(d.total_nitrogen.to_numpy(float), 3)
    P, _, _ = C.causal_fill(d.total_phosphorus.to_numpy(float), 3)
    return d.date.to_numpy(), T, N, P, d.par_umol.to_numpy(float), d.chlorophyll_a.to_numpy(float)


def build_pairs(df, stations, k, split):
    s0, s1 = np.datetime64(C.SPLITS[split][0]), np.datetime64(C.SPLITS[split][1]); rows = []
    for st in stations:
        dates, T, N, P, I, Cc = station_series(df, st)
        for i in range(len(dates) - k):
            if not (s0 <= dates[i] and dates[i + k] <= s1) or np.isnan(Cc[i]) or np.isnan(Cc[i + k]):
                continue
            seg = slice(i, i + k)
            if np.isnan(T[seg]).any() or np.isnan(N[seg]).any() or np.isnan(P[seg]).any() or np.isnan(I[seg]).any():
                continue
            rows.append((C.STATION_IDS[st], np.log(Cc[i + k] + 1e-3) - np.log(Cc[i] + 1e-3), T[seg], N[seg], P[seg], I[seg]))
    sid = torch.tensor([r[0] for r in rows], device=dev); y = torch.tensor([r[1] for r in rows], dtype=torch.float32, device=dev)
    return sid, y, [torch.tensor(np.stack([r[j] for r in rows]), dtype=torch.float32, device=dev) for j in (2, 3, 4, 5)]


class Kin(torch.nn.Module):
    def __init__(self, mode):
        super().__init__(); self.mode = mode
        self.raw = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.zeros((), device=dev)) for k in KKEYS}); self.mu = torch.nn.Parameter(torch.zeros(6, device=dev))

    def v(self, k, raw):
        if self.mode == "sigmoid":
            lo, hi = (LIT["r"][0] + LIT["m"][0], LIT["r"][1] + LIT["m"][1]) if k == "rm" else (LIT | {"mu_max": (0.1, 1.5)})[k]
            return lo + torch.sigmoid(raw) * (hi - lo)
        return CENTER[k] * torch.exp(raw)

    def cum_net(self, sid, T, N, P, I):
        mu = self.v("mu_max", self.mu)[sid][:, None]; N, P, I = F.relu(N), F.relu(P), F.relu(I)
        f = 1.047 ** (torch.clamp(T, 0, 40) - 20) * N / (self.v("K_N", self.raw["K_N"]) + N) * P / (self.v("K_P", self.raw["K_P"]) + P) * I / (self.v("K_I", self.raw["K_I"]) + I)
        return (mu * f - self.v("rm", self.raw["rm"])).sum(1)


def multiday(df):
    def r2(y, p):
        y, p = y.detach().cpu().numpy(), p.detach().cpu().numpy(); return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    rows = []
    for group, stations in {"rivers (source)": C.SOURCE, "Yongdam Reservoir": ["용담호"], "Daecheong Reservoir": ["대청호"]}.items():
        for k in (1, 7, 14):
            tr, te = build_pairs(df, stations, k, "train"), build_pairs(df, stations, k, "test")
            for mode in ("sigmoid", "free"):
                torch.manual_seed(0); kin = Kin(mode); opt = torch.optim.Adam(kin.parameters(), lr=0.05)
                for _ in range(3000):
                    l = F.smooth_l1_loss(kin.cum_net(tr[0], *tr[2]), tr[1]); opt.zero_grad(); l.backward(); opt.step()
                with torch.no_grad():
                    rows.append({"group": group, "k_days": k, "mode": mode, "n_train": int(len(tr[1])), "n_test": int(len(te[1])),
                                 "r2_train": r2(tr[1], kin.cum_net(tr[0], *tr[2])), "r2_test": r2(te[1], kin.cum_net(te[0], *te[2])),
                                 **{kk: float(kin.v(kk, kin.raw[kk])) for kk in KKEYS}})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True); torch.manual_seed(0)
    rt = run_table(); rt.to_csv(OUT / "run_parameters.csv", index=False)
    df = C.load_daily()
    trm = ((df["date"] >= C.SPLITS["train"][0]) & (df["date"] <= C.SPLITS["train"][1])).to_numpy()
    scaler = C.Scaler().fit(df, trm, C.SOURCE); arr = C.build_arrays(df, scaler, list(C.STATION_IDS), C.SEQ_LEN, C.HORIZONS)
    tr = C.split_indices(arr, "train", C.SOURCE); tr = tr[C.mask_targets_to_split(arr, tr, "train").sum(1) > 0]
    pf = profiles(arr, tr, scaler); pf.to_csv(OUT / "profiles.csv", index=False)
    ps = profile_summary(pf); ps.to_csv(OUT / "profile_summary.csv", index=False)
    md = multiday(df); md.to_csv(OUT / "kinetic_multiday.csv", index=False)

    src = rt[(rt.target == "") & (rt.w_max == 0.3) & (rt.kn_max == 0.5)]
    lines = ["# Identifiability of the kinetic parameters (v4)", "",
             "## 1. Source runs at physics weight 0.3: joint vs converged estimates (mean over seeds; position 0 = lower, 1 = upper literature bound)", "",
             md_table(src.groupby(["physics", "estimate"])[["K_N", "K_P", "K_I", "r_plus_m", "pos_K_N", "pos_K_P", "pos_K_I", "kin_r2_train", "kin_r2_test"]].mean()), "",
             "Seed spread of the converged values:", "", md_table(src[src.estimate == "converged"].groupby("physics")[["K_N", "K_P", "K_I"]].agg(["min", "max"])), "",
             "## 2. Profiles (loss range inside the literature range; approximate 95% interval for the kinetic regression, per seed)", "",
             md_table(ps[["seed", "objective", "param", "rel_range_in_literature", "rel_range_grid", "argmin", "ci95_lo", "ci95_hi", "ci_spans_literature_range"]].sort_values(["objective", "param", "seed"]).reset_index(drop=True)), "",
             "## 3. Kinetic explanatory power on the test years (R2 of ln C(t+k) - ln C(t))", "",
             md_table(md.pivot_table(index=["group", "mode"], columns="k_days", values="r2_test")), ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8"); print("\n".join(lines))


if __name__ == "__main__":
    main()
