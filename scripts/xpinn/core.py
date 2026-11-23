"""Automatic search for an accurate and explainable process-based forecaster of chlorophyll-a: a pilot system of
automatic experiments (the first search for the process model).

A network (or one constant per station) gives the parameters of a daily mass balance; the mass balance, iterated along the
river network, makes the 1-14-day forecasts. Variant factors: growth law (g), parameter structure (p), transport (t),
residual (r), driver trajectory (d), kinetic constants (k). Black-box references (networks without process terms): bb-C1 (joint xLSTM, the joint network of the paper) and bb-C2 (bb-C1 with
the upstream stations' Chl-a inputs). Data and training follow the Nakdong pilot (scripts/national/chain_nakdong.py).
Usage: core.py --variant NAME --net nakdong --split dev --record 24 --seed 42 [--placebo]
(--placebo: kinetic placebo, the same run with the driver trajectories shuffled across days within each station before
training; the network inputs are unchanged)"""
from __future__ import annotations
import argparse, copy, json, os, random, sys, time
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "clean_revision")); sys.path.insert(0, str(ROOT / "scripts" / "national"))
import nat as N                                                                  # noqa: E402
C = N.patch()
from models import build_backbone                                                # noqa: E402
from chain_nakdong import CHAIN, UP                                              # noqa: E402

OUT = Path(os.environ["XPINN_OUT"]) if os.environ.get("XPINN_OUT") else ROOT / "results" / "xpinn_20261027"   # code tests write elsewhere
H, HMAX, L = [1, 3, 7, 14], 14, 7                     # horizons; rollout days; days of upstream history (max travel time)
THETA, FIXED_K = 1.047, {"KN": 0.255, "KP": 0.0255, "KI": 525.0}
RANGE = {"mu": (0.1, 1.5), "loss": (0.015, 0.3), "K": (5.0, 300.0), "kappa": (0.0, 0.5), "w": (0.0, 0.9), "share": (0.0, 1.0),
         "tau1": (0.0, 5.0), "tau2": (0.0, 5.0), "v": (5.0, 100.0), "KN": (0.01, 0.5), "KP": (0.001, 0.05), "KI": (50.0, 1000.0)}
EPS_MAX, EPS_PEN = 0.1, 10.0                           # bounded residual of the daily log growth rate, and its penalty
SPLITS = {"dev": ("2022-12-31", "2023-01-01", "2023-12-31"), "test": ("2023-12-31", "2024-01-01", "2025-12-31")}  # record end, evaluation
NETWORKS = {"nakdong": (CHAIN, UP), **{b.lower(): (N.FOLDS[b], {}) for b in ("Han", "Geum", "Yeongsan")}}
# Stage 5: optional input length (a shorter input than 30 days) and station list (stations whose data cover the whole
# period). Both stay off unless the environment sets them, so every earlier run is unchanged.
if os.environ.get("XPINN_SEQ_LEN"):
    C.SEQ_LEN = int(os.environ["XPINN_SEQ_LEN"]); assert C.SEQ_LEN >= 7, "the 7-day driver means need at least 7 input days"
if os.environ.get("XPINN_STATIONS"):
    _keep = set(Path(os.environ["XPINN_STATIONS"]).read_text().split())
    NETWORKS = {n: ([x for x in st if x in _keep], {x: [u for u in us if u in _keep] for x, us in up.items() if x in _keep and any(u in _keep for u in us)})
                for n, (st, up) in NETWORKS.items()}
FACTORS = {"g": ("exp", "ricker", "relax"), "p": ("daily", "station", "anchored"), "t": ("none", "free", "distance"),
           "r": ("none", "bounded"), "d": ("hold", "clim"), "k": ("fixed", "learned")}
DRIVERS = ["water_temperature", "total_nitrogen", "total_phosphorus", "par_umol"]


@dataclass(frozen=True)
class Variant:
    g: str = "exp"; p: str = "daily"; t: str = "free"; r: str = "none"; d: str = "hold"; k: str = "fixed"; bb: str = ""

    @staticmethod
    def parse(name: str) -> "Variant":
        if name.startswith("bb-"):
            return Variant(bb=name[3:])
        v = Variant(**dict(x.split("-", 1) for x in name.split(".")))
        assert all(getattr(v, f) in lv for f, lv in FACTORS.items()), name
        return v

    @property
    def name(self) -> str:
        return f"bb-{self.bb}" if self.bb else ".".join(f"{f}-{getattr(self, f)}" for f in FACTORS)

    def params(self) -> list[str]:
        """Parameters given per station (and day, unless p = station)."""
        p = ["mu", "loss"] + (["K"] if self.g == "ricker" else []) + (["kappa"] if self.g == "relax" else [])
        p += ["w", "share"] if self.t != "none" else []
        return p + (["tau1", "tau2"] if self.t == "free" else [])

    def n_daily(self) -> int:
        """Number of quantities that change from day to day (parsimony; tie-break only)."""
        return (len(self.params()) if self.p != "station" else 0) + (1 if self.r == "bounded" else 0)

    def without_transport(self) -> "Variant":
        return replace(self, t="none") if not self.bb else self


BASELINE = Variant()


def tag(v: str, net: str, split: str, rec: int, seed: int, placebo: bool = False) -> str:
    return f"{v}__{net}_{split}_r{rec}_s{seed}" + ("__placebo" if placebo else "")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def fourier(doy, K=2):
    doy = np.asarray(doy, float); cols = [np.ones_like(doy)]
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    return np.stack(cols, -1)


def haversine(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    return float(2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)))


# ------------------------------------------------------------------------------------------------------------- data
def prepare(net: str, split: str, rec: int) -> dict:
    stations, up = NETWORKS[net]
    rec_end, ev0, ev1 = (pd.Timestamp(x) for x in SPLITS[split])
    df = N.load_daily(); df = df[df.station.isin(stations)].reset_index(drop=True)
    start = rec_end - pd.DateOffset(months=rec) + pd.Timedelta(days=1)
    days = pd.date_range(start, rec_end); vstart = days[-max(1, int(round(len(days) * N.VAL_FRAC)))]; tr_end = vstart - pd.Timedelta(days=1)
    sc = C.Scaler().fit(df, ((df.date >= start) & (df.date <= rec_end)).to_numpy(), stations)          # inside the record only
    arrs = {s: C.build_arrays(df, sc, [s], C.SEQ_LEN, H) for s in stations}
    anc = arrs[stations[0]]["anchor"]
    for s in stations[1:]:
        anc = np.intersect1d(anc, arrs[s]["anchor"])
    assert (np.diff(anc) == np.timedelta64(1, "D")).all(), "anchors must be consecutive days"
    for s in stations:
        keep = np.isin(arrs[s]["anchor"], anc); arrs[s] = {k: (a[keep] if isinstance(a, np.ndarray) else a) for k, a in arrs[s].items()}
    S, T = len(stations), len(anc); idx = {s: i for i, s in enumerate(stations)}
    fn = arrs[stations[0]]["feat_names"]
    X = np.stack([arrs[s]["X"] for s in stations], 1).astype(np.float32)                                 # (T, S, 30, F)
    ch = [fn.index(c) for c in ("chlorophyll_a", "chlorophyll_a__mask", "chlorophyll_a__age", "chlorophyll_a__last_obs")]
    upx = np.zeros(X.shape[:3] + (8,), np.float32)                                                     # upstream Chl-a inputs (bb-C2)
    for s, us in up.items():
        for k, u in enumerate(us):
            upx[:, idx[s], :, 4 * k:4 * k + 4] = X[:, idx[u]][:, :, ch]
    yraw = np.stack([arrs[s]["y_raw"] for s in stations], 1); ym = np.stack([arrs[s]["y_mask"] for s in stations], 1)
    base = np.maximum(np.stack([sc.y_unscale(arrs[s]["base"].astype(float)) for s in stations], 1), 1e-3)   # last measured Chl-a
    pad = np.concatenate([np.repeat(base[:1], L, 0), base], 0)
    hist = np.stack([pad[j:j + T] for j in range(L + 1)], -1)                                            # (T, S, L+1): t-L .. t
    pr = np.stack([arrs[s]["phys_raw"][:, -7:, :] for s in stations], 1); pv = np.stack([arrs[s]["phys_valid"][:, -7:, :] for s in stations], 1)
    den = pv.sum(2); drv7 = np.where(den > 0, (pr * pv).sum(2) / np.maximum(den, 1), np.nan)[..., [0, 1, 2, 4]]   # 7-day means T, N, P, PAR
    inrec = (anc >= np.datetime64(start.date())) & (anc <= np.datetime64(rec_end.date()))
    for c in range(4):
        col = drv7[inrec][:, :, c]; med = np.array([np.nanmedian(col[:, i]) if np.isfinite(col[:, i]).any() else np.nan for i in range(S)])
        med = np.where(np.isnan(med), np.nanmedian(col), med); drv7[:, :, c] = np.where(np.isnan(drv7[:, :, c]), med[None, :], drv7[:, :, c])
    # seasonal climatologies fitted on the record (annual Fourier, K = 2): drivers, and log Chl-a for the relaxation target
    ad = pd.DatetimeIndex(anc)
    Fk = fourier(np.stack([(ad + pd.Timedelta(days=k)).dayofyear for k in range(1, HMAX + 1)], 1))          # (T, 14, 5)
    Fw = fourier(np.stack([(ad - pd.Timedelta(days=j)).dayofyear for j in range(7)], 1))                    # (T, 7, 5)
    decay = np.exp(-np.arange(1, HMAX + 1) / 7.0)
    drv_clim = np.zeros((T, S, HMAX, 4)); mclim = np.zeros((T, S, HMAX))
    for s in stations:
        d = df[df.station == s].set_index("date").sort_index(); fit = ((d.index >= start) & (d.index <= rec_end)); Xd = fourier(d.index.dayofyear)
        for c, col in enumerate(DRIVERS):
            y = d[col].to_numpy(float); m = fit & np.isfinite(y)
            beta = np.linalg.lstsq(Xd[m], y[m], rcond=None)[0] if m.sum() >= 60 else np.r_[np.median(y[m]) if m.any() else drv7[:, idx[s], c].mean(), np.zeros(4)]
            drv_clim[:, idx[s], :, c] = Fk @ beta + (drv7[:, idx[s], c] - (Fw @ beta).mean(1))[:, None] * decay[None, :]
        y = np.log(np.maximum(d.chlorophyll_a.to_numpy(float), 0.1)); m = fit & np.isfinite(y)
        beta = np.linalg.lstsq(Xd[m], y[m], rcond=None)[0] if m.sum() >= 60 else np.r_[np.median(y[m]) if m.any() else 0.0, np.zeros(4)]
        mclim[:, idx[s], :] = Fk @ beta
    drv_clim[..., 0] = np.clip(drv_clim[..., 0], 0, 40); drv_clim[..., 1:] = np.maximum(drv_clim[..., 1:], 0)
    xy = pd.read_csv(N.NAT / "weis_auto_stations.csv", comment="#", header=None, names=["code", "name", "lat", "lon"], dtype={"code": str}).set_index("code")
    dist = np.zeros((S, 2), np.float32)
    for s, us in up.items():
        for k, u in enumerate(us):
            dist[idx[s], k] = haversine(xy.loc[u, ["lat", "lon"]].to_numpy(float), xy.loc[s, ["lat", "lon"]].to_numpy(float))
    tgt = arrs[stations[0]]["target"]
    sel = lambda lo, hi: np.where((anc >= np.datetime64(lo.date())) & (anc <= np.datetime64(hi.date())))[0]
    ylog = np.log1p(np.nan_to_num(yraw, nan=0.0)).astype(np.float32)
    # Stage 5b: network inputs of the listed variables only (selection of the input variables). A water-quality variable brings
    # its value, measured flag and age (Chl-a also its last measured value). Off unless the environment sets it, so every earlier run is unchanged.
    if os.environ.get("XPINN_INPUTS"):
        _v = set(os.environ["XPINN_INPUTS"].split(",")); assert _v <= set(C.VARS + C.ERA5), sorted(_v - set(C.VARS + C.ERA5))
        _k = [i for i, c in enumerate(fn) if c.split("__")[0] in _v]; X, fn = np.ascontiguousarray(X[..., _k]), [fn[i] for i in _k]
    D = {"stations": stations, "up": up, "dist": dist, "X": X, "up_x": upx, "feat": list(fn), "base": base.astype(np.float32), "hist": hist.astype(np.float32),
         "drv_hold": np.repeat(drv7[:, :, None, :], HMAX, 2).astype(np.float32), "drv_clim": drv_clim.astype(np.float32), "mclim": mclim.astype(np.float32),
         "ylog": ylog, "yraw": yraw, "anc": anc, "tgt": tgt,
         "m_tr": (ym * (tgt <= np.datetime64(tr_end.date()))[:, None, :]).astype(np.float32),
         "m_va": (ym * (tgt <= np.datetime64(rec_end.date()))[:, None, :]).astype(np.float32),
         "m_ev": (ym * (tgt <= np.datetime64(ev1.date()))[:, None, :]).astype(np.float32),
         "tr": sel(start, tr_end), "va": sel(vstart, rec_end), "ev": sel(ev0, ev1),
         "record": {"start": str(start.date()), "val_start": str(vstart.date()), "end": str(rec_end.date()), "eval": [str(ev0.date()), str(ev1.date())]}}
    dl = (ylog - np.log1p(base)[:, :, None])[D["tr"]]; m = D["m_tr"][D["tr"]] > 0
    var = np.array([dl[..., k][m[..., k]].var() if m[..., k].sum() > 2 else 1.0 for k in range(len(H))])
    hw = 1 / np.maximum(var, 1e-6); D["hw"] = (hw / hw.mean()).astype(np.float32)
    chl = df.dropna(subset=["chlorophyll_a"])
    D["counts"] = {s: {"chl_train": int(((chl.station == s) & (chl.date >= start) & (chl.date <= tr_end)).sum()),
                       "chl_val": int(((chl.station == s) & (chl.date >= vstart) & (chl.date <= rec_end)).sum()),
                       "eval_targets": [int(D["m_ev"][D["ev"], i, k].sum()) for k in range(len(H))]} for i, s in enumerate(stations)}
    return D


# ------------------------------------------------------------------------------------------------------------- model
def bound(z, name):
    lo, hi = RANGE[name]
    return lo + torch.sigmoid(z) * (hi - lo)


class XNet(nn.Module):
    def __init__(self, v: Variant, n_feat: int, stations: list, up: dict, dist):
        super().__init__()
        self.v, self.stations, self.up, self.S = v, stations, up, len(stations)
        self.idx = {s: i for i, s in enumerate(stations)}
        self.register_buffer("sid", torch.tensor([C.STATION_IDS[s] for s in stations]))
        self.register_buffer("dist", torch.tensor(dist, dtype=torch.float32))
        self.pn = [] if v.bb else v.params()
        self.n_nn = len(self.pn) if (not v.bb and v.p != "station") else 0
        n_out = 4 if v.bb else self.n_nn + (1 if v.r == "bounded" else 0)
        self.bb = build_backbone("xLSTM", n_feat + (8 if v.bb == "C2" else 0), len(C.STATION_IDS), C.SEQ_LEN, n_out) if n_out else None
        self.zs = nn.Parameter(torch.zeros(self.S, len(self.pn))) if (not v.bb and v.p != "daily") else None
        self.zk = nn.Parameter(torch.zeros(self.S, 3)) if (not v.bb and v.k == "learned") else None
        self.zv = nn.Parameter(torch.zeros(())) if (not v.bb and v.t == "distance") else None

    def constants(self):
        return [p for p in (self.zs, self.zk, self.zv) if p is not None]

    def raw(self, x):
        B = x.shape[0]
        o = self.bb(x.reshape(B * self.S, *x.shape[2:]), self.sid.repeat(B))
        return (o[0] if isinstance(o, tuple) else o).reshape(B, self.S, -1)

    def physical(self, x):
        """Parameters (B, S) by name, the residual (B, S) or None, kinetic constants and the travel speed."""
        B = x.shape[0]; z = self.raw(x) if self.bb is not None else None
        if self.v.p == "daily":
            zp = z[..., :self.n_nn]
        elif self.v.p == "station":
            zp = self.zs.expand(B, -1, -1)
        else:                                                                  # anchored: station constant +- 1 logit per day
            zp = self.zs + torch.tanh(z[..., :self.n_nn])
        P = {n: bound(zp[..., j], n) for j, n in enumerate(self.pn)}
        eps = EPS_MAX * torch.tanh(z[..., -1]) if self.v.r == "bounded" else None
        kin = ({n: bound(self.zk[:, j], n) for j, n in enumerate(("KN", "KP", "KI"))} if self.zk is not None
               else {n: torch.full((self.S,), FIXED_K[n], device=x.device) for n in ("KN", "KP", "KI")})
        if self.zv is not None:
            P["v"] = bound(self.zv, "v")
        return P, eps, kin

    def forward(self, x, base, hist, drv, mclim, transport=True, resid=True):
        """log1p forecasts (B, S, 4) and the physical quantities (None for the black boxes)."""
        if self.v.bb:
            return torch.log1p(base)[..., None] + self.raw(x), None
        P, eps, kin = self.physical(x)
        T, Nn, Pp, I = drv[..., 0].clamp(0, 40), drv[..., 1].clamp_min(0), drv[..., 2].clamp_min(0), drv[..., 3].clamp_min(0)
        KN, KP, KI = (kin[n][None, :, None] for n in ("KN", "KP", "KI"))
        phi = THETA ** (T - 20) * Nn / (KN + Nn) * Pp / (KP + Pp) * I / (KI + I)                     # (B, S, 14)
        gpos = P["mu"][..., None] * phi
        use_t = self.v.t != "none" and transport
        tau_d = (self.dist / P["v"]).clamp(0, L) if (use_t and self.v.t == "distance") else None
        state = [hist[:, i, :] for i in range(self.S)]; out = []
        for k in range(1, HMAX + 1):
            for i, s in enumerate(self.stations):                             # stations are listed upstream first
                own = state[i][:, -1]
                if use_t and s in self.up:
                    inflow, us = torch.zeros_like(own), self.up[s]
                    for slot, u in enumerate(us):
                        tau = (P["tau1"][:, i] if slot == 0 else P["tau2"][:, i]) if tau_d is None else tau_d[i, slot].expand_as(own)
                        sj = state[self.idx[u]]; pos = (L + k) - tau             # upstream already advanced to t + k
                        lo = pos.floor().long().clamp(0, sj.shape[1] - 1); hi = (lo + 1).clamp(max=sj.shape[1] - 1); fr = pos - lo.float()
                        val = (1 - fr) * sj.gather(1, lo[:, None]).squeeze(1) + fr * sj.gather(1, hi[:, None]).squeeze(1)
                        wgt = torch.ones_like(own) if len(us) == 1 else (P["share"][:, i] if slot == 0 else 1 - P["share"][:, i])
                        inflow = inflow + wgt * val
                    mixed = (1 - P["w"][:, i]) * own + P["w"][:, i] * inflow
                else:
                    mixed = own
                gk = gpos[:, i, k - 1]
                lg = (gk * (1 - mixed / P["K"][:, i]) if self.v.g == "ricker" else gk) - P["loss"][:, i]
                if self.v.g == "relax":
                    lg = lg - P["kappa"][:, i] * (torch.log(mixed.clamp_min(1e-3)) - mclim[:, i, k - 1])
                if eps is not None and resid:
                    lg = lg + eps[:, i]
                state[i] = torch.cat([state[i], (mixed * torch.exp(lg.clamp(-1.0, 1.0))).clamp(1e-3, 1e4)[:, None]], 1)
            if k in H:
                out.append(torch.stack([state[i][:, -1] for i in range(self.S)], 1))
        return torch.log1p(torch.stack(out, -1)), {"P": P, "eps": eps, "kin": kin, "tau_d": tau_d}


# ------------------------------------------------------------------------------------------------------------- training
def wmse(pred, y, m, hw):
    return ((pred - y) ** 2 * m * hw).sum() / (m * hw).sum().clamp_min(1.0)


def inputs(D, j, v, dev, drv=None):
    x = D["X"][j]
    if v.bb == "C2":
        x = np.concatenate([x, D["up_x"][j]], -1)
    drv = D["drv_clim" if v.d == "clim" else "drv_hold"][j] if drv is None else drv
    return [torch.as_tensor(a, device=dev) for a in (x, D["base"][j], D["hist"][j], drv, D["mclim"][j])]


def val_loss(net, D, idx, mask, v, dev, hw, bs=128):
    net.eval(); num = den = 0.0
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            j = idx[b:b + bs]; p, _ = net(*inputs(D, j, v, dev))
            y, m = torch.as_tensor(D["ylog"][j], device=dev), torch.as_tensor(mask[j], device=dev)
            num += float(((p - y) ** 2 * m * hw).sum()); den += float((m * hw).sum())
    return num / max(den, 1.0)


def train(v, D, lr, seed, dev, epochs=100, patience=15, bs=32):
    seed_all(seed); net = XNet(v, D["X"].shape[-1], D["stations"], D["up"], D["dist"]).to(dev); hw = torch.as_tensor(D["hw"], device=dev)
    cid = {id(p) for p in net.constants()}; others = [p for p in net.parameters() if id(p) not in cid]
    groups = ([{"params": others, "lr": lr, "weight_decay": 1e-4}] if others else []) + \
             ([{"params": net.constants(), "lr": min(10 * lr, 1e-2), "weight_decay": 0.0}] if cid else [])
    opt = torch.optim.Adam(groups); rng = np.random.RandomState(seed)
    tr = D["tr"][D["m_tr"][D["tr"]].sum((1, 2)) > 0]
    best, best_state, bad, ep_best = np.inf, copy.deepcopy(net.state_dict()), 0, -1
    for ep in range(epochs):
        net.train(); perm = rng.permutation(tr)
        for b in range(0, len(perm), bs):
            j = perm[b:b + bs]; p, aux = net(*inputs(D, j, v, dev))
            loss = wmse(p, torch.as_tensor(D["ylog"][j], device=dev), torch.as_tensor(D["m_tr"][j], device=dev), hw)
            if aux is not None and aux["eps"] is not None:
                loss = loss + EPS_PEN * (aux["eps"] ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        vl = val_loss(net, D, D["va"], D["m_va"], v, dev, hw)
        if vl < best - 1e-6:
            best, best_state, bad, ep_best = vl, copy.deepcopy(net.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, best, ep_best


# ------------------------------------------------------------------------------------------------------------- evaluation
def evaluate(net, D, v, dev, drv_eval=None, bs=128):
    """Forecasts at the evaluation anchors; for process variants also without the residual and without transport, and the
    physical quantities. drv_eval replaces the driver trajectories of the evaluation anchors (placebo)."""
    ev = D["ev"]; drv_all = D["drv_clim" if v.d == "clim" else "drv_hold"][ev] if drv_eval is None else drv_eval
    full, mech, local, Q = [], [], [], []
    net.eval()
    with torch.no_grad():
        for b in range(0, len(ev), bs):
            j = ev[b:b + bs]; args = inputs(D, j, v, dev, drv_all[b:b + bs])
            p, aux = net(*args); full.append(p.cpu().numpy())
            if aux is not None:
                mech.append(net(*args, resid=False)[0].cpu().numpy() if aux["eps"] is not None else full[-1])
                local.append(net(*args, transport=False)[0].cpu().numpy() if v.t != "none" else full[-1])
                q = {n: t.cpu().numpy() for n, t in aux["P"].items() if t.dim() == 2}
                if aux["eps"] is not None:
                    q["eps"] = aux["eps"].cpu().numpy()
                for n, t in aux["kin"].items():
                    q[n] = np.repeat(t.cpu().numpy()[None, :], len(j), 0)
                if "v" in aux["P"]:
                    q["v"] = np.full((len(j), net.S), float(aux["P"]["v"]))
                    tdist = aux["tau_d"].cpu().numpy() if aux["tau_d"] is not None else (D["dist"] / float(aux["P"]["v"])).clip(0, L)
                    q["tau1"], q["tau2"] = (np.repeat(tdist[None, :, c], len(j), 0) for c in (0, 1))
                Q.append(q)
    out = {"full": np.concatenate(full)}
    if Q:
        out["mech"], out["local"] = np.concatenate(mech), np.concatenate(local)
        out["par"] = {n: np.concatenate([q[n] for q in Q]) for n in Q[0]}
    return out


def frames(D, v, res, stations):
    ev = D["ev"]; S = len(stations); n = len(ev)
    st = np.tile(np.array(stations), n); ad = np.repeat(D["anc"][ev].astype(str), S)
    rows = []
    for k, h in enumerate(H):
        m = D["m_ev"][ev][:, :, k].ravel() > 0
        rows.append(pd.DataFrame({"station": st, "anchor_date": ad, "horizon": h, "target_date": np.repeat(D["tgt"][ev][:, k].astype(str), S), "observed": m,
                                  "y_true_raw": np.where(m, D["yraw"][ev][:, :, k].ravel(), np.nan), "y_pred_raw": np.expm1(res["full"][:, :, k].ravel()),
                                  "base_raw": D["base"][ev].ravel()}))
    pred = pd.concat(rows, ignore_index=True)
    if "par" not in res:
        return pred, None
    lb = np.log1p(D["base"][ev])
    par = pd.DataFrame({"station": st, "anchor_date": ad, **{n: a.ravel() for n, a in res["par"].items()},
                        "d7_full": (res["full"][:, :, 2] - lb).ravel(), "d7_mech": (res["mech"][:, :, 2] - lb).ravel(), "d7_local": (res["local"][:, :, 2] - lb).ravel()})
    nup = {s: len(D["up"].get(s, [])) for s in stations}                     # quantities that do not apply are left empty
    for col, need in (("w", 1), ("tau1", 1), ("share", 2), ("tau2", 2)):
        if col in par:
            par.loc[par.station.map(nup) < need, col] = np.nan
    return pred, par


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); ap.add_argument("--net", default="nakdong")
    ap.add_argument("--split", default="dev", choices=list(SPLITS)); ap.add_argument("--record", type=int, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--placebo", action="store_true", help="train with the driver trajectories shuffled across days within each station")
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("XPINN_EPOCHS", 100)), help="XPINN_EPOCHS: short code tests only"); a = ap.parse_args()
    v = Variant.parse(a.variant); assert not (a.placebo and v.bb)
    rd = OUT / "runs" / tag(v.name, a.net, a.split, a.record, a.seed, a.placebo); rd.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    if (rd / "DONE").exists():
        print("skip", rd.name); return
    D = prepare(a.net, a.split, a.record)
    if a.placebo:                                                             # one permutation of the days per station, both trajectories
        rng = np.random.RandomState(1000 + a.seed)
        for i in range(len(D["stations"])):
            perm = rng.permutation(len(D["anc"]))
            for key in ("drv_hold", "drv_clim"):
                D[key][:, i] = D[key][perm, i]
    runs = []
    for lr in (1e-3, 3e-4, 1e-4):
        net, best, ep = train(v, D, lr, a.seed, dev, epochs=a.epochs); runs.append((lr, net, best, ep))
        print(f"[{rd.name}] lr {lr:g} val {best:.4f} epoch {ep}", flush=True)
    lr, net, best, ep = min(runs, key=lambda r: r[2])
    torch.save(net.state_dict(), rd / "model.pt")
    pred, par = frames(D, v, evaluate(net, D, v, dev), D["stations"])
    pred.to_csv(rd / "predictions_eval.csv", index=False, float_format="%.6g")
    if par is not None:
        par.to_csv(rd / "params_eval.csv", index=False, float_format="%.6g")
    json.dump({"variant": v.name, "placebo": a.placebo, "net": a.net, "split": a.split, "record_months": a.record, "seed": a.seed, "lr_selected": lr, "best_val": best,
               "best_epoch": ep, "lr_table": [{"lr": r[0], "best_val": r[2], "best_epoch": r[3]} for r in runs], "record": D["record"],
               "counts": D["counts"], "time_s": time.time() - t0, "stations": D["stations"], "upstream": D["up"], "distance_km": D["dist"].tolist()},
              open(rd / "config.json", "w"), indent=1)
    (rd / "DONE").write_text("ok"); print(f"[{rd.name}] done {(time.time() - t0) / 60:.1f} min, lr {lr:g}", flush=True)


if __name__ == "__main__":
    main()
