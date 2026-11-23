"""Exploration of accurate and explainable process-based forecasters, second search of the process-equation design (Supplementary Note S5): designs that keep the strengths of
persistence and add named process terms, tried in many variants.

The forecast starts from the last measurement (the strength of persistence) and changes it day by day only through named
terms: C(t+k) = M(t+k) · exp(Δ), Δ = growth − loss − flushing − κ (ln M − target) [+ ε], M = own value mixed with the
upstream station's value when transport is on. A network (xLSTM) or one constant per station gives the term parameters.
Building blocks (factor: levels):
  anc   anchor: last (last measured value) | smooth (blend of the last value and the 7-day mean of measurements, weight α)
        | filter (stage 4: the anomaly from the target is filtered over the 30-day window, a <- (1 − κ) a each day and
        a <- a + g (measured anomaly − a) on measured days, gain g a station constant; it takes the place of smoothing
        and ageing: with g = 1 it is the last value aged over the days since the last measurement)
  age   no | yes (before the forecast, the anchor relaxes toward the target over the days since the last measurement)
  tgt   relaxation target: none | clim (seasonal climatology of the record) | recent (mean of the last 30 days of
        measurements) | blend (ω · clim + (1 − ω) · recent) | mean (mean of the record, batch 2) | mrec (ω · mean + (1 − ω) ·
        recent, batch 2) | mnet (record mean + a bounded shift from the network, ±0.75 in ln units, batch 3) | stage 3, the
        record mean moved toward the seasonal climatology, mean + w · (clim − mean): msy (w = (Y − 1)/Y, Y = calendar years
        of the record with at least 60 measurements, so 0 with one year) | meb (empirical-Bayes w = 1 − (year-to-year
        variance of the seasonal Fourier coefficients / Y) / (mean squared coefficient), 0 with one year: the target follows
        the season only as far as the years agree on it) | mcf (w = ω learned per station, with the climatology of the
        record's other years for anchors inside the record, so ω is fitted out of year; the record mean with one year)
  gro   growth: none | monod (μ θ^(T−20) N/(K_N+N) P/(K_P+P) I/(K_I+I) − ℓ) | temp (μ θ^(T−20) − ℓ) | anom (μa · [Monod factor
        of the driver trajectory − Monod factor of the drivers' seasonal climatology], growth anomaly, no ℓ) | sea (μa ·
        [Monod factor of the driver trajectory − its mean over the record's days], batch 2) | seaT (the same with the
        temperature factor θ^(T−20) only, batch 2)
  flush no | yes (loss β · rain, rain = mean of the last 7 days decaying with an e-folding time of 3 days over the horizon)
  par   parameter structure: daily (the network gives every parameter each day) | station (one constant per station, no
        network) | mixed (growth rates daily from the network, all other parameters station constants) | mod (every parameter
        a station constant moved each day by the network within ±1 on the logit scale, batch 3) | kmod (the same for the
        relaxation rate κ only, batch 3) | mlp, kmlp, emlp (batch 4: a small network of the drivers on the forecast day
        alone: water temperature, log N, log P, log light and log rain, standardized, in place of the xLSTM, so its response
        can be drawn: mlp moves every station constant, kmlp the relaxation rate only, emlp gives only the target shift of
        mnet) | pooled (batch 4: one constant for all stations, no network)
  kt    no | yes (relaxation rate depending on water temperature, κ θκ^(T − 20) with θκ a station constant, batch 3; left out
        of the name when no, so earlier names are unchanged)
  fit   fitting criterion (batch 4, left out of the name when log): log (squared error of ln(1 + C), horizons weighted by the
        inverse variance of the log change, so the 1-day forecast weighs most; as before) | flat (the same with equal
        weights at 3, 7 and 14 days and none at 1 day, the horizons of the score) | raw (squared error of C divided by the
        station's standard deviation of C on the training days, inverse-variance weights) | rawflat (raw with the flat weights)
  gate  no | yes (batch 5, left out of the name when no: every network output is multiplied by a learned gate that starts
        at 0, so training starts from the design without the network and the network enters only as far as the data ask;
        the gate decays toward 0 with the network's weight decay)
  tr    transport: none | dist (upstream mixing w, travel time = straight-line distance / one speed v)
  res   none | bounded (|ε| ≤ 0.1 per day, penalty 10 · mean ε²)
  drv   driver trajectory: hold (7-day means held) | clim (seasonal climatology plus the anomaly decaying over 7 days)
  kin   K_N, K_P, K_I: fixed | learned (station constants)
  pre   no | nat (stage 4, left out of the name when no: the network and the starting values of the station constants come
        from joint training on the other three basins, and the network is frozen at the target; runs through hybrid.py)
Steps: stages 3 and 4 are design steps of the process model and of the hybrid model (seasonal.py, hybrid.py); stages
5a and 5b choose the input length and the input variables of the joint network (inputlen.py, inputvars.py); stage 6
runs the process model and the hybrid model on that input (newinput.py); stage 7 runs the comparison models
(newinput_bench.py); stage 8 changes one part of the hybrid model at a time (newinput_ablate.py). Their results are in
the stage folders of results/input_design_20261114 and of results/xpinn_explore_20261027.
Usage: core2.py --spec NAME --record 24 --seed 42 [--split dev] [--net nakdong] [--placebo]"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "xpinn"))
import core as K1                                                                # noqa: E402  (data, splits, networks of the first search)
from core import C, N, build_backbone                                            # noqa: E402

OUT = Path(os.environ["XPINN2_OUT"]) if os.environ.get("XPINN2_OUT") else ROOT / "results" / "xpinn_explore_20261027"
H, HMAX, L, THETA, FIXED_K = K1.H, K1.HMAX, K1.L, K1.THETA, K1.FIXED_K
RANGE = {"mu": (0.1, 1.5), "loss": (0.015, 0.3), "mua": (0.0, 2.0), "kappa": (0.0, 0.5), "omega": (0.0, 1.0), "beta": (0.0, 0.05),
         "alpha": (0.0, 1.0), "w": (0.0, 0.9), "share": (0.0, 1.0), "v": (5.0, 100.0), "KN": (0.01, 0.5), "KP": (0.001, 0.05), "KI": (50.0, 1000.0),
         "thk": (0.9, 1.1), "tshift": (-0.75, 0.75), "gain": (0.0, 1.0)}
EPS_MAX, EPS_PEN, RAIN_EFOLD, MOD_C, KAPPA_MAX = 0.1, 10.0, 3.0, 1.0, 0.9
FACTORS = {"anc": ("last", "smooth", "filter"), "age": ("no", "yes"), "tgt": ("none", "clim", "recent", "blend", "mean", "mrec", "mnet", "msy", "meb", "mcf"),
           "gro": ("none", "monod", "temp", "anom", "sea", "seaT"),
           "flush": ("no", "yes"), "par": ("daily", "station", "mixed", "mod", "kmod", "mlp", "kmlp", "emlp", "pooled"), "tr": ("none", "dist"), "res": ("none", "bounded"),
           "drv": ("hold", "clim"), "kin": ("fixed", "learned"), "kt": ("no", "yes"),
           "fit": ("log", "flat", "raw", "rawflat"), "gate": ("no", "yes"), "pre": ("no", "nat")}
OPTIONAL = {"kt": "no", "fit": "log", "gate": "no", "pre": "no"}                                                           # factors added later: named only when not at this level
GROWTH = ("mu", "loss", "mua")


@dataclass(frozen=True)
class Spec:
    anc: str = "last"; age: str = "no"; tgt: str = "none"; gro: str = "none"; flush: str = "no"; par: str = "daily"
    tr: str = "none"; res: str = "none"; drv: str = "hold"; kin: str = "fixed"; kt: str = "no"; fit: str = "log"; gate: str = "no"; pre: str = "no"; bb: str = ""

    @staticmethod
    def parse(name: str) -> "Spec":
        if name.startswith("bb-"):
            return Spec(bb=name[3:])
        s = Spec(**dict(x.split("-", 1) for x in name.split(".")))
        assert all(getattr(s, f) in lv for f, lv in FACTORS.items()), name
        return s.normal()

    def normal(self) -> "Spec":
        """Equivalent specifications get one name: without growth rates, 'mixed' is 'station'; without the age term or a
        target, the anchor ages nothing; kinetic constants and driver trajectories matter only with kinetics."""
        s = self
        if s.par == "mixed" and s.gro == "none":
            s = replace(s, par="station")
        if s.anc == "filter":                                                 # the filter needs a target and replaces ageing
            s = replace(s, anc="last") if s.tgt == "none" else replace(s, age="no")
        if s.tgt == "none" and s.age == "yes":
            s = replace(s, age="no")
        if s.gro in ("none", "temp", "seaT") and s.kin == "learned":
            s = replace(s, kin="fixed")
        if s.gro == "none" and s.kt == "no" and s.drv == "clim":
            s = replace(s, drv="hold")
        if s.tgt == "none" and (s.kt == "yes" or s.par in ("kmod", "kmlp")):
            s = replace(s, kt="no", par="station" if s.par in ("kmod", "kmlp") else s.par)
        if s.par == "emlp" and s.tgt != "mnet":
            s = replace(s, par="station")
        assert not (s.par in ("mlp", "kmlp", "emlp") and s.res != "none"), "the driver network gives no residual"
        if s.gate == "yes" and not s.n_daily():                                # nothing to gate without a network
            s = replace(s, gate="no")
        return s

    @property
    def name(self) -> str:
        return f"bb-{self.bb}" if self.bb else ".".join(f"{f}-{getattr(self, f)}" for f in FACTORS if OPTIONAL.get(f) != getattr(self, f))

    def params(self) -> list[str]:
        p = (["mu", "loss"] if self.gro in ("monod", "temp") else []) + (["mua"] if self.gro in ("anom", "sea", "seaT") else [])
        p += (["kappa"] if self.tgt != "none" else []) + (["omega"] if self.tgt in ("blend", "mrec", "mcf") else []) + (["tshift"] if self.tgt == "mnet" else [])
        p += ["thk"] if self.kt == "yes" else []
        p += (["beta"] if self.flush == "yes" else []) + (["alpha"] if self.anc == "smooth" else []) + (["gain"] if self.anc == "filter" else [])
        return p + (["w", "share"] if self.tr == "dist" else [])

    def daily(self) -> list[str]:
        """Parameters given by the network alone each day (the target shift is always one)."""
        if self.par == "daily":
            return self.params()
        return [n for n in self.params() if n in GROWTH or n == "tshift"] if self.par == "mixed" else (["tshift"] if self.tgt == "mnet" else [])

    def modulated(self) -> list[str]:
        """Station constants that the network moves each day within ±MOD_C on the logit scale."""
        if self.par in ("mod", "mlp"):
            return [n for n in self.params() if n not in self.daily()]
        return ["kappa"] if self.par in ("kmod", "kmlp") and "kappa" in self.params() else []

    def n_daily(self) -> int:
        return len(self.daily()) + len(self.modulated()) + (1 if self.res == "bounded" else 0)


def tag(spec: str, net: str, split: str, rec: int, seed: int, placebo: bool = False) -> str:
    return K1.tag(spec, net, split, rec, seed, placebo)


# ------------------------------------------------------------------------------------------------------------- data
def prepare(net: str, split: str, rec: int) -> dict:
    """The first search's arrays plus the persistence features (recent mean, 7-day mean, age of the last measurement),
    rain, and the pure seasonal climatologies of the drivers and of Chl-a at the forecast day."""
    D = K1.prepare(net, split, rec)
    stations = D["stations"]; S, T = len(stations), len(D["anc"]); idx = {s: i for i, s in enumerate(stations)}
    rec_end = pd.Timestamp(K1.SPLITS[split][0]); start = rec_end - pd.DateOffset(months=rec) + pd.Timedelta(days=1)
    df = N.load_daily(); df = df[df.station.isin(stations)]
    ad = pd.DatetimeIndex(D["anc"]); F0 = K1.fourier(ad.dayofyear)
    Fk = K1.fourier(np.stack([(ad + pd.Timedelta(days=k)).dayofyear for k in range(1, HMAX + 1)], 1))
    recent, smooth7, age, rain, mclim0, mmean = (np.full((T, S), np.nan) for _ in range(6)); dclim = np.zeros((T, S, HMAX, 4))
    rdays = pd.date_range(start, rec_end); drec = np.full((S, len(rdays), 4), np.nan)
    wsy, web, mclim_cf, mclim0_cf, sea = np.zeros(S), np.zeros(S), np.zeros((T, S, HMAX)), np.zeros((T, S)), {}
    W = C.SEQ_LEN                                                            # stage 4: the 30-day window of measurements for the filter
    Fwin = K1.fourier(np.stack([(ad - pd.Timedelta(days=W - 1 - j)).dayofyear for j in range(W)], 1))
    wobs, wmask, wclim, wclim_cf = (np.zeros((T, S, W)) for _ in range(4))
    for s in stations:
        d = df[df.station == s].set_index("date").sort_index(); d = d.reindex(pd.date_range(d.index.min(), d.index.max())); i = idx[s]
        fit = (d.index >= start) & (d.index <= rec_end); Xd = K1.fourier(d.index.dayofyear)
        for c, col in enumerate(K1.DRIVERS):
            y = d[col].to_numpy(float); m = fit & np.isfinite(y)
            beta = np.linalg.lstsq(Xd[m], y[m], rcond=None)[0] if m.sum() >= 60 else np.r_[np.median(y[m]) if m.any() else 0.0, np.zeros(4)]
            dclim[:, i, :, c] = Fk @ beta
        lc = np.log(np.maximum(d.chlorophyll_a.to_numpy(float), 0.1)); m = fit & np.isfinite(lc)
        beta = np.linalg.lstsq(Xd[m], lc[m], rcond=None)[0] if m.sum() >= 60 else np.r_[np.median(lc[m]) if m.any() else 0.0, np.zeros(4)]
        mclim0[:, i] = F0 @ beta; mmean[:, i] = lc[m].mean() if m.any() else beta[0]
        yrs = [y for y in range(start.year, rec_end.year + 1) if (m & (d.index.year == y)).sum() >= 60]   # stage 3: years of the record
        Y, inyr = len(yrs), lambda ys: m & np.isin(d.index.year, ys)
        cfk, cf0 = np.full((T, HMAX), mmean[0, i]), np.full(T, mmean[0, i])                       # one year: the record mean
        if Y >= 2:
            a = np.stack([np.linalg.lstsq(Xd[inyr([y])], lc[inyr([y])], rcond=None)[0][1:] for y in yrs])   # seasonal coefficients per year
            sig = float((a.mean(0) ** 2).mean()); web[i] = float(np.clip(1 - a.var(0, ddof=1).mean() / Y / sig, 0, 1)) if sig > 0 else 0.0
            cfk, cf0 = Fk @ beta, F0 @ beta                                                       # after the record: the record's climatology
            ay, inrec = ad.year.to_numpy(), (ad >= start) & (ad <= rec_end)
            for y in yrs + [None]:                                                                # inside the record: the other years only
                sel = inrec & (ay == y) if y is not None else inrec & ~np.isin(ay, yrs)
                oth = inyr([z for z in yrs if z != y]); b_ = np.linalg.lstsq(Xd[oth], lc[oth], rcond=None)[0]
                cfk[sel], cf0[sel] = Fk[sel] @ b_, F0[sel] @ b_
        wsy[i] = (Y - 1) / Y if Y else 0.0; mclim_cf[:, i], mclim0_cf[:, i] = cfk, cf0
        wk = Fwin @ beta; cfw = np.full((T, W), mmean[0, i])                                      # climatology on the window days
        if Y >= 2:
            cfw = wk.copy()
            for y in yrs + [None]:
                sel = inrec & (ay == y) if y is not None else inrec & ~np.isin(ay, yrs)
                oth = inyr([z for z in yrs if z != y]); cfw[sel] = Fwin[sel] @ np.linalg.lstsq(Xd[oth], lc[oth], rcond=None)[0]
        widx = d.index.get_indexer(ad)[:, None] - (W - 1 - np.arange(W))[None, :]; wv = np.where(widx >= 0, lc[np.clip(widx, 0, None)], np.nan)
        wobs[:, i], wmask[:, i], wclim[:, i], wclim_cf[:, i] = np.nan_to_num(wv, nan=0.0), np.isfinite(wv), wk, cfw
        sea[s] = {"years": yrs, "w_sy": round(float(wsy[i]), 4), "w_eb": round(float(web[i]), 4)}
        drec[i] = d[K1.DRIVERS].rolling(7, min_periods=1).mean().reindex(rdays).ffill().bfill().to_numpy(float)
        ls = pd.Series(lc, index=d.index); pos = d.index.get_indexer(ad)
        recent[:, i] = ls.rolling(30, min_periods=1).mean().to_numpy()[pos]           # measured days up to the forecast day only
        smooth7[:, i] = ls.rolling(7, min_periods=1).mean().to_numpy()[pos]
        last = pd.Series(np.where(np.isfinite(lc), np.arange(len(d)), np.nan), index=d.index).ffill().to_numpy()
        age[:, i] = np.clip(np.nan_to_num(np.arange(len(d)) - last, nan=30.0), 0, 30)[pos]
        rain[:, i] = d.era5_tp_mm.rolling(7, min_periods=1).mean().to_numpy()[pos]
    lb = np.log(np.maximum(D["base"], 1e-3))
    D["recent"] = np.where(np.isfinite(recent), recent, mclim0).astype(np.float32)
    D["smooth7"] = np.where(np.isfinite(smooth7), smooth7, lb).astype(np.float32)
    D["age"] = age.astype(np.float32); D["rain"] = np.nan_to_num(rain, nan=0.0).astype(np.float32)
    D["mclim0"] = mclim0.astype(np.float32); D["mmean"] = mmean.astype(np.float32)
    D["wsy"], D["web"] = (np.repeat(w[None, :], T, 0).astype(np.float32) for w in (wsy, web))
    D["mclim_cf"], D["mclim0_cf"], D["sea"] = mclim_cf.astype(np.float32), mclim0_cf.astype(np.float32), sea
    D["wobs"], D["wmask"], D["wclim"], D["wclim_cf"] = (a.astype(np.float32) for a in (wobs, wmask, wclim, wclim_cf))
    for c in range(4):                                                    # a driver never measured in a station's record: median of the others
        drec[..., c] = np.where(np.isnan(drec[..., c]), np.nanmedian(drec[..., c]), drec[..., c])
    drec[..., 0] = np.clip(drec[..., 0], 0, 40); drec[..., 1:] = np.maximum(drec[..., 1:], 0); D["drec"] = drec.astype(np.float32)
    dclim[..., 0] = np.clip(dclim[..., 0], 0, 40); dclim[..., 1:] = np.maximum(dclim[..., 1:], 0); D["dclim"] = dclim.astype(np.float32)
    return D


# ------------------------------------------------------------------------------------------------------------- model
def bound(z, name):
    lo, hi = RANGE[name]
    return lo + torch.sigmoid(z) * (hi - lo)


def monod(drv, kin):
    T, Nn, P, I = drv[..., 0].clamp(0, 40), drv[..., 1].clamp_min(0), drv[..., 2].clamp_min(0), drv[..., 3].clamp_min(0)
    KN, KP, KI = (kin[n][None, :, None] for n in ("KN", "KP", "KI"))
    return THETA ** (T - 20) * Nn / (KN + Nn) * P / (KP + P) * I / (KI + I)


class XNet2(nn.Module):
    def __init__(self, s: Spec, n_feat: int, stations: list, up: dict, dist, drec=None):
        super().__init__()
        self.s, self.stations, self.up, self.S = s, stations, up, len(stations)
        self.idx = {st: i for i, st in enumerate(stations)}
        sid = [C.STATION_IDS[st] for st in stations]
        if os.environ.get("XPINN_STATION_ID") == "0":                               # stage 5b: one identifier for every station, so the
            sid = [0] * len(sid)                                                    # embedding holds no station information; off unless set
        self.register_buffer("sid", torch.tensor(sid))
        self.register_buffer("dist", torch.tensor(dist, dtype=torch.float32))
        self.register_buffer("drec",torch.tensor(drec if drec is not None else np.zeros((self.S, 1, 4)), dtype=torch.float32), persistent=False)  # (S, record days, 4)
        self.dn = [] if s.bb else s.daily()
        self.cn = [] if s.bb else [n for n in s.params() if n not in self.dn]
        self.mn = [] if s.bb else s.modulated()                                      # among cn, moved daily by the network
        n_out = 4 if s.bb else len(self.dn) + len(self.mn) + (1 if s.res == "bounded" else 0)
        small = not s.bb and s.par in ("mlp", "kmlp", "emlp")
        self.bb = build_backbone("xLSTM", n_feat + (8 if s.bb == "C2" else 0), len(C.STATION_IDS), C.SEQ_LEN, n_out) if n_out and not small else None
        self.mlp = nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, n_out)) if n_out and small else None
        self.register_buffer("fmu", torch.zeros(5)); self.register_buffer("fsd", torch.ones(5))      # standardisation of the driver features
        self.gate = nn.Parameter(torch.zeros(n_out)) if (not s.bb and s.gate == "yes" and n_out) else None
        self.zs = nn.Parameter(torch.zeros(1 if s.par == "pooled" else self.S, len(self.cn))) if self.cn else None
        self.zk = nn.Parameter(torch.zeros(self.S, 3)) if (not s.bb and s.kin == "learned") else None
        self.zv = nn.Parameter(torch.zeros(())) if (not s.bb and s.tr == "dist") else None

    def constants(self):
        return [p for p in (self.zs, self.zk, self.zv) if p is not None]

    def raw(self, x):
        B = x.shape[0]; o = self.bb(x.reshape(B * self.S, *x.shape[2:]), self.sid.repeat(B))
        return (o[0] if isinstance(o, tuple) else o).reshape(B, self.S, -1)

    @staticmethod
    def drivers(b):
        """Driver features on the forecast day (B, S, 5): water temperature, log N, log P, log light, log rain."""
        d = b["drv"][..., 0, :]
        return torch.stack([d[..., 0], torch.log(d[..., 1].clamp_min(0) + 0.01), torch.log(d[..., 2].clamp_min(0) + 0.001),
                            torch.log1p(d[..., 3].clamp_min(0)), torch.log1p(b["rain"].clamp_min(0))], -1)

    def physical(self, b):
        x = b["x"]; B = x.shape[0]
        z = self.mlp((self.drivers(b) - self.fmu) / self.fsd) if self.mlp is not None else (self.raw(x) if self.bb is not None else None)
        if self.gate is not None:
            z = z * self.gate
        P = {n: bound(z[..., j], n) for j, n in enumerate(self.dn)}
        P.update({n: bound(self.zs[:, j], n).expand(B, self.S) for j, n in enumerate(self.cn) if n not in self.mn})
        P.update({n: bound(self.zs[:, j] + MOD_C * torch.tanh(z[..., len(self.dn) + self.mn.index(n)]), n) for j, n in enumerate(self.cn) if n in self.mn})
        eps = EPS_MAX * torch.tanh(z[..., -1]) if self.s.res == "bounded" else None
        kin = ({n: bound(self.zk[:, j], n) for j, n in enumerate(("KN", "KP", "KI"))} if self.zk is not None
               else {n: torch.full((self.S,), FIXED_K[n], device=x.device) for n in ("KN", "KP", "KI")})
        if self.zv is not None:
            P["v"] = bound(self.zv, "v")
        return P, eps, kin

    def forward(self, b, transport=True, resid=True, growth=True):
        """b: batch dict of tensors. Returns log1p forecasts (B, S, 4) and the physical quantities."""
        s = self.s
        if s.bb:
            return torch.log1p(b["base"])[..., None] + self.raw(b["x"]), None
        P, eps, kin = self.physical(b)
        lb = torch.log(b["base"].clamp_min(1e-3))
        la = P["alpha"] * lb + (1 - P["alpha"]) * b["smooth7"] if s.anc == "smooth" else lb               # anchor
        if s.tgt == "clim":
            tk, t0 = b["mclim"], b["mclim0"]
        elif s.tgt == "recent":
            tk, t0 = b["recent"][..., None].expand_as(b["mclim"]), b["recent"]
        elif s.tgt == "blend":
            om = P["omega"]; tk = om[..., None] * b["mclim"] + (1 - om)[..., None] * b["recent"][..., None]; t0 = om * b["mclim0"] + (1 - om) * b["recent"]
        elif s.tgt == "mean":
            tk, t0 = b["mmean"][..., None].expand_as(b["mclim"]), b["mmean"]
        elif s.tgt == "mrec":
            t0 = P["omega"] * b["mmean"] + (1 - P["omega"]) * b["recent"]; tk = t0[..., None].expand_as(b["mclim"])
        elif s.tgt == "mnet":
            t0 = b["mmean"] + P["tshift"]; tk = t0[..., None].expand_as(b["mclim"])
        elif s.tgt in ("msy", "meb"):                                         # the record mean moved toward the climatology (stage 3)
            w = b["wsy" if s.tgt == "msy" else "web"]
            tk = b["mmean"][..., None] + w[..., None] * (b["mclim"] - b["mmean"][..., None]); t0 = b["mmean"] + w * (b["mclim0"] - b["mmean"])
        elif s.tgt == "mcf":
            om = P["omega"]
            tk = b["mmean"][..., None] + om[..., None] * (b["mclim_cf"] - b["mmean"][..., None]); t0 = b["mmean"] + om * (b["mclim0_cf"] - b["mmean"])
        if s.tgt != "none":                                                   # relaxation rate over the horizon (B, S, HMAX)
            kap = P["kappa"][..., None].expand_as(b["mclim"])
            if s.kt == "yes":
                kap = (kap * P["thk"][..., None] ** (b["drv"][..., 0].clamp(0, 40) - 20)).clamp(max=KAPPA_MAX)
        if s.age == "yes":                                                    # the anchor ages toward the target since the last measurement
            la = t0 + (la - t0) * (1 - kap[..., 0]) ** b["age"]
        if s.anc == "filter":                                                 # filtered anomaly over the window (stage 4)
            mm = b["mmean"][..., None]
            tw = (b["wclim"] if s.tgt == "clim" else mm + P["omega"][..., None] * (b["wclim_cf"] - mm) if s.tgt == "mcf"
                  else mm + b["wsy" if s.tgt == "msy" else "web"][..., None] * (b["wclim"] - mm) if s.tgt in ("msy", "meb")
                  else t0[..., None].expand_as(b["wobs"]))
            phi, g, a, seen = 1 - kap[..., 0], P["gain"], torch.zeros_like(t0), torch.zeros_like(t0)
            for j in range(b["wobs"].shape[-1]):
                ap = phi * a; ob = b["wobs"][..., j] - tw[..., j]; mj = b["wmask"][..., j]
                a = torch.where(mj > 0, torch.where(seen > 0, ap + g * (ob - ap), ob), ap); seen = torch.maximum(seen, mj)
            la = torch.where(seen > 0, t0 + a, t0 + (lb - t0) * phi ** b["age"])
        if s.gro == "monod":
            g = P["mu"][..., None] * monod(b["drv"], kin) - P["loss"][..., None]
        elif s.gro == "temp":
            g = P["mu"][..., None] * THETA ** (b["drv"][..., 0].clamp(0, 40) - 20) - P["loss"][..., None]
        elif s.gro == "anom":
            g = P["mua"][..., None] * (monod(b["drv"], kin) - monod(b["dclim"], kin))
        elif s.gro == "sea":                                                  # Monod factor relative to its mean over the record's days
            g = P["mua"][..., None] * (monod(b["drv"], kin) - monod(self.drec[None], kin).mean(-1)[..., None])
        elif s.gro == "seaT":
            tf = lambda d: THETA ** (d[..., 0].clamp(0, 40) - 20)
            g = P["mua"][..., None] * (tf(b["drv"]) - tf(self.drec[None]).mean(-1)[..., None])
        else:
            g = torch.zeros_like(b["mclim"])
        if not growth:                                                        # decomposition: without the growth term (loss kept)
            g = -P["loss"][..., None].expand_as(g) if s.gro in ("monod", "temp") else torch.zeros_like(g)
        if s.flush == "yes":
            dec = torch.exp(-torch.arange(1, HMAX + 1, device=lb.device, dtype=lb.dtype) / RAIN_EFOLD)
            g = g - P["beta"][..., None] * b["rain"][..., None] * dec
        use_t = s.tr == "dist" and transport
        tau_d = (self.dist / P["v"]).clamp(0, L) if use_t else None
        hist = b["hist"].clone(); hist[:, :, -1] = torch.exp(la)
        state = [hist[:, i, :] for i in range(self.S)]; out = []
        for k in range(1, HMAX + 1):
            for i, st in enumerate(self.stations):
                own = state[i][:, -1]
                if use_t and st in self.up:
                    inflow, us = torch.zeros_like(own), self.up[st]
                    for slot, u in enumerate(us):
                        sj = state[self.idx[u]]; pos = (L + k) - tau_d[i, slot].expand_as(own)
                        lo = pos.floor().long().clamp(0, sj.shape[1] - 1); hi = (lo + 1).clamp(max=sj.shape[1] - 1); fr = pos - lo.float()
                        val = (1 - fr) * sj.gather(1, lo[:, None]).squeeze(1) + fr * sj.gather(1, hi[:, None]).squeeze(1)
                        wgt = torch.ones_like(own) if len(us) == 1 else (P["share"][:, i] if slot == 0 else 1 - P["share"][:, i])
                        inflow = inflow + wgt * val
                    mixed = (1 - P["w"][:, i]) * own + P["w"][:, i] * inflow
                else:
                    mixed = own
                lg = g[:, i, k - 1]
                if s.tgt != "none":
                    lg = lg - kap[:, i, k - 1] * (torch.log(mixed.clamp_min(1e-3)) - tk[:, i, k - 1])
                if eps is not None and resid:
                    lg = lg + eps[:, i]
                state[i] = torch.cat([state[i], (mixed * torch.exp(lg.clamp(-1.0, 1.0))).clamp(1e-3, 1e4)[:, None]], 1)
            if k in H:
                out.append(torch.stack([state[i][:, -1] for i in range(self.S)], 1))
        return torch.log1p(torch.stack(out, -1)), {"P": P, "eps": eps, "kin": kin, "tau_d": tau_d}


# ------------------------------------------------------------------------------------------------------------- training
KEYS = ("base", "hist", "mclim", "mclim0", "recent", "smooth7", "age", "rain", "dclim", "mmean", "wsy", "web", "mclim_cf", "mclim0_cf",
        "wobs", "wmask", "wclim", "wclim_cf")


def batch(D, j, s, dev, drv=None):
    x = D["X"][j]
    if s.bb == "C2":
        x = np.concatenate([x, D["up_x"][j]], -1)
    b = {"x": x, "drv": D["drv_clim" if s.drv == "clim" else "drv_hold"][j] if drv is None else drv, **{k: D[k][j] for k in KEYS}}
    return {k: torch.as_tensor(v, device=dev) for k, v in b.items()}


def weights(D, s, dev):
    """Horizon weights of the fitting criterion and the station scale of C (for the raw criteria)."""
    hw = torch.as_tensor(D["hw"], device=dev)
    if s.fit in ("flat", "rawflat"):
        hw = torch.tensor([0.0 if h == 1 else 1.0 for h in H], device=dev)
    m = D["m_tr"][D["tr"]] > 0; yr = D["yraw"][D["tr"]]
    sd = np.array([yr[:, i][m[:, i]].std() if m[:, i].sum() > 2 else 1.0 for i in range(yr.shape[1])])
    return hw, torch.as_tensor(np.maximum(sd, 1e-3), dtype=torch.float32, device=dev)


def criterion(p, D, j, mask, s, dev, hw, csd):
    if s.fit in ("raw", "rawflat"):
        r = (torch.expm1(p) - torch.as_tensor(D["yraw"][j], device=dev).nan_to_num()) / csd[None, :, None]
    else:
        r = p - torch.as_tensor(D["ylog"][j], device=dev)
    m = torch.as_tensor(mask[j], device=dev)
    return (r ** 2 * m * hw).sum(), (m * hw).sum()


def val_loss(net, D, idx, mask, s, dev, hw, csd=None, bs=128):
    net.eval(); num = den = 0.0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]; p, _ = net(batch(D, j, s, dev))
            if s.fit == "log":                                                # as before
                y, m = torch.as_tensor(D["ylog"][j], device=dev), torch.as_tensor(mask[j], device=dev)
                num += float(((p - y) ** 2 * m * hw).sum()); den += float((m * hw).sum())
            else:
                a, b_ = criterion(p, D, j, mask, s, dev, hw, csd); num += float(a); den += float(b_)
    return num / max(den, 1.0)


def train(s, D, lr, seed, dev, epochs=100, patience=15, bs=32):
    K1.seed_all(seed); net = XNet2(s, D["X"].shape[-1], D["stations"], D["up"], D["dist"], D["drec"]).to(dev); hw, csd = weights(D, s, dev)
    if net.mlp is not None:                                                   # standardise the driver features on the training days
        with torch.no_grad():
            f = XNet2.drivers(batch(D, D["tr"], s, dev)).reshape(-1, 5); net.fmu.copy_(f.mean(0)); net.fsd.copy_(f.std(0).clamp_min(1e-3))
    cid = {id(p) for p in net.constants()}; others = [p for p in net.parameters() if id(p) not in cid]
    groups = ([{"params": others, "lr": lr, "weight_decay": 1e-4}] if others else []) + \
             ([{"params": net.constants(), "lr": min(10 * lr, 1e-2), "weight_decay": 0.0}] if cid else [])
    opt = torch.optim.Adam(groups); rng = np.random.RandomState(seed)
    tr = D["tr"][D["m_tr"][D["tr"]].sum((1, 2)) > 0]
    best, best_state, bad, ep_best = np.inf, copy.deepcopy(net.state_dict()), 0, -1
    for ep in range(epochs):
        net.train(); perm = rng.permutation(tr)
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]; p, aux = net(batch(D, j, s, dev))
            if s.fit == "log":                                                # as before
                loss = K1.wmse(p, torch.as_tensor(D["ylog"][j], device=dev), torch.as_tensor(D["m_tr"][j], device=dev), hw)
            else:
                a, b_ = criterion(p, D, j, D["m_tr"], s, dev, hw, csd); loss = a / b_.clamp_min(1.0)
            if aux is not None and aux["eps"] is not None:
                loss = loss + EPS_PEN * (aux["eps"] ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        vl = val_loss(net, D, D["va"], D["m_va"], s, dev, hw, csd)
        if vl < best - 1e-6:
            best, best_state, bad, ep_best = vl, copy.deepcopy(net.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, best, ep_best


def evaluate(net, D, s, dev, bs=128):
    """Forecasts at the evaluation anchors; for process models also without the residual, without transport and without
    the growth term, and the physical quantities."""
    ev = D["ev"]; full, mech, local, nogro, Q = [], [], [], [], []
    net.eval()
    with torch.no_grad():
        for i in range(0, len(ev), bs):
            j = ev[i:i + bs]; b = batch(D, j, s, dev); p, aux = net(b); full.append(p.cpu().numpy())
            if aux is None:
                continue
            mech.append(net(b, resid=False)[0].cpu().numpy() if aux["eps"] is not None else full[-1])
            local.append(net(b, transport=False)[0].cpu().numpy() if s.tr != "none" else full[-1])
            nogro.append(net(b, growth=False)[0].cpu().numpy() if s.gro != "none" else full[-1])
            q = {n: t.cpu().numpy() for n, t in aux["P"].items() if t.dim() == 2}
            if aux["eps"] is not None:
                q["eps"] = aux["eps"].cpu().numpy()
            if s.kin == "learned":
                q.update({n: np.repeat(t.cpu().numpy()[None, :], len(j), 0) for n, t in aux["kin"].items()})
            if "v" in aux["P"]:
                q["v"] = np.full((len(j), net.S), float(aux["P"]["v"]))
                td = aux["tau_d"].cpu().numpy(); q["tau1"], q["tau2"] = (np.repeat(td[None, :, c], len(j), 0) for c in (0, 1))
            Q.append(q)
    out = {"full": np.concatenate(full)}
    if Q:
        out.update(mech=np.concatenate(mech), local=np.concatenate(local), nogro=np.concatenate(nogro), par={n: np.concatenate([q[n] for q in Q]) for n in Q[0]})
    return out


def frames(D, s, res):
    pred, _ = K1.frames(D, None, {"full": res["full"]}, D["stations"])
    if "par" not in res:
        return pred, None
    ev = D["ev"]; S = len(D["stations"]); st = np.tile(np.array(D["stations"]), len(ev)); lb = np.log1p(D["base"][ev])
    par = pd.DataFrame({"station": st, "anchor_date": np.repeat(D["anc"][ev].astype(str), S), **{n: a.ravel() for n, a in res["par"].items()},
                        "d7_full": (res["full"][:, :, 2] - lb).ravel(), "d7_mech": (res["mech"][:, :, 2] - lb).ravel(),
                        "d7_local": (res["local"][:, :, 2] - lb).ravel(), "d7_nogro": (res["nogro"][:, :, 2] - lb).ravel(),
                        "rain": D["rain"][ev].ravel(), "age": D["age"][ev].ravel()})
    nup = {st_: len(D["up"].get(st_, [])) for st_ in D["stations"]}
    for col, need in (("w", 1), ("tau1", 1), ("share", 2), ("tau2", 2)):
        if col in par:
            par.loc[par.station.map(nup) < need, col] = np.nan
    return pred, par


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--net", default="nakdong")
    ap.add_argument("--split", default="dev", choices=list(K1.SPLITS)); ap.add_argument("--record", type=int, required=True); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--placebo", action="store_true", help="train with the driver trajectories shuffled across days within each station")
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("XPINN_EPOCHS", 100))); a = ap.parse_args()
    s = Spec.parse(a.spec); assert s.pre == "no", "designs with national pre-training run through hybrid.py"
    rd = OUT / "runs" / tag(s.name, a.net, a.split, a.record, a.seed, a.placebo); rd.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    if (rd / "DONE").exists():
        print("skip", rd.name); return
    D = prepare(a.net, a.split, a.record)
    if a.placebo:
        rng = np.random.RandomState(1000 + a.seed)
        for i in range(len(D["stations"])):
            perm = rng.permutation(len(D["anc"]))
            for key in ("drv_hold", "drv_clim"):
                D[key][:, i] = D[key][perm, i]
    runs = []
    for lr in (1e-3, 3e-4, 1e-4):
        net, best, ep = train(s, D, lr, a.seed, dev, epochs=a.epochs); runs.append((lr, net, best, ep))
        print(f"[{rd.name}] lr {lr:g} val {best:.4f} epoch {ep}", flush=True)
    lr, net, best, ep = min(runs, key=lambda r: r[2])
    torch.save(net.state_dict(), rd / "model.pt")
    pred, par = frames(D, s, evaluate(net, D, s, dev))
    pred.to_csv(rd / "predictions_eval.csv", index=False, float_format="%.6g")
    if par is not None:
        par.to_csv(rd / "params_eval.csv", index=False, float_format="%.6g")
    json.dump({"spec": s.name, "placebo": a.placebo, "net": a.net, "split": a.split, "record_months": a.record, "seed": a.seed, "lr_selected": lr,
               "best_val": best, "best_epoch": ep, "lr_table": [{"lr": r[0], "best_val": r[2], "best_epoch": r[3]} for r in runs], "record": D["record"],
               "counts": D["counts"], "seasonal_weights": D["sea"], "time_s": time.time() - t0, "stations": D["stations"], "upstream": D["up"], "distance_km": D["dist"].tolist(),
               "seq_len": C.SEQ_LEN, "inputs": D["feat"], "station_identifier": os.environ.get("XPINN_STATION_ID") != "0"},
              open(rd / "config.json", "w"), indent=1)
    (rd / "DONE").write_text("ok"); print(f"[{rd.name}] done {(time.time() - t0) / 60:.1f} min, lr {lr:g}", flush=True)


if __name__ == "__main__":
    main()
