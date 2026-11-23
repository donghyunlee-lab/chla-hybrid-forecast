"""Shared data pipeline, metrics and I/O of the six-station experiments (Geum River basin) in scripts/clean_revision."""
from __future__ import annotations
import json, hashlib, os, time, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd

VARS = ["water_temperature", "ph", "electrical_conductivity", "dissolved_oxygen", "turbidity",
        "total_organic_carbon", "chlorophyll_a", "total_nitrogen", "total_phosphorus"]
LOG_VARS = {"chlorophyll_a", "turbidity"}            # log1p before scaling (input channels only)
ERA5 = ["era5_ssrd_mj", "era5_t2m", "era5_tp_mm", "era5_wind_max"]
STATION_IDS = {"현도": 0, "남면": 1, "공주": 2, "부여": 3, "용담호": 4, "대청호": 5}
SOURCE = ["현도", "남면", "공주", "부여"]
TARGETS = ["용담호", "대청호"]
HORIZONS = [1, 3, 7, 14]
SEQ_LEN = 30
MAX_FFILL = 3
AGE_CAP = 30
SPLITS = {"train": ("2021-01-01", "2022-12-31"), "val": ("2023-01-01", "2023-12-31"), "test": ("2024-01-01", "2025-12-31")}
PAR_FRACTION = 0.45          # PAR share of shortwave
MJ_TO_UMOL_S = 52.9          # MJ/m2/day of PAR -> mean umol/m2/s (4.57 umol/J / 86400 s)
BLOOM_THRESHOLDS = [15.0, 25.0]   # mg/m3, Korean water-quality forecast tiers
PHYS_VARS = ["T", "N", "P", "C", "PAR"]

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "clean"
RESULTS = Path(os.environ["PINN_RESULTS_DIR"]) if os.environ.get("PINN_RESULTS_DIR", "").startswith("/") else ROOT / os.environ.get("PINN_RESULTS_DIR", "results/clean_revision_v1")


def load_daily() -> pd.DataFrame:
    obs = pd.read_csv(CLEAN / "geum_daily_observed.csv", parse_dates=["date"])
    era = pd.read_csv(CLEAN / "era5_daily.csv", parse_dates=["date"])
    df = obs.merge(era, on=["station", "date"], how="left")
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    df["par_umol"] = df["era5_ssrd_mj"] * PAR_FRACTION * MJ_TO_UMOL_S
    return df


def causal_fill(v: np.ndarray, max_ffill: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-only fill up to max_ffill days. Returns (filled, observed_mask, age_days)."""
    obs = ~np.isnan(v)
    filled = np.full_like(v, np.nan, dtype=float)
    age = np.full(len(v), np.inf)
    last, last_i = np.nan, None
    for i in range(len(v)):
        if obs[i]:
            last, last_i = v[i], i
            filled[i], age[i] = v[i], 0
        elif last_i is not None:
            age[i] = i - last_i
            if age[i] <= max_ffill:
                filled[i] = last
    return filled, obs, age


class Scaler:
    """Train-period statistics only. Inputs z-scored; target log1p min-max."""
    def __init__(self):
        self.mean, self.std, self.median = {}, {}, {}
        self.y_lo = self.y_hi = None

    def fit(self, df: pd.DataFrame, train_mask: np.ndarray, stations: list[str] | None = None):
        """Statistics from the training period of the given stations only (source-only by default in v3)."""
        m = np.asarray(train_mask, bool)
        if stations is not None:
            m = m & df["station"].isin(stations).to_numpy()
        sub = df[m]
        for v in VARS + ERA5:
            x = sub[v].to_numpy(dtype=float)
            if v in LOG_VARS:
                x = np.log1p(np.clip(x, 0, None))
            x = x[~np.isnan(x)]
            self.mean[v], self.std[v], self.median[v] = float(x.mean()), float(x.std() + 1e-8), float(np.median(x))
        y = np.log1p(np.clip(sub["chlorophyll_a"].to_numpy(dtype=float), 0, None))
        y = y[~np.isnan(y)]
        self.y_lo, self.y_hi = float(y.min()), float(y.max())
        return self

    def y_scale(self, raw):
        return (np.log1p(np.clip(raw, 0, None)) - self.y_lo) / (self.y_hi - self.y_lo)

    def y_unscale(self, s):
        return np.expm1(s * (self.y_hi - self.y_lo) + self.y_lo)

    def to_dict(self):
        return {"mean": self.mean, "std": self.std, "median": self.median, "y_lo": self.y_lo, "y_hi": self.y_hi}


def build_arrays(df: pd.DataFrame, scaler: Scaler, stations: list[str], seq_len: int = SEQ_LEN, horizons=HORIZONS):
    """Return dict of arrays for all anchors of the given stations (all splits)."""
    X, SID, PR, PV, Y, YR, YM, AD, TD, BS, BV = [], [], [], [], [], [], [], [], [], [], []
    feat_names = []
    for v in VARS:
        feat_names += [f"{v}", f"{v}__mask", f"{v}__age"]
    feat_names += ERA5 + ["chlorophyll_a__last_obs"]
    for st in stations:
        d = df[df["station"] == st].reset_index(drop=True)
        n = len(d)
        chans, phys_raw, phys_valid = [], [], []
        for v in VARS:
            raw = d[v].to_numpy(dtype=float)
            x = np.log1p(np.clip(raw, 0, None)) if v in LOG_VARS else raw.copy()
            filled, obs, age = causal_fill(x, MAX_FFILL)
            filled = np.where(np.isnan(filled), scaler.median[v], filled)
            chans += [(filled - scaler.mean[v]) / scaler.std[v], obs.astype(float), np.minimum(age, AGE_CAP) / AGE_CAP]
            if v in ("water_temperature", "total_nitrogen", "total_phosphorus"):
                f_raw, _, _ = causal_fill(raw, MAX_FFILL)          # physics uses raw units, causal <= 3 days
                phys_raw.append(np.where(np.isnan(f_raw), 0.0, f_raw)); phys_valid.append(~np.isnan(f_raw))
            if v == "chlorophyll_a":
                phys_raw.append(np.where(np.isnan(raw), 0.0, raw)); phys_valid.append(~np.isnan(raw))  # observed only
        for v in ERA5:
            e = d[v].to_numpy(dtype=float)
            e = np.where(np.isnan(e), scaler.median[v], e)
            chans.append((e - scaler.mean[v]) / scaler.std[v])
        # unbounded forward fill of log1p chlorophyll-a: explicit persistence information for the network
        chl_raw = d["chlorophyll_a"].to_numpy(dtype=float)
        ff_unb, _, age_unb = causal_fill(np.log1p(np.clip(chl_raw, 0, None)), 10 ** 6)
        ff_unb_f = np.where(np.isnan(ff_unb), scaler.median["chlorophyll_a"], ff_unb)
        chans.append((ff_unb_f - scaler.mean["chlorophyll_a"]) / scaler.std["chlorophyll_a"])
        par = d["par_umol"].to_numpy(dtype=float)
        phys_raw.append(np.where(np.isnan(par), 0.0, par)); phys_valid.append(~np.isnan(par))
        # order phys as T, N, P, C, PAR
        order = {"T": 0, "N": 1, "P": 2, "C": 3, "PAR": 4}
        pr = np.stack([phys_raw[0], phys_raw[2], phys_raw[3], phys_raw[1], phys_raw[4]], axis=1)  # built order: T, C, N, P, PAR
        pv = np.stack([phys_valid[0], phys_valid[2], phys_valid[3], phys_valid[1], phys_valid[4]], axis=1)
        Xst = np.stack(chans, axis=1).astype(np.float32)   # (n_days, F)
        dates = d["date"].to_numpy()
        y_med = (scaler.median["chlorophyll_a"] - scaler.y_lo) / (scaler.y_hi - scaler.y_lo)   # median already in log1p space
        for i in range(seq_len - 1, n):
            ys, yr, ym, td = [], [], [], []
            for h in horizons:
                j = i + h
                if j < n and not np.isnan(chl_raw[j]):
                    yr.append(chl_raw[j]); ys.append(scaler.y_scale(chl_raw[j])); ym.append(1.0); td.append(dates[j])
                else:
                    yr.append(np.nan); ys.append(0.0); ym.append(0.0); td.append(dates[min(j, n - 1)])
            # persistence base: last observed chlorophyll-a inside the window (same rule as the persistence baseline)
            if not np.isnan(ff_unb[i]) and age_unb[i] <= seq_len - 1:
                BS.append((ff_unb[i] - scaler.y_lo) / (scaler.y_hi - scaler.y_lo)); BV.append(True)
            else:
                BS.append(y_med); BV.append(False)
            X.append(Xst[i - seq_len + 1:i + 1]); SID.append(STATION_IDS[st])
            PR.append(pr[i - seq_len + 1:i + 1]); PV.append(pv[i - seq_len + 1:i + 1])
            Y.append(ys); YR.append(yr); YM.append(ym); AD.append(dates[i]); TD.append(td)
    out = dict(X=np.asarray(X, np.float32), sid=np.asarray(SID, np.int64), phys_raw=np.asarray(PR, np.float32),
               phys_valid=np.asarray(PV, bool), y=np.asarray(Y, np.float32), y_raw=np.asarray(YR, np.float64),
               y_mask=np.asarray(YM, np.float32), anchor=np.asarray(AD, "datetime64[D]"), target=np.asarray(TD, "datetime64[D]"),
               base=np.asarray(BS, np.float32), base_valid=np.asarray(BV, bool), feat_names=feat_names)
    return out


def split_indices(arr: dict, split: str, stations: list[str] | None = None) -> np.ndarray:
    s, e = np.datetime64(SPLITS[split][0]), np.datetime64(SPLITS[split][1])
    m = (arr["anchor"] >= s) & (arr["anchor"] <= e)
    if stations is not None:
        ids = np.array([STATION_IDS[x] for x in stations])
        m &= np.isin(arr["sid"], ids)
    return np.where(m)[0]


def record_subset(arr: dict, idx: np.ndarray, months: int) -> np.ndarray:
    """Keep anchors within the first `months` months of the training period (record-length scarcity)."""
    end = (pd.Timestamp(SPLITS["train"][0]) + pd.DateOffset(months=months)).to_datetime64()
    return idx[arr["anchor"][idx] < end]


def mask_targets_to_split(arr: dict, idx: np.ndarray, split: str) -> np.ndarray:
    """Zero the target mask where the target date falls after the split end (avoids cross-split targets)."""
    e = np.datetime64(SPLITS[split][1])
    ym = arr["y_mask"][idx].copy()
    ym[arr["target"][idx] > e] = 0.0
    return ym


def mask_targets_after(arr: dict, idx: np.ndarray, end) -> np.ndarray:
    """Zero the target mask where the target date falls after `end` (any date)."""
    e = np.datetime64(pd.Timestamp(end).date())
    ym = arr["y_mask"][idx].copy()
    ym[arr["target"][idx] > e] = 0.0
    return ym


def record_window(arr: dict, idx: np.ndarray, months: int, end: str, val_frac: float):
    """v4 record-length scarcity. Keep the `months` months that end on `end`, so a short record holds the latest season
    (6 months ending in December = July-December, bloom season included). The last `val_frac` of the record's anchor days
    is held out for early stopping and learning-rate choice, as a forecaster who has only this record would do; training targets
    stop the day before that block. Returns (train_idx, val_idx or None, train_end, record_end)."""
    end_ts = pd.Timestamp(end)
    start_ts = end_ts - pd.DateOffset(months=months) + pd.Timedelta(days=1)
    a = arr["anchor"][idx]
    rec = idx[(a >= np.datetime64(start_ts.date())) & (a <= np.datetime64(end_ts.date()))]
    if val_frac <= 0:
        return rec, None, end_ts, end_ts
    days = np.unique(arr["anchor"][rec])
    val_start = days[-max(1, int(round(len(days) * val_frac)))]
    tr, va = rec[arr["anchor"][rec] < val_start], rec[arr["anchor"][rec] >= val_start]
    return tr, va, pd.Timestamp(val_start) - pd.Timedelta(days=1), end_ts


# ---------------- metrics ----------------
def _safe(y, p):
    m = ~(np.isnan(y) | np.isnan(p))
    return y[m], p[m]


def kge(y, p):
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    r = np.corrcoef(y, p)[0, 1]
    alpha = np.std(p) / np.std(y); beta = np.mean(p) / np.mean(y) if np.mean(y) != 0 else np.nan
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def basic_metrics(y, p) -> dict:
    y, p = _safe(np.asarray(y, float), np.asarray(p, float))
    if len(y) == 0:
        return {"n": 0}
    err = p - y
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(err ** 2) / sst if sst > 0 else np.nan
    ly, lp = np.log1p(np.clip(y, 0, None)), np.log1p(np.clip(p, 0, None))
    lsst = np.sum((ly - ly.mean()) ** 2)
    return {"n": int(len(y)), "r2": float(r2), "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)), "kge": float(kge(y, p)),
            "mape": float(np.mean(np.abs(err[y > 0] / y[y > 0])) * 100) if np.any(y > 0) else np.nan,
            "log_rmse": float(np.sqrt(np.mean((lp - ly) ** 2))),
            "log_r2": float(1 - np.sum((lp - ly) ** 2) / lsst) if lsst > 0 else np.nan,
            "pearson_r": float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0 else np.nan}


def bloom_metrics(y, p, dates=None) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = ~(np.isnan(y) | np.isnan(p)); y, p = y[m], p[m]
    out = {}
    for thr in BLOOM_THRESHOLDS:
        oy, op = y >= thr, p >= thr
        tp, fp, fn = int(np.sum(oy & op)), int(np.sum(~oy & op)), int(np.sum(oy & ~op))
        prec = tp / (tp + fp) if tp + fp else np.nan; rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (tp + fp) and (tp + fn) and (prec + rec) else np.nan
        out[f"thr{int(thr)}"] = {"events_obs": int(oy.sum()), "events_pred": int(op.sum()), "tp": tp, "fp": fp, "fn": fn,
                                 "precision": prec, "recall": rec, "f1": f1}
    if len(y):
        k = int(np.argmax(y)); kp = int(np.argmax(p))
        out["peak"] = {"obs_peak": float(y[k]), "pred_at_obs_peak": float(p[k]), "peak_rel_error": float((p[k] - y[k]) / y[k]) if y[k] else np.nan,
                       "pred_peak": float(p[kp]), "peak_timing_error_days": float((dates[kp] - dates[k]) / np.timedelta64(1, "D")) if dates is not None else np.nan}
    return out


def evaluate_predictions(pred: pd.DataFrame) -> dict:
    """pred columns: station, horizon, target_date, y_true_raw, y_pred_raw (rows with observed targets only)."""
    res = {}
    for h, g in pred.groupby("horizon"):
        res[f"h{h}"] = {"pooled": basic_metrics(g.y_true_raw, g.y_pred_raw) | {"bloom": bloom_metrics(g.y_true_raw, g.y_pred_raw, g.target_date.to_numpy())}}
        for st, gs in g.groupby("station"):
            res[f"h{h}"][st] = basic_metrics(gs.y_true_raw, gs.y_pred_raw) | {"bloom": bloom_metrics(gs.y_true_raw, gs.y_pred_raw, gs.target_date.to_numpy())}
    return res


def write_json(obj, path: Path):
    def conv(o):
        if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.datetime64, pd.Timestamp)): return str(o)
        if isinstance(o, float) and np.isnan(o): return None
        raise TypeError(str(type(o)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=conv, ensure_ascii=False, indent=1), encoding="utf-8")


def append_index(row: dict):
    """Append one row, aligning columns with the existing index (union of columns). Locked, so the GPU queue and the
    CPU baselines can write to the same index at the same time."""
    import fcntl
    p = RESULTS / "runs_index.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        new = pd.DataFrame([row])
        if p.exists():
            old = pd.read_csv(p)
            new = pd.concat([old, new], ignore_index=True, sort=False)
        new.to_csv(p, index=False)


