"""National benchmark: the 37 stations, the leave-one-basin-out folds, the periods and the models, plus a patch
of the paper's data module (scripts/clean_revision/common.py) so that its loader, scaler and arrays run unchanged on the
national data."""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "clean_revision"))
import common as C                                                               # noqa: E402

NAT = ROOT / os.environ.get("PINN_NATIONAL_DIR", "data/national")     # data/national_v2: the same file plus the 2025 values of eight stations
OUT = ROOT / "results" / "national_20261025"
STATIONS = pd.read_csv(NAT / "stations.csv", dtype={"code": str}).sort_values("code").reset_index(drop=True)
CODES = STATIONS.code.tolist()
BASINS = ["Han", "Nakdong", "Geum", "Yeongsan"]
BASIN_OF, NAME_OF, TYPE_OF = (dict(zip(STATIONS.code, STATIONS[c])) for c in ("basin", "name_kr", "type"))
FOLDS = {b: [c for c in CODES if BASIN_OF[c] == b] for b in BASINS}
SPLITS = {"train": ("2021-01-01", "2023-12-31"), "val": ("2023-01-01", "2023-12-31"), "test": ("2024-01-01", "2025-12-31")}
RECORD_END, VAL_FRAC, RECORDS, SOURCE_RECORD = "2023-12-31", 0.2, (6, 12, 24, 36), 36
SEEDS = (42, 123, 456, 1, 2)
PHYS = ["--phys_param", "exp", "--phys_wd", "0", "--phys_lr", "1e-2", "--warmup", "0", "--curriculum", "0", "--converge_physics", "1", "--w_max", "0.3"]
NETS = {"xLSTM": ["--backbone", "xLSTM", "--physics", "off"], "LSTM": ["--backbone", "LSTM", "--physics", "off"],
        "iTransformer": ["--backbone", "iTransformer", "--physics", "off"], "ModernTCN": ["--backbone", "ModernTCN", "--physics", "off"],
        "Mamba": ["--backbone", "Mamba", "--physics", "off"], "PINN-sigmoid": ["--backbone", "xLSTM", "--physics", "sigmoid"] + PHYS,
        "PINN-soft": ["--backbone", "xLSTM", "--physics", "soft"] + PHYS, "PINN-none": ["--backbone", "xLSTM", "--physics", "none"] + PHYS,
        "control": ["--backbone", "xLSTM", "--physics", "sigmoid"] + PHYS + ["--phys_null", "1"]}


def load_daily() -> pd.DataFrame:
    obs = pd.read_csv(NAT / "daily_observed.csv", parse_dates=["date"], dtype={"station": str})
    era = pd.read_csv(NAT / "era5_daily.csv", parse_dates=["date"], dtype={"station": str})
    df = obs.merge(era, on=["station", "date"], how="left").sort_values(["station", "date"]).reset_index(drop=True)
    df["par_umol"] = df["era5_ssrd_mj"] * C.PAR_FRACTION * C.MJ_TO_UMOL_S
    return df


def patch():
    """Point the paper's data module at the national data (station keys are WEIS codes)."""
    C.STATION_IDS = {c: i for i, c in enumerate(CODES)}
    C.SPLITS = dict(SPLITS)
    C.load_daily = load_daily
    C.SOURCE, C.TARGETS = list(CODES), []
    return C


def set_fold(fold: str):
    """Leave one basin out: the held-out basin's stations are the targets, all other stations the source."""
    if fold:
        C.TARGETS = list(FOLDS[fold]); C.SOURCE = [c for c in CODES if c not in C.TARGETS]
    return C.SOURCE, C.TARGETS

MIN_CHL_TRAIN, MIN_CHL_VAL = 30, 10          # eligibility of a station and record length


def record_counts() -> pd.DataFrame:
    """Measured Chl-a days in the training part and in the validation part (last 20 % of days) of each station's record."""
    d = pd.read_csv(NAT / "daily_observed.csv", parse_dates=["date"], dtype={"station": str}, usecols=["station", "date", "chlorophyll_a"])
    end = pd.Timestamp(RECORD_END); rows = []
    for m in RECORDS:
        start = end - pd.DateOffset(months=m) + pd.Timedelta(days=1)
        days = pd.date_range(start, end); vstart = days[-max(1, int(round(len(days) * VAL_FRAC)))]
        for st, g in d.groupby("station"):
            w = g[(g.date >= start) & (g.date <= end) & g.chlorophyll_a.notna()]
            rows.append({"station": st, "record": m, "chl_train": int((w.date < vstart).sum()), "chl_val": int((w.date >= vstart).sum())})
    r = pd.DataFrame(rows); r["eligible"] = (r.chl_train >= MIN_CHL_TRAIN) & (r.chl_val >= MIN_CHL_VAL)
    return r


def eligible_set() -> set:
    r = record_counts()
    return set(zip(r[r.eligible].station, r[r.eligible].record))
