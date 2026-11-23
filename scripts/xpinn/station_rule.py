"""Stations with the whole period: only stations whose data cover the whole period are used, records cut off in the middle are not.
Rule: measured chlorophyll-a runs from January 2021 to December 2025, every calendar year has it on at least half of the days, and no
gap between measured days is longer than 120 days. Input: data/national/daily_observed.csv. For the eight Yeongsan-Seomjin stations the
2025 values are read from the institutional copy of the WEIS daily values, where the 2025 files carry no basin prefix and have
other station IDs (ALIAS_2025);
the first build of the daily file kept only the IDs of stations.csv and so stops at 2024-12-31 for them.
Output (results/input_design_20261114/): station_coverage.csv (all 37 stations, the numbers and the reason) and stations.txt.
Usage: station_rule.py [output folder]"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "input_design_20261114"
CORE = Path(os.environ.get("WEIS_DATA_CORE_ROOT", os.path.expanduser("weis-data-core")))
START, END, MIN_YEAR, MAX_GAP = "2021-01-01", "2025-12-31", 0.5, 120
ALIAS_2025 = {"S04001": "AUT-dbd1abfb269d4d", "S04002": "AUT-5c258e9cd9395e", "S04004": "AUT-54db41b4ca50a2", "S04005": "AUT-1885c8d145ea84",
              "S04007": "AUT-303799d7e6824e", "S04008": "AUT-a30c6ff18489e0", "S04009": "AUT-9bd8fc04b61947", "S04012": "AUT-5a4669e6f5847e"}


def chl_2025_from_core() -> dict:
    """Measured chlorophyll-a days of 2025 for the stations of ALIAS_2025 (numeric observations of the automatic daily network)."""
    import pyarrow.dataset as ds, pyarrow.compute as pc
    d = ds.dataset(str(CORE / "data/silver/observations/automatic/daily"), format="parquet")
    t = d.to_table(columns=["station_id", "date_local"], filter=pc.field("station_id").isin(list(ALIAS_2025.values())) & (pc.field("variable_id") == "chlorophyll_a")
                   & (pc.field("is_numeric_observation") == True)).to_pandas()
    t["date"] = pd.to_datetime(t.date_local).dt.normalize(); back = {v: k for k, v in ALIAS_2025.items()}
    return {back[sid]: pd.DatetimeIndex(sorted(g.date.unique())) for sid, g in t.groupby("station_id")}


def main():
    nat = ROOT / os.environ.get("PINN_NATIONAL_DIR", "data/national")     # with data/national_v2 the 2025 values are already in the file
    st = pd.read_csv(nat / "stations.csv", dtype={"code": str})
    d = pd.read_csv(nat / "daily_observed.csv", parse_dates=["date"], dtype={"station": str}, usecols=["station", "date", "chlorophyll_a"])
    days = pd.date_range(START, END); extra = chl_2025_from_core(); rows = []
    for _, r in st.iterrows():
        x = d[d.station == r.code].set_index("date").chlorophyll_a.reindex(days).notna().to_numpy(); n25 = 0
        if r.code in extra:                                              # 2025 from the copy of the WEIS daily values (not in the daily file)
            add = days.isin(extra[r.code]) & (days.year == 2025); n25 = int(add.sum()); x = x | add
        idx = np.flatnonzero(x); gaps = np.diff(idx) - 1; gi = int(gaps.argmax())
        yr = {y: float(x[days.year == y].mean()) for y in range(2021, 2026)}
        first, last, maxgap = days[idx[0]], days[idx[-1]], int(gaps.max())
        why = []
        if first > pd.Timestamp("2021-01-31"): why.append(f"starts {first.date()}")
        if last < pd.Timestamp("2025-12-01"): why.append(f"ends {last.date()}")
        if min(yr.values()) < MIN_YEAR: why.append(f"{min(yr, key=yr.get)}: {min(yr.values()):.0%} of days")
        if maxgap > MAX_GAP: why.append(f"gap of {maxgap} days from {(days[idx[gi]] + pd.Timedelta(days=1)).date()}")
        rows.append({"code": r.code, "name_kr": r.name_kr, "basin": r.basin, "type": r.type, "first": str(first.date()), "last": str(last.date()),
                     **{f"share_{y}": round(v, 3) for y, v in yr.items()}, "longest_gap_days": maxgap, "chl_days_2025_from_core": n25,
                     "included": not why, "reason_excluded": "; ".join(why)})
    T = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "station_coverage.csv", index=False); (OUT / "stations.txt").write_text("\n".join(T[T.included].code) + "\n")
    print(T[["code", "name_kr", "basin", "included", "reason_excluded"]].to_string(index=False))
    print(int(T.included.sum()), "stations:", T[T.included].groupby("basin").size().to_dict())


if __name__ == "__main__":
    main()
