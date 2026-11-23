"""Station table of the manuscript (description of the data, no model): for the 27 stations of the input design, the
basin, the type (river or lake), the share of days of 2021-2025 with a measured value of Chl-a, water temperature, TN and TP,
and the Chl-a level (median, 90th percentile, share of measured days at or above 15 and 25 mg m-3), from the second build of
the daily file. Writes output/results/station_table.csv (compared with the bundled results/input_design_20261114/station_table.csv).
Usage: station_table.py"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "input_design_20261114"

if __name__ == "__main__":
    V2 = ROOT / "data" / "national_v2"
    d = pd.read_csv(V2 / "daily_observed.csv", dtype={"station": str}, parse_dates=["date"])
    d = d.merge(pd.read_csv(V2 / "era5_daily.csv", dtype={"station": str}, parse_dates=["date"]), on=["station", "date"], how="left")
    keep = (OUT / "stations.txt").read_text().split(); st = pd.read_csv(ROOT / "data" / "national" / "stations.csv", dtype=str).set_index("code")
    d = d[d.station.isin(keep) & (d.date >= "2021-01-01") & (d.date <= "2025-12-31")]; n = (pd.Timestamp("2025-12-31") - pd.Timestamp("2021-01-01")).days + 1; rows = []
    for s in keep:
        g = d[d.station == s]; c = g.chlorophyll_a.dropna()
        rows.append({"station": s, "name_kr": st.loc[s, "name_kr"], "basin": st.loc[s, "basin"], "type": st.loc[s, "type"], "lat": st.loc[s, "lat"], "lon": st.loc[s, "lon"],
                     "chl_share": len(c) / n, "wt_share": g.water_temperature.notna().sum() / n, "tn_share": g.total_nitrogen.notna().sum() / n,
                     "tp_share": g.total_phosphorus.notna().sum() / n, "chl_days": len(c), "chl_median": c.median(), "chl_p90": c.quantile(0.9), "chl_max": c.max(),
                     "share_ge15": (c >= 15).mean(), "share_ge25": (c >= 25).mean(), "wt_median": g.water_temperature.median(), "tn_median": g.total_nitrogen.median(),
                     "tp_median": g.total_phosphorus.median(), "rain_mm_per_day": g.era5_tp_mm.mean()})
    T = pd.DataFrame(rows); (ROOT / "output" / "results").mkdir(parents=True, exist_ok=True); T.to_csv(ROOT / "output" / "results" / "station_table.csv", index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 100)
    print(sorted(d.columns)); print(T.round(3).to_string(index=False))
    for k in ("basin", "type"):
        print(T.groupby(k)[["chl_share", "wt_share", "tn_share", "tp_share", "chl_median", "chl_p90", "share_ge15", "share_ge25"]].agg(["min", "median", "max"]).round(3).T.to_string())
    print(T[["chl_share", "wt_share", "tn_share", "tp_share", "chl_median", "chl_p90"]].agg(["min", "median", "max"]).round(3).to_string())
