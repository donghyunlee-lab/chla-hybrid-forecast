"""Daily dataset of the national benchmark (first build): the 37 stations of data/national/stations.csv from an institutional
copy of the daily values of the WEIS automatic monitoring network (numeric observations only), 2020-12-01 to 2025-12-31, the
nine variables of the paper. Duplicates of a station, day and variable are averaged and nothing is filled; station keys are the
WEIS codes. Output: data/national/{daily_observed.csv, observed_rates.csv, dataset_manifest.json}."""
from __future__ import annotations
import argparse, json, os, sys, datetime as dt
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds, pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[2]
VARS = ["water_temperature", "ph", "electrical_conductivity", "dissolved_oxygen", "turbidity",
        "total_organic_carbon", "chlorophyll_a", "total_nitrogen", "total_phosphorus"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=os.environ.get("WEIS_DATA_CORE_ROOT", os.path.expanduser("weis-data-core")))
    ap.add_argument("--start", default="2020-12-01"); ap.add_argument("--end", default="2025-12-31")
    a = ap.parse_args()
    nat = ROOT / "data" / "national"
    st = pd.read_csv(nat / "stations.csv", dtype={"code": str})
    silver = Path(a.core) / "data/silver/observations/automatic/daily"
    d = ds.dataset(str(silver), format="parquet")
    names = st.name_kr.tolist()
    flt = pc.field("station_name_raw").isin(names) & pc.field("variable_id").isin(VARS) & (pc.field("is_numeric_observation") == True)
    cols = ["station_name_raw", "date_local", "variable_id", "value_numeric"] + (["station_id"] if "station_id" in d.schema.names else [])
    df = d.to_table(columns=cols, filter=flt).to_pandas()
    if "station_id" in df.columns:                                   # keep exactly the stations chosen in stations.csv
        df = df[df.station_id.isin(st.core_station_id)]
    df["date"] = pd.to_datetime(df["date_local"]).dt.normalize()
    df = df[(df["date"] >= a.start) & (df["date"] <= a.end)]
    code = dict(zip(st.name_kr, st.code)); df["station"] = df.station_name_raw.map(code)
    n_dup = int((df.groupby(["station", "date", "variable_id"]).size() > 1).sum())
    agg = df.groupby(["station", "date", "variable_id"])["value_numeric"].mean().unstack("variable_id")
    grid = pd.date_range(a.start, a.end, freq="D"); frames = []
    for c in st.code:
        sub = (agg.loc[c] if c in agg.index.get_level_values(0) else pd.DataFrame(index=grid)).reindex(grid).reindex(columns=VARS)
        sub.index.name = "date"; sub.insert(0, "station", c); frames.append(sub.reset_index())
    wide = pd.concat(frames, ignore_index=True)[["station", "date"] + VARS]
    wide["date"] = wide["date"].dt.strftime("%Y-%m-%d")
    wide.to_csv(nat / "daily_observed.csv", index=False, float_format="%.6g")
    w = wide.copy(); w["year"] = w["date"].str[:4]
    rates = w.groupby(["station", "year"])[VARS].apply(lambda x: x.notna().mean()).round(3).reset_index()
    rates.to_csv(nat / "observed_rates.csv", index=False)
    meta = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "source": "an institutional copy of the daily values of the WEIS automatic monitoring network",
            "silver_dir": str(silver), "date_range": [a.start, a.end], "stations": len(st), "variables": VARS, "rows": int(len(wide)),
            "numeric_cells": int(wide[VARS].notna().sum().sum()), "duplicate_station_date_variable_cells_averaged": n_dup,
            "rules": ["numeric observations only", "duplicates within station-date-variable averaged", "no filling, interpolation or smoothing"]}
    (nat / "dataset_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("stations", "rows", "numeric_cells", "duplicate_station_date_variable_cells_averaged")}))
    print(rates[rates.year.isin(["2021", "2023", "2024", "2025"])].pivot_table(index="station", columns="year", values="chlorophyll_a").round(2).to_string())


if __name__ == "__main__":
    main()
