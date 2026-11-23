"""National daily dataset, second build: the first build (data/national) plus the 2025 values of the eight
Yeongsan-Seomjin stations. Their 2025 files carry no basin prefix in the institutional copy of the WEIS daily values, so they
have other station IDs there (ALIAS_2025) and build_national_dataset.py, which keeps only the IDs of stations.csv, left 2025 empty for them.
Same rules as before (automatic network, daily values, numeric observations only, duplicates averaged, nothing filled). Rows are
assigned to stations by station ID, not by name. The first build (data/national) is not touched.
Output: data/national_v2/{daily_observed.csv, observed_rates.csv, stations.csv, dataset_manifest.json} and unchanged copies of
era5_daily.csv, weis_auto_stations.csv and record_eligibility.csv (records end 2023-12-31, so eligibility is the same)."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, shutil
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds, pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[2]
VARS = ["water_temperature", "ph", "electrical_conductivity", "dissolved_oxygen", "turbidity",
        "total_organic_carbon", "chlorophyll_a", "total_nitrogen", "total_phosphorus"]
ALIAS_2025 = {"S04001": "AUT-dbd1abfb269d4d", "S04002": "AUT-5c258e9cd9395e", "S04004": "AUT-54db41b4ca50a2", "S04005": "AUT-1885c8d145ea84",
              "S04007": "AUT-303799d7e6824e", "S04008": "AUT-a30c6ff18489e0", "S04009": "AUT-9bd8fc04b61947", "S04012": "AUT-5a4669e6f5847e"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=os.environ.get("WEIS_DATA_CORE_ROOT", os.path.expanduser("weis-data-core")))
    ap.add_argument("--start", default="2020-12-01"); ap.add_argument("--end", default="2025-12-31"); ap.add_argument("--out", default=str(ROOT / "data" / "national_v2"))
    a = ap.parse_args()
    v1, out = ROOT / "data" / "national", Path(a.out); out.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(v1 / "stations.csv", dtype={"code": str})
    ids = {**dict(zip(st.core_station_id, st.code)), **{v: k for k, v in ALIAS_2025.items()}}            # station ID -> station code
    silver = Path(a.core) / "data/silver/observations/automatic/daily"
    d = ds.dataset(str(silver), format="parquet")
    flt = pc.field("station_id").isin(list(ids)) & pc.field("variable_id").isin(VARS) & (pc.field("is_numeric_observation") == True)
    df = d.to_table(columns=["station_id", "station_name_raw", "date_local", "variable_id", "value_numeric"], filter=flt).to_pandas()
    df["station"] = df.station_id.map(ids); df["date"] = pd.to_datetime(df["date_local"]).dt.normalize()
    df = df[(df["date"] >= a.start) & (df["date"] <= a.end)]
    name = dict(zip(st.code, st.name_kr)); bad = df[df.station_name_raw != df.station.map(name)]
    assert bad.empty, f"station names differ from stations.csv: {bad[['station', 'station_name_raw']].drop_duplicates().to_dict('records')}"
    al = df[df.station_id.isin(list(ALIAS_2025.values()))]
    assert (al.date.dt.year == 2025).all(), "the alias IDs must hold 2025 only"
    both = df[df.station.isin(ALIAS_2025)].groupby(["station", "date", "variable_id"]).station_id.nunique()
    assert (both == 1).all(), "a station-day-variable is covered by two station IDs"
    n_dup = int((df.groupby(["station", "date", "variable_id"]).size() > 1).sum())
    agg = df.groupby(["station", "date", "variable_id"])["value_numeric"].mean().unstack("variable_id")
    grid = pd.date_range(a.start, a.end, freq="D"); frames = []
    for c in st.code:
        sub = (agg.loc[c] if c in agg.index.get_level_values(0) else pd.DataFrame(index=grid)).reindex(grid).reindex(columns=VARS)
        sub.index.name = "date"; sub.insert(0, "station", c); frames.append(sub.reset_index())
    wide = pd.concat(frames, ignore_index=True)[["station", "date"] + VARS]
    wide["date"] = wide["date"].dt.strftime("%Y-%m-%d")
    wide.to_csv(out / "daily_observed.csv", index=False, float_format="%.6g")
    w = wide.copy(); w["year"] = w["date"].str[:4]
    rates = w.groupby(["station", "year"])[VARS].apply(lambda x: x.notna().mean()).round(3).reset_index()
    rates.to_csv(out / "observed_rates.csv", index=False)
    s2 = st.copy(); s2["core_station_id_2025"] = s2.code.map(ALIAS_2025).fillna(""); s2.loc[s2.code.isin(ALIAS_2025), "test_end"] = a.end
    s2.to_csv(out / "stations.csv", index=False)
    for f in ("era5_daily.csv", "weis_auto_stations.csv", "record_eligibility.csv"):
        shutil.copyfile(v1 / f, out / f)
    # comparison with the first build: equal everywhere except the 2025 rows of the alias stations
    a1 = pd.read_csv(v1 / "daily_observed.csv", dtype={"station": str}); a2 = pd.read_csv(out / "daily_observed.csv", dtype={"station": str})
    assert (a1[["station", "date"]].to_numpy() == a2[["station", "date"]].to_numpy()).all()
    new = a2.station.isin(ALIAS_2025) & (a2.date >= "2025-01-01")
    x1, x2 = a1.loc[~new, VARS], a2.loc[~new, VARS]
    same_old = bool(((x1 == x2) | (x1.isna() & x2.isna())).all().all())
    added = a2.loc[new].groupby("station")[VARS].apply(lambda x: x.notna().sum())
    meta = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "source": "an institutional copy of the daily values of the WEIS automatic monitoring network",
            "silver_dir": str(silver), "date_range": [a.start, a.end], "stations": len(st), "variables": VARS, "rows": int(len(wide)),
            "numeric_cells": int(wide[VARS].notna().sum().sum()), "duplicate_station_date_variable_cells_averaged": n_dup,
            "rules": ["numeric observations only", "duplicates within station-date-variable averaged", "no filling, interpolation or smoothing",
                      "rows assigned by station ID; the 2025 IDs of the eight Yeongsan-Seomjin stations are added (alias_2025)"],
            "alias_2025": ALIAS_2025, "first_build": {"daily_observed_sha256": sha(v1 / "daily_observed.csv"), "cells_outside_the_added_rows_identical": same_old,
                                                      "cells_in_the_added_rows_first_build": int(a1.loc[new, VARS].notna().sum().sum())},
            "added_2025_measured_days": {k: {v: int(n) for v, n in r.items()} for k, r in added.iterrows()},
            "sha256": {f: sha(out / f) for f in ("daily_observed.csv", "era5_daily.csv", "stations.csv", "weis_auto_stations.csv", "record_eligibility.csv")}}
    (out / "dataset_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("stations", "rows", "numeric_cells", "duplicate_station_date_variable_cells_averaged")}))
    print("identical to the first build outside the added rows:", same_old, "| cells of the first build in the added rows:", meta["first_build"]["cells_in_the_added_rows_first_build"])
    print(added.to_string())


if __name__ == "__main__":
    main()
