"""ERA5 daily weather at the official coordinates of the 37 stations (national benchmark), from the Open-Meteo
archive (model era5), with the variables and units of data/clean/era5_daily.csv (the six Geum stations). Output: data/national/era5_daily.csv
and data/national/era5_raw/ (raw responses and a manifest)."""
from __future__ import annotations
import json, time, datetime as dt, urllib.request, urllib.parse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY = "shortwave_radiation_sum,temperature_2m_mean,precipitation_sum,wind_speed_10m_max"


def main(start="2020-12-01", end="2025-12-31", model="era5"):
    nat = ROOT / "data" / "national"; raw_dir = nat / "era5_raw"; raw_dir.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(nat / "stations.csv", dtype={"code": str})
    manifest = {"retrieved_at": dt.datetime.now().isoformat(timespec="seconds"), "model": model, "daily": DAILY, "stations": {}}
    rows = []
    for r in st.itertuples(index=False):
        f = raw_dir / f"{r.code}.json"
        q = dict(latitude=r.lat, longitude=r.lon, start_date=start, end_date=end, daily=DAILY, timezone="Asia/Seoul", models=model)
        url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(q)
        if not f.exists():
            for attempt in range(6):
                try:
                    with urllib.request.urlopen(url, timeout=120) as resp:
                        f.write_bytes(resp.read())
                    break
                except Exception as e:
                    print("retry", r.code, attempt, e, flush=True); time.sleep(15 * (attempt + 1))
            else:
                raise SystemExit(f"failed {r.code}")
            time.sleep(1.5)
        js = json.loads(f.read_bytes()); d = js["daily"]
        df = pd.DataFrame({"station": r.code, "date": d["time"], "era5_ssrd_mj": d["shortwave_radiation_sum"], "era5_t2m": d["temperature_2m_mean"],
                           "era5_tp_mm": d["precipitation_sum"], "era5_wind_max": d["wind_speed_10m_max"]})
        rows.append(df)
        manifest["stations"][r.code] = {"name": r.name_kr, "lat": r.lat, "lon": r.lon, "url": url, "grid_lat": js.get("latitude"), "grid_lon": js.get("longitude"),
                                        "n_days": len(d["time"]), "n_missing_ssrd": int(pd.isna(df.era5_ssrd_mj).sum())}
        print(r.code, r.name_kr, js.get("latitude"), js.get("longitude"), len(d["time"]), "missing:", manifest["stations"][r.code]["n_missing_ssrd"], flush=True)
    pd.concat(rows, ignore_index=True).to_csv(nat / "era5_daily.csv", index=False)
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
