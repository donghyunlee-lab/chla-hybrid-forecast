"""Harmful cyanobacteria cell counts of the algae alert samples (is chlorophyll-a a proxy for algal cells?).

Source: WEIS, the national water-environment information system (water.nier.go.kr), menu "data search > green algae
information > past water-quality data" (the Korean menu path is written to the manifest). Public, no login. The page allows
daily-data queries of up to two years; this script sends the same query the page sends, one site and one window at a time.

Writes
  data/external/algae_alert/raw/<site>_<start>_<end>.json   response bodies as received
  data/external/algae_alert/manifest.json                   request parameters, time received, HTTP status, SHA-256
  data/clean/algae_alert_weekly.csv                         tidy table, one row per site and sampling date
"""
from __future__ import annotations
import hashlib, http.cookiejar, json, time, urllib.parse, urllib.request, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "external" / "algae_alert" / "raw"
CLEAN = ROOT / "data" / "clean"
BASE = "https://water.nier.go.kr"
PAGE = BASE + "/web/algaePreMeasure?pMENU_NO=111"
QUERY = BASE + "/web/algaePreMeasure/toastList"
UA = "Mozilla/5.0 (X11; Linux x86_64) pinn-hab-reproducibility/1.0"
# group as coded on the page (S: drinking-water source sections, B: algae observation sites at the weirs),
# WEIS site code, site name, automatic monitoring station the site is paired with in this study
SITES = [("S", "3008G50", "대청호(문의)", "대청호"), ("S", "3008G30", "대청호(회남)", "대청호"), ("S", "3008G40", "대청호(추동)", "대청호"),
         ("S", "3001G20", "용담호(취수탑)", "용담호"), ("S", "3001G40", "용담호(댐앞)", "용담호"),
         ("B", "3012A32", "공주보", "공주"), ("B", "3012A42", "백제보", "부여"), ("B", "3012A07", "세종보", "남면")]
# The service filters on a month label given to each sampling week (the month of the week's Thursday), not on the
# sampling date: 2025-12-29 is labelled 2026-01, 2020-11-30 is labelled 2020-12. Windows therefore run one month past
# the study period on each side, stay within the page's two-year limit, and rows are then cut to DATE_RANGE by date.
WINDOWS = [("202012", "202212"), ("202301", "202412"), ("202501", "202601")]
DATE_RANGE = ("2020-12-01", "2025-12-31")
COLS = {"IEM_WTRTP": "water_temp", "IEM_PH": "ph", "IEM_DOC": "do", "IEM_TRP": "transparency_m", "IEM_TUR": "turbidity",
        "IEM_CHLA": "chla", "IEM_BGALAGE_CELL_CO": "cyano_cells", "IEM_BGALAGE_MICROSTS": "microcystis", "IEM_BGALAGE_ANBA": "anabaena",
        "IEM_BGALAGE_OSRTRIA": "oscillatoria", "IEM_BGALAGE_APZO": "aphanizomenon", "IEM_GEOSM": "geosmin_ng_l", "IEM_MIB2": "mib2_ng_l",
        "IEM_MICROSTLR": "mc_lr", "IEM_MICROSTRR": "mc_rr", "IEM_MICROSTLA": "mc_la", "IEM_MICROSTYR": "mc_yr", "IEM_MICROSTLF": "mc_lf", "IEM_MICROSTLY": "mc_ly"}


def form(group: str, code: str, s: str, e: str) -> dict:
    """The page's serialised search form for a daily-data query (querySn 53)."""
    return {"dFielNm": "과거수질자료", "excelSnsNos": "", "pMENU_NO": "111", "page": "", "ATTR_2": s, "ATTR_3": e, "querySn": "53",
            "serchRadio": group, "type_date": "1", "s_year": s[:4], "s_month": s[4:], "e_year": e[:4], "e_month": e[4:], "stationRadio": "1",
            "sido": "", "sigun": "", "middle": "", "small": "", "searchNm": "", "pageSize": "10", "station_code": code}


def to_num(x):
    if x is None:
        return np.nan, ""
    try:
        return float(str(x).replace(",", "").strip()), ""
    except ValueError:
        return np.nan, str(x).strip()          # e.g. the flag for a value below the limit of quantification (kept as given)


def main():
    RAW.mkdir(parents=True, exist_ok=True); CLEAN.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    opener.addheaders = [("User-Agent", UA)]
    opener.open(PAGE, timeout=30).read()                         # session cookie, as a browser would have
    manifest, rows = [], []
    for group, code, name, paired in SITES:
        for s, e in WINDOWS:
            params = form(group, code, s, e)
            req = urllib.request.Request(QUERY, data=urllib.parse.urlencode(params).encode("utf-8"), method="POST",
                                         headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                                                  "X-Requested-With": "XMLHttpRequest", "Referer": PAGE})
            with opener.open(req, timeout=60) as r:
                body, status = r.read(), r.status
            received = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds")
            f = RAW / f"{code}_{s}_{e}.json"; f.write_bytes(body)
            recs = json.loads(body.decode("utf-8")).get("list", [])
            manifest.append({"file": f"raw/{f.name}", "url": QUERY, "method": "POST", "params": params, "http_status": status,
                             "received_kst": received, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "records": len(recs)})
            for rec in recs:
                row = {"site_code": rec.get("SWMN_CODE"), "group": group, "site_name": name, "site_name_weis": rec.get("SWMN_NM"),
                       "water_body": rec.get("SWMN_DETAIL_NM"), "sampling_location": rec.get("DETAIL_ADRES"), "river_or_lake": rec.get("RIVER_LKMH_SE"),
                       "paired_station": paired, "date": pd.to_datetime(rec.get("CHCK_DE"), format="%Y.%m.%d")}
                for k, c in COLS.items():
                    row[c], flag = to_num(rec.get(k))
                    if flag:
                        row[f"{c}_flag"] = flag
                rows.append(row)
            print(code, name, s, e, status, len(recs), flush=True)
            time.sleep(1.0)
    df = pd.DataFrame(rows).sort_values(["site_code", "date"]).reset_index(drop=True)
    n_all = len(df)
    df = df[(df.date >= DATE_RANGE[0]) & (df.date <= DATE_RANGE[1])].drop_duplicates().reset_index(drop=True)
    dup = df.duplicated(["site_code", "date"], keep=False)
    if dup.any():
        print("WARNING: differing rows for the same site and date:\n", df[dup].to_string())
    print(f"{n_all} rows received, {len(df)} kept within {DATE_RANGE}")
    flags = sorted(c for c in df.columns if c.endswith("_flag"))
    df = df[[c for c in df.columns if not c.endswith("_flag")] + flags]
    out = CLEAN / "algae_alert_weekly.csv"; df.to_csv(out, index=False)
    meta = {"source": "WEIS 물환경정보시스템 (water.nier.go.kr), 자료조회 > 녹조(조류)정보 > 과거수질자료, daily data (querySn 53)",
            "page": PAGE, "retrieved_kst": manifest[0]["received_kst"] if manifest else None, "script": "scripts/clean_revision/fetch_algae_alert.py",
            "window_rule": "service filters on the month label of each sampling week (month of the week's Thursday); windows extend one month past the study period and rows are cut by sampling date",
            "date_range": DATE_RANGE,
            "output_csv": str(out.relative_to(ROOT)), "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "requests": manifest}
    (RAW.parent / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(df)} rows -> {out}")
    print(df.groupby(["site_code", "site_name", df.date.dt.year]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
