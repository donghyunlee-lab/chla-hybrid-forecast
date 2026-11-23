# Data

This document describes every UTF-8 comma-separated file under `data/`, including its source, build rules, columns and physical units. Dates follow Korea Standard Time (`YYYY-MM-DD`), and blank cells denote unmeasured days without any gap filling.

## Sources

| Source | Used for |
|---|---|
| Automatic water quality monitoring network of the Water Environment Information System (WEIS, https://water.nier.go.kr), National Institute of Environmental Research, Republic of Korea | Daily values of nine water quality variables; official station codes and coordinates |
| Weekly samples of the Korean algae alert system, published in the same system (menu for green algae information, past water quality data) | Laboratory Chl-a and cell counts of harmful cyanobacteria at eight sites of the Geum basin (Supplementary Note S4, Figure S1) |
| ERA5 reanalysis of the Copernicus Climate Change Service, daily values retrieved through the Open-Meteo historical weather API (`archive-api.open-meteo.com`, model `era5`, time zone Asia/Seoul) | Daily weather at each station (`scripts/national/fetch_era5_national.py`) |

## How the daily files were built

Two scripts (`scripts/national/build_national_dataset.py` and `scripts/national/build_national_v2.py`) assemble the daily water-quality tables from a local archive of public WEIS daily monitoring records under four rules:

1. Keep numeric measurements only and drop non-numeric source flags.
2. Average multiple measurements recorded on the same calendar day for the same station and variable.
3. Leave unmeasured station-days blank without interpolation or temporal smoothing.
4. Emit a complete daily grid from `2020-12-01` through `2025-12-31` across all 37 candidate stations (`1,857` days, `68,709` rows).

Model data loaders handle missing inputs dynamically at run time by carrying the latest measurement forward for up to 3 days, recording binary presence and observation-age channels, and substituting the pre-2024 training median after longer gaps. All training losses and evaluation metrics score measured Chl-a days exclusively.

Out of those 37 candidate stations, 27 continuous stations pass the completeness screen in `results/input_design_20261114/stations.txt`, while `results/input_design_20261114/station_coverage.csv` records the exclusion reason for each of the other ten stations.

### Two builds of the daily file

| Folder | Build | Used for |
|---|---|---|
| `data/national/` | First build | Read by some scripts (station coordinates for Figure 1, and parts of the Level B check) |
| `data/national_v2/` | Second build: the first build plus the 2025 values of eight Yeongsan–Seomjin stations | All models and results of the paper |

In the local WEIS archive, 2025 records for eight Yeongsan–Seomjin stations (`S04001`, `S04002`, `S04004`, `S04005`, `S04007`, `S04008`, `S04009` and `S04012`) appear under alternate station identifiers (`alias_2025`). Because the first build (`data/national/`) matched only the primary identifiers in `stations.csv`, those eight stations had empty 2025 rows there. The second build (`data/national_v2/`) maps those 2025 identifiers back to their stations while leaving every other cell unchanged. Through `2024-12-31`, both builds match cell for cell and produce identical development-year runs, and the package ships both folders because auxiliary scripts reference paths in `data/national/`. `data/national_v2/dataset_manifest.json` logs the identifier map, added 2025 observation counts and SHA-256 digests.

## Files

### `data/national_v2/` and `data/national/`

`daily_observed.csv`: daily water quality across 37 stations.

| Column | Meaning | Unit |
|---|---|---|
| `station` | WEIS station code (for example `S01001`) | |
| `date` | Day | |
| `water_temperature` | Water temperature | °C |
| `ph` | pH | |
| `electrical_conductivity` | Electrical conductivity | µS/cm |
| `dissolved_oxygen` | Dissolved oxygen | mg/L |
| `turbidity` | Turbidity | NTU |
| `total_organic_carbon` | Total organic carbon | mg/L |
| `chlorophyll_a` | Chlorophyll-a (sensor), the forecast target | mg/m³ |
| `total_nitrogen` | Total nitrogen (TN) | mg/L |
| `total_phosphorus` | Total phosphorus (TP) | mg/L |

`era5_daily.csv`: daily ERA5 weather extracted at the coordinates of the 37 stations in `stations.csv` from `2020-12-01` to `2025-12-31` (identical across both folders).

| Column | Meaning | Unit |
|---|---|---|
| `station`, `date` | As above | |
| `era5_ssrd_mj` | Shortwave radiation, daily sum | MJ/m² |
| `era5_t2m` | Air temperature at 2 m, daily mean | °C |
| `era5_tp_mm` | Precipitation, daily sum | mm |
| `era5_wind_max` | Wind speed at 10 m, daily maximum | km/h |

`observed_rates.csv`: annual fraction of measured days per station and variable (`2020` covers December only).

`stations.csv`: metadata for the 37 candidate stations.

| Column | Meaning |
|---|---|
| `code` | WEIS station code |
| `name_kr` | Station name in Korean (see the glossary below) |
| `basin` | `Han`, `Nakdong`, `Geum` or `Yeongsan` (the Yeongsan–Seomjin basin) |
| `type` | `river` or `lake` (reservoir) |
| `lat`, `lon` | Latitude and longitude, rounded (degrees). The official coordinates are in `weis_auto_stations.csv` |
| `core_station_id` | Identifier of the station in the source copy of the daily values |
| `chl_rate`, `tn_rate`, `tp_rate`, `wt_rate` | Shares of days with a value of Chl-a, TN, TP and water temperature, from the screening of the stations in the source copy |
| `n_chl_2021_2023`, `n_chl_2024`, `n_chl_2025` | Numbers of Chl-a records in the source copy in these years, from the same screening |
| `test_end` | Last day with values in the source copy: 2025-12-31 (2024-12-31 for the eight Yeongsan–Seomjin stations in `data/national/`) |
| `core_station_id_2025` | `data/national_v2` only: the other identifier under which the source copy stores the 2025 values of the eight Yeongsan–Seomjin stations |

Columns `chl_rate` through `n_chl_2025` record initial source-extract counts and are unused by package scripts. Manuscript coverage statistics come from `observed_rates.csv` and `results/input_design_20261114/station_table.csv`.

`weis_auto_stations.csv`: official code (`code`), Korean name (`name`), latitude (`lat`) and longitude (`lon`) for all 77 automatic WEIS stations retrieved on `2026-10-25` (identical across both folders, with endpoint details in the file header).

`record_eligibility.csv` (`data/national/` only): counts of measured Chl-a days in the first 80% training split (`chl_train`) and final 20% validation split (`chl_val`) for 6-, 12-, 24- and 36-month records ending `2023-12-31`, marking `eligible = True` when `chl_train >= 30` and `chl_val >= 10`.

### `data/clean/`

Six-station Geum River files read by the shared loader in `scripts/clean_revision/` and by the weekly algae-alert evaluation:

| File | Content |
|---|---|
| `geum_daily_observed.csv` | Daily water quality of six stations, columns and units as `daily_observed.csv`. The values are identical to those of `data/national_v2/daily_observed.csv` for the same stations |
| `era5_daily.csv` | Daily ERA5 weather of the six stations, columns and units as above. The values are not identical to those of `data/national/era5_daily.csv` for the same stations |
| `observed_rates.csv` | Share of days with a measured value, as above |
| `algae_alert_weekly.csv` | Weekly samples of the algae alert system, 2020-12-01 to 2025-12-29, eight sites, 2,165 rows (`scripts/clean_revision/fetch_algae_alert.py`) |

These Geum files index stations by their Korean names:

| Key | Station | WEIS code |
|---|---|---|
| 현도 | Hyeondo | S03009 |
| 남면 | Nammyeon | S03013 |
| 공주 | Gongju | S03004 |
| 부여 | Buyeo | S03005 |
| 용담호 | Yongdam | S03008 |
| 대청호 | Daecheong | S03003 |

`algae_alert_weekly.csv` stores one row per sampling site and date:

| Column | Meaning | Unit |
|---|---|---|
| `site_code` | WEIS code of the sampling site | |
| `group` | `S`: section of a drinking water reservoir; `B`: algae observation site at a weir | |
| `site_name`, `site_name_weis`, `water_body`, `sampling_location` | Names of the site, as given by the source (Korean; see the glossary) | |
| `river_or_lake` | 호소 (lake or reservoir) or 하천 (river) | |
| `paired_station` | Automatic monitoring station paired with the site in this study (Korean key as in the table above) | |
| `date` | Sampling day | |
| `water_temp` | Water temperature | °C |
| `ph` | pH | |
| `do` | Dissolved oxygen | mg/L |
| `transparency_m` | Transparency | m |
| `turbidity` | Turbidity | NTU |
| `chla` | Chl-a measured in the laboratory | mg/m³ |
| `cyano_cells` | Cell count of harmful cyanobacteria | cells/mL |
| `microcystis`, `anabaena`, `oscillatoria`, `aphanizomenon` | Cell counts of the four genera | cells/mL |
| `geosmin_ng_l`, `mib2_ng_l` | Geosmin and 2-MIB | ng/L |
| `mc_lr`, `mc_rr`, `mc_la`, `mc_yr`, `mc_lf`, `mc_ly` | Microcystin variants LR, RR, LA, YR, LF and LY | as reported by the source |
| `*_flag` | Text entry of the source in place of a number for the variable named in the column (see the glossary); the value column is then empty | |

## Glossary of Korean values

Source tables retain original Korean strings from WEIS.

Station names of the 27 study stations (`name_kr` in `stations.csv` and `station_table.csv`):

| Basin | Korean | Name in the paper |
|---|---|---|
| Han | 가평, 서상, 의암호, 능서, 양평 | Gapyeong, Seosang, Uiam, Neungseo, Yangpyeong |
| Nakdong | 적포, 청암, 다산, 칠곡, 상동, 남강 | Jeokpo, Cheongam, Dasan, Chilgok, Sangdong, Namgang |
| Geum | 대청호, 공주, 부여, 장계, 용담호, 현도, 남면, 유구천 | Daecheong, Gongju, Buyeo, Janggye, Yongdam, Hyeondo, Nammyeon, Yugucheon |
| Yeongsan–Seomjin | 주암호, 나주, 옥정호, 용봉, 탐진호, 동복호, 우치, 지석천 | Juam, Naju, Okjeong, Yongbong, Tamjin, Dongbok, Uchi, Jiseokcheon |

The Korean suffix 호 denotes a reservoir or lake.

Algae-alert sampling sites (`algae_alert_weekly.csv`):

| `site_code` | `site_name` | Meaning | `water_body` | `sampling_location` | `paired_station` |
|---|---|---|---|---|---|
| 3008G50 | 대청호(문의) | Daecheong Reservoir, Munui | 문의 (Munui) | 취수구 50m청주: 50 m from the intake for Cheongju | 대청호 (Daecheong) |
| 3008G40 | 대청호(추동) | Daecheong Reservoir, Chudong | 추동 (Chudong) | 취수구 50m대전: 50 m from the intake for Daejeon | 대청호 (Daecheong) |
| 3008G30 | 대청호(회남) | Daecheong Reservoir, Hoenam | 회남 (Hoenam) | 대청호: Daecheong Reservoir | 대청호 (Daecheong) |
| 3001G20 | 용담호(취수탑) | Yongdam Reservoir, intake tower | 취수탑 (intake tower) | 취수구 앞: in front of the intake | 용담호 (Yongdam) |
| 3001G40 | 용담호(댐앞) | Yongdam Reservoir, in front of the dam | 댐앞 (in front of the dam) | 취수구 앞: in front of the intake | 용담호 (Yongdam) |
| 3012A07 | 세종보 | Sejong Weir | 연기 (Yeongi) | 보 상류 1.1Km: 1.1 km upstream of the weir | 남면 (Nammyeon) |
| 3012A32 | 공주보 | Gongju Weir | 금강 (Geum River) | 보 상류 500m: 500 m upstream of the weir | 공주 (Gongju) |
| 3012A42 | 백제보 | Baekje Weir | 부여 (Buyeo) | 보 상류 500m: 500 m upstream of the weir | 부여 (Buyeo) |

`site_name_weis` names the water body: 대청호 (Daecheong Reservoir), 용담호 (Yongdam Reservoir), 세종보 (Sejong Weir), 공주보 (Gongju Weir) and 백제보 (Baekje Weir).

Flag strings in `*_flag` columns:

| Value | Meaning |
|---|---|
| 정량한계미만 | Below the limit of quantification |
| 분석중 | Analysis not yet finished at the time of retrieval |

## Not included

- This package omits the raw multi-station WEIS archive folder from which `build_national_dataset.py`, `build_national_v2.py` and `station_rule.py` read unmerged station exports, while shipping all compiled CSV outputs and pointing readers to public WEIS downloads at https://water.nier.go.kr.
- Raw JSON payloads from the Open-Meteo and WEIS algae-alert HTTP endpoints are omitted in favor of the assembled CSV tables.
- External GIS shapefiles for Figure 1 (HydroBASINS, national rivers and provincial boundaries) are obtained directly from HydroSHEDS and https://www.data.go.kr as documented in `README.md`.
