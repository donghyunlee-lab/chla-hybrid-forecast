# chla-hybrid-forecast

Code, daily monitoring data and result tables for the paper:

> Donghyun Lee. Hard constraints make an accurate hybrid chlorophyll-a forecast interpretable: a process equation with an
> xLSTM correction. Submitted to *Ecological Informatics*.

This study forecasts daily chlorophyll-a (Chl-a) 1, 3, 7 and 14 days ahead across 27 stations of South Korea's automatic
water-quality monitoring network during the 2024–2025 test years. At each station, the hybrid model steps a five-constant
Gompertz-type process equation in ln Chl-a inside fixed physical parameter intervals and adds an xLSTM daily adjustment
bounded within ±0.1 per day.

You can verify the paper at three reproducibility levels:

| Level | What it does | What you need | Time |
|---|---|---|---|
| A | Rebuilds every generated table and figure of the paper from the result tables | This package, Python 3.12, no GPU | About 1 minute |
| B | Recomputes the result tables from the saved forecasts and model files | This package plus three archives (about 730 MB to download, 3.2 GB unpacked) | About 7 minutes on a CPU |
| C | Trains the models again | This package, one NVIDIA GPU | About 29 hours on one RTX 4090 |

Repository: https://github.com/donghyunlee-lab/chla-hybrid-forecast
Archive record (package and large archives): https://doi.org/10.5281/zenodo.24470518

## Contents

| Path | Content |
|---|---|
| `README.md`, `DATA.md` | Package guide and column-by-column description of every data file |
| `LICENSE`, `LICENSE-DATA` | MIT license for code and CC BY 4.0 license plus third-party attributions for data and results |
| `CITATION.cff` | Citation metadata |
| `reproduce_A.sh` | Level A runner that writes all generated tables and figures to `output/` |
| `reproduce_B.sh` | Level B runner that recomputes all 141 summary tables from saved runs and checks agreement |
| `environment/` | Pinned package versions: `requirements-A.txt` (Level A), `requirements-A-map.txt` (extra map packages for Figure 1), `requirements-train.txt` (Levels B and C) and `system.txt` (Linux, GPU, driver, CUDA and Python environment used for training) |
| `data/` | Daily water-quality measurements, daily ERA5 weather, station metadata and weekly algae-alert samples (`DATA.md`) |
| `src/pinn_hab/` | Neural network architectures (xLSTM and baseline backbones) and Monod ODE loss modules for the loss-term PINN |
| `scripts/xpinn/` | Process and hybrid models (`core2.py`, `hybrid.py`), experiment drivers, and table/figure generators (`paper_tables.py`, `paper_supp.py`, `paper_figures.py`) |
| `scripts/national/` | Dataset builders and national benchmark comparison models (`train_nat.py`, `baselines_nat.py`) |
| `scripts/clean_revision/` | Shared data loader, weekly algae-alert correlation analysis (`algae_proxy_analysis.py`) and Figure S1 script (`make_figures_v4.py`) |
| `results/input_design_20261114/` | Summary CSV tables behind the paper: forecast accuracy, paired tests, term shares, fitted constants, alert metrics, driver perturbations and station coverage |
| `results/xpinn_explore_20261027/` | Development-year equation-search leaderboards reported in Supplementary Note S5 (Table S15) |
| `results/clean_revision_v4/FREEZE_20261010/results/e9_algae_proxy/summary.csv` | Chl-a versus harmful cyanobacterial cell counts in the Geum basin (Supplementary Note S4, Figure S1) |
| `results/FILES.tsv`, `results/FILES_B.tsv` | Byte sizes and SHA-256 checksums for shipped result tables (`FILES.tsv`) and for the 141 tables verified by Level B (`FILES_B.tsv`) |
| `manuscript/expected/` | Reference LaTeX tables and PDF/PNG figures matching the manuscript |
| `earlier_design/` | Archived code and tables for the causal five-station rerun summarized in Table R1 of the response to reviewers |
| `MANIFEST.sha256` | SHA-256 checksum, byte size and relative path of every file in this package |

## Names in the code and in the paper

Scripts use compact internal identifiers that map directly to the model and stage names in the paper:

| Code | Paper |
|---|---|
| `hybrid`, `main`, `H-res-all` | Hybrid model |
| `core`, `P-mcf-all-kT` | Process model (the hybrid model without the network) |
| `bb-nut` (`bb-C1` with the `nut-id` input) | Joint network, 17 channels |
| `bb-chl` | Joint network, Chl-a only |
| `bb-all` | Joint network, 32 channels |
| `bb-C1` | Joint network (xLSTM trained on the record of the target basin) |
| `PINN-sigmoid` | Loss-term PINN |
| `xLSTM zero-shot`, `xLSTM fine-tuned` | xLSTM zero-shot, xLSTM fine-tuned |
| `XGBoost zero-shot` | XGBoost zero-shot |
| `damped anomaly`, `harmonic ARIMA`, `climatology`, `persistence` | Damped anomaly, Harmonic ARIMA, Climatology, Persistence |
| `mean of six` | Mean of six |
| `free-eps` | Hybrid model without the bound on the correction |
| `free-const` | Hybrid model without the ranges of the constants |
| `no-pre` | Hybrid model without training on other basins |
| `growth` | Hybrid model with the growth term |
| `n17`, `nut` | The 17 input channels |
| `all` (input set) | All 32 input channels |
| `kappa`, `thk`, `omega`, `beta`, `alpha` | Station constants κ, θ, ω, β, α (Table 2) |
| `eps` | Correction of the network, ε |
| `pre-nat`, pre-training | First training step, on the three source basins |
| `dev` | Development year 2023 (records ending December 2022) |
| `test` | Test years 2024–2025 (records ending December 2023) |
| `r36`, `r24`, `r12` | Records of 36, 24 and 12 months |
| `nakdong`, `han`, `geum`, `yeongsan` | Basins (`yeongsan` denotes the Yeongsan–Seomjin basin) |
| stage 5a, 5b | Choice of the input length and of the input variables (Supplementary Note S1, Table S2) |
| stage 6 | Process model and hybrid model |
| stage 7 | Comparison models (Tables 4 and 7, S4 and S5) |
| stage 8 | Runs with one part of the hybrid model changed (Table 6) |

## Requirements

- All published outputs come from Python 3.12.13. Any Python 3.12 build runs the scripts.
- Level A installs `environment/requirements-A.txt`, which pins `pandas==3.0.6` because `pandas 3.1.0` parses three CSV floats differently in the last decimal place of Tables S1 and S3. Rebuilding Figure 1 additionally requires `environment/requirements-A-map.txt`.
- Matplotlib renders figure labels in Arial. Systems without Arial fall back to a substitute font with slightly different glyph widths.
- Levels B and C install `environment/requirements-train.txt` (`torch==2.11.0+cu128`). We trained all models and verified Level B on the Ubuntu Linux workstation documented in `environment/system.txt`.

## Level A: tables and figures from the result tables

```
python3.12 -m venv venv-a
venv-a/bin/pip install -r environment/requirements-A.txt
PYTHON=venv-a/bin/python ./reproduce_A.sh
```

`reproduce_A.sh` runs on CPU without model training and populates `output/`:

| Output | Content |
|---|---|
| `output/tables/` | Tables 1, 4, 5, 6 and 7 (`tab1_stations.tex`, `tab4_accuracy.tex`, `tab5_constants.tex`, `tab6_parts.tex`, `tab7_accuracy12.tex`) |
| `output/tables/supp/` | Supplementary Tables S1 to S15 |
| `output/figures/` | Figures 2 to 8 (`fig_framework`, `fig2_model` to `fig7_limits`) and Figure S1 (`fig6_algae_proxy`, while Figure 7 is `fig6_constraints`), as PDF and PNG |
| `output/results/` | Rebuilt station summary CSV and weekly Geum algae-alert correlation summary |
| `output/logs/` | Per-script execution logs |

You can diff `output/` directly against `manuscript/expected/`. Under `requirements-A.txt` with Arial installed, all 20 generated `.tex` tables match `manuscript/expected/tables/` byte for byte, the two rebuilt CSV summaries match `results/`, and all generated PNG figures match `manuscript/expected/figures/` pixel for pixel (PDF files carry creation timestamps, so compare rendered pages).

Two items sit outside the automatic Level A build:

- Hand-authored specification tables (Tables 2 and 3) live directly in `manuscript/expected/tables/`.
- Figure 1 (`fig1_study_area`) requires external GIS shapefiles that are not redistributed here. After placing those three layers under `src/gis/` as listed below and installing `environment/requirements-A-map.txt`, re-run `reproduce_A.sh` or execute `venv-a/bin/python scripts/xpinn/paper_figures.py fig1`.

| Layer | Source | Place it as |
|---|---|---|
| River basins | HydroBASINS, level 6, Asia (`hybas_as_lev06_v1c`), from HydroSHEDS (https://www.hydrosheds.org) | `src/gis/HydroBASINS/hybas_as_lev06_v1c.shp` with its side files |
| National river network | Public Data Portal of Korea (https://www.data.go.kr) | one shapefile in `src/gis/Rivers_National/` |
| Province boundaries | Public Data Portal of Korea (https://www.data.go.kr) | one shapefile in `src/gis/시도/` (the Korean word for provinces) |

Figure 1 reads official monitoring coordinates from `data/national/weis_auto_stations.csv`.

## Level B: result tables from the saved forecasts and model files

Download the three companion archives from Zenodo (https://doi.org/10.5281/zenodo.24470518) and verify their SHA-256 digests:

| Archive | Content | Download | Unpacked | SHA-256 |
|---|---|---|---|---|
| `pinn-hab-B-i-forecasts-constants.tar.xz` | Saved forecasts, saved station constants, run settings and the summary tables of every run behind the paper (15,228 files) | 261 MB | 2.11 GB | `4d40e38c01ee0c4672b0307df8b36dfd3e109f79cba4d660c67155cfab007868` |
| `pinn-hab-B-ii-model-files.tar.xz` | Model files of these runs (2,612 files) | 394 MB | 556 MB | `60d36dc0b38aecb1a7646d3135216e8b6043e02d1f163bdd013ebd574480852d` |
| `pinn-hab-B-iii-design-steps-S15.tar.xz` | Runs of the design steps behind Table S15 (1,989 files) | 78 MB | 509 MB | `cd4982860d04f7adf26c06a01b637074a45d1d18ba2458648b51169067e0a10d` |

Extract all three archives at the package root, create the Python environment and run `./reproduce_B.sh`. Archive `i` includes identical copies of the 28 summary files already present under `results/`.

```
tar -xJf pinn-hab-B-i-forecasts-constants.tar.xz
tar -xJf pinn-hab-B-ii-model-files.tar.xz
tar -xJf pinn-hab-B-iii-design-steps-S15.tar.xz
python3.12 -m venv .venv
.venv/bin/pip install -r environment/requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu128
PYTHON=.venv/bin/python ./reproduce_B.sh
```

Debian and Ubuntu systems need the `python3.12-venv` apt package before creating `.venv`. Alternatively, `uv` builds the same environment via `uv venv --python 3.12.13 .venv` followed by `uv pip install --python .venv/bin/python -r environment/requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match`.

Level B runs on CPU without retraining any model. First, `reproduce_B.sh` backs up all 141 shipped tables listed in `results/FILES_B.tsv` into `output/B_expected/`. Next, the scoring scripts recompute every accuracy table, paired Wilcoxon test, secondary metric, six-term log-space split, station constant table, alert score, case trajectory, seed-spread summary, rain/temperature perturbation table and development-year leaderboard from the raw prediction files. At the end, the script checks every rebuilt file against `output/B_expected/`, reports `identical` or `equal within 1e-9` and finishes with `141 of 141 files reproduced` in roughly 7 minutes on a 64-core CPU (`output/logs_B/`).

`reproduce_B.sh` pins `OMP_NUM_THREADS=16` during the rain and temperature perturbation step to match the archived CPU run. Other thread counts reorder 32-bit floating-point reductions and shift perturbation cells by at most `2e-7`.

Two screening files (`results/input_design_20261114/station_coverage.csv` and `stations.txt`, produced by `scripts/xpinn/station_rule.py`) stay fixed during Level B because `station_rule.py` reads raw station-level WEIS files that are not bundled here (`DATA.md`). All 2025 daily measurements for those stations appear in `data/national_v2/daily_observed.csv`.

## Level C: training the models again

All experiment drivers invoke `.venv/bin/python` relative to the package root, so build `.venv` there as shown in Level B. Run Level C in a clean copy of the package without unpacking the Level B archives, because the drivers skip any run directory that already contains finished outputs. Execute Steps 1 through 9 from the package root in order:

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `.venv/bin/python scripts/xpinn/inputlen.py all --workers 16` | Input length of the joint network (7, 14, 21 or 30 days) on the development year (seeds 42, 123, 456) | Steps 1 to 4: about 5 h |
| 2 | `.venv/bin/python scripts/xpinn/inputlen.py run --split test --lens 14 21 30 --seeds 42 123 456 1 2 --workers 16`, then the same command with `score` in place of `run` | Check of the chosen length on the test years | |
| 3 | `.venv/bin/python scripts/xpinn/inputvars.py all --workers 16` | Four sets of input variables, with and without the station identifier, on the development year | |
| 4 | `.venv/bin/python scripts/xpinn/inputvars.py run --split test --configs all-id nut-id chl-id --seeds 42 123 456 1 2 --workers 16`, then the same command with `score` in place of `run` | Check of the chosen input on the test years | |
| 5 | `.venv/bin/python scripts/xpinn/newinput.py all --workers 16` | Process model and hybrid model: pre-training on the source basins, fit at the target basin, scores on the development year and the test years, and a repeated run | About 5 h |
| 6 | `.venv/bin/python scripts/xpinn/newinput_bench.py all --slots 16 --workers 8`, then `.venv/bin/python scripts/xpinn/newinput_bench_more.py` | All comparison models, with 17 and with 32 input channels, scores and paired tests, followed by secondary scores | About 10 h |
| 7 | `.venv/bin/python scripts/xpinn/newinput_ablate.py all --workers 16` | The hybrid model with one part changed (`free-eps`, `free-const`, `no-pre`, `growth`), and scores | About 8.5 h |
| 8 | `.venv/bin/python scripts/xpinn/newinput_explain.py all`; `newinput_case.py`; `newinput_strength.py`; `newinput_strength.py by_type`; `newinput_ablate_more.py all`; `OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/xpinn/newinput_response.py --models main,core,bb-nut,free-eps,free-const` (all in `scripts/xpinn/`) | Term split, constants, alert scores, case series, spread between runs, results by station type, response tests (no training) | Minutes |
| 9 | `PYTHON=venv-a/bin/python ./reproduce_A.sh` | Tables and figures | About 1 minute |

Test-period evaluations use five random seeds (`42`, `123`, `456`, `1` and `2`), while development-year runs in Steps 1, 3 and 5 use three seeds (`42`, `123` and `456`).

On a single NVIDIA RTX 4090 GPU (24 GB VRAM) running 10 to 16 worker processes concurrently, Steps 1 through 9 finish in roughly 29 hours, with one hybrid run taking 15 to 20 minutes. Reduce `--workers` and `--slots` on GPUs with smaller memory. Supplementary Note S5 equation-search stages (Table S15) use `scripts/xpinn/explore.py`, `seasonal.py` and `hybrid.py` as documented at the top of each script, with saved outputs stored in archive `iii`.

After retraining, compare the regenerated summary CSVs against `results/FILES_B.tsv` and compare individual `predictions_eval.csv` or `predictions_test.csv` files against archive `i`. On the reference workstation in `environment/system.txt`, repeating a run produced identical forecast and constant files, whereas different GPUs or CUDA libraries introduce minor floating-point variations. For a single-basin spot check (~27 minutes on one RTX 4090), run the Geum fold on seed 42:

```
NEWINPUT_NETS=geum NEWINPUT_TEST_SEEDS=42 .venv/bin/python scripts/xpinn/newinput.py run --split test --workers 4
```

That command pre-trains the xLSTM on the three non-Geum basins and fits the process and hybrid models across the Geum stations for both 36- and 12-month records (`results/input_design_20261114/stage6/runs/`).

## Earlier design

`earlier_design/` preserves the code and result tables from the strictly causal five-station rerun reported in Table R1 of the response to reviewers. None of the manuscript's main or supplementary tables depend on `earlier_design/`.

## Data sources and licenses

- Daily automatic water-quality observations and weekly algae-alert records come from the Water Environment Information System (WEIS, https://water.nier.go.kr) operated by the National Institute of Environmental Research, Republic of Korea.
- Daily meteorological series come from the Copernicus Climate Change Service ERA5 reanalysis retrieved at station coordinates via the Open-Meteo Historical Weather API.
- External GIS layers for Figure 1 come from HydroSHEDS (HydroBASINS) and the Public Data Portal of Korea.

Source code carries the MIT license (`LICENSE`), while compiled datasets and result tables carry CC BY 4.0 alongside provider attributions (`LICENSE-DATA`). `DATA.md` documents the construction and schema of every data file.

## How to cite

Please cite the paper (`CITATION.cff`):

Donghyun Lee. Hard constraints make an accurate hybrid chlorophyll-a forecast interpretable: a process equation with an xLSTM
correction. Submitted to *Ecological Informatics*. Package DOI: https://doi.org/10.5281/zenodo.24470518.

## Contact

Donghyun Lee, Ph.D., Division of Social Science & AI, Hankuk University of Foreign Studies, 107 Imun-ro, Dongdaemun-gu, Seoul 02450, Republic of Korea, and R&D Center, AI Korea inc., Incheon, Republic of Korea (`donghyun.lee@hufs.ac.kr`). Code questions are also welcome via repository issues (https://github.com/donghyunlee-lab/chla-hybrid-forecast/issues).
