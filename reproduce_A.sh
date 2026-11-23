#!/usr/bin/env bash
# Level A: rebuild every generated table and figure of the manuscript from the bundled data and result tables.
# No model is trained. Writes output/tables, output/figures and output/results;
# script messages go to output/logs. Usage: ./reproduce_A.sh   (PYTHON=/path/to/python to choose the interpreter)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
rm -rf output
mkdir -p output/tables/supp output/figures output/results output/logs

run() { local log="output/logs/$1.log"; shift; echo "  $*"; "$PY" "$@" > "$log" 2>&1 || { echo "FAILED: $* (see $log)"; tail -20 "$log"; exit 1; }; }

echo "station table from the daily data"
run station_table scripts/xpinn/station_table.py
echo "tables of the main text and Tables S1, S4-S7"
run paper_tables scripts/xpinn/paper_tables.py all
echo "Tables S2, S3, S8-S15"
run paper_supp scripts/xpinn/paper_supp.py
echo "figures"
if [ -f src/gis/HydroBASINS/hybas_as_lev06_v1c.shp ] && [ -d src/gis/Rivers_National ] && [ -d src/gis/시도 ]; then
  run fig1 scripts/xpinn/paper_figures.py fig1
else
  echo "  SKIPPED Figure 1 (study area map): the map layers are not part of the bundle."
  echo "  Place the HydroBASINS level-6 Asia basins, the national river network and the province boundaries in src/gis/"
  echo "  as the map-layer table in README.md says, then run: $PY scripts/xpinn/paper_figures.py fig1"
fi
for f in fig_framework fig2 fig3 fig4 fig5 fig6 fig7; do
  run "$f" scripts/xpinn/paper_figures.py "$f"
done
echo "Chl-a as a proxy for cyanobacteria, from the algae alert samples (input of Figure S1)"
run algae_proxy scripts/clean_revision/algae_proxy_analysis.py
echo "Figure S1 (fig6_algae_proxy)"
run figS1 scripts/clean_revision/make_figures_v4.py fig6
echo "done: output/"
