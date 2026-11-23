#!/usr/bin/env bash
# Level B: recompute the summary result tables from the saved runs (saved forecasts, saved constants, model files and run
# settings). No model is trained. Needs the core bundle plus the archives of saved forecasts and model files unpacked at the
# bundle root (results/...).
#
# The scripts write their tables in place, next to the shipped ones (results/.../dev, test, explain, strength, response,
# stage*/dev|test|more and the exploration leaderboards). Before the run the shipped tables listed in results/FILES_B.tsv are
# copied to output/B_expected/; after the run every listed file is compared with that copy: identical, or numerically equal
# within 1e-9. Script messages go to output/logs_B/. Some scripts also write a file ledger.jsonl that lists their runs; it is
# not compared.
#
# CPU is enough (no step trains a model; a GPU, if present, is only used to load arrays). About 7 minutes on 64 cores.
# The response to rain and temperature runs on the CPU with 16 threads (OMP_NUM_THREADS=16), as the archived run did; with
# another thread count its float32 sums differ in the 7th digit (largest difference about 2e-7).
#
# Not recomputed here: results/input_design_20261114/station_coverage.csv and stations.txt (scripts/xpinn/station_rule.py)
# need the 2025 Chl-a days of eight Yeongsan-Seomjin stations from a copy of the WEIS daily values that is not redistributed; the 2025
# values themselves are in data/national_v2/daily_observed.csv.
#
# Usage: ./reproduce_B.sh   (PYTHON=/path/to/python to choose the interpreter; environment/requirements-train.txt)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
LIST=results/FILES_B.tsv
[ -f "$LIST" ] || { echo "missing $LIST (unpack the saved-forecast archive at the bundle root)"; exit 1; }

rm -rf output/B_expected output/logs_B
mkdir -p output/B_expected output/logs_B
tail -n +2 "$LIST" | cut -f1 | while read -r f; do mkdir -p "output/B_expected/$(dirname "$f")"; cp -p "$f" "output/B_expected/$f"; done
DONE_B5=results/xpinn_explore_20261027/DONE_b5; cp -p "$DONE_B5" output/B_expected/DONE_b5.marker   # explore.py score rewrites it

run() { local name=$1; shift; local t0=$SECONDS; echo "  $name: $*"
  "$@" > "output/logs_B/$name.log" 2>&1 || { echo "FAILED: $* (see output/logs_B/$name.log)"; tail -20 "output/logs_B/$name.log"; exit 1; }
  echo "    $((SECONDS - t0)) s"; }

echo "stage 5a, input length: development year, then the check on the test years"
run 5a_dev  "$PY" scripts/xpinn/inputlen.py score --split dev
run 5a_test "$PY" scripts/xpinn/inputlen.py score --split test --lens 14 21 30 --seeds 42 123 456 1 2
echo "stage 5b, input variables and station identifier"
run 5b_dev  "$PY" scripts/xpinn/inputvars.py score --split dev
run 5b_test "$PY" scripts/xpinn/inputvars.py score --split test --configs all-id nut-id chl-id --seeds 42 123 456 1 2
echo "stage 6, process core and hybrid: station and pooled scores, constants, process share"
run 6_dev   "$PY" scripts/xpinn/newinput.py score --split dev
run 6_test  "$PY" scripts/xpinn/newinput.py score --split test
echo "stage 7, all models of the national benchmark: scores, ranks, paired differences, by basin; secondary scores"
run 7_score "$PY" scripts/xpinn/newinput_bench.py score
run 7_more  "$PY" scripts/xpinn/newinput_bench_more.py
echo "stage 8, runs with a part removed"
run 8_score "$PY" scripts/xpinn/newinput_ablate.py score
echo "explanation materials: fit, term split, constants by station, alert scores, diagnosis; case series"
run explain "$PY" scripts/xpinn/newinput_explain.py all
run case    "$PY" scripts/xpinn/newinput_case.py
echo "strength materials: single-run spread, fitted weights; by station type"
run strength      "$PY" scripts/xpinn/newinput_strength.py
run strength_type "$PY" scripts/xpinn/newinput_strength.py by_type
echo "stage 8 further materials: spread, constants, term split"
run ablate_more "$PY" scripts/xpinn/newinput_ablate_more.py all
echo "response to rain and water temperature (five models, CPU, 16 threads)"
run response env OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= "$PY" scripts/xpinn/newinput_response.py --models main,core,bb-nut,free-eps,free-const
echo "design steps of the exploration (Table S15): leaderboard of batches 1-5, seasonal and hybrid leaderboards"
run explore_score "$PY" scripts/xpinn/explore.py score --batch b5
cp -p output/B_expected/DONE_b5.marker "$DONE_B5"
run lb_seasonal "$PY" - <<'EOF'
import sys, json
sys.path.insert(0, "scripts/xpinn")
import seasonal as S                       # seasonal.py step1 and step2 without their runs (all runs are saved)
s1 = list(S.STEP1.values()) + list(S.REF.values())
S.leaderboard(s1, 1)
t = json.loads((S.ST / "target.json").read_text())["target"]
S.leaderboard(s1 + list(S.step2_specs(t).values()), 2)
EOF
run lb_hybrid "$PY" - <<'EOF'
import sys
sys.path.insert(0, "scripts/xpinn")
import hybrid as H                         # hybrid.py all, steps 1 and 2, without their runs (all runs are saved)
lb, _ = H.dev_board(list(H.CORES.values()), "step1"); core = H.choose_core(lb)
H.dev_board(list(H.CORES.values()) + list(H.hybrids("filter" if "filter" in core else "all").values()), "step2")
EOF

echo "comparing with the shipped tables"
"$PY" - "$LIST" <<'EOF'
import sys, json, math, filecmp
from pathlib import Path
import numpy as np, pandas as pd
TOL = 1e-9

def frames(a, b):
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return None
    worst = 0.0
    for c in a.columns:
        x, y = a[c], b[c]; xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        if (xn.notna() | x.isna()).all() and (yn.notna() | y.isna()).all():
            xv, yv = xn.to_numpy(float), yn.to_numpy(float)
            if (np.isnan(xv) != np.isnan(yv)).any():
                return None
            d = np.abs(xv - yv)[~np.isnan(xv)]; worst = max(worst, float(d.max()) if d.size else 0.0)
        elif (x.fillna("<NA>").to_numpy() != y.fillna("<NA>").to_numpy()).any():
            return None
    return worst

def js(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b): return None
        r = [js(a[k], b[k]) for k in a]; return None if None in r else max(r, default=0.0)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): return None
        r = [js(x, y) for x, y in zip(a, b)]; return None if None in r else max(r, default=0.0)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return 0.0 if (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) else abs(float(a) - float(b))
    return 0.0 if a == b else None

bad = 0; rows = [l.split("\t")[0] for l in Path(sys.argv[1]).read_text().splitlines()[1:] if l.strip()]
for f in rows:
    new, ref = Path(f), Path("output/B_expected") / f
    if filecmp.cmp(new, ref, shallow=False):
        st = "identical"
    else:
        try:
            if new.suffix == ".json":
                d = js(json.loads(new.read_text()), json.loads(ref.read_text()))
            else:
                rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False).replace("", np.nan)
                d = frames(rd(new), rd(ref))
        except Exception as e:
            d = None
        st = "DIFFERENT" if d is None or d > TOL else f"equal within 1e-9 (largest difference {d:.2g})"
    bad += st == "DIFFERENT"; print(f"{st}\t{f}")
print(f"{len(rows) - bad} of {len(rows)} files reproduced")
sys.exit(1 if bad else 0)
EOF
echo "done"
