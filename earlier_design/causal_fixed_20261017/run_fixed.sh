#!/usr/bin/env bash
# Original design on the past-only dataset with the corrected implementation (the corrections are marked 'rerun fix'
# in the code). Five jobs in parallel:
#   h01   phase 1 (steps 1-3), phase 2 step 2 (transfer), phase 3 (steps 1-4), baselines, all metrics   (1 day ahead)
#   h03 / h07 / h14   phase 1 steps 1-2, baselines, all metrics                                          (3 / 7 / 14 days)
#   voc0  phase 1 steps 1-2 without the volatile organic compounds (1 day ahead)
# Extra arguments go to every training script (short test: --epochs 1 --seeds 42).
# Usage, from this folder: ./link_shared_modules.sh once, then  mkdir -p logs; bash run_fixed.sh > logs/run_fixed.log 2>&1
# (PY=/path/to/python to choose the interpreter).
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
SEEDS_CSV="${BASELINE_SEEDS:-42,123,456}"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
prep() { mkdir -p "$1/logs"; for x in src scripts data; do ln -sfn "../$x" "$1/$x"; done; }
step() {   # step <dir> <h> <log> <script and args...>
  local d=$1 h=$2 log=$3; shift 3
  echo "[$(stamp)] $d: $*"
  (cd "$d" && PINN_HORIZON=$h "$PY" "$@" > "logs/$log.log" 2>&1) || { echo "[$(stamp)] $d: FAILED $*"; return 1; }
}
evaluate() {
  local d=$1 h=$2
  (cd "$d" && PINN_HORIZON=$h BASELINE_SEEDS=$SEEDS_CSV "$PY" ../../baselines_bootstrap.py . > logs/baselines.log 2>&1) || { echo "[$(stamp)] $d: FAILED baselines"; return 1; }
  (cd "$d" && "$PY" ../../all_metrics_bootstrap.py . > logs/all_metrics.log 2>&1) || { echo "[$(stamp)] $d: FAILED all metrics"; return 1; }
  echo "[$(stamp)] $d: evaluated"
}
full_h1() {
  prep h01
  step h01 1 phase1 scripts/run_phase1_backbone_contest.py "$@" || return 1
  step h01 1 phase2 scripts/run_phase2_scarcity.py --step 2 "$@" || return 1
  step h01 1 phase3 scripts/run_phase3_xai.py "$@" || return 1
  evaluate h01 1
}
lead() {
  local h=$1; shift; local d; d=$(printf "h%02d" "$h"); prep "$d"
  step "$d" "$h" step1 scripts/run_phase1_backbone_contest.py --step 1 "$@" || return 1
  step "$d" "$h" step2 scripts/run_phase1_backbone_contest.py --step 2 "$@" || return 1
  evaluate "$d" "$h"
}
novoc() {
  prep voc0
  (export PINN_DROP_VOC=1
   step voc0 1 step1 scripts/run_phase1_backbone_contest.py --step 1 "$@" && step voc0 1 step2 scripts/run_phase1_backbone_contest.py --step 2 "$@")
}
echo "[$(stamp)] start; args: $*; data sha256 $(sha256sum data/imputed_daily_data.csv | cut -c1-16)"
full_h1 "$@" &
for h in 3 7 14; do lead "$h" "$@" & done
novoc "$@" &
wait
echo "[$(stamp)] all jobs finished"
