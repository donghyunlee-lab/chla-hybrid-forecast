#!/usr/bin/env bash
# Completes src/pinn_hab of the earlier design with the modules it shares, byte for byte, with the main code
# (src/pinn_hab at the bundle root). Each one is a relative symbolic link. Run once before run_fixed.sh,
# baselines_bootstrap.py, all_metrics_bootstrap.py or posthoc_fixed.py.
# Usage: ./link_shared_modules.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p "src/pinn_hab"; ln -sfn "../../../../src/pinn_hab/__init__.py" "src/pinn_hab/__init__.py"
mkdir -p "src/pinn_hab/models"; ln -sfn "../../../../../src/pinn_hab/models/__init__.py" "src/pinn_hab/models/__init__.py"
mkdir -p "src/pinn_hab/models"; ln -sfn "../../../../../src/pinn_hab/models/base.py" "src/pinn_hab/models/base.py"
mkdir -p "src/pinn_hab/models"; ln -sfn "../../../../../src/pinn_hab/models/factory.py" "src/pinn_hab/models/factory.py"
mkdir -p "src/pinn_hab/models"; ln -sfn "../../../../../src/pinn_hab/models/itransformer.py" "src/pinn_hab/models/itransformer.py"
mkdir -p "src/pinn_hab/models"; ln -sfn "../../../../../src/pinn_hab/models/mamba.py" "src/pinn_hab/models/mamba.py"
mkdir -p "src/pinn_hab/models"; ln -sfn "../../../../../src/pinn_hab/models/modern_tcn.py" "src/pinn_hab/models/modern_tcn.py"
mkdir -p "src/pinn_hab/models"; ln -sfn "../../../../../src/pinn_hab/models/xlstm.py" "src/pinn_hab/models/xlstm.py"
mkdir -p "src/pinn_hab/physics"; ln -sfn "../../../../../src/pinn_hab/physics/__init__.py" "src/pinn_hab/physics/__init__.py"
mkdir -p "src/pinn_hab/physics"; ln -sfn "../../../../../src/pinn_hab/physics/wrapper.py" "src/pinn_hab/physics/wrapper.py"
mkdir -p "src/pinn_hab/utils"; ln -sfn "../../../../../src/pinn_hab/utils/__init__.py" "src/pinn_hab/utils/__init__.py"
mkdir -p "src/pinn_hab/utils"; ln -sfn "../../../../../src/pinn_hab/utils/device.py" "src/pinn_hab/utils/device.py"
mkdir -p "src/pinn_hab/utils"; ln -sfn "../../../../../src/pinn_hab/utils/korean_mapping.py" "src/pinn_hab/utils/korean_mapping.py"
mkdir -p "src/pinn_hab/utils"; ln -sfn "../../../../../src/pinn_hab/utils/memory.py" "src/pinn_hab/utils/memory.py"
echo "linked 14 shared modules into src/pinn_hab"
