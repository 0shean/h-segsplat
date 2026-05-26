#!/bin/bash
# Stage 3: build per-level banks, index maps, mask features tables, mask directories.
set -euo pipefail

SCENE_DIR="$1"
SCENE_NAME="$2"
REPO_ROOT="$3"

source "$REPO_ROOT/envs/hsegsplat/venv/bin/activate"
cd "$REPO_ROOT"

# Outputs land directly in the scene directory.
python scripts/build_hsegsplat_inputs.py \
    --scene_dir "$SCENE_DIR" \
    --output_dir "$SCENE_DIR" \
    --scene_key "$SCENE_NAME"

deactivate
