#!/bin/bash
# Stage 4: compute containment-based parent chain dict.
set -euo pipefail

SCENE_DIR="$1"
SCENE_NAME="$2"
REPO_ROOT="$3"

source "$REPO_ROOT/envs/hsegsplat/venv/bin/activate"
cd "$REPO_ROOT"

python scripts/compute_parent_chain.py \
    --scene_dir "$SCENE_DIR" \
    --output_dir "$SCENE_DIR"


deactivate
