#!/bin/bash
# Stage 2: per-mask SigLIP feature extraction.
set -euo pipefail

SCENE_DIR="$1"
SCENE_NAME="$2"
REPO_ROOT="$3"

source "$REPO_ROOT/envs/siglip/venv/bin/activate"
cd "$REPO_ROOT"

python scripts/run_siglip.py \
    --data_dir "$(dirname "$SCENE_DIR")" \
    --scene "$SCENE_NAME"

deactivate
