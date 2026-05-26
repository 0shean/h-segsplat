#!/bin/bash
# Stage 1: SemanticSAM masks at three granularity levels.
set -euo pipefail
unset MPLBACKEND

SCENE_DIR="$1"
SCENE_NAME="$2"
SAM_CHECKPOINT="$3"
REPO_ROOT="$4"

source "$REPO_ROOT/envs/sam/venv/bin/activate"

# Run from repo root so the script's default --semantic_sam_dir resolution works.
cd "$REPO_ROOT"

python scripts/run_semantic_sam.py \
    --data_dir "$(dirname "$SCENE_DIR")" \
    --scene "$SCENE_NAME" \
    --checkpoint "$SAM_CHECKPOINT"

deactivate
