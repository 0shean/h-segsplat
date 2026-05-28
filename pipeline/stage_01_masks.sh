#!/bin/bash
# Stage 1: SemanticSAM masks at three granularity levels.
#
# By default runs all three granularities (1, 3, 6). If $HSEGSPLAT_SEMSAM_LEVELS
# is set (space-separated, e.g. "3 6"), only those levels are emitted. This
# env var is used by the orchestrator when a different model (e.g. SAM ViT-H
# via stage_01a_sam_vith.sh) is supplying the lvl-1 masks.
set -euo pipefail
unset MPLBACKEND

SCENE_DIR="$1"
SCENE_NAME="$2"
SAM_CHECKPOINT="$3"
REPO_ROOT="$4"

source "$REPO_ROOT/envs/sam/venv/bin/activate"

# Run from repo root so the script's default --semantic_sam_dir resolution works.
cd "$REPO_ROOT"

# Build the --levels argument from $HSEGSPLAT_SEMSAM_LEVELS (default: "1 3 6").
SEMSAM_LEVELS="${HSEGSPLAT_SEMSAM_LEVELS:-1 3 6}"
echo "[stage 1] semantic-SAM levels: $SEMSAM_LEVELS"

python scripts/run_semantic_sam.py \
    --data_dir "$(dirname "$SCENE_DIR")" \
    --scene "$SCENE_NAME" \
    --checkpoint "$SAM_CHECKPOINT" \
    --levels $SEMSAM_LEVELS

deactivate
