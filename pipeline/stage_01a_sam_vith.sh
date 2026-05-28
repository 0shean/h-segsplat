#!/bin/bash
# Stage 1a: SAM ViT-H masks at level 1 (replaces Semantic-SAM lvl 1).
#
# Runs Meta's original SAM (NOT SAM 2) via SamAutomaticMaskGenerator. The output
# layout is identical to Semantic-SAM at lvl 1, so downstream stages don't care
# which model emitted the masks_lvl_1/ folder.
#
# Use in tandem with stage_01_masks.sh:
#     bash pipeline/stage_01a_sam_vith.sh  <SCENE_DIR> <SCENE_NAME> <SAM_VITH_CKPT> <REPO_ROOT>
#     HSEGSPLAT_SEMSAM_LEVELS="3 6" bash pipeline/stage_01_masks.sh ...
set -euo pipefail
unset MPLBACKEND

SCENE_DIR="$1"
SCENE_NAME="$2"
SAM_VITH_CHECKPOINT="$3"
REPO_ROOT="$4"

source "$REPO_ROOT/envs/sam/venv/bin/activate"

cd "$REPO_ROOT"

python scripts/run_sam_vith_lvl1.py \
    --data_dir "$(dirname "$SCENE_DIR")" \
    --scene "$SCENE_NAME" \
    --checkpoint "$SAM_VITH_CHECKPOINT"

deactivate