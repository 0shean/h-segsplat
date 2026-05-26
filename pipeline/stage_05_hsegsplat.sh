#!/bin/bash
# Stage 5: H-SegSplat inference — DepthSplat encoder + gsplat rasterizer.
#
# Reproduces the long Hydra command that the original Colab notebook used.
# All file-system paths point at the scene directory; outputs land there.
set -euo pipefail
unset MPLBACKEND

SCENE_DIR="$1"
SCENE_NAME="$2"
DEPTHSPLAT_CHECKPOINT="$3"
REPO_ROOT="$4"

source "$REPO_ROOT/envs/hsegsplat/venv/bin/activate"

# Hydra resolves config_path relative to the script's cwd. The inference script
# uses `config_path="config"` which points at DepthSplat's config dir, so we
# must cd into depthsplat/ before running.
cd "$REPO_ROOT/depthsplat"

# Datasets / eval index references inside the scene dir.
DATASET_ROOT="$SCENE_DIR/datasets/${SCENE_NAME}_2view"
EVAL_INDEX="$SCENE_DIR/assets/${SCENE_NAME}_2view_eval.json"

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "[stage 5] ERROR: dataset chunk not found at $DATASET_ROOT"
    echo "[stage 5]        stages 3+4 should have produced it"
    exit 1
fi

python "$REPO_ROOT/scripts/run_hsegsplat_inference.py" \
    +experiment=dl3dv \
    dataset.test_chunk_interval=1 \
    "dataset.roots=[$DATASET_ROOT]" \
    "dataset.image_shape=[${HSEGSPLAT_TARGET_H:-640},${HSEGSPLAT_TARGET_W:-960}]" \
    "dataset.ori_image_shape=[${HSEGSPLAT_TARGET_H:-640},${HSEGSPLAT_TARGET_W:-960}]" \
    dataset.max_fov=130.0 \
    model.encoder.num_scales=2 \
    model.encoder.upsample_factor=4 \
    model.encoder.lowest_feature_resolution=8 \
    model.encoder.monodepth_vit_type=vitb \
    model.encoder.gaussian_adapter.gaussian_scale_max=0.2 \
    "checkpointing.pretrained_model=$DEPTHSPLAT_CHECKPOINT" \
    mode=test \
    dataset/view_sampler=evaluation \
    "dataset.view_sampler.index_path=$EVAL_INDEX" \
    data_loader.test.num_workers=0 \
    "++segsplat.assets_dir=$SCENE_DIR" \
    "++segsplat.output_dir=$SCENE_DIR" \
    "++segsplat.levels=[1,3,6]"

deactivate

echo "[stage 5] gaussians.pt written to $SCENE_DIR/gaussians.pt"
