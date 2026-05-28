#!/bin/bash
#SBATCH --output=logs/render_midpoint_%x_%j.log
#SBATCH --error=logs/render_midpoint_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=00:30:00

# One-shot midpoint render for a single MultiScan scene.
# Submit with:
#   sbatch --job-name=midrender_<scene> pipeline/euler/render_midpoint.sbatch <scene>
# where <scene> is e.g. scene_00044_00.

set -e
SCENE_ARG="$1"
if [[ -z "$SCENE_ARG" ]]; then
    echo "Usage: sbatch pipeline/euler/render_midpoint.sbatch <scene_name>"
    exit 1
fi

echo "Render midpoint for scene $SCENE_ARG on node $(hostname): $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

module load stack/2024-06
module load python/3.10.13
module load cuda/12.1.1
module load gcc/12.2.0
module load eth_proxy

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8

REPO_ROOT="/cluster/project/cvg/students/sergejsz/h-segsplat"
cd "$REPO_ROOT"
mkdir -p logs

source envs/hsegsplat/venv/bin/activate

GAUSSIANS="data/$SCENE_ARG/gaussians.pt"
OUT_DIR="data/$SCENE_ARG/midpoint_render"

if [[ ! -f "$GAUSSIANS" ]]; then
    echo "[render_midpoint] ERROR: $GAUSSIANS not found"
    exit 1
fi

mkdir -p "$OUT_DIR"

# The renderer computes the midpoint pose from gaussians.pt's extrinsics
# directly — no extra JSON required. (A midpoint_pose.json from a local
# sanity_render_midpoint_gt.py run can be passed with --midpoint_pose_json
# if you want to override the pose, but for the standard
# midpoint-of-the-two-input-views protocol, this is unnecessary.)
python scripts/evaluation_scripts/multiscan/render_midpoint_gsplat.py \
    --gaussians_pt "$GAUSSIANS" \
    --out_dir "$OUT_DIR" \
    --device cuda \
    --levels 1 3 6

deactivate

echo "Render midpoint for scene $SCENE_ARG done: $(date)"
ls -lh "$OUT_DIR/midpoint_rgb.png" "$OUT_DIR/midpoint_feature_map_lvl"*.npy 2>/dev/null || true
