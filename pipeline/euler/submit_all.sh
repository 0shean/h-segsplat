#!/bin/bash
# Submit the full pipeline on Euler: 3 env setups + ckpt download, then
# all 20 scenes once those four prerequisites finish successfully.
#
# Usage (from the cluster, inside the repo root):
#   bash pipeline/euler/submit_all.sh
#
# Optional first arg = "skip_setup" to skip the four setup/ckpt jobs and
# only submit the 20 scene jobs (use this for re-runs):
#   bash pipeline/euler/submit_all.sh skip_setup

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

SCENES=(
    3dovs_bed
    3dovs_bench
    3dovs_blue_sofa
    3dovs_covered_desk
    3dovs_lawn
    3dovs_office_desk
    3dovs_room
    3dovs_snacks
    3dovs_sofa
    3dovs_table
    scene_00005_00
    scene_00006_00
    scene_00018_00
    scene_00019_00
    scene_00022_00
    scene_00030_00
    scene_00044_00
    scene_00048_00
    scene_00051_00
    scene_00055_00
)

DEPS=""
if [[ "${1:-}" != "skip_setup" ]]; then
    echo "=== Submitting setup jobs ==="
    JID_SAM=$(sbatch --parsable pipeline/euler/setup_sam.sbatch)
    echo "  setup_sam      -> $JID_SAM"
    JID_SIGLIP=$(sbatch --parsable pipeline/euler/setup_siglip.sbatch)
    echo "  setup_siglip   -> $JID_SIGLIP"
    JID_HSEG=$(sbatch --parsable pipeline/euler/setup_hsegsplat.sbatch)
    echo "  setup_hsegsplat-> $JID_HSEG"
    JID_CKPT=$(sbatch --parsable pipeline/euler/download_checkpoints.sbatch)
    echo "  ckpt download  -> $JID_CKPT"
    DEPS="--dependency=afterok:${JID_SAM}:${JID_SIGLIP}:${JID_HSEG}:${JID_CKPT}"
    echo "Scene jobs will start only after these four succeed."
else
    echo "=== Skipping setup; submitting scenes immediately ==="
fi

echo ""
echo "=== Submitting ${#SCENES[@]} scene jobs ==="
for s in "${SCENES[@]}"; do
    if [[ ! -d "data/$s" ]]; then
        echo "  WARN: data/$s does not exist, skipping"
        continue
    fi
    JID=$(sbatch --parsable $DEPS \
        --job-name="hseg_$s" \
        pipeline/euler/run_scene.sbatch "data/$s")
    echo "  $s -> $JID"
done

echo ""
echo "=== All jobs submitted. Monitor with: ==="
echo "  squeue -u \$USER"
echo ""
echo "Per-scene timings will accumulate in:"
echo "  data/pipeline_timings.csv"
