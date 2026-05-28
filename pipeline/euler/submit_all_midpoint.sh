#!/bin/bash
# Submit the midpoint render job for all 10 MultiScan scenes.
#
# Prerequisite (on the cluster):
#   - data/<scene>/gaussians.pt exists from the main pipeline run.
#
# The renderer computes the midpoint pose from gaussians.pt's extrinsics
# directly, so no extra inputs need to be uploaded.
#
# Usage:
#   bash pipeline/euler/submit_all_midpoint.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

SCENES=(
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

echo "=== Submitting midpoint render jobs for ${#SCENES[@]} scenes ==="
for s in "${SCENES[@]}"; do
    if [[ ! -f "data/$s/gaussians.pt" ]]; then
        echo "  [skip] data/$s/gaussians.pt missing"
        continue
    fi
    JID=$(sbatch --parsable \
        --job-name="midrender_$s" \
        pipeline/euler/render_midpoint.sbatch "$s")
    echo "  $s -> $JID"
done

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Once done, the artifacts live in data/<scene>/midpoint_render/:"
echo "  midpoint_rgb.png"
echo "  midpoint_rgb.npy"
echo "  midpoint_feature_map_lvl{1,3,6}.npy"
echo ""
echo "Pull them to laptop with:"
echo "  rsync -avh --include='*/' --include='midpoint_render/**' --exclude='*' \\"
echo "    sergejsz@euler.ethz.ch:/cluster/project/cvg/students/sergejsz/h-segsplat/data/ \\"
echo "    /Users/sergeyzhahovskiy/Desktop/h-segsplat-repo/results/euler_midpoint/"
