#!/bin/bash
#SBATCH --output=logs/run_%x_%j.log
#SBATCH --error=logs/run_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=04:00:00

# Per-scene runner. Submit with:
#   sbatch --job-name=hseg_<scene> pipeline/euler/run_scene.sbatch data/<scene>
#
# Reads the scene-dir as positional arg $1, then defers to the existing
# pipeline/run_pipeline.sh which writes per-stage timings to:
#   <scene>/pipeline_timings.csv
#   data/pipeline_timings.csv  (aggregate across all jobs)

set -e
SCENE_ARG="$1"
if [[ -z "$SCENE_ARG" ]]; then
    echo "Usage: sbatch pipeline/euler/run_scene.sbatch data/<scene>"
    exit 1
fi

echo "Run scene $SCENE_ARG on node $(hostname): $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

module load stack/2024-06
module load python/3.10.13
module load cuda/12.1.1
module load gcc/12.2.0
module load eth_proxy

# Euler login + compute nodes default to C/POSIX locale (ASCII). torch's
# JIT cpp_extension versioner calls open(...).read() with the system
# encoding when hashing CUDA source files; if any file (e.g. gsplat's
# csrc/*.cu / *.h) contains a non-ASCII byte in a comment, it crashes
# with UnicodeDecodeError. Force a UTF-8 locale before anything imports
# torch.
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8

REPO_ROOT="/cluster/project/cvg/students/sergejsz/h-segsplat"
cd "$REPO_ROOT"
mkdir -p logs

bash pipeline/run_pipeline.sh "$SCENE_ARG"

echo "Run scene $SCENE_ARG done: $(date)"
