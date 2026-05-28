#!/bin/bash
#SBATCH --job-name=hseg_setup_hsegsplat
#SBATCH --output=logs/setup_hsegsplat_%j.log
#SBATCH --error=logs/setup_hsegsplat_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=02:00:00

# Builds envs/hsegsplat/venv. DepthSplat is pinned to torch 2.4.0 + cu124.
# gsplat compiles a CUDA op on first import; this script verifies that
# import at the end so we want a GPU + matching CUDA module visible.
# eth_proxy needed for pip + git.

set -ex
echo "Setup H-SegSplat venv: $(date)"

module load stack/2024-06
module load python/3.10.13
module load cuda/12.1.1
module load gcc/12.2.0
module load eth_proxy
module list

REPO_ROOT="/cluster/project/cvg/students/sergejsz/h-segsplat"
cd "$REPO_ROOT"
mkdir -p logs

bash envs/hsegsplat/setup.sh

echo "Setup H-SegSplat done: $(date)"
