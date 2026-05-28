#!/bin/bash
#SBATCH --job-name=hseg_setup_sam
#SBATCH --output=logs/setup_sam_%j.log
#SBATCH --error=logs/setup_sam_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=02:00:00

# Builds envs/sam/venv. Needs a GPU because the Semantic-SAM
# MultiScaleDeformableAttention CUDA op compiles against the active CUDA.
# Needs eth_proxy for pip/git to reach pypi + github.

set -ex
echo "Setup SAM venv: $(date)"

module load stack/2024-06
module load python/3.10.13
module load cuda/12.1.1
module load gcc/12.2.0
module load eth_proxy
module list

REPO_ROOT="/cluster/project/cvg/students/sergejsz/h-segsplat"
cd "$REPO_ROOT"
mkdir -p logs

bash envs/sam/setup.sh

echo "Setup SAM done: $(date)"
