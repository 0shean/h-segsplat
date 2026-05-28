#!/bin/bash
#SBATCH --job-name=hseg_setup_siglip
#SBATCH --output=logs/setup_siglip_%j.log
#SBATCH --error=logs/setup_siglip_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=01:00:00

# Builds envs/siglip/venv. GPU not strictly required for install, but
# the verification step at the end of setup.sh runs a CUDA check.
# eth_proxy needed for pip.

set -ex
echo "Setup SigLIP venv: $(date)"

module load stack/2024-06
module load python/3.10.13
module load cuda/12.1.1
module load gcc/12.2.0
module load eth_proxy
module list

REPO_ROOT="/cluster/project/cvg/students/sergejsz/h-segsplat"
cd "$REPO_ROOT"
mkdir -p logs

bash envs/siglip/setup.sh

echo "Setup SigLIP done: $(date)"
