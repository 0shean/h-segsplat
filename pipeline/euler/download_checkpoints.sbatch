#!/bin/bash
#SBATCH --job-name=hseg_ckpts
#SBATCH --output=logs/ckpts_%j.log
#SBATCH --error=logs/ckpts_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=00:30:00

# Downloads the two model checkpoints into the repo root.
# Needs GPU because Euler only routes internet (via eth_proxy) from GPU
# nodes per the project setup.

set -ex
echo "Download checkpoints: $(date)"

module load stack/2024-06
module load eth_proxy

REPO_ROOT="/cluster/project/cvg/students/sergejsz/h-segsplat"
cd "$REPO_ROOT"
mkdir -p logs depthsplat/pretrained

DEPTHSPLAT_CKPT="depthsplat/pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"
SAM_CKPT="swinl_only_sam_many2many.pth"

if [[ ! -f "$DEPTHSPLAT_CKPT" ]]; then
    wget -O "$DEPTHSPLAT_CKPT" \
        https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth
else
    echo "DepthSplat ckpt already present, skipping."
fi

if [[ ! -f "$SAM_CKPT" ]]; then
    wget -O "$SAM_CKPT" \
        https://huggingface.co/UX-Decoder/Semantic-SAM/resolve/main/swinl_only_sam_many2many.pth
else
    echo "SAM ckpt already present, skipping."
fi

ls -lh "$DEPTHSPLAT_CKPT" "$SAM_CKPT"
echo "Download checkpoints done: $(date)"
