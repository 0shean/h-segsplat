#!/bin/bash
# envs/siglip/setup.sh
#
# Creates the SigLIP venv. Works on both ETH Euler and Colab / generic Linux.
# Mirrors scripts_from_euler/setup_siglip_venv.sh.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO_ROOT/envs/siglip/venv"

echo "[envs/siglip] repo root: $REPO_ROOT"
echo "[envs/siglip] venv path: $VENV"

if command -v module &> /dev/null; then
    echo "[envs/siglip] loading Euler modules"
    module load stack/2024-06
    module load python/3.10.13
    module load cuda/12.1.1
    module load gcc/12.2.0
    module load eth_proxy
fi

PYTHON="${PYTHON:-python3.10}"
if ! command -v "$PYTHON" &> /dev/null; then
    PYTHON=python3
fi
echo "[envs/siglip] python: $($PYTHON --version) at $(which $PYTHON)"

rm -rf "$VENV"
"$PYTHON" -m venv --clear "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip

# Single pip install so the resolver respects numpy<2 across all packages.
# opencv-python pinned <4.10 to dodge its numpy 2 hard requirement.
pip install \
    "numpy<2" \
    torch==2.1.2 torchvision==0.16.2 \
    open-clip-torch \
    transformers sentencepiece \
    tqdm "opencv-python<4.10" pillow \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "[envs/siglip] verification:"
"$PYTHON" -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
"$PYTHON" -c "import torch; print(f'  Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
"$PYTHON" -c "import open_clip; print('  open_clip OK')"

echo "[envs/siglip] done. activate with: source $VENV/bin/activate"
