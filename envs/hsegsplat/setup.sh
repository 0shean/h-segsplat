#!/bin/bash
# envs/hsegsplat/setup.sh
#
# Creates the H-SegSplat venv. Works on both ETH Euler and Colab / generic Linux.
#
# Mirrors the Colab cells that previously did this manually:
#   pip install torch==2.4.0 torchvision==0.19.0 --index-url cu124
#   pip install -r depthsplat/requirements.txt
#   pip install gsplat imageio
# plus a numpy pin patch (depthsplat pins exactly 1.24.4 which conflicts with newer
# minor versions on the cu124 wheels).

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO_ROOT/envs/hsegsplat/venv"

echo "[envs/hsegsplat] repo root: $REPO_ROOT"
echo "[envs/hsegsplat] venv path: $VENV"

if command -v module &> /dev/null; then
    echo "[envs/hsegsplat] loading Euler modules"
    module load stack/2024-06
    module load python/3.10.13
    # DepthSplat uses CUDA 12.4; cu124 wheels exist for both 12.4 and 12.1 module loads
    # in practice (PyTorch wheels are forward-compatible with newer driver). On Euler we
    # use the 12.1 module since it's the closest stable; PyTorch's cu124 wheels still load.
    module load cuda/12.1.1
    module load gcc/12.2.0
    module load eth_proxy
fi

PYTHON="${PYTHON:-python3.10}"
if ! command -v "$PYTHON" &> /dev/null; then
    PYTHON=python3
fi
echo "[envs/hsegsplat] python: $($PYTHON --version) at $(which $PYTHON)"

rm -rf "$VENV"
"$PYTHON" -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip wheel

# DepthSplat's pinned torch + CUDA 12.4.
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Apply the numpy patch that the Colab cell did (loosen the exact pin).
sed -i.bak 's/numpy==1.24.4/numpy>=1.24.4,<2.0/' "$REPO_ROOT/depthsplat/requirements.txt" || true

# DepthSplat's requirements. The git URL for diff-gaussian-rasterization-modified
# needs CUDA at build time — keep that in mind if it fails on Colab.
pip install -r "$REPO_ROOT/depthsplat/requirements.txt"

# H-SegSplat's own deps on top of DepthSplat: gsplat (rasterizer) + imageio (PNG writing).
pip install gsplat imageio

# Add the vendored depthsplat dir to a .pth file so `from src.X import Y` works
# inside our scripts without us having to cd into depthsplat/.
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$REPO_ROOT/depthsplat" > "$SITE_PACKAGES/depthsplat.pth"

echo "[envs/hsegsplat] verification:"
python -c "import torch; print(f'  Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import gsplat; print(f'  gsplat OK')"
python -c "import sys; sys.path.insert(0, '$REPO_ROOT/depthsplat'); import src.model.encoder.encoder_depthsplat as _; print('  depthsplat import OK')"

echo "[envs/hsegsplat] done. activate with: source $VENV/bin/activate"
