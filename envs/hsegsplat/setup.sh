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

# Build-time prereqs the fresh venv lacks but Colab's base Python has by default.
# diff-gaussian-rasterization-modified's setup.py uses setuptools + torch's
# CUDA extension builder which calls ninja. Both must be in the venv.
pip install "setuptools<70" ninja

# DepthSplat's pinned torch + CUDA 12.4.
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Apply the numpy patch that the Colab cell did (loosen the exact pin).
sed -i.bak 's/numpy==1.24.4/numpy>=1.24.4,<2.0/' "$REPO_ROOT/depthsplat/requirements.txt" || true

# DepthSplat's requirements include diff-gaussian-rasterization-modified, which is a
# CUDA extension built from source. It needs CUDA_HOME set. Mirrors the SAM setup.
if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -d /usr/local/cuda ]]; then
        export CUDA_HOME=/usr/local/cuda
    elif command -v nvcc &> /dev/null; then
        export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    else
        echo "[envs/hsegsplat] ERROR: no CUDA toolkit found (CUDA_HOME unset, nvcc not in PATH)."
        exit 1
    fi
    echo "[envs/hsegsplat] auto-set CUDA_HOME=$CUDA_HOME"
fi
# --no-build-isolation: use the venv's torch+ninja for setup.py invocations
# (the diff-gaussian-rasterization-modified source build needs them).
# -v: surface the real error if one occurs, instead of "No available output".
pip install --no-build-isolation -v -r "$REPO_ROOT/depthsplat/requirements.txt"

# H-SegSplat's own deps on top of DepthSplat: gsplat (rasterizer) + imageio (PNG writing).
# matplotlib_inline: not used by us, but Colab exports MPLBACKEND=module://matplotlib_inline...
# which causes matplotlib to crash if the backend module isn't importable. Installing it
# makes the venv robust to that env-var leak.
pip install gsplat imageio matplotlib_inline

# Add the vendored depthsplat dir to a .pth file so `from src.X import Y` works
# inside our scripts without us having to cd into depthsplat/.
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$REPO_ROOT/depthsplat" > "$SITE_PACKAGES/depthsplat.pth"

echo "[envs/hsegsplat] verification:"
# Unset Colab's MPLBACKEND for the duration of the verification — it points at a module
# that isn't installed by default. matplotlib_inline (installed above) handles the
# pipeline case; this just keeps the smoke test clean.
unset MPLBACKEND
python -c "import torch; print(f'  Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import gsplat; print(f'  gsplat OK')"
python -c "import sys; sys.path.insert(0, '$REPO_ROOT/depthsplat'); import src.model.encoder.encoder_depthsplat as _; print('  depthsplat import OK')"

touch "$VENV/.setup_complete"
echo "[envs/hsegsplat] done. activate with: source $VENV/bin/activate"
