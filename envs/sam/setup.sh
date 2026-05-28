#!/bin/bash
# envs/sam/setup.sh
#
# Creates the SemanticSAM venv. Works on both ETH Euler (with `module load`)
# and Colab / generic Linux (without modules).
#
# Mirrors scripts_from_euler/setup_semanticsam_venv.sh from the original Euler
# layout. Key invariants:
#   - Python 3.10
#   - torch 2.1.2 + torchvision 0.16.2 + CUDA 12.1
#   - numpy<2  (Semantic-SAM's deps are pre-NumPy-2)
#   - detectron2 built from source (no prebuilt wheels for this combo)
#   - tokenizers==0.13.3, transformers==4.24.0
#   - SemanticSAM's MultiScaleDeformableAttention CUDA op compiled in-place
#
# Run from anywhere; venv lives at <repo_root>/envs/sam/venv/.

set -ex

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO_ROOT/envs/sam/venv"

echo "[envs/sam] repo root: $REPO_ROOT"
echo "[envs/sam] venv path: $VENV"

# Optional: load Euler modules if `module` exists (i.e., we're on the cluster).
if command -v module &> /dev/null; then
    echo "[envs/sam] module command detected — loading Euler modules"
    module load stack/2024-06
    module load python/3.10.13
    module load cuda/12.1.1
    module load gcc/12.2.0
    module load eth_proxy
fi

# Pick the right python. On Colab the default python3 is 3.10 or 3.11; on Euler
# the loaded module sets python -> python3.10. On macOS/local we expect the user
# to point at a 3.10 explicitly via PYTHON env var.
PYTHON="${PYTHON:-python3.10}"
if ! command -v "$PYTHON" &> /dev/null; then
    PYTHON=python3
fi
echo "[envs/sam] python: $($PYTHON --version) at $(which $PYTHON)"

# Always start from a clean venv to avoid stale CUDA op artifacts.
rm -rf "$VENV"
"$PYTHON" -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip wheel
pip install "setuptools<70"
pip install "numpy<2"

# PyTorch first (everything else depends on it).
pip install --force-reinstall \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Detectron2 (no-build-isolation so it picks up the just-installed torch).
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# SemanticSAM's pinned transformers stack.
pip install tokenizers==0.13.3 transformers==4.24.0 'shapely>=2.0.1'

# Install the vendored Semantic-SAM in editable mode.
cd "$REPO_ROOT/Semantic-SAM"
pip install -e .
pip install tyro==0.9.35

# Compile the MultiScaleDeformableAttention CUDA op in-place.
# SemanticSAM's setup.py requires CUDA_HOME — set it if the env doesn't have one.
# Euler's `module load cuda/...` already exports it; Colab does not.
if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -d /usr/local/cuda ]]; then
        export CUDA_HOME=/usr/local/cuda
    elif command -v nvcc &> /dev/null; then
        # Derive from nvcc location: /<prefix>/bin/nvcc -> /<prefix>
        export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    else
        echo "[envs/sam] ERROR: no CUDA toolkit found. CUDA_HOME unset and nvcc not in PATH."
        echo "[envs/sam]        Install CUDA 12.x and re-run, or set CUDA_HOME=<path> manually."
        exit 1
    fi
    echo "[envs/sam] auto-set CUDA_HOME=$CUDA_HOME"
fi
cd "$REPO_ROOT/Semantic-SAM/semantic_sam/body/encoder/ops"
"$PYTHON" setup.py build install

# Lightweight runtime deps used by our scripts.
pip install opencv-python tqdm pillow

# Meta's original SAM (NOT SAM 2). Co-installed in this same venv because
# scripts/run_sam_vith_lvl1.py drops in as a replacement for Semantic-SAM at
# granularity 1 (matching SegSplat's protocol). The two models coexist:
# segment_anything has no torch version pin beyond >=1.7, so our torch 2.1.2
# is fine. The vit-h checkpoint must be downloaded separately and pointed at
# via $SAM_VITH_CHECKPOINT (or the default $REPO_ROOT/sam_vit_h_4b8939.pth).
pip install 'git+https://github.com/facebookresearch/segment-anything.git'

# Sentinel file written only on success — the notebook checks for this to decide
# whether to skip a re-run. Without it, a half-finished venv directory looks
# "done" when it isn't.
touch "$VENV/.setup_complete"

echo "[envs/sam] done. activate with: source $VENV/bin/activate"
