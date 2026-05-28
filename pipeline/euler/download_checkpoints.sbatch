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

DEPTHSPLAT_URL="https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"
SAM_URL="https://huggingface.co/UX-Decoder/Semantic-SAM/resolve/main/swinl_only_sam_many2many.pth?download=true"

DEPTHSPLAT_CKPT="depthsplat/pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"
SAM_CKPT="swinl_only_sam_many2many.pth"

# Validate a checkpoint by size + magic bytes. A truncated or HTML-error-page
# download will fail one of these checks loudly instead of cascading into
# torch.load() with an inscrutable EOFError at pipeline runtime.
validate_ckpt() {
    local path="$1"
    local min_mb="$2"
    if [[ ! -f "$path" ]]; then
        echo "[ckpts] ERROR: $path missing after download"
        return 1
    fi
    local size_mb
    size_mb=$(du -m "$path" | cut -f1)
    if (( size_mb < min_mb )); then
        echo "[ckpts] ERROR: $path is ${size_mb} MB, expected at least ${min_mb} MB."
        echo "[ckpts]        First 200 bytes follow — if HTML, the download was redirected to an error page:"
        head -c 200 "$path" | xxd | head -10
        return 1
    fi
    # Torch checkpoints are zip archives ('PK\x03\x04') or legacy pickle
    # (b'\x80\x02' / b'\x80\x04'). Either way the file should NOT start with
    # '<' (HTML) or 'Not Found' etc.
    local magic
    magic=$(head -c 2 "$path" | od -An -tx1 | tr -d ' \n')
    if [[ "$magic" != "504b" && "$magic" != "8002" && "$magic" != "8003" && "$magic" != "8004" ]]; then
        echo "[ckpts] ERROR: $path has unexpected magic bytes 0x$magic (not a torch checkpoint)."
        head -c 200 "$path" | xxd | head -10
        return 1
    fi
    echo "[ckpts] $path validated: ${size_mb} MB, magic 0x$magic"
}

# Force a clean download if the existing file is suspicious.
ensure_ckpt() {
    local path="$1" url="$2" min_mb="$3"
    if [[ -f "$path" ]] && validate_ckpt "$path" "$min_mb"; then
        echo "[ckpts] $path already present and valid, skipping."
        return 0
    fi
    rm -f "$path"
    # -L: follow redirects (huggingface 302s to CDN)
    # --fail-with-body: error on HTTP >=400, but still surface response body for diagnosis
    # --retry 3: transient errors via eth_proxy aren't uncommon
    curl -L --fail --retry 3 --retry-delay 5 -o "$path" "$url"
    validate_ckpt "$path" "$min_mb"
}

ensure_ckpt "$DEPTHSPLAT_CKPT" "$DEPTHSPLAT_URL" 100
ensure_ckpt "$SAM_CKPT"        "$SAM_URL"        500

ls -lh "$DEPTHSPLAT_CKPT" "$SAM_CKPT"
echo "Download checkpoints done: $(date)"
