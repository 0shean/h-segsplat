#!/bin/bash
# pipeline/run_pipeline.sh
#
# End-to-end H-SegSplat pipeline driver.
#
# Usage:
#   bash pipeline/run_pipeline.sh data/<scene>
#
# Where <scene> is a subdirectory of data/ containing:
#   data/<scene>/dslr/nerfstudio/transforms.json
#   data/<scene>/dslr/resized_images/<frame>.JPG
#
# Stages (each is idempotent — skips work that's already done):
#   1. SAM masks       -> data/<scene>/masks_lvl_{1,3,6}/<frame>/...
#   2. SigLIP features -> data/<scene>/masks_lvl_{1,3,6}/<frame>/siglip_embeddings.npy
#   3. Build inputs    -> data/<scene>/level_{1,3,6}/*, datasets/, assets/, meta.json
#   4. Parent chain    -> data/<scene>/per_mask/parents.json
#   5. H-SegSplat run  -> data/<scene>/gaussians.pt + rendered_feature_map_lvl{1,3,6}.npy
#
# Stage 1 requires CUDA. Stage 2 requires CUDA. Stages 3 and 4 are CPU.
# Stage 5 requires CUDA and a DepthSplat checkpoint.

set -euo pipefail

# Colab exports MPLBACKEND=module://matplotlib_inline.backend_inline so notebook
# matplotlib renders inline. That module isn't present in our venvs, which causes
# matplotlib.__init__ to crash with ValueError. Unset it for the whole pipeline.
unset MPLBACKEND

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 data/<scene>"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_DIR_ARG="$1"
# Allow either a path relative to repo root or an absolute path.
if [[ "$SCENE_DIR_ARG" = /* ]]; then
    SCENE_DIR="$SCENE_DIR_ARG"
else
    SCENE_DIR="$REPO_ROOT/$SCENE_DIR_ARG"
fi
SCENE_NAME="$(basename "$SCENE_DIR")"

# Where to find the SAM checkpoint. Defaults to repo_root; can be overridden by env var.
SAM_CHECKPOINT="${SAM_CHECKPOINT:-$REPO_ROOT/swinl_only_sam_many2many.pth}"

# Where to find the DepthSplat checkpoint.
DEPTHSPLAT_CHECKPOINT="${DEPTHSPLAT_CHECKPOINT:-$REPO_ROOT/depthsplat/pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth}"

if [[ ! -d "$SCENE_DIR" ]]; then
    echo "[pipeline] ERROR: $SCENE_DIR does not exist"
    exit 1
fi
if [[ ! -f "$SCENE_DIR/dslr/nerfstudio/transforms.json" ]]; then
    echo "[pipeline] ERROR: $SCENE_DIR/dslr/nerfstudio/transforms.json missing"
    exit 1
fi

echo "[pipeline] repo root  : $REPO_ROOT"
echo "[pipeline] scene dir  : $SCENE_DIR"
echo "[pipeline] scene name : $SCENE_NAME"

# ------------------------------------------------------------
# Per-stage timing CSVs:
#   $SCENE_DIR/pipeline_timings.csv      (per scene, fresh each run)
#   $REPO_ROOT/data/pipeline_timings.csv (aggregate, appended across runs)
# ------------------------------------------------------------
SCENE_TIMINGS="$SCENE_DIR/pipeline_timings.csv"
AGG_TIMINGS="$REPO_ROOT/data/pipeline_timings.csv"
CSV_HEADER="scene,stage,start_iso,end_iso,duration_seconds,status"
echo "$CSV_HEADER" > "$SCENE_TIMINGS"
if [[ ! -f "$AGG_TIMINGS" ]]; then
    echo "$CSV_HEADER" > "$AGG_TIMINGS"
fi

# Run a stage and record its wallclock time + exit status to the CSVs.
# Usage: run_stage <stage_label> <command...>
run_stage() {
    local label="$1"; shift
    local start_iso start_ns end_iso end_ns dur status
    start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    start_ns=$(date +%s%N)
    set +e
    "$@"
    status=$?
    set -e
    end_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    end_ns=$(date +%s%N)
    # bash arithmetic on big ints
    dur=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN{printf "%.3f", (e - s) / 1e9}')
    local result="ok"
    [[ $status -ne 0 ]] && result="failed"
    local row="${SCENE_NAME},${label},${start_iso},${end_iso},${dur},${result}"
    echo "$row" >> "$SCENE_TIMINGS"
    echo "$row" >> "$AGG_TIMINGS"
    echo "[pipeline] [$label] $result in ${dur}s"
    if [[ $status -ne 0 ]]; then
        echo "[pipeline] ABORT: stage '$label' failed (exit $status). See $SCENE_TIMINGS"
        exit $status
    fi
}

# Mark overall start time so we can also log a TOTAL row.
PIPELINE_START_NS=$(date +%s%N)
PIPELINE_START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ------------------------------------------------------------
# Stage 1: SAM
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "[pipeline] Stage 1: SemanticSAM"
echo "============================================================"
if [[ ! -f "$SAM_CHECKPOINT" ]]; then
    echo "[pipeline] ERROR: SAM checkpoint not found at $SAM_CHECKPOINT"
    echo "[pipeline]        Download swinl_only_sam_many2many.pth and place it there,"
    echo "[pipeline]        or set SAM_CHECKPOINT=<path>."
    exit 1
fi
run_stage stage1_sam \
    bash "$REPO_ROOT/pipeline/stage_01_masks.sh" \
    "$SCENE_DIR" "$SCENE_NAME" "$SAM_CHECKPOINT" "$REPO_ROOT"

# ------------------------------------------------------------
# Stage 2: SigLIP
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "[pipeline] Stage 2: SigLIP features"
echo "============================================================"
run_stage stage2_siglip \
    bash "$REPO_ROOT/pipeline/stage_02_features.sh" \
    "$SCENE_DIR" "$SCENE_NAME" "$REPO_ROOT"

# ------------------------------------------------------------
# Stage 3: Build H-SegSplat inputs
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "[pipeline] Stage 3: build H-SegSplat inputs (banks, index_maps, ...)"
echo "============================================================"
run_stage stage3_build \
    bash "$REPO_ROOT/pipeline/stage_03_build.sh" \
    "$SCENE_DIR" "$SCENE_NAME" "$REPO_ROOT"

# ------------------------------------------------------------
# Stage 4: Parent chain (containment dict)
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "[pipeline] Stage 4: compute parent chain"
echo "============================================================"
run_stage stage4_parents \
    bash "$REPO_ROOT/pipeline/stage_04_parents.sh" \
    "$SCENE_DIR" "$SCENE_NAME" "$REPO_ROOT"

# ------------------------------------------------------------
# Stage 5: H-SegSplat inference (DepthSplat + gsplat)
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "[pipeline] Stage 5: H-SegSplat inference"
echo "============================================================"
if [[ ! -f "$DEPTHSPLAT_CHECKPOINT" ]]; then
    echo "[pipeline] ERROR: DepthSplat checkpoint not found at $DEPTHSPLAT_CHECKPOINT"
    echo "[pipeline]        Place it there, or set DEPTHSPLAT_CHECKPOINT=<path>."
    exit 1
fi
run_stage stage5_hsegsplat \
    bash "$REPO_ROOT/pipeline/stage_05_hsegsplat.sh" \
    "$SCENE_DIR" "$SCENE_NAME" "$DEPTHSPLAT_CHECKPOINT" "$REPO_ROOT"

# Total row
PIPELINE_END_NS=$(date +%s%N)
PIPELINE_END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOTAL_DUR=$(awk -v s="$PIPELINE_START_NS" -v e="$PIPELINE_END_NS" 'BEGIN{printf "%.3f", (e - s) / 1e9}')
TOTAL_ROW="${SCENE_NAME},total,${PIPELINE_START_ISO},${PIPELINE_END_ISO},${TOTAL_DUR},ok"
echo "$TOTAL_ROW" >> "$SCENE_TIMINGS"
echo "$TOTAL_ROW" >> "$AGG_TIMINGS"

echo ""
echo "============================================================"
echo "[pipeline] All stages complete."
echo "[pipeline] Final artifact: $SCENE_DIR/gaussians.pt"
echo "[pipeline] Timings (this scene):"
column -t -s, "$SCENE_TIMINGS" 2>/dev/null || cat "$SCENE_TIMINGS"
echo ""
echo "[pipeline] Aggregate timings across all scenes this session:"
echo "[pipeline]   $AGG_TIMINGS"
echo "============================================================"
