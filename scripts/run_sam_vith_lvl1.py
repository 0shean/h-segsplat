#!/usr/bin/env python3
"""
Stage 1a: Meta's original SAM (ViT-H) automatic mask generation, replacing
Semantic-SAM's granularity-1 masks.

This script is a drop-in for the level-1 part of run_semantic_sam.py. The
output layout is identical:

    <data_dir>/<scene>/masks_lvl_1/<frame_stem>/
        mask_0.png, mask_1.png, ...
        metadata.json   (id, mask_file, bbox, area, predicted_iou, stability_score, level)

It is intended to be run BEFORE run_semantic_sam.py, with the latter then
restricted to --levels 3 6 so its own lvl-1 pass is skipped:

    bash pipeline/stage_01a_sam_vith.sh    <SCENE_DIR> <SCENE_NAME> <SAM_VITH_CKPT> <REPO_ROOT>
    bash pipeline/stage_01_masks.sh        <SCENE_DIR> <SCENE_NAME> <SEM_SAM_CKPT>  <REPO_ROOT>  # --levels 3 6

The downstream stages (SigLIP, build, etc.) consume masks_lvl_1/ exactly as
they do today, so no changes needed past this script + the orchestrator.

Why this script exists:
    SegSplat's §3.1 step 1 specifies SAM ViT-H + NMS at "whole object" granularity.
    Semantic-SAM at granularity 1 is a different model and produces, on some
    scenes, multiple overlapping "object + context" masks (e.g. Pikachu plus
    surrounding sofa) that get classified by SigLIP as the salient object and
    cause spillover at render time. This script replaces lvl-1 with the SAM
    that the paper actually used. Levels 3 and 6 keep using Semantic-SAM since
    that model is uniquely suited to multi-granularity prompting.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# Import nms_masks from the Semantic-SAM script (same module-level function).
# We use the SAME thresholds via the same global slot so the two paths agree.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from run_semantic_sam import nms_masks  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root containing one or more scene dirs. Each scene must have "
                        "dslr/resized_images/ inside.")
    p.add_argument("--scene", type=str, default=None,
                   help="Process a single scene (subdirectory of data_dir). "
                        "If omitted, processes all subdirs of data_dir.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to sam_vit_h_4b8939.pth.")
    p.add_argument("--model_type", type=str, default="vit_h",
                   choices=["vit_h", "vit_l", "vit_b"],
                   help="SAM backbone size. vit_h matches SegSplat's setup.")
    p.add_argument("--target_w", type=int,
                   default=int(os.environ.get("HSEGSPLAT_TARGET_W", "960")))
    p.add_argument("--target_h", type=int,
                   default=int(os.environ.get("HSEGSPLAT_TARGET_H", "640")))
    # Defaults below are tuned for "whole object" granularity, matching SegSplat's
    # §3.1 step 1 framing. SAM's stock defaults (32/0.88/0.95/10) produce a
    # mask at every visible structure, including small parts — too fine for our
    # lvl-1 purpose. Empirically the stock settings on the 3D-OVS sofa scene
    # yielded ~38 masks (more than Semantic-SAM at lvl 6), one for each small
    # detail (a single button on the Xbox controller, etc.). The values below
    # bias toward a sparser prompt grid and drop very small masks outright.
    p.add_argument("--points_per_side", type=int, default=16,
                   help="SAM samples points_per_side^2 prompts uniformly. 16 -> "
                        "256 prompts; 32 (SAM default) -> 1024. Lower = coarser.")
    p.add_argument("--pred_iou_thresh", type=float, default=0.88,
                   help="SAM's confidence filter (mask quality).")
    p.add_argument("--stability_score_thresh", type=float, default=0.95,
                   help="SAM's stability filter.")
    p.add_argument("--min_mask_region_area", type=int, default=6000,
                   help="Drop masks below this many pixels (default ~1%% of "
                        "960x640 = 6144). Kills small-part masks like a single "
                        "button. Set to 0 for SAM's default of 10.")
    p.add_argument("--nms_iou_thresh", type=float, default=0.5,
                   help="Same as run_semantic_sam.py: drop the lower-confidence "
                        "mask of any pair with IoU >= this.")
    p.add_argument("--nms_iom_thresh", type=float, default=0.85,
                   help="Same as run_semantic_sam.py: drop the lower-confidence "
                        "mask of any pair with Intersection-over-Min >= this.")
    return p.parse_args()


def import_sam():
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError as e:
        print("[sam-vith] segment_anything is not installed in the active venv.\n"
              "           Install with: pip install segment-anything", file=sys.stderr)
        raise
    return sam_model_registry, SamAutomaticMaskGenerator


def collect_scenes(data_dir: Path, single_scene: str | None) -> list[Path]:
    if single_scene:
        return [data_dir / single_scene]
    return [d for d in sorted(data_dir.iterdir())
            if d.is_dir() and (d / "dslr" / "resized_images").is_dir()]


def iter_images(scene_dir: Path) -> list[Path]:
    img_dir = scene_dir / "dslr" / "resized_images"
    return sorted([p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")])


def process_scene(scene_dir: Path, generator, target_size: tuple) -> None:
    level_root = scene_dir / "masks_lvl_1"
    level_root.mkdir(parents=True, exist_ok=True)

    images = iter_images(scene_dir)
    if not images:
        print(f"  [sam-vith] WARNING: no images found in {scene_dir}/dslr/resized_images")
        return
    print(f"  [sam-vith] {len(images)} images to process")

    for img_path in tqdm(images, desc=f"  {scene_dir.name} lvl1(SAM)", leave=True):
        stem = img_path.stem
        out_dir = level_root / stem
        meta_path = out_dir / "metadata.json"
        if meta_path.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        # Resize to the standard pipeline resolution. SAM accepts uint8 HxWxC RGB.
        img_pil = Image.open(img_path).convert("RGB").resize(target_size, Image.BICUBIC)
        img_np = np.array(img_pil)
        outputs_raw = generator.generate(img_np)
        n_raw = len(outputs_raw)
        # Same NMS routine + thresholds as Semantic-SAM lvl-3 / 6.
        outputs = nms_masks(outputs_raw)
        n_kept = len(outputs)
        if n_kept != n_raw:
            print(f"    [{stem}] NMS lvl1: kept {n_kept}/{n_raw} masks "
                  f"(dropped {n_raw - n_kept} overlapping)")

        metadata = []
        for i, ann in enumerate(outputs):
            mask = ann["segmentation"]
            cv2.imwrite(str(out_dir / f"mask_{i}.png"),
                        (mask * 255).astype(np.uint8))
            metadata.append({
                "id": i,
                "mask_file": f"mask_{i}.png",
                "bbox": list(ann["bbox"]),  # SAM uses XYWH already
                "area": int(ann["area"]),
                "predicted_iou": float(ann["predicted_iou"]),
                "stability_score": float(ann["stability_score"]),
                "level": 1,
                "source": "sam_vit_h",
            })
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"    [{stem}] {len(outputs)} masks (lvl 1, SAM ViT-H)")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    checkpoint = str(Path(args.checkpoint).resolve())

    if not torch.cuda.is_available():
        raise RuntimeError("SAM ViT-H requires CUDA. No GPU available.")
    target_size = (args.target_w, args.target_h)

    print(f"[sam-vith] data_dir         = {data_dir}")
    print(f"[sam-vith] checkpoint       = {checkpoint}")
    print(f"[sam-vith] target_size      = {target_size}")
    print(f"[sam-vith] model_type       = {args.model_type}")
    print(f"[sam-vith] NMS              = IoU>={args.nms_iou_thresh} OR IoM>={args.nms_iom_thresh}")

    # Same trick as run_semantic_sam.py: stash NMS thresholds on the helper.
    nms_masks._iou_thresh = args.nms_iou_thresh
    nms_masks._iom_thresh = args.nms_iom_thresh

    scenes = collect_scenes(data_dir, args.scene)
    if not scenes:
        raise RuntimeError(f"No scenes with dslr/resized_images/ found under {data_dir}")
    print(f"[sam-vith] scenes to process: {[s.name for s in scenes]}")

    print("[sam-vith] loading SAM ViT-H ...")
    sam_model_registry, SamAutomaticMaskGenerator = import_sam()
    sam = sam_model_registry[args.model_type](checkpoint=checkpoint).cuda()
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )

    for scene_dir in scenes:
        print(f"\n[sam-vith] === scene {scene_dir.name} ===")
        process_scene(scene_dir, generator, target_size)

    print("\n[sam-vith] done.")


if __name__ == "__main__":
    main()
