#!/usr/bin/env python3
"""
Stage 1: SemanticSAM automatic mask generation at three granularity levels.

For each scene under <data_dir>/<scene>/dslr/resized_images/, runs the
SemanticSamAutomaticMaskGenerator once per level [1, 3, 6] and writes the
resulting masks + metadata into

    <data_dir>/<scene>/masks_lvl_<L>/<frame_stem>/
        mask_0.png, mask_1.png, ...
        metadata.json

This is the entry point of the H-SegSplat pipeline. The next stage
(`run_siglip.py`) reads these folders and produces per-mask SigLIP features.

Idempotent: skips frame-level dirs that already contain a metadata.json.

The script expects to find Semantic-SAM at <repo_root>/Semantic-SAM/, which is
where the monorepo vendors it. Run from the repo root or set --semantic_sam_dir.
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root containing one or more scene dirs. Each scene must have "
                        "dslr/resized_images/ inside.")
    p.add_argument("--scene", type=str, default=None,
                   help="Process a single scene (subdirectory of data_dir). "
                        "If omitted, processes all subdirs of data_dir.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to swinl_only_sam_many2many.pth (or another SemanticSAM checkpoint).")
    p.add_argument("--config", type=str, default=None,
                   help="Path to SemanticSAM config YAML. Defaults to "
                        "<semantic_sam_dir>/configs/semantic_sam_only_sa-1b_swinL.yaml.")
    p.add_argument("--semantic_sam_dir", type=str, default=None,
                   help="Path to vendored Semantic-SAM/ directory. "
                        "Defaults to <repo_root>/Semantic-SAM where <repo_root> is the parent of scripts/.")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 3, 6],
                   help="Granularity levels to extract (default 1 3 6).")
    p.add_argument("--target_w", type=int, default=960)
    p.add_argument("--target_h", type=int, default=640)
    p.add_argument("--points_per_side", type=int, default=32)
    p.add_argument("--pred_iou_thresh", type=float, default=0.88)
    p.add_argument("--stability_score_thresh", type=float, default=0.92)
    p.add_argument("--min_mask_region_area", type=int, default=10)
    return p.parse_args()


def resolve_semantic_sam_dir(arg_value: str | None) -> Path:
    if arg_value:
        d = Path(arg_value).resolve()
    else:
        # Default: <repo_root>/Semantic-SAM (this script lives at <repo_root>/scripts/).
        d = (Path(__file__).resolve().parent.parent / "Semantic-SAM").resolve()
    if not (d / "semantic_sam").is_dir():
        raise FileNotFoundError(
            f"SemanticSAM not found at {d}. Pass --semantic_sam_dir or vendor Semantic-SAM "
            f"at <repo_root>/Semantic-SAM/."
        )
    return d


def import_semantic_sam(sam_dir: Path):
    """Import Semantic-SAM from the vendored copy. Prefers an existing editable install
    if one is present (envs/sam/setup.sh runs `pip install -e .`), otherwise falls back
    to sys.path injection."""
    sys.path.insert(0, str(sam_dir))
    from semantic_sam.BaseModel import BaseModel
    from semantic_sam import build_model
    from utils.arguments import load_opt_from_config_file
    from tasks.automatic_mask_generator import SemanticSamAutomaticMaskGenerator
    return BaseModel, build_model, load_opt_from_config_file, SemanticSamAutomaticMaskGenerator


def load_model(sam_dir: Path, checkpoint: str, config: str):
    BaseModel, build_model, load_opt_from_config_file, _ = import_semantic_sam(sam_dir)
    opt = load_opt_from_config_file(config)
    model = BaseModel(opt, build_model(opt)).from_pretrained(checkpoint).eval().cuda()
    return model


def collect_scenes(data_dir: Path, single_scene: str | None) -> list[Path]:
    if single_scene:
        scene_path = data_dir / single_scene
        if not scene_path.is_dir():
            raise FileNotFoundError(f"--scene {single_scene} not found under {data_dir}")
        return [scene_path]
    return [d for d in sorted(data_dir.iterdir())
            if d.is_dir() and (d / "dslr" / "resized_images").is_dir()]


def iter_images(scene_dir: Path) -> list[Path]:
    img_dir = scene_dir / "dslr" / "resized_images"
    return sorted([p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")])


def process_scene_level(scene_dir: Path,
                        level: int,
                        model,
                        sam_dir: Path,
                        target_size: tuple,
                        gen_kwargs: dict):
    """Run the auto mask generator at one granularity level for one scene."""
    _, _, _, AutoGen = import_semantic_sam(sam_dir)

    gen = AutoGen(
        model,
        level=[level],   # single level per generator invocation
        **gen_kwargs,
    )
    level_root = scene_dir / f"masks_lvl_{level}"
    level_root.mkdir(parents=True, exist_ok=True)

    images = iter_images(scene_dir)
    if not images:
        print(f"  [sam lvl{level}] WARNING: no images found in {scene_dir}/dslr/resized_images")
        return
    print(f"  [sam lvl{level}] {len(images)} images to process")

    for img_path in tqdm(images, desc=f"  {scene_dir.name} lvl{level}", leave=True):
        stem = img_path.stem
        out_dir = level_root / stem
        meta_path = out_dir / "metadata.json"
        if meta_path.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        img_pil = Image.open(img_path).convert("RGB").resize(target_size, Image.BICUBIC)
        img_np = np.array(img_pil)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).cuda()
        outputs = gen.generate(img_tensor)

        metadata = []
        for i, ann in enumerate(outputs):
            mask = ann["segmentation"]
            cv2.imwrite(str(out_dir / f"mask_{i}.png"),
                        (mask * 255).astype(np.uint8))
            metadata.append({
                "id": i,
                "mask_file": f"mask_{i}.png",
                "bbox": list(ann["bbox"]),
                "area": int(ann["area"]),
                "predicted_iou": float(ann["predicted_iou"]),
                "stability_score": float(ann["stability_score"]),
                "level": int(level),
            })
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"    [{stem}] {len(outputs)} masks")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    sam_dir = resolve_semantic_sam_dir(args.semantic_sam_dir)
    config = args.config or str(sam_dir / "configs" / "semantic_sam_only_sa-1b_swinL.yaml")
    checkpoint = str(Path(args.checkpoint).resolve())

    if not torch.cuda.is_available():
        raise RuntimeError("SemanticSAM requires CUDA. No GPU available.")

    target_size = (args.target_w, args.target_h)
    gen_kwargs = dict(
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )

    print(f"[sam] data_dir         = {data_dir}")
    print(f"[sam] semantic_sam_dir = {sam_dir}")
    print(f"[sam] checkpoint       = {checkpoint}")
    print(f"[sam] config           = {config}")
    print(f"[sam] target_size      = {target_size}")
    print(f"[sam] levels           = {args.levels}")

    scenes = collect_scenes(data_dir, args.scene)
    if not scenes:
        raise RuntimeError(f"No scenes with dslr/resized_images/ found under {data_dir}")
    print(f"[sam] scenes to process: {[s.name for s in scenes]}")

    print("[sam] loading model...")
    model = load_model(sam_dir, checkpoint, config)

    for scene_dir in scenes:
        print(f"\n[sam] === scene {scene_dir.name} ===")
        for level in args.levels:
            process_scene_level(scene_dir, level, model, sam_dir, target_size, gen_kwargs)

    print("\n[sam] done.")


if __name__ == "__main__":
    main()
