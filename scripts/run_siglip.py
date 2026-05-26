#!/usr/bin/env python3
"""
Stage 2: per-mask SigLIP feature extraction.

For each scene under <data_dir>/<scene>/ and each level in [1, 3, 6], reads the
masks produced by stage 1 (run_semantic_sam.py) and writes one SigLIP feature
per mask:

    <data_dir>/<scene>/masks_lvl_<L>/<frame_stem>/siglip_embeddings.npy

The embeddings are the per-mask average of two crops (mask-only + bbox-with-bg)
as in the Euler `run_siglip2.py`. PROJECT_PLAN.md §6.6 lists Search3D-style
bbox expansion as a future change to this script.

Idempotent: skips frame dirs that already have siglip_embeddings.npy.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
except ImportError:
    print("open_clip is not installed. Activate envs/siglip/ first.", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root containing one or more scene dirs (output of run_semantic_sam.py).")
    p.add_argument("--scene", type=str, default=None,
                   help="Single scene to process. If omitted, all subdirs of data_dir are processed.")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 3, 6])
    p.add_argument("--target_w", type=int,
                   default=int(os.environ.get("HSEGSPLAT_TARGET_W", "960")))
    p.add_argument("--target_h", type=int,
                   default=int(os.environ.get("HSEGSPLAT_TARGET_H", "640")))
    p.add_argument("--model", type=str, default="ViT-SO400M-14-SigLIP")
    p.add_argument("--pretrained", type=str, default="webli")
    return p.parse_args()


class MaskCropEncoder:
    """Per-mask SigLIP encoding. Two strategies depending on the granularity level:

    * **Level 1 (whole objects):** original two-crop average — (a) mask-only with
      non-mask pixels zeroed at the tight bbox, (b) tight bbox with background kept.
      Whole-object masks already include plenty of context, so no expansion is needed.

    * **Levels 3, 6 (sub-parts / fine parts):** Search3D-style §III-B single 10%
      expanded bbox, background kept, no zeroing of non-mask pixels. The expansion
      gives SigLIP enough surrounding context to disambiguate small parts (e.g. a
      "leg" crop with some chair around it tells the encoder "chair leg", not
      "rectangular brown thing").

    The bbox expansion is **strictly a SigLIP-input artifact**. The mask PNGs on disk,
    the per-pixel mask_id_maps, the containment dict, and the per-Gaussian finest-mask
    references all use the original unchanged binary masks — only the *feature
    embedding* of each mask changes.
    """

    # Per-level expansion ratio (Search3D §III-B "k_exp"). 0 = use original two-crop method.
    DEFAULT_EXPANSION = {1: 0.0, 3: 0.1, 6: 0.1}

    def __init__(self, model_name: str, pretrained: str, device: str,
                 expansion: dict | None = None):
        self.device = device
        self.expansion = dict(self.DEFAULT_EXPANSION)
        if expansion:
            self.expansion.update(expansion)
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device,
        )
        self.model.eval()
        self.preprocess_vlm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # SigLIP feature dim — introspect via a forward pass on a dummy crop.
        # open_clip's various model wrappers (TimmModel, custom VisionTransformer, ...)
        # expose this in different ways; a forward pass is the portable check.
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224), device=device)
            self.D = int(self.model.encode_image(dummy).shape[-1])

    def _pad_square(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        l = max(w, h)
        pad = torch.zeros((3, l, l), dtype=torch.uint8)
        if h > w:
            pad[:, :, (h - w) // 2: (h - w) // 2 + w] = img
        else:
            pad[:, (w - h) // 2: (w - h) // 2 + h, :] = img
        return pad

    def _tight_bbox(self, mask: torch.Tensor) -> tuple | None:
        """Returns (y1, y2, x1, x2) inclusive for the tight bbox around the mask, or None
        if the mask is empty."""
        rows = torch.any(mask, dim=2)
        cols = torch.any(mask, dim=1)
        if not torch.any(rows):
            return None
        y1, y2 = torch.where(rows[0])[0][[0, -1]]
        x1, x2 = torch.where(cols[0])[0][[0, -1]]
        return int(y1), int(y2), int(x1), int(x2)

    def _crop(self, mask: torch.Tensor, image: torch.Tensor, with_background: bool):
        """Original two-crop helper for level 1 (no bbox expansion)."""
        image = image.clone()
        if not with_background:
            image[mask.expand(image.shape) == 0] = 0
        bbox = self._tight_bbox(mask)
        if bbox is None:
            return None
        y1, y2, x1, x2 = bbox
        return image[:, y1:y2 + 1, x1:x2 + 1]

    def _expanded_crop(self, mask: torch.Tensor, image: torch.Tensor, k_exp: float):
        """Search3D-style single expanded bbox crop, background kept.

        Expands the tight bbox by `k_exp` on each side (so total width/height grows
        by 2 × k_exp = 20% at k_exp=0.1), clamped to image bounds. No zeroing of
        non-mask pixels — the whole expanded rectangle is fed to SigLIP.
        """
        _, H, W = image.shape
        bbox = self._tight_bbox(mask)
        if bbox is None:
            return None
        y1, y2, x1, x2 = bbox
        h = y2 - y1 + 1
        w = x2 - x1 + 1
        dy = int(round(h * k_exp))
        dx = int(round(w * k_exp))
        y1e = max(0, y1 - dy)
        y2e = min(H - 1, y2 + dy)
        x1e = max(0, x1 - dx)
        x2e = min(W - 1, x2 + dx)
        return image[:, y1e:y2e + 1, x1e:x2e + 1]

    def embed_all(self, img_uint8: torch.Tensor, masks_bool: torch.Tensor,
                  level: int) -> np.ndarray:
        """Returns (N, D) per-mask SigLIP features. Encoding strategy depends on `level`
        per `self.expansion`."""
        k_exp = self.expansion.get(level, 0.0)
        out = []
        for i in range(len(masks_bool)):
            mask = masks_bool[i].unsqueeze(0)

            if k_exp > 0.0:
                # Search3D-style: single expanded bbox crop, background kept.
                crop = self._expanded_crop(mask, img_uint8, k_exp)
                if crop is None:
                    out.append(torch.zeros(self.D))
                    continue
                crop = self._pad_square(crop)
                inp = self.preprocess_vlm(crop.float() / 255.0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.model.encode_image(inp)[0]
            else:
                # Original: two crops (mask-only + bbox-with-bg), averaged.
                crop_mask = self._crop(mask, img_uint8, with_background=False)
                if crop_mask is None:
                    out.append(torch.zeros(self.D))
                    continue
                crop_mask = self._pad_square(crop_mask)
                crop_bg = self._pad_square(self._crop(mask, img_uint8, with_background=True))
                crops = torch.stack([
                    self.preprocess_vlm(crop_mask.float() / 255.0),
                    self.preprocess_vlm(crop_bg.float() / 255.0),
                ]).to(self.device)
                with torch.no_grad():
                    feat = self.model.encode_image(crops).mean(dim=0)

            out.append(feat.cpu())
        return torch.stack(out).numpy()


def collect_scenes(data_dir: Path, single_scene: str | None) -> list[Path]:
    if single_scene:
        return [data_dir / single_scene]
    return [d for d in sorted(data_dir.iterdir())
            if d.is_dir() and (d / "dslr" / "resized_images").is_dir()]


def process_scene_level(scene_dir: Path, level: int, encoder: MaskCropEncoder,
                        target_size: tuple):
    """Compute SigLIP features for all masks at one level for one scene."""
    img_root = scene_dir / "dslr" / "resized_images"
    level_root = scene_dir / f"masks_lvl_{level}"
    if not level_root.is_dir():
        print(f"  [skip] {scene_dir.name} lvl{level}: no {level_root.name}/ — stage 1 not run for this level")
        return

    frames = sorted([d for d in level_root.iterdir() if d.is_dir()])
    for frame_dir in tqdm(frames, desc=f"{scene_dir.name} lvl{level}", leave=False):
        embed_path = frame_dir / "siglip_embeddings.npy"
        if embed_path.exists():
            continue
        meta_path = frame_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            metadata = json.load(f)
        if not metadata:
            np.save(embed_path, np.zeros((0, encoder.D), dtype=np.float32))
            continue

        # Find the source image. Frame dir name matches image stem.
        stem = frame_dir.name
        candidates = [img_root / f"{stem}.JPG", img_root / f"{stem}.jpg",
                      img_root / f"{stem}.jpeg", img_root / f"{stem}.png"]
        img_path = next((c for c in candidates if c.exists()), None)
        if img_path is None:
            print(f"  [warn] no source image for {frame_dir} (looked for {[c.name for c in candidates]})")
            continue

        img_pil = Image.open(img_path).convert("RGB").resize(target_size, Image.BICUBIC)
        img_uint8 = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)

        masks = []
        for entry in metadata:
            m = cv2.imread(str(frame_dir / entry["mask_file"]), cv2.IMREAD_GRAYSCALE)
            masks.append(torch.from_numpy(m > 127))
        masks_t = torch.stack(masks)

        feats = encoder.embed_all(img_uint8, masks_t, level)
        np.save(embed_path, feats.astype(np.float32))


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    target_size = (args.target_w, args.target_h)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[siglip] warning: no GPU. SigLIP forward passes will be slow.")
    print(f"[siglip] data_dir = {data_dir}")
    print(f"[siglip] device   = {device}")
    print(f"[siglip] model    = {args.model} ({args.pretrained})")
    print(f"[siglip] levels   = {args.levels}")

    scenes = collect_scenes(data_dir, args.scene)
    if not scenes:
        raise RuntimeError(f"No scenes found under {data_dir}")
    print(f"[siglip] scenes to process: {[s.name for s in scenes]}")

    print("[siglip] loading model...")
    encoder = MaskCropEncoder(args.model, args.pretrained, device)
    print(f"[siglip] feature dim D = {encoder.D}")
    print(f"[siglip] crop strategy per level:")
    for lvl in sorted(encoder.expansion):
        k = encoder.expansion[lvl]
        mode = (f"single expanded bbox (k_exp={k:.2f}, Search3D §III-B)"
                if k > 0 else "two-crop average (original)")
        print(f"           lvl {lvl}: {mode}")

    for scene_dir in scenes:
        print(f"\n[siglip] === scene {scene_dir.name} ===")
        for level in args.levels:
            process_scene_level(scene_dir, level, encoder, target_size)

    print("\n[siglip] done.")


if __name__ == "__main__":
    main()
