#!/usr/bin/env python3
"""
Ground-truth-ish visualization of per-mask SigLIP labels at each granularity level.

For each frame and each level in {1, 3, 6}, produce one PNG showing:
  - the input RGB
  - mask outlines (one color per mask, colors recycled if necessary)
  - the best-matching class label from a candidate list printed at each mask centroid,
    along with the cosine score, e.g. "chair (0.34)".

Candidate classes are read from --classes (one per line; default: repo root semantic_classes.txt).

This is the H-SegSplat equivalent of "what would SigLIP call each mask?" — useful to pick
realistic query terms before running query_hsegsplat.py.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import open_clip


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", type=str, required=True,
                   help="Scene dir with dslr/resized_images/ + masks_lvl_{1,3,6}/")
    p.add_argument("--classes", type=str, required=True,
                   help="Path to a newline-separated list of class names (semantic_classes.txt).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where to write the labeled PNGs.")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 3, 6])
    p.add_argument("--min-area-for-label", type=int, default=400,
                   help="Skip drawing text on masks smaller than this many pixels (outline still drawn).")
    p.add_argument("--model-name", type=str, default="ViT-SO400M-14-SigLIP")
    p.add_argument("--pretrained", type=str, default="webli")
    p.add_argument("--top-k", type=int, default=1,
                   help="How many top labels per mask to print (1 prints just the best).")
    return p.parse_args()


def palette(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 179, n, endpoint=False).astype(np.uint8)
    rng.shuffle(hues)
    hsv = np.stack([hues, np.full(n, 230, np.uint8), np.full(n, 240, np.uint8)], axis=-1)
    bgr = cv2.cvtColor(hsv[None, :, :], cv2.COLOR_HSV2BGR)[0]
    return bgr


def mask_outline(mask_bool: np.ndarray) -> np.ndarray:
    """1-pixel outline of a bool mask, dilated for visibility."""
    m = mask_bool.astype(np.uint8) * 255
    edge = cv2.Canny(m, 50, 150)
    edge = cv2.dilate(edge, np.ones((2, 2), np.uint8))
    return edge > 0


def draw_label(img: np.ndarray, text: str, x: int, y: int,
               text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with a small filled rectangle behind it."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45
    th = 1
    (tw, t_h), baseline = cv2.getTextSize(text, font, fs, th)
    # Clamp into image.
    H, W = img.shape[:2]
    x = max(0, min(x, W - tw - 4))
    y = max(t_h + 4, min(y, H - 4))
    cv2.rectangle(img, (x - 2, y - t_h - 4), (x + tw + 2, y + baseline - 2),
                  bg_color, -1)
    cv2.putText(img, text, (x, y - 3), font, fs, text_color, th, cv2.LINE_AA)


def encode_text(model, tokenizer, texts, batch_size: int = 256):
    """Encode a list of text prompts in batches, return L2-normalized (N, D) numpy array."""
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            tokens = tokenizer(chunk)
            emb = model.encode_text(tokens).float()
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    image_dir = scene_dir / "dslr" / "resized_images"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [c.strip() for c in Path(args.classes).read_text().splitlines() if c.strip()]
    print(f"Loaded {len(classes)} candidate classes from {args.classes}")

    print(f"Loading {args.model_name} ({args.pretrained}) ...")
    model, _, _ = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    print("Encoding class names ...")
    T = encode_text(model, tokenizer, classes)  # (C, D) L2-normalized
    print(f"Class embeddings: {T.shape}")

    # Resolve frame names (any file in resized_images sorted; matches build step's order).
    frame_files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"Found {len(frame_files)} frames.")

    for level in args.levels:
        masks_root = scene_dir / f"masks_lvl_{level}"
        if not masks_root.exists():
            print(f"  skip level {level}: no folder {masks_root}")
            continue

        for frame_path in frame_files:
            frame_stem = frame_path.stem
            mdir = masks_root / frame_stem
            if not mdir.exists():
                print(f"  skip {frame_stem} lvl {level}: no mask dir")
                continue

            with open(mdir / "metadata.json") as f:
                meta = json.load(f)
            feats = np.load(mdir / "siglip_embeddings.npy")
            if feats.shape[0] != len(meta):
                raise ValueError(f"{mdir}: feature count {feats.shape[0]} != mask count {len(meta)}")

            # L2-normalize per-mask features so the dot product is cosine sim.
            feats_unit = feats / np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-12)
            sims = feats_unit @ T.T  # (N, C)

            top_indices = np.argsort(-sims, axis=1)[:, :args.top_k]

            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"  warning: missing image {frame_path}")
                continue

            H, W = img.shape[:2]
            colors = palette(max(len(meta), 12), seed=level * 17)

            for i, entry in enumerate(meta):
                mask = cv2.imread(str(mdir / entry["mask_file"]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                mask_bool = mask > 127
                if mask_bool.shape != (H, W):
                    mask_bool = cv2.resize(mask_bool.astype(np.uint8), (W, H),
                                            interpolation=cv2.INTER_NEAREST) > 0
                color = tuple(int(c) for c in colors[i % len(colors)])

                # Outline
                outline = mask_outline(mask_bool)
                img[outline] = color

                area = int(entry["area"])
                if area < args.min_area_for_label:
                    continue

                # Centroid (mean of mask pixels) — works for arbitrarily shaped blobs.
                ys, xs = np.where(mask_bool)
                if ys.size == 0:
                    continue
                cy = int(ys.mean())
                cx = int(xs.mean())

                top_k_text = []
                for j in range(args.top_k):
                    ci = top_indices[i, j]
                    top_k_text.append(f"{classes[ci]} ({sims[i, ci]:.2f})")
                text = " | ".join(top_k_text)
                draw_label(img, text, cx - 30, cy)

            out_path = output_dir / f"labels_lvl{level}_{frame_stem}.png"
            cv2.imwrite(str(out_path), img)
            print(f"  wrote {out_path}  (N_masks={len(meta)})")

    print(f"\nAll outputs under: {output_dir}/")


if __name__ == "__main__":
    main()
