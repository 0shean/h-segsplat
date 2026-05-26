#!/usr/bin/env python3
"""
Sanity-check visualization of build_hsegsplat_inputs.py outputs.

For each level in {1, 3, 6}, renders:
  level_<N>/vis/cluster_<frame>.png         per-pixel cluster index colored (0=black bg)
  level_<N>/vis/cluster_overlay_<frame>.png same, alpha-blended on the input image
  level_<N>/vis/mask_id_<frame>.png         per-pixel mask id (debug, pre-cluster)
  level_<N>/vis/legend_clusters.png
  level_<N>/vis/legend_masks.png

Plus a combined panel per frame (RGB | level1 | level3 | level6) under <hsegsplat_dir>/vis/.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", type=str, required=True,
                   help="Original scene dir (with dslr/resized_images/<frame>.JPG).")
    p.add_argument("--hsegsplat_dir", type=str, required=True,
                   help="Output of build_hsegsplat_inputs.py (contains level_1/, level_3/, level_6/).")
    p.add_argument("--alpha", type=float, default=0.55,
                   help="Overlay alpha for the cluster color (0=image only, 1=color only).")
    return p.parse_args()


def palette(n: int, seed: int = 0) -> np.ndarray:
    """Distinct BGR colors. Index 0 reserved for background (black)."""
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 179, n, endpoint=False).astype(np.uint8)
    rng.shuffle(hues)
    hsv = np.stack([hues, np.full(n, 230, np.uint8), np.full(n, 240, np.uint8)], axis=-1)
    bgr = cv2.cvtColor(hsv[None, :, :], cv2.COLOR_HSV2BGR)[0]
    out = np.zeros((n + 1, 3), dtype=np.uint8)
    out[1:] = bgr
    return out


def colorize(idx_map: np.ndarray, pal: np.ndarray) -> np.ndarray:
    return pal[idx_map.clip(min=0, max=pal.shape[0] - 1)]


def make_legend(pal: np.ndarray, names: list, out_path: Path,
                row_h: int = 40, sw_w: int = 60, txt_pad: int = 12):
    H = row_h * len(pal)
    W = 380
    canvas = np.full((H, W, 3), 30, dtype=np.uint8)
    for i, color in enumerate(pal):
        y0 = i * row_h
        cv2.rectangle(canvas, (5, y0 + 5), (5 + sw_w, y0 + row_h - 5), color.tolist(), -1)
        label = names[i] if i < len(names) else f"id {i}"
        cv2.putText(canvas, label, (5 + sw_w + txt_pad, y0 + row_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def visualize_level(level_dir: Path, image_dir: Path, alpha: float):
    """Render per-level images. Returns (cluster_overlays_by_frame, frame_names) for the combined panel."""
    with open(level_dir / "meta.json") as f:
        meta = json.load(f)

    index_maps = np.load(level_dir / "index_maps.npy")
    mask_id_maps = np.load(level_dir / "mask_id_maps.npy")
    M = meta["M"]
    frame_names = meta["frame_order"]

    vis_dir = level_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    cluster_pal = palette(M, seed=0)
    cluster_names = ["bg"] + [f"c{i}" for i in range(1, M + 1)]

    max_mask_id = int(mask_id_maps.max())
    mask_pal = palette(max_mask_id, seed=1)
    mask_names = ["bg"] + [f"m{i-1}" for i in range(1, max_mask_id + 1)]

    overlays = {}
    for v_idx, fname in enumerate(frame_names):
        idx = index_maps[v_idx]
        mid = mask_id_maps[v_idx]

        cluster_img = colorize(idx, cluster_pal)
        mask_img = colorize(mid, mask_pal)

        cv2.imwrite(str(vis_dir / f"cluster_{Path(fname).stem}.png"), cluster_img)
        cv2.imwrite(str(vis_dir / f"mask_id_{Path(fname).stem}.png"), mask_img)

        rgb = cv2.imread(str(image_dir / fname))
        if rgb is None:
            print(f"    warning: source image missing {image_dir / fname}; skipping overlay")
            continue
        if rgb.shape[:2] != idx.shape:
            rgb = cv2.resize(rgb, (idx.shape[1], idx.shape[0]), interpolation=cv2.INTER_AREA)

        fg = (idx > 0)[:, :, None]
        blended = (alpha * cluster_img + (1 - alpha) * rgb).astype(np.uint8)
        overlay = np.where(fg, blended, rgb)
        cv2.imwrite(str(vis_dir / f"cluster_overlay_{Path(fname).stem}.png"), overlay)
        overlays[fname] = overlay

        bg_pix = int((idx == 0).sum())
        total = idx.size
        present = sorted(int(c) for c in np.unique(idx) if c != 0)
        print(f"    [{v_idx}] {fname}: bg={100*bg_pix/total:.1f}% | clusters present: {len(present)}")

    make_legend(cluster_pal, cluster_names, vis_dir / "legend_clusters.png")
    make_legend(mask_pal[:max_mask_id + 1], mask_names[:max_mask_id + 1],
                vis_dir / "legend_masks.png")

    return overlays, frame_names


def make_combined_panel(rgb: np.ndarray,
                        overlays_by_level: dict,
                        levels: list,
                        out_path: Path,
                        label_h: int = 32):
    """Side-by-side panel: RGB | overlay@lvl1 | overlay@lvl3 | overlay@lvl6, with labels."""
    panels = [("RGB", rgb)] + [(f"lvl {lvl}", overlays_by_level[lvl]) for lvl in levels]
    H, W = rgb.shape[:2]

    canvas = np.full((H + label_h, W * len(panels), 3), 30, dtype=np.uint8)
    for i, (label, img) in enumerate(panels):
        x0 = i * W
        canvas[label_h:label_h + H, x0:x0 + W] = img
        cv2.putText(canvas, label, (x0 + 10, label_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def main():
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    hsegsplat_dir = Path(args.hsegsplat_dir)

    with open(hsegsplat_dir / "meta.json") as f:
        top_meta = json.load(f)
    levels = top_meta["levels"]
    frame_names = top_meta["frame_order"]
    image_dir = scene_dir / "dslr" / "resized_images"

    overlays_per_level = {}  # level -> {frame: overlay}
    for lvl in levels:
        level_dir = hsegsplat_dir / f"level_{lvl}"
        print(f"  [level {lvl}]")
        overlays, _ = visualize_level(level_dir, image_dir, args.alpha)
        overlays_per_level[lvl] = overlays

    # Combined panels.
    combined_dir = hsegsplat_dir / "vis"
    combined_dir.mkdir(parents=True, exist_ok=True)
    for fname in frame_names:
        rgb = cv2.imread(str(image_dir / fname))
        if rgb is None:
            print(f"  warning: source image missing {image_dir / fname}; skipping combined panel")
            continue
        overlays_by_level = {}
        skip = False
        for lvl in levels:
            ov = overlays_per_level[lvl].get(fname)
            if ov is None:
                skip = True
                break
            if ov.shape[:2] != rgb.shape[:2]:
                rgb = cv2.resize(rgb, (ov.shape[1], ov.shape[0]), interpolation=cv2.INTER_AREA)
            overlays_by_level[lvl] = ov
        if skip:
            continue
        out_path = combined_dir / f"levels_panel_{Path(fname).stem}.png"
        make_combined_panel(rgb, overlays_by_level, levels, out_path)

    print(f"\nPer-level visualizations: {hsegsplat_dir}/level_<N>/vis/")
    print(f"Combined panels:          {combined_dir}/")


if __name__ == "__main__":
    main()
