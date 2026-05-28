#!/usr/bin/env python3
"""
Visualize H-SegSplat per-class predictions on 3D-OVS target views.

For each labeled target view of a scene, computes per-class LERF relevancy at
the requested levels (same as eval_3dovs_miou.py), does an argmax over
classes, and produces three PNGs:

    <out>/<scene>/<view_id>__rgb.png      target view RGB (GT-aligned)
    <out>/<scene>/<view_id>__pred.png     predicted class color map (no GT, no labels)
    <out>/<scene>/<view_id>__gt.png       same color map computed from GT masks (for compare)
    <out>/<scene>/<view_id>__overlay.png  predicted color map alpha-blended onto RGB

One color per class, consistent across views and across pred/gt for visual comparison.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "multiscan" / "eval"))
from hsegsplat_offline_state import SigLIPTextEncoder  # noqa: E402


# Distinct colors (BGR) — first 12 from a colorblind-friendly palette.
PALETTE_BGR = np.array([
    [220, 220, 220],  # 0 = background (light grey)
    [ 26,  28, 228],  # red
    [ 74, 175,  77],  # green
    [184, 126, 55 ],  # blue
    [163,  78, 152],  # purple
    [  0, 127, 255],  # orange
    [191, 255, 255],  # yellow
    [191, 207, 166],  # cyan-greyish
    [153, 153, 247],  # pink
    [153, 153, 153],  # grey
    [ 64, 224, 208],  # turquoise
    [127,   0, 255],  # magenta
], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--colab_root", type=Path, required=True)
    p.add_argument("--ingested_root", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--levels", nargs="+", type=int, default=[3])
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--mass_threshold", type=float, default=0.05)
    p.add_argument("--out_dir", type=Path, default=Path("data/3D-OVS/eval_results/visualizations"))
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Blend weight of the pred color map on the overlay.")
    return p.parse_args()


def colorize(class_map: np.ndarray) -> np.ndarray:
    """class_map: (H, W) int with -1 = unknown/background, >=0 = class index.
    Returns (H, W, 3) uint8 BGR."""
    H, W = class_map.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    # background colour (light grey).
    out[:] = PALETTE_BGR[0]
    for c in range(int(class_map.max()) + 1):
        if c < 0: continue
        col = PALETTE_BGR[1 + (c % (len(PALETTE_BGR) - 1))]
        out[class_map == c] = col
    return out


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("[viz] loading SigLIP ...")
    encoder = SigLIPTextEncoder(device="cpu")

    for scene in args.scenes:
        print(f"\n[viz] === {scene} ===")
        scene_out = args.out_dir / scene
        scene_out.mkdir(parents=True, exist_ok=True)
        tv = json.load(open(args.ingested_root / scene / "target_views.json"))
        classes = tv["classes"]
        targets = tv["targets"]
        gt_root = args.ingested_root / scene / "gt_masks"

        g = torch.load(args.colab_root / scene / "gaussians.pt",
                       map_location="cpu", weights_only=False)
        banks = {}
        for lvl in args.levels:
            b = g["banks"][lvl].numpy().astype(np.float32)[1:]
            b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
            banks[lvl] = b
        feat_per_lvl = {lvl: np.load(args.colab_root / scene / f"rendered_feature_map_targets_lvl{lvl}.npy")
                        for lvl in args.levels}
        T, H, W, _ = feat_per_lvl[args.levels[0]].shape

        phi_canon = encoder.phi_canonical
        class_embs = {cls: encoder(cls) for cls in classes}
        # Save the class -> color legend for this scene.
        legend_path = scene_out / "_legend.png"
        legend = np.full((30 * len(classes) + 30, 600, 3), 255, dtype=np.uint8)
        for i, cls in enumerate(classes):
            col = PALETTE_BGR[1 + (i % (len(PALETTE_BGR) - 1))]
            cv2.rectangle(legend, (10, 10 + 30*i), (40, 30 + 30*i),
                          [int(c) for c in col], -1)
            cv2.putText(legend, cls, (50, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(legend_path), legend)

        for t_idx, tgt in enumerate(targets):
            view_id = tgt["view_id"]
            combined = None
            fg_mass_any = None
            for lvl in args.levels:
                E = feat_per_lvl[lvl][t_idx]
                E_real = E[..., 1:]
                fg = E_real.sum(-1)
                fg_mass_any = fg if fg_mass_any is None else np.maximum(fg_mass_any, fg)
                F = E_real.reshape(-1, banks[lvl].shape[0]) @ banks[lvl]
                F = F / np.maximum(np.linalg.norm(F, axis=-1, keepdims=True), 1e-9)
                sim_c = F @ phi_canon.T
                class_rels = []
                for cls in classes:
                    sim_q = F @ class_embs[cls]
                    diff = args.tau * (sim_q[:, None] - sim_c)
                    pair = 1.0 / (1.0 + np.exp(-diff))
                    class_rels.append(pair.min(axis=1))
                class_rels = np.stack(class_rels, axis=0).reshape(-1, H, W)
                combined = class_rels if combined is None else np.maximum(combined, class_rels)

            valid = fg_mass_any >= args.mass_threshold
            # weight by mass + argmax
            pred = (combined * fg_mass_any[None]).argmax(0)
            pred[~valid] = -1
            pred_rgb = colorize(pred)

            # GT colorized via per-class binary masks.
            gt_map = np.full((H, W), -1, dtype=np.int32)
            for c_i, cls in enumerate(classes):
                p = gt_root / view_id / f"{cls}.png"
                if not p.exists(): continue
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0
                gt_map[m] = c_i
            gt_rgb = colorize(gt_map)

            # RGB
            rgb_path = gt_root / view_id / "_rgb.jpg"
            rgb_bgr = cv2.imread(str(rgb_path))

            # Overlay = alpha blend pred on RGB.
            overlay = cv2.addWeighted(rgb_bgr, 1 - args.alpha, pred_rgb, args.alpha, 0)

            cv2.imwrite(str(scene_out / f"{view_id}__rgb.png"), rgb_bgr)
            cv2.imwrite(str(scene_out / f"{view_id}__pred.png"), pred_rgb)
            cv2.imwrite(str(scene_out / f"{view_id}__gt.png"), gt_rgb)
            cv2.imwrite(str(scene_out / f"{view_id}__overlay.png"), overlay)
            print(f"  wrote 4 PNGs for view {view_id}")
        print(f"  wrote {scene_out}/_legend.png")


if __name__ == "__main__":
    main()
