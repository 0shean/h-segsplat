#!/usr/bin/env python3
"""
Per-level prediction overlays on 3D-OVS target views.

For each labeled target view, computes per-class LERF relevancy independently
at each level (1, 3, 6) and writes one overlay PNG per level alongside the GT
overlay and RGB:

    <out>/<scene>/<view_id>__rgb.png
    <out>/<scene>/<view_id>__gt_overlay.png
    <out>/<scene>/<view_id>__pred_lvl1.png
    <out>/<scene>/<view_id>__pred_lvl3.png
    <out>/<scene>/<view_id>__pred_lvl6.png

Pred = per-pixel argmax over class relevancies (mass-weighted). Same colour
palette is used across levels for visual comparison.
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


PALETTE_BGR = np.array([
    [220, 220, 220],  # 0 = background
    [ 26,  28, 228],  # red
    [ 74, 175,  77],  # green
    [184, 126,  55],  # blue
    [163,  78, 152],  # purple
    [  0, 127, 255],  # orange
    [191, 255, 255],  # yellow
    [191, 207, 166],
    [153, 153, 247],
    [153, 153, 153],
    [ 64, 224, 208],
    [127,   0, 255],
], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--colab_root", type=Path, required=True)
    p.add_argument("--ingested_root", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--mass_threshold", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--out_dir", type=Path,
                   default=Path("data/3D-OVS/eval_results/level_visualizations"))
    return p.parse_args()


def colorize(class_map: np.ndarray) -> np.ndarray:
    out = np.full((*class_map.shape, 3), 220, dtype=np.uint8)
    n_classes = int(class_map.max()) + 1
    for c in range(max(n_classes, 0)):
        col = PALETTE_BGR[1 + (c % (len(PALETTE_BGR) - 1))]
        out[class_map == c] = col
    return out


def predict_one_level(feat_t: np.ndarray, bank: np.ndarray,
                       class_embs: dict, phi_canon: np.ndarray,
                       tau: float, mass_thresh: float,
                       classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pred_class_map[H,W] int, fg_mass[H,W] float)."""
    E_real = feat_t[..., 1:]
    H, W, M = E_real.shape
    fg_mass = E_real.sum(-1)
    F = E_real.reshape(-1, M) @ bank
    F = F / np.maximum(np.linalg.norm(F, axis=-1, keepdims=True), 1e-9)
    sim_c = F @ phi_canon.T
    rels = []
    for cls in classes:
        sim_q = F @ class_embs[cls]
        diff = tau * (sim_q[:, None] - sim_c)
        rels.append((1.0 / (1.0 + np.exp(-diff))).min(axis=1))
    rels = np.stack(rels, axis=0).reshape(-1, H, W)
    rels = rels * fg_mass[None]
    pred = rels.argmax(0)
    valid = fg_mass >= mass_thresh
    pred[~valid] = -1
    return pred, fg_mass


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("[viz] loading SigLIP ...")
    encoder = SigLIPTextEncoder(device="cpu")

    LEVELS = [1, 3, 6]
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
        for lvl in LEVELS:
            b = g["banks"][lvl].numpy().astype(np.float32)[1:]
            b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
            banks[lvl] = b
        feat_per_lvl = {lvl: np.load(args.colab_root / scene / f"rendered_feature_map_targets_lvl{lvl}.npy")
                        for lvl in LEVELS}

        # Legend
        legend = np.full((30 * len(classes) + 30, 600, 3), 255, dtype=np.uint8)
        for i, cls in enumerate(classes):
            col = PALETTE_BGR[1 + (i % (len(PALETTE_BGR) - 1))]
            cv2.rectangle(legend, (10, 10 + 30*i), (40, 30 + 30*i),
                          [int(c) for c in col], -1)
            cv2.putText(legend, cls, (50, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(scene_out / "_legend.png"), legend)

        phi_canon = encoder.phi_canonical
        class_embs = {cls: encoder(cls) for cls in classes}

        for t_idx, tgt in enumerate(targets):
            view_id = tgt["view_id"]
            rgb_path = gt_root / view_id / "_rgb.jpg"
            rgb_bgr = cv2.imread(str(rgb_path))
            H, W = rgb_bgr.shape[:2]

            # GT overlay
            gt_map = np.full((H, W), -1, dtype=np.int32)
            for c_i, cls in enumerate(classes):
                p = gt_root / view_id / f"{cls}.png"
                if not p.exists(): continue
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0
                gt_map[m] = c_i
            gt_rgb = colorize(gt_map)
            gt_overlay = cv2.addWeighted(rgb_bgr, 1 - args.alpha, gt_rgb, args.alpha, 0)

            cv2.imwrite(str(scene_out / f"{view_id}__rgb.png"), rgb_bgr)
            cv2.imwrite(str(scene_out / f"{view_id}__gt_overlay.png"), gt_overlay)

            for lvl in LEVELS:
                pred, _ = predict_one_level(feat_per_lvl[lvl][t_idx], banks[lvl],
                                             class_embs, phi_canon,
                                             args.tau, args.mass_threshold, classes)
                pred_rgb = colorize(pred)
                overlay = cv2.addWeighted(rgb_bgr, 1 - args.alpha, pred_rgb, args.alpha, 0)
                cv2.imwrite(str(scene_out / f"{view_id}__pred_lvl{lvl}.png"), overlay)
            print(f"  view {view_id}: wrote rgb + gt + 3 level overlays")


if __name__ == "__main__":
    main()
