#!/usr/bin/env python3
"""
Per-level prediction overlays on MultiScan context views.

Same idea as scripts_3dovs/visualize_levels.py but without target_views.json
(MultiScan doesn't have it). The class list is fixed: common indoor objects
that should appear in scene_00006_00 (sofa, table, etc.). The visualizer
renders the rendered RGB at each of the 2 context views and overlays
per-class argmax for each of the 3 levels.

Outputs:
    <colab_outputs>/<scan>/level_overlays/
        _legend.png
        view0__rgb.png
        view0__pred_lvl1.png
        view0__pred_lvl3.png
        view0__pred_lvl6.png
        view1__... (same)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "scripts_multiscan" / "eval"))
from hsegsplat_offline_state import SigLIPTextEncoder


# Re-use the 3D-OVS palette but extended.
PALETTE_BGR = np.array([
    [220, 220, 220],
    [ 26,  28, 228],
    [ 74, 175,  77],
    [184, 126,  55],
    [163,  78, 152],
    [  0, 127, 255],
    [191, 255, 255],
    [191, 207, 166],
    [153, 153, 247],
    [153, 153, 153],
    [ 64, 224, 208],
    [127,   0, 255],
    [203, 192, 255],
    [128,   0, 128],
], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", type=str, required=True)
    p.add_argument("--colab_outputs_root", type=Path,
                   default=Path("data/Multiscan/colab_outputs"))
    p.add_argument("--classes", nargs="+", required=True,
                   help="Class names to query (space-separated). Use quotes for "
                        "multi-word classes, e.g. --classes sofa table 'coffee table' picture")
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--mass_threshold", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.6)
    return p.parse_args()


def colorize(class_map: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.full((*class_map.shape, 3), 220, dtype=np.uint8)
    for c in range(n_classes):
        col = PALETTE_BGR[1 + (c % (len(PALETTE_BGR) - 1))]
        out[class_map == c] = col
    return out


def main():
    args = parse_args()
    scan = args.scan
    classes = args.classes
    n_cls = len(classes)
    print(f"[{scan}] classes: {classes}")

    g = torch.load(args.colab_outputs_root / scan / "gaussians.pt",
                   map_location="cpu", weights_only=False)
    levels = g["levels"]
    H, W = g["image_HW"]
    banks = {}
    for lvl in levels:
        b = g["banks"][lvl].numpy().astype(np.float32)[1:]
        b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
        banks[lvl] = b

    feat_per_lvl = {lvl: np.load(args.colab_outputs_root / scan
                                  / f"rendered_feature_map_lvl{lvl}.npy")
                     for lvl in levels}
    V = feat_per_lvl[levels[0]].shape[0]
    print(f"[{scan}] V={V}, H={H}, W={W}, levels={levels}")

    encoder = SigLIPTextEncoder(device="cpu")
    phi_canon = encoder.phi_canonical
    class_embs = np.stack([encoder(c) for c in classes], axis=0)

    out_dir = args.colab_outputs_root / scan / "level_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Legend
    legend = np.full((30 * n_cls + 30, 600, 3), 255, dtype=np.uint8)
    for i, cls in enumerate(classes):
        col = PALETTE_BGR[1 + (i % (len(PALETTE_BGR) - 1))]
        cv2.rectangle(legend, (10, 10 + 30 * i), (40, 30 + 30 * i),
                      [int(c) for c in col], -1)
        cv2.putText(legend, cls, (50, 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "_legend.png"), legend)

    rgb = np.load(args.colab_outputs_root / scan / "rendered_rgb.npy")
    for v in range(V):
        rgb_img = (rgb[v] * 255).clip(0, 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"view{v}__rgb.png"), rgb_bgr)

        for lvl in levels:
            E = feat_per_lvl[lvl][v]
            E_real = E[..., 1:]
            fg_mass = E_real.sum(-1)
            F = E_real.reshape(-1, banks[lvl].shape[0]) @ banks[lvl]
            F = F / np.maximum(np.linalg.norm(F, axis=-1, keepdims=True), 1e-9)
            sim_c = F @ phi_canon.T
            rels = []
            for k in range(n_cls):
                sim_q = F @ class_embs[k]
                diff = args.tau * (sim_q[:, None] - sim_c)
                pair = 1.0 / (1.0 + np.exp(-diff))
                rels.append(pair.min(axis=1))
            rels = np.stack(rels, axis=0).reshape(-1, H, W)
            rels = rels * fg_mass[None, :, :]
            pred = rels.argmax(axis=0)
            valid = fg_mass >= args.mass_threshold
            pred[~valid] = -1
            pred_rgb = colorize(pred, n_cls)
            overlay = cv2.addWeighted(rgb_bgr, 1 - args.alpha, pred_rgb, args.alpha, 0)
            cv2.imwrite(str(out_dir / f"view{v}__pred_lvl{lvl}.png"), overlay)
        print(f"  view {v}: wrote rgb + {len(levels)} level overlays")
    print(f"[{scan}] outputs at {out_dir}/")


if __name__ == "__main__":
    main()
