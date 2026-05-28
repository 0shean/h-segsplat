#!/usr/bin/env python3
"""
2D mIoU eval for H-SegSplat against a mesh-rasterized GT class map.

Inputs:
    --gt_dir         folder containing gt_class_map.npy + label_map.json
                     (output of render_gt_mesh.py)
    --colab_output   folder containing gaussians.pt + rendered_feature_map_lvl{1,3,6}.npy
    --levels         which H-SegSplat granularity levels to use (default: 3)

For each view in gt_class_map.npy:
    1.  Determine the class set: unique objectIds visible in this view, mapped
        to class names via label_map.json (e.g. "sofa", "wall", "floor").
        Class instances are collapsed by class name — multiple "wall" oids
        merge into one class.
    2.  For each class name, run LERF relevancy against the rendered
        feature map (per-cluster bank lookup -> dense F -> LERF) at the
        requested level(s).
    3.  Argmax over classes per pixel -> predicted class map.
    4.  Restrict the comparison to pixels with foreground mass >= threshold
        (option "b" from the design conversation).
    5.  Per-class IoU and scene mIoU.

Outputs:
    <out_dir>/per_class_iou.json     dict of {view -> {class_name: iou}}
    <out_dir>/per_class_iou.csv      one row per (view, class)
    <out_dir>/pred_overlay_v<v>.png  predicted-class overlay on rendered RGB
    Console table: per-view + scene-level mIoU.
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
from hsegsplat_offline_state import SigLIPTextEncoder


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
    [144, 238, 144],
    [  0, 215, 255],
    [180, 105, 255],
    [128, 128,   0],
    [255, 191,   0],
    [  0,   0, 139],
], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", type=Path, required=True,
                   help="Folder with gt_class_map.npy + label_map.json (from render_gt_mesh.py).")
    p.add_argument("--colab_output", type=Path, required=True,
                   help="Folder with gaussians.pt + rendered_feature_map_lvl{1,3,6}.npy.")
    p.add_argument("--levels", nargs="+", type=int, default=[3])
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--mass_threshold", type=float, default=0.05,
                   help="Pixels with summed E_real below this are dropped from "
                        "both the prediction and the evaluation (option (b)).")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Overlay blend strength for the pred PNG.")
    p.add_argument("--skip_classes", nargs="*", default=["remove", "<id=0>"],
                   help="Class names to skip during eval (junk/unknown).")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Defaults to <gt_dir>/../2d_miou_eval/")
    return p.parse_args()


def colorize(class_map: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.full((*class_map.shape, 3), 220, dtype=np.uint8)
    for c in range(n_classes):
        col = PALETTE_BGR[1 + (c % (len(PALETTE_BGR) - 1))]
        out[class_map == c] = col
    return out


def compute_iou(p: np.ndarray, g: np.ndarray) -> float:
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return float("nan")
    return inter / union


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = args.gt_dir.parent / "2d_miou_eval"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] loading GT from {args.gt_dir}")
    gt_class_map = np.load(args.gt_dir / "gt_class_map.npy")   # (V, H, W) int32
    V, H, W = gt_class_map.shape
    label_map_path = args.gt_dir / "label_map.json"
    if label_map_path.exists():
        oid_to_name = {int(k): v for k, v in json.load(open(label_map_path)).items()}
    else:
        oid_to_name = {}

    # Aggregate visible class names across all views, collapsing instances.
    visible_classes = sorted({
        oid_to_name.get(int(oid), f"<id={int(oid)}>")
        for v in range(V)
        for oid in np.unique(gt_class_map[v])
        if int(oid) >= 0
    })
    print(f"[eval] {len(visible_classes)} unique class names across {V} views: {visible_classes}")
    # Drop user-marked junk classes
    skip = set(args.skip_classes)
    visible_classes = [c for c in visible_classes if c not in skip]
    print(f"[eval] {len(visible_classes)} after skipping {sorted(skip)}: {visible_classes}")

    # Map each gt oid to its class index in `visible_classes` (or -1 if skipped)
    oid_to_class_idx = {}
    name_to_idx = {n: i for i, n in enumerate(visible_classes)}
    for oid, name in oid_to_name.items():
        if name in name_to_idx:
            oid_to_class_idx[oid] = name_to_idx[name]

    # Reduce per-pixel gt oid -> per-pixel class index, collapsing instances.
    gt_class_idx = np.full_like(gt_class_map, -1, dtype=np.int32)
    for oid, cidx in oid_to_class_idx.items():
        gt_class_idx[gt_class_map == oid] = cidx
    # Pixels with oid not in visible_classes (e.g. dropped "remove") stay -1.

    # Load gaussians.pt + per-level feature maps
    print(f"[eval] loading H-SegSplat from {args.colab_output}")
    g = torch.load(args.colab_output / "gaussians.pt", map_location="cpu",
                   weights_only=False)
    banks = {}
    for lvl in args.levels:
        b = g["banks"][lvl].numpy().astype(np.float32)[1:]   # drop bg row
        b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
        banks[lvl] = b
    feat_per_lvl = {}
    for lvl in args.levels:
        f = np.load(args.colab_output / f"rendered_feature_map_lvl{lvl}.npy")
        if f.shape[0] != V:
            raise ValueError(f"feature map V={f.shape[0]} != GT V={V}")
        feat_per_lvl[lvl] = f

    # Load rendered RGB for overlays
    rgb_arr_path = args.colab_output / "rendered_rgb.npy"
    if rgb_arr_path.exists():
        rgb_arr = np.load(rgb_arr_path)
    else:
        rgb_arr = None

    # SigLIP text encoder
    print(f"[eval] loading SigLIP ...")
    encoder = SigLIPTextEncoder(device="cpu")
    phi_canon = encoder.phi_canonical
    class_embs = np.stack(
        [encoder(c.replace("_", " ")) for c in visible_classes], axis=0)

    # Eval loop
    results = {}
    csv_rows = []
    n_cls = len(visible_classes)
    for v in range(V):
        # Combine per-level per-class relevancy via per-pixel max
        combined = None
        fg_mass_any = None
        for lvl in args.levels:
            E = feat_per_lvl[lvl][v]                # (H, W, M+1)
            E_real = E[..., 1:]
            fg = E_real.sum(-1)
            fg_mass_any = fg if fg_mass_any is None else np.maximum(fg_mass_any, fg)
            F = E_real.reshape(-1, banks[lvl].shape[0]) @ banks[lvl]
            F = F / np.maximum(np.linalg.norm(F, axis=-1, keepdims=True), 1e-9)
            sim_c = F @ phi_canon.T
            class_rels = []
            for k in range(n_cls):
                sim_q = F @ class_embs[k]
                diff = args.tau * (sim_q[:, None] - sim_c)
                pair = 1.0 / (1.0 + np.exp(-diff))
                class_rels.append(pair.min(axis=1))
            class_rels = np.stack(class_rels, axis=0).reshape(-1, H, W)
            combined = class_rels if combined is None else np.maximum(combined, class_rels)

        # Mass-weight + argmax
        per_class = combined * fg_mass_any[None, :, :]
        pred = per_class.argmax(axis=0).astype(np.int32)
        valid_pred = fg_mass_any >= args.mass_threshold
        pred[~valid_pred] = -1

        # IoU only over pixels that have BOTH a GT class AND a valid prediction.
        valid_gt = gt_class_idx[v] >= 0
        eval_mask = valid_gt & valid_pred
        n_eval = int(eval_mask.sum())
        n_gt = int(valid_gt.sum())
        n_pred = int(valid_pred.sum())
        print(f"\nview {v}: gt_pixels={n_gt}, pred_pixels={n_pred}, eval_pixels={n_eval}")

        ious = {}
        for ci, name in enumerate(visible_classes):
            g_bool = (gt_class_idx[v] == ci) & eval_mask
            p_bool = (pred == ci) & eval_mask
            iou = compute_iou(p_bool.astype(bool), g_bool.astype(bool))
            ious[name] = iou
            csv_rows.append({"view": v, "class": name,
                             "iou": iou if not np.isnan(iou) else "",
                             "gt_pixels": int(g_bool.sum()),
                             "pred_pixels": int(p_bool.sum())})
        # Per-view mIoU = mean over classes that have at least 1 GT pixel.
        # ALSO compute a weighted mIoU = sum(IoU * gt_area) / sum(gt_area) so
        # bigger classes get more say. Both numbers are reported.
        scored = [iou for iou in ious.values() if not np.isnan(iou)]
        miou_v = float(np.mean(scored)) if scored else float("nan")
        weighted_num = 0.0; weighted_den = 0
        for ci, name in enumerate(visible_classes):
            iou = ious[name]
            if np.isnan(iou): continue
            gt_area = int(((gt_class_idx[v] == ci) & eval_mask).sum())
            weighted_num += iou * gt_area
            weighted_den += gt_area
        miou_w = (weighted_num / weighted_den) if weighted_den else float("nan")
        results[f"view_{v}"] = {"per_class": ious, "miou": miou_v,
                                 "miou_weighted_by_area": miou_w,
                                 "n_eval_pixels": n_eval,
                                 "fraction_predictable": n_pred / (H * W)}
        for n, iou in sorted(ious.items(), key=lambda x: -(x[1] if not np.isnan(x[1]) else -1)):
            if not np.isnan(iou):
                print(f"  {n:<25s}: {iou:.3f}")
        print(f"  -> view {v} mIoU (unweighted) = {miou_v:.4f}")
        print(f"  -> view {v} mIoU (area-weighted) = {miou_w:.4f}")

        # Save overlays
        if rgb_arr is not None:
            rgb_img = (rgb_arr[v] * 255).clip(0, 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            pred_color = colorize(pred, n_cls)
            blend = cv2.addWeighted(rgb_bgr, 1 - args.alpha, pred_color, args.alpha, 0)
            cv2.imwrite(str(args.out_dir / f"pred_overlay_v{v}.png"), blend)

    # Cross-view mIoU = mean of per-view mIoUs
    view_mious = [r["miou"] for r in results.values() if isinstance(r, dict)
                  and not np.isnan(r.get("miou", float("nan")))]
    view_mious_w = [r["miou_weighted_by_area"] for r in results.values()
                    if isinstance(r, dict)
                    and not np.isnan(r.get("miou_weighted_by_area", float("nan")))]
    scene_miou = float(np.mean(view_mious)) if view_mious else float("nan")
    scene_miou_w = float(np.mean(view_mious_w)) if view_mious_w else float("nan")
    results["scene_miou"] = scene_miou
    results["scene_miou_weighted"] = scene_miou_w
    print(f"\n[eval] scene mIoU (unweighted, mean of per-view mIoUs): {scene_miou:.4f}")
    print(f"[eval] scene mIoU (area-weighted): {scene_miou_w:.4f}")

    # Persist
    with open(args.out_dir / "per_class_iou.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: None)
    import csv as _csv
    with open(args.out_dir / "per_class_iou.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["view", "class", "iou", "gt_pixels", "pred_pixels"])
        w.writeheader(); w.writerows(csv_rows)
    print(f"[eval] wrote {args.out_dir / 'per_class_iou.json'} and per_class_iou.csv")

    # Class legend
    legend = np.full((30 * n_cls + 30, 600, 3), 255, dtype=np.uint8)
    for i, cls in enumerate(visible_classes):
        col = PALETTE_BGR[1 + (i % (len(PALETTE_BGR) - 1))]
        cv2.rectangle(legend, (10, 10 + 30*i), (40, 30 + 30*i),
                      [int(c) for c in col], -1)
        cv2.putText(legend, cls, (50, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.out_dir / "_legend.png"), legend)


if __name__ == "__main__":
    main()
