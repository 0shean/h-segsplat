#!/usr/bin/env python3
"""
3D-OVS mIoU eval using v2 per-mask SigLIP features (instead of per-cluster bank).

Why this can help: the cluster bank at level 1 has very few entries (M_1 = 5–6)
because there are few whole-object SAM masks across 2 views. With 6 classes in
the scene, multiple objects collapse into the same cluster centroid, and any
text query against that mixed centroid biases toward whichever object SigLIP
finds most visually salient. The result is "bleed" — e.g. the Pikachu cluster's
mask membership also includes a generic-foreground mask that ends up covering
sofa-looking pixels.

The v2 payload stores the **per-mask** SigLIP features, which are sharper than
cluster centroids. We re-use the rendered per-pixel `E_real[c]` (one weight per
cluster) but compute per-class scores via the per-mask vectors of each mask
that belongs to that cluster:

    cluster_to_masks[c] = list of mask indices m with closest-cluster == c
    per_cluster_class_score[c, k] = aggregate over mask_features[m] @ class_text[k]
                                    for m in cluster_to_masks[c]

Two aggregation choices:
    --aggregate mean : average per-mask LERF inside the cluster
    --aggregate max  : take the per-mask LERF max

Then at each pixel: score[k] = sum_c E_real[c] * per_cluster_class_score[c, k]

This is mathematically the same shape as the original (per-pixel argmax over
classes), but the per-cluster class score is now informed by individual SAM
masks rather than the cluster centroid. Re-uses existing rendered features ---
no new GPU runs needed.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "scripts_multiscan" / "eval"))
from hsegsplat_offline_state import SigLIPTextEncoder  # noqa: E402


PALETTE_BGR = np.array([
    [220, 220, 220],  # bg
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
], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--colab_root", type=Path, required=True)
    p.add_argument("--ingested_root", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--levels", nargs="+", type=int, default=[3])
    p.add_argument("--aggregate", choices=["mean", "max"], default="max")
    p.add_argument("--per_level_normalize", action="store_true",
                   help="Normalize each per-level score map by its level-max foreground "
                        "mass before combining across levels. Lets level-6 (sparse) "
                        "contribute even when its absolute scores are small.")
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--mass_threshold", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out_dir", type=Path, default=Path("data/3D-OVS/eval_results"))
    p.add_argument("--visualize", action="store_true",
                   help="Also write color-overlay PNGs to <out_dir>/v2_visualizations/<scene>/")
    p.add_argument("--alpha", type=float, default=0.6)
    return p.parse_args()


def colorize(class_map: np.ndarray) -> np.ndarray:
    out = np.full((*class_map.shape, 3), 220, dtype=np.uint8)
    n = int(class_map.max()) + 1
    for c in range(max(n, 0)):
        col = PALETTE_BGR[1 + (c % (len(PALETTE_BGR) - 1))]
        out[class_map == c] = col
    return out


def compute_iou(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


def assign_masks_to_clusters(mask_features: np.ndarray, bank_real: np.ndarray) -> np.ndarray:
    """Returns (N_masks,) int: cluster index in [0, M) for each mask."""
    mf = mask_features / np.maximum(np.linalg.norm(mask_features, axis=-1, keepdims=True), 1e-9)
    return (mf @ bank_real.T).argmax(axis=1)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("[v2-eval] loading SigLIP ...")
    encoder = SigLIPTextEncoder(device=args.device)

    overall = {}
    per_class_results = {}
    for scene in args.scenes:
        print(f"\n[v2-eval] === {scene} ===")
        tv_path = args.ingested_root / scene / "target_views.json"
        g_pt = args.colab_root / scene / "gaussians.pt"
        gt_root = args.ingested_root / scene / "gt_masks"
        feat_paths = {lvl: args.colab_root / scene / f"rendered_feature_map_targets_lvl{lvl}.npy"
                      for lvl in args.levels}
        missing = [str(p) for p in [tv_path, g_pt, gt_root] + list(feat_paths.values())
                   if not p.exists()]
        if missing:
            print(f"  [skip] missing: {missing}")
            continue
        tv = json.load(open(tv_path))
        classes = tv["classes"]
        targets = tv["targets"]

        g = torch.load(g_pt, map_location="cpu", weights_only=False)
        if "v2" not in g:
            print(f"  [skip] gaussians.pt has no v2 payload")
            continue

        phi_canon = encoder.phi_canonical
        class_embs = np.stack([encoder(c) for c in classes], 0)  # (C, D)

        # Pre-compute per-cluster per-class score for each requested level.
        # Aggregation = "mean" or "max" of per-mask LERF over the masks in that cluster.
        cluster_class_score_per_lvl = {}
        for lvl in args.levels:
            bank = g["banks"][lvl].numpy().astype(np.float32)[1:]   # (M, D)
            bank_real = bank / np.maximum(np.linalg.norm(bank, axis=-1, keepdims=True), 1e-9)
            mf = g["v2"]["mask_features"][lvl].cpu().numpy().astype(np.float32)
            mf_norm = mf / np.maximum(np.linalg.norm(mf, axis=-1, keepdims=True), 1e-9)
            mask_to_cluster = assign_masks_to_clusters(mf, bank_real)
            # Per-mask LERF vs each class.
            sim_q = mf_norm @ class_embs.T  # (N_masks, C)
            sim_c = mf_norm @ phi_canon.T   # (N_masks, num_canon)
            diff = args.tau * (sim_q[:, :, None] - sim_c[:, None, :])  # (N_masks, C, num_canon)
            pair = 1.0 / (1.0 + np.exp(-diff))
            mask_class_lerf = pair.min(axis=-1)                        # (N_masks, C)

            M = bank_real.shape[0]
            C = len(classes)
            cluster_class_score = np.zeros((M, C), dtype=np.float32)
            for c_idx in range(M):
                in_c = (mask_to_cluster == c_idx)
                if not in_c.any():
                    continue
                if args.aggregate == "max":
                    cluster_class_score[c_idx] = mask_class_lerf[in_c].max(axis=0)
                else:
                    cluster_class_score[c_idx] = mask_class_lerf[in_c].mean(axis=0)
            cluster_class_score_per_lvl[lvl] = cluster_class_score
            print(f"  lvl {lvl}: {mf.shape[0]} masks -> {M} clusters; "
                  f"mask->cluster histogram = {np.bincount(mask_to_cluster, minlength=M).tolist()}")
            print(f"    per-cluster best class:")
            for c_idx in range(M):
                best = int(cluster_class_score[c_idx].argmax())
                top = cluster_class_score[c_idx, best]
                print(f"      cluster {c_idx}: {classes[best]:<35s} score={top:.3f}")

        per_class_iou = {c: [] for c in classes}
        vis_dir = args.out_dir / "v2_visualizations" / scene
        if args.visualize:
            vis_dir.mkdir(parents=True, exist_ok=True)
            # legend
            legend = np.full((30 * len(classes) + 30, 600, 3), 255, dtype=np.uint8)
            for i, cls in enumerate(classes):
                col = PALETTE_BGR[1 + (i % (len(PALETTE_BGR) - 1))]
                cv2.rectangle(legend, (10, 10 + 30*i), (40, 30 + 30*i),
                              [int(c) for c in col], -1)
                cv2.putText(legend, cls, (50, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(vis_dir / "_legend.png"), legend)

        for t_idx, tgt in enumerate(targets):
            view_id = tgt["view_id"]
            combined = None
            fg_mass_any = None
            for lvl in args.levels:
                feat = np.load(feat_paths[lvl])
                E = feat[t_idx]                         # (H, W, M+1)
                E_real = E[..., 1:]                     # (H, W, M)
                H, W, M = E_real.shape
                fg_mass = E_real.sum(-1)
                fg_mass_any = fg_mass if fg_mass_any is None else np.maximum(fg_mass_any, fg_mass)
                # Per-pixel per-class score = E_real @ cluster_class_score   (H, W, C)
                per_class = (E_real.reshape(-1, M)
                             @ cluster_class_score_per_lvl[lvl]).reshape(H, W, -1)
                # Move to (C, H, W)
                per_class = per_class.transpose(2, 0, 1)
                if args.per_level_normalize:
                    # Normalize by the level's fg mass at each pixel so sparse
                    # levels (6) aren't drowned out when they're the only level
                    # firing there.
                    norm = np.maximum(fg_mass, 1e-6)
                    per_class = per_class / norm[None, :, :]
                combined = per_class if combined is None else np.maximum(combined, per_class)

            # Mass-weight + argmax
            valid = fg_mass_any >= args.mass_threshold
            pred_cls = (combined * fg_mass_any[None]).argmax(0)
            pred_cls[~valid] = -1

            ious = {}
            for c_i, cls in enumerate(classes):
                gt_path = gt_root / view_id / f"{cls}.png"
                if not gt_path.exists():
                    continue
                gt_m = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                gt_bool = gt_m > 0
                pred_bool = (pred_cls == c_i)
                iou = compute_iou(pred_bool, gt_bool)
                ious[cls] = iou
                per_class_iou[cls].append(iou)
            print(f"  view {view_id}: " +
                  ", ".join(f"{c}={v:.3f}" for c, v in ious.items()))

            if args.visualize:
                rgb_bgr = cv2.imread(str(gt_root / view_id / "_rgb.jpg"))
                pred_rgb = colorize(pred_cls)
                overlay = cv2.addWeighted(rgb_bgr, 1 - args.alpha, pred_rgb, args.alpha, 0)
                cv2.imwrite(str(vis_dir / f"{view_id}__pred_v2.png"), overlay)

        scene_class_iou = {c: float(np.nanmean(per_class_iou[c]))
                            if per_class_iou[c] else float("nan") for c in classes}
        valid_ious = [v for v in scene_class_iou.values() if not np.isnan(v)]
        scene_miou = float(np.mean(valid_ious)) if valid_ious else float("nan")
        print(f"  -> scene mIoU = {scene_miou:.4f}")
        overall[scene] = scene_miou
        per_class_results[scene] = scene_class_iou

    print("\n[v2-eval] SUMMARY")
    print(f"  {'scene':<14} {'mIoU':>8}")
    for s, v in overall.items():
        print(f"  {s:<14} {v:>8.4f}")
    valid = [v for v in overall.values() if not np.isnan(v)]
    if valid:
        print(f"  {'Overall':<14} {np.mean(valid):>8.4f}")

    levels_tag = "lvl" + "_".join(str(l) for l in sorted(args.levels))
    scene_tag = "_".join(args.scenes) if len(args.scenes) <= 4 else f"{len(args.scenes)}scenes"
    out_path = args.out_dir / f"miou_v2__{scene_tag}__{levels_tag}__{args.aggregate}__tau{int(args.tau)}.json"
    with open(out_path, "w") as f:
        json.dump({"levels": list(args.levels), "aggregate": args.aggregate,
                   "tau": args.tau, "mass_threshold": args.mass_threshold,
                   "per_scene_miou": overall, "per_class": per_class_results,
                   "overall_miou": float(np.mean(valid)) if valid else None}, f, indent=2)
    print(f"\n[v2-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
