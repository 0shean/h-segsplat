#!/usr/bin/env python3
"""
Hierarchical 2D mIoU eval for H-SegSplat against the mesh-rasterized GT.

Inputs:
    --gt_dir         folder containing gt_objectId_map.npy + gt_partId_map.npy
                     + channels.json + the multiscan annotations.json
                     (output of render_gt_mesh.py invoked with
                      --face_label_attrs objectId partId)
    --colab_output   folder containing gaussians.pt + rendered_feature_map_lvl{1,3,6}.npy
    --annotations    path to scene's <scene>.annotations.json
    --object_levels  H-SegSplat levels used for the OBJECT class score (default: 1 3)
    --part_levels    H-SegSplat levels used for the PART class score (default: 6)

For each view:
    1. Gather class names from the annotations file:
         - object classes = unique label stems (e.g. "chair", "table") over
           the OBJECTS that appear in this view at all.
         - part classes = unique label stems over the PARTS that appear in
           this view at all.
       Per-pixel GT (obj_class_idx, part_class_idx) is computed by mapping
       objectId -> object class and partId -> part class.
    2. LERF relevancy per OBJECT class at object_levels (max over levels),
       LERF relevancy per PART class at part_levels.
    3. Per-pixel argmax -> pred_obj_class, pred_part_class.
       Pixels below mass_threshold get pred=-1.
    4. Restrict comparison to pixels with valid_gt AND valid_pred for the
       channel under test.

Reported metrics (per view + scene average):
    - object mIoU       (unweighted + area-weighted)
    - part mIoU         (unweighted + area-weighted)
    - joint (obj, part) mIoU over all (obj_class, part_class) tuples that
      appear as GT in this view
    - hierarchy-consistency rate = fraction of validly-predicted pixels
      where the predicted (obj_class, part_class) tuple is a tuple that
      actually exists in the GT annotations for this scene (the part of
      class P is plausibly a part of an object of class O somewhere).

Outputs:
    <out_dir>/hier_iou.json          full per-view / scene results
    <out_dir>/hier_pred_obj_v<v>.png object-class overlay
    <out_dir>/hier_pred_part_v<v>.png part-class overlay
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
    p.add_argument("--gt_dir", type=Path, required=True)
    p.add_argument("--colab_output", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True,
                   help="Scene annotations.json (objects + parts).")
    p.add_argument("--object_levels", nargs="+", type=int, default=[1, 3])
    p.add_argument("--part_levels", nargs="+", type=int, default=[6])
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--mass_threshold", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--skip_classes", nargs="*", default=["remove", "unknown"])
    p.add_argument("--out_dir", type=Path, default=None)
    return p.parse_args()


def class_stem(label: str) -> str:
    """`chair.3` -> `chair`. Underscores kept (e.g. `coffee_table`)."""
    if "." in label:
        return label.rsplit(".", 1)[0]
    return label


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


def lerf_per_class(F: np.ndarray, phi_c: np.ndarray, class_embs: np.ndarray,
                   tau: float) -> np.ndarray:
    """F: (N, D) unit; returns (n_classes, N) LERF relevancy."""
    sim_c = F @ phi_c.T   # (N, 3)
    out = np.empty((class_embs.shape[0], F.shape[0]), dtype=np.float32)
    for k in range(class_embs.shape[0]):
        sim_q = F @ class_embs[k]                  # (N,)
        diff = tau * (sim_q[:, None] - sim_c)
        pair = 1.0 / (1.0 + np.exp(-diff))
        out[k] = pair.min(axis=1)
    return out


def predict_classes(feat_per_lvl: dict, banks: dict, levels, classes,
                    class_embs, phi_c, tau, H, W,
                    mass_threshold: float):
    """Returns (pred_class_idx (H,W), fg_mass (H,W))."""
    combined = None
    fg_mass_any = None
    for lvl in levels:
        E = feat_per_lvl[lvl]                # (H, W, M+1) for this view
        E_real = E[..., 1:]
        fg = E_real.sum(-1)
        fg_mass_any = fg if fg_mass_any is None else np.maximum(fg_mass_any, fg)
        F = E_real.reshape(-1, banks[lvl].shape[0]) @ banks[lvl]
        F = F / np.maximum(np.linalg.norm(F, axis=-1, keepdims=True), 1e-9)
        class_rels = lerf_per_class(F, phi_c, class_embs, tau).reshape(
            -1, H, W)
        combined = class_rels if combined is None else np.maximum(
            combined, class_rels)
    per_class = combined * fg_mass_any[None, :, :]
    pred = per_class.argmax(axis=0).astype(np.int32)
    pred[fg_mass_any < mass_threshold] = -1
    return pred, fg_mass_any


def build_id_to_class(items: list, key_id: str) -> dict:
    """items: list of {id_key, 'label', ...}. Returns id -> class stem."""
    out = {}
    for it in items:
        out[int(it[key_id])] = class_stem(it["label"])
    return out


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = args.gt_dir.parent / "hier_eval"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load GT channel maps
    gt_obj = np.load(args.gt_dir / "gt_objectId_map.npy")     # (V, H, W) int32
    gt_part = np.load(args.gt_dir / "gt_partId_map.npy")      # (V, H, W) int32
    V, H, W = gt_obj.shape
    print(f"[hier] GT shape: V={V} H={H} W={W}")

    # Annotations -> {objectId -> obj_class, partId -> part_class, plus
    # the set of valid (obj_class, part_class) tuples that exist in this scene}.
    ann = json.load(open(args.annotations))
    obj_id_to_class = build_id_to_class(ann["objects"], "objectId")
    part_id_to_class = build_id_to_class(ann["parts"], "partId")
    valid_tuples = set()           # (obj_class, part_class)
    for o in ann["objects"]:
        oc = class_stem(o["label"])
        for pid in o.get("partIds", []):
            pc = part_id_to_class.get(int(pid))
            if pc is not None:
                valid_tuples.add((oc, pc))
    print(f"[hier] {len(obj_id_to_class)} objects, {len(part_id_to_class)} parts, "
          f"{len(valid_tuples)} unique (obj_class, part_class) tuples in annotations")

    skip = set(args.skip_classes)

    # Visible OBJECT classes = stems of objects that appear in any view
    visible_obj = sorted({
        obj_id_to_class.get(int(oid))
        for v in range(V) for oid in np.unique(gt_obj[v])
        if int(oid) >= 0 and obj_id_to_class.get(int(oid)) not in skip
        and obj_id_to_class.get(int(oid)) is not None
    })
    visible_part = sorted({
        part_id_to_class.get(int(pid))
        for v in range(V) for pid in np.unique(gt_part[v])
        if int(pid) >= 0 and part_id_to_class.get(int(pid)) not in skip
        and part_id_to_class.get(int(pid)) is not None
    })
    print(f"[hier] visible obj classes ({len(visible_obj)}): {visible_obj}")
    print(f"[hier] visible part classes ({len(visible_part)}): {visible_part}")

    obj_name_to_idx = {n: i for i, n in enumerate(visible_obj)}
    part_name_to_idx = {n: i for i, n in enumerate(visible_part)}

    # Per-pixel GT class indices
    gt_obj_idx = np.full_like(gt_obj, -1, dtype=np.int32)
    for oid, oc in obj_id_to_class.items():
        if oc in obj_name_to_idx:
            gt_obj_idx[gt_obj == oid] = obj_name_to_idx[oc]
    gt_part_idx = np.full_like(gt_part, -1, dtype=np.int32)
    for pid, pc in part_id_to_class.items():
        if pc in part_name_to_idx:
            gt_part_idx[gt_part == pid] = part_name_to_idx[pc]

    # Load H-SegSplat
    print(f"[hier] loading {args.colab_output / 'gaussians.pt'}")
    g = torch.load(args.colab_output / "gaussians.pt", map_location="cpu",
                   weights_only=False)
    needed_levels = sorted(set(args.object_levels) | set(args.part_levels))
    banks = {}
    feat_per_lvl = {}
    for lvl in needed_levels:
        b = g["banks"][lvl].numpy().astype(np.float32)[1:]
        b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
        banks[lvl] = b
        f = np.load(args.colab_output / f"rendered_feature_map_lvl{lvl}.npy")
        if f.shape[0] != V:
            raise ValueError(f"lvl{lvl} feature map V={f.shape[0]} != GT V={V}")
        feat_per_lvl[lvl] = f

    rgb_arr = None
    if (args.colab_output / "rendered_rgb.npy").exists():
        rgb_arr = np.load(args.colab_output / "rendered_rgb.npy")

    print("[hier] loading SigLIP ...")
    encoder = SigLIPTextEncoder(device="cpu")
    phi_c = encoder.phi_canonical
    obj_embs = np.stack(
        [encoder(c.replace("_", " ")) for c in visible_obj], axis=0)
    part_embs = np.stack(
        [encoder(c.replace("_", " ")) for c in visible_part], axis=0)

    results = {}

    obj_view_mious, obj_view_mious_w = [], []
    part_view_mious, part_view_mious_w = [], []
    joint_view_mious = []
    consistency_rates = []

    for v in range(V):
        per_view_feat = {lvl: feat_per_lvl[lvl][v] for lvl in needed_levels}

        pred_obj, fg_obj = predict_classes(
            per_view_feat, banks, args.object_levels, visible_obj, obj_embs,
            phi_c, args.tau, H, W, args.mass_threshold)
        pred_part, fg_part = predict_classes(
            per_view_feat, banks, args.part_levels, visible_part, part_embs,
            phi_c, args.tau, H, W, args.mass_threshold)

        # ----- Object mIoU -----
        valid_obj = (pred_obj >= 0) & (gt_obj_idx[v] >= 0)
        ious_obj, areas_obj = {}, {}
        for ci, name in enumerate(visible_obj):
            gb = (gt_obj_idx[v] == ci) & valid_obj
            pb = (pred_obj == ci) & valid_obj
            ious_obj[name] = compute_iou(pb, gb)
            areas_obj[name] = int(gb.sum())
        scored = [x for x in ious_obj.values() if not np.isnan(x)]
        miou_obj = float(np.mean(scored)) if scored else float("nan")
        wn, wd = 0.0, 0
        for n, iou in ious_obj.items():
            if np.isnan(iou): continue
            wn += iou * areas_obj[n]; wd += areas_obj[n]
        miou_obj_w = (wn / wd) if wd else float("nan")
        obj_view_mious.append(miou_obj); obj_view_mious_w.append(miou_obj_w)

        # ----- Part mIoU -----
        valid_part = (pred_part >= 0) & (gt_part_idx[v] >= 0)
        ious_part, areas_part = {}, {}
        for ci, name in enumerate(visible_part):
            gb = (gt_part_idx[v] == ci) & valid_part
            pb = (pred_part == ci) & valid_part
            ious_part[name] = compute_iou(pb, gb)
            areas_part[name] = int(gb.sum())
        scored = [x for x in ious_part.values() if not np.isnan(x)]
        miou_part = float(np.mean(scored)) if scored else float("nan")
        wn, wd = 0.0, 0
        for n, iou in ious_part.items():
            if np.isnan(iou): continue
            wn += iou * areas_part[n]; wd += areas_part[n]
        miou_part_w = (wn / wd) if wd else float("nan")
        part_view_mious.append(miou_part); part_view_mious_w.append(miou_part_w)

        # ----- Joint (obj, part) mIoU -----
        valid_joint = valid_obj & valid_part
        # tuples that appear in GT for this view
        gt_pairs = set()
        gt_pair_codes = gt_obj_idx[v].astype(np.int64) * 10000 + gt_part_idx[v]
        # exclude codes where either component is invalid
        pair_valid_gt = (gt_obj_idx[v] >= 0) & (gt_part_idx[v] >= 0)
        unique_gt_codes = np.unique(gt_pair_codes[pair_valid_gt])
        ious_joint = {}
        for code in unique_gt_codes:
            oc = int(code // 10000); pc = int(code % 10000)
            gb = (gt_pair_codes == code) & valid_joint
            pred_code = pred_obj.astype(np.int64) * 10000 + pred_part
            pb = (pred_code == code) & valid_joint
            ious_joint[f"{visible_obj[oc]}|{visible_part[pc]}"] = compute_iou(
                pb, gb)
        scored = [x for x in ious_joint.values() if not np.isnan(x)]
        miou_joint = float(np.mean(scored)) if scored else float("nan")
        joint_view_mious.append(miou_joint)

        # ----- Hierarchy-consistency rate -----
        # Of pixels with both pred_obj>=0 and pred_part>=0, what fraction
        # form a (obj_class, part_class) tuple that actually exists in the
        # scene's annotation graph?
        pred_pair_valid = (pred_obj >= 0) & (pred_part >= 0)
        n_pred = int(pred_pair_valid.sum())
        if n_pred:
            consistent = np.zeros((H, W), dtype=bool)
            for (oc, pc) in valid_tuples:
                if oc in obj_name_to_idx and pc in part_name_to_idx:
                    consistent |= ((pred_obj == obj_name_to_idx[oc]) &
                                   (pred_part == part_name_to_idx[pc]))
            consistent &= pred_pair_valid
            cons_rate = int(consistent.sum()) / n_pred
        else:
            cons_rate = float("nan")
        consistency_rates.append(cons_rate)

        results[f"view_{v}"] = {
            "object": {"per_class": ious_obj, "miou": miou_obj,
                       "miou_weighted": miou_obj_w},
            "part":   {"per_class": ious_part, "miou": miou_part,
                       "miou_weighted": miou_part_w},
            "joint":  {"per_pair": ious_joint, "miou": miou_joint,
                       "n_unique_pairs": len(unique_gt_codes)},
            "hierarchy_consistency": cons_rate,
        }

        print(f"\nview {v}:")
        print(f"  object  mIoU = {miou_obj:.4f}  (weighted {miou_obj_w:.4f})")
        print(f"  part    mIoU = {miou_part:.4f}  (weighted {miou_part_w:.4f})")
        print(f"  joint   mIoU = {miou_joint:.4f}  over {len(unique_gt_codes)} pairs")
        print(f"  hierarchy consistency = {cons_rate:.4f}")

        # Overlays
        if rgb_arr is not None:
            rgb = (rgb_arr[v] * 255).clip(0, 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for tag, pred, n in (("obj", pred_obj, len(visible_obj)),
                                 ("part", pred_part, len(visible_part))):
                col = colorize(pred, n)
                blend = cv2.addWeighted(rgb_bgr, 1 - args.alpha, col,
                                        args.alpha, 0)
                cv2.imwrite(str(args.out_dir / f"hier_pred_{tag}_v{v}.png"), blend)

    def _mean(xs):
        xs = [x for x in xs if not np.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    scene = {
        "object_miou":         _mean(obj_view_mious),
        "object_miou_weighted": _mean(obj_view_mious_w),
        "part_miou":           _mean(part_view_mious),
        "part_miou_weighted":  _mean(part_view_mious_w),
        "joint_miou":          _mean(joint_view_mious),
        "hierarchy_consistency": _mean(consistency_rates),
        "n_views": V,
        "n_visible_obj_classes": len(visible_obj),
        "n_visible_part_classes": len(visible_part),
    }
    results["scene"] = scene
    print("\n[hier] scene summary:")
    for k, v in scene.items():
        print(f"  {k}: {v}")

    with open(args.out_dir / "hier_iou.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: None)
    print(f"\n[hier] wrote {args.out_dir / 'hier_iou.json'}")


if __name__ == "__main__":
    main()
