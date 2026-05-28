#!/usr/bin/env python3
"""
3D-OVS mIoU eval driver for H-SegSplat outputs.

Inputs per scene (downloaded from Colab):
    --colab_root/<scene>/rendered_feature_map_targets_lvl1.npy   (T, H, W, M_1+1)
    --colab_root/<scene>/gaussians.pt                            (for the level-1 bank)
    --ingested_root/<scene>/target_views.json                    (classes list)
    --ingested_root/<scene>/gt_masks/<view_id>/<class>.png       (binary GT)

What it does:
    1. For each scene:
        - Load rendered_feature_map_targets_lvl1.npy (level-1 one-hots at each target view).
        - Load the level-1 bank from gaussians.pt → recover SigLIP feature map per pixel.
        - Encode each class name + canonical phrases via SigLIP.
        - Compute LERF relevancy per pixel per class.
        - Argmax over classes → predicted class map per target view.
        - Compute per-class IoU vs GT masks; average → per-scene mIoU.
    2. Report per-scene + overall mIoU in a table matching SegSplat / N2F2 Table 3.
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
from hsegsplat_offline_state import SigLIPTextEncoder, lerf_per_mask  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--colab_root", type=Path, required=True,
                   help="Folder containing <scene>/{gaussians.pt, "
                        "rendered_feature_map_targets_lvl{1,3,6}.npy}.")
    p.add_argument("--ingested_root", type=Path, required=True,
                   help="data/3D-OVS/ingested/  (has target_views.json + gt_masks per scene)")
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--levels", nargs="+", type=int, default=[1, 3],
                   help="Granularity levels to score. Multi-level uses the per-pixel MAX "
                        "of per-class relevancy across levels (richer features cover more "
                        "of the class vocab). Default [1, 3] since level 1 alone is too "
                        "coarse for 3D-OVS object classes; level 6 is too fine.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="LERF relevancy threshold for the per-class binary mask "
                        "(SegSplat / LangSplat convention).")
    p.add_argument("--mass_threshold", type=float, default=0.05,
                   help="Drop pixels whose foreground-channel mass is below this "
                        "(low-confidence background). Only used in 'argmax' mode.")
    p.add_argument("--mode", choices=["per_class_threshold", "argmax", "oracle"],
                   default="per_class_threshold",
                   help="per_class_threshold (DEFAULT, matches SegSplat/LangSplat/N2F2): "
                        "for each class, compute LERF relevancy, threshold at --threshold, "
                        "compare to GT. Classes may overlap or be absent. "
                        "argmax: assign each valid pixel the highest-relevancy class. "
                        "oracle: per pixel per class predict positive if ANY level "
                        "fires above --threshold (upper bound for the per-level-max design).")
    p.add_argument("--tau", type=float, default=100.0,
                   help="LERF temperature. Higher = more saturated (binary). "
                        "100 = LERF default, 10 = LangSplat default.")
    p.add_argument("--normalize", choices=["none", "median", "minmax"], default="none",
                   help="Per-class relevancy normalization before thresholding. "
                        "median: subtract per-image median (LangSplat-style). "
                        "minmax: rescale to [0, 1] per image.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out_dir", type=Path, default=Path("data/3D-OVS/eval_results"))
    p.add_argument("--target_view", nargs="+", default=None,
                   help="Restrict eval to the given target view IDs. Pass either a "
                        "single view ID applied to all scenes ('--target_view 09') "
                        "or per-scene 'scene:view' pairs "
                        "('--target_view bed:00 lawn:09 sofa:23'). Omit to use all "
                        "labelled views in target_views.json (default; matches "
                        "SegSplat / 3D-OVS protocol).")
    return p.parse_args()


def parse_target_view_arg(arg, scenes):
    """Returns dict {scene -> set(view_id) or None}. None means 'use all views'."""
    if not arg:
        return {s: None for s in scenes}
    pairs = [a for a in arg if ":" in a]
    if pairs:
        out = {s: None for s in scenes}
        for p in arg:
            if ":" not in p:
                raise ValueError(
                    f"--target_view: mix of bare and scene:view forms not supported ({p})")
            scene, view = p.split(":", 1)
            if out.get(scene) is None:
                out[scene] = set()
            out[scene].add(view)
        return out
    s = set(arg)
    return {sc: s for sc in scenes}


def compute_iou(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    if union == 0:
        return float("nan")  # no GT, no pred — skip
    return float(inter) / float(union)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("[3dovs] loading SigLIP ...")
    encoder = SigLIPTextEncoder(device=args.device)

    view_filter = parse_target_view_arg(args.target_view, args.scenes)

    overall = {}  # scene -> mIoU
    per_class_results = {}  # scene -> {class: iou}
    for scene in args.scenes:
        print(f"\n[3dovs] === {scene} ===")
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
        with open(tv_path) as f:
            tv = json.load(f)
        classes = tv["classes"]
        targets = tv["targets"]

        # Filter to the requested target view(s) for this scene, if any.
        # The .npy feature maps were written in target order from target_views.json,
        # so we need to remember each kept target's ORIGINAL index for indexing.
        target_iter = list(enumerate(targets))  # [(orig_idx, tgt)] over original order
        wanted = view_filter.get(scene)
        if wanted is not None:
            target_iter = [(i, t) for (i, t) in target_iter if t["view_id"] in wanted]
            if not target_iter:
                avail = [t["view_id"] for t in targets]
                print(f"  [skip] none of requested views {sorted(wanted)} "
                      f"are present. Available: {avail}")
                continue
            print(f"  restricting to {len(target_iter)}/{len(targets)} target view(s): "
                  f"{[t['view_id'] for _, t in target_iter]}")

        # Load per-level banks (drop bg row, L2-normalize).
        g = torch.load(g_pt, map_location="cpu", weights_only=False)
        banks = {}
        for lvl in args.levels:
            b = g["banks"][lvl].cpu().numpy().astype(np.float32)[1:]
            b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
            banks[lvl] = b
        feat_per_lvl = {lvl: np.load(feat_paths[lvl]) for lvl in args.levels}
        T, H, W, _ = feat_per_lvl[args.levels[0]].shape
        print(f"  T={T} target views, H={H}, W={W}, levels={args.levels} "
              f"with M={ {lvl: banks[lvl].shape[0] for lvl in args.levels} }")

        # Pre-encode every class once.
        phi_canon = encoder.phi_canonical
        class_embs = {cls: encoder(cls) for cls in classes}

        per_class_per_v_iou = {c: [] for c in classes}
        # Per-level oracle bookkeeping: how often each level provides the "winning"
        # positive prediction (i.e. fires above threshold for a GT-positive pixel
        # when no other level does). Only filled when mode == "oracle".
        oracle_level_wins = {lvl: 0 for lvl in args.levels}
        oracle_level_unique_wins = {lvl: 0 for lvl in args.levels}
        for t_idx, tgt in target_iter:
            # Per-level per-class relevancy tensor, kept so oracle mode can
            # OR them at threshold time. For per_class_threshold and argmax we
            # max-reduce immediately into `combined_rels` to save memory.
            per_lvl_rels = {}            # lvl -> (C, H, W) relevancies in [0,1]
            combined_rels = None         # (C, H, W) running max across levels
            fg_mass_any = None           # (H, W) max foreground mass across levels
            for lvl in args.levels:
                E = feat_per_lvl[lvl][t_idx]                  # (H, W, M+1)
                E_real = E[..., 1:]                           # (H, W, M)
                fg_mass = E_real.sum(-1)                      # (H, W)
                if fg_mass_any is None:
                    fg_mass_any = fg_mass.copy()
                else:
                    fg_mass_any = np.maximum(fg_mass_any, fg_mass)

                # Dense SigLIP feature per pixel via cluster bank.
                F = E_real.reshape(-1, banks[lvl].shape[0]) @ banks[lvl]  # (HW, D)
                F = F / np.maximum(np.linalg.norm(F, axis=-1, keepdims=True), 1e-9)
                sim_c = F @ phi_canon.T                                   # (HW, 3)
                # Per-class LERF (relevancy formula from LERF / SegSplat eq. 4).
                class_rels = []
                for cls in classes:
                    sim_q = F @ class_embs[cls]
                    diff = args.tau * (sim_q[:, None] - sim_c)
                    pair = 1.0 / (1.0 + np.exp(-diff))
                    class_rels.append(pair.min(axis=1))
                class_rels = np.stack(class_rels, axis=0).reshape(-1, H, W)
                per_lvl_rels[lvl] = class_rels
                if combined_rels is None:
                    combined_rels = class_rels
                else:
                    combined_rels = np.maximum(combined_rels, class_rels)

            ious = {}
            if args.mode == "per_class_threshold":
                # SegSplat / LangSplat / N2F2 protocol: per-class binary mask.
                # Each class is thresholded independently; pixels may belong to
                # multiple classes or none.
                # Optional per-class normalization to compensate for SigLIP's
                # narrower cosine distribution vs CLIP.
                for c_i, cls in enumerate(classes):
                    gt_path = gt_root / tgt["view_id"] / f"{cls}.png"
                    if not gt_path.exists():
                        continue
                    gt_m = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                    gt_bool = gt_m > 0
                    rel = combined_rels[c_i]
                    if args.normalize == "median":
                        rel = rel - float(np.median(rel))
                    elif args.normalize == "minmax":
                        lo, hi = float(rel.min()), float(rel.max())
                        rel = (rel - lo) / max(hi - lo, 1e-9)
                    pred_bool = rel >= args.threshold
                    iou = compute_iou(pred_bool, gt_bool)
                    ious[cls] = iou
                    per_class_per_v_iou[cls].append(iou)
            elif args.mode == "argmax":
                # argmax: assign each valid pixel the single highest-relevancy class.
                valid = fg_mass_any >= args.mass_threshold
                # Weight by fg_mass to suppress empty regions.
                combined_rels_w = combined_rels * fg_mass_any[None, :, :]
                pred_cls = combined_rels_w.argmax(axis=0)
                pred_cls[~valid] = -1
                for c_i, cls in enumerate(classes):
                    gt_path = gt_root / tgt["view_id"] / f"{cls}.png"
                    if not gt_path.exists():
                        continue
                    gt_m = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                    gt_bool = gt_m > 0
                    pred_bool = (pred_cls == c_i)
                    iou = compute_iou(pred_bool, gt_bool)
                    ious[cls] = iou
                    per_class_per_v_iou[cls].append(iou)
            else:  # oracle
                # Per pixel per class: predict positive if ANY level fires above
                # --threshold. This is an upper bound — it says: "if we had a
                # perfect per-level selector, how good could the per-pixel-max
                # combiner be?"
                # We also count, per level, how often it provides the only
                # positive vote at a GT-positive pixel ("unique win"). That
                # diagnostic shows whether dropping a level would lose information.
                for c_i, cls in enumerate(classes):
                    gt_path = gt_root / tgt["view_id"] / f"{cls}.png"
                    if not gt_path.exists():
                        continue
                    gt_m = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                    gt_bool = gt_m > 0
                    # OR over per-level binary predictions.
                    per_lvl_bin = {lvl: per_lvl_rels[lvl][c_i] >= args.threshold
                                   for lvl in args.levels}
                    pred_bool = np.zeros((H, W), dtype=bool)
                    for lvl in args.levels:
                        pred_bool |= per_lvl_bin[lvl]
                    iou = compute_iou(pred_bool, gt_bool)
                    ious[cls] = iou
                    per_class_per_v_iou[cls].append(iou)
                    # Bookkeeping: per-level utilization at GT-positive pixels.
                    for lvl in args.levels:
                        wins = int((per_lvl_bin[lvl] & gt_bool).sum())
                        # "unique" = this level fires but the others don't.
                        others = np.zeros((H, W), dtype=bool)
                        for olvl in args.levels:
                            if olvl != lvl:
                                others |= per_lvl_bin[olvl]
                        unique = int((per_lvl_bin[lvl] & gt_bool & ~others).sum())
                        oracle_level_wins[lvl] += wins
                        oracle_level_unique_wins[lvl] += unique
            print(f"  view {tgt['view_id']}: " +
                  ", ".join(f"{c}={v:.3f}" if not np.isnan(v) else f"{c}=nan"
                            for c, v in ious.items()))

        # Average per class across the T target views.
        scene_class_iou = {c: float(np.nanmean(per_class_per_v_iou[c]))
                           if per_class_per_v_iou[c] else float("nan")
                           for c in classes}
        # Scene mIoU = mean over classes (NaN-safe).
        valid_ious = [v for v in scene_class_iou.values() if not np.isnan(v)]
        scene_miou = float(np.mean(valid_ious)) if valid_ious else float("nan")
        print(f"  -> per-class IoU: {scene_class_iou}")
        print(f"  -> scene mIoU = {scene_miou:.4f}")
        if args.mode == "oracle":
            total_wins = sum(oracle_level_wins.values()) or 1
            for lvl in args.levels:
                pct = 100.0 * oracle_level_wins[lvl] / total_wins
                upct = 100.0 * oracle_level_unique_wins[lvl] / max(1, oracle_level_wins[lvl])
                print(f"  -> lvl{lvl}: fires at {oracle_level_wins[lvl]} GT-pos pixels "
                      f"({pct:.1f}% of total), {upct:.1f}% of which are unique to this level")
        overall[scene] = scene_miou
        per_class_results[scene] = scene_class_iou

    # Summary
    print("\n[3dovs] SUMMARY")
    print(f"  {'scene':<14} {'mIoU':>8}")
    valid = [v for v in overall.values() if not np.isnan(v)]
    for s, v in overall.items():
        print(f"  {s:<14} {v:>8.4f}")
    if valid:
        print(f"  {'Overall':<14} {np.mean(valid):>8.4f}")

    # Tag the output file with the levels + mode + threshold so runs don't overwrite.
    levels_tag = "lvl" + "_".join(str(l) for l in sorted(args.levels))
    mode_tag = "argmax" if args.mode == "argmax" else f"thresh{args.threshold:.2f}"
    tau_tag = f"tau{int(args.tau)}"
    scene_tag = "_".join(args.scenes) if len(args.scenes) <= 4 else f"{len(args.scenes)}scenes"
    out_path = args.out_dir / f"miou__{scene_tag}__{levels_tag}__{mode_tag}__{tau_tag}.json"
    with open(out_path, "w") as f:
        json.dump({"levels": list(args.levels),
                   "mode": args.mode,
                   "threshold": args.threshold,
                   "mass_threshold": args.mass_threshold,
                   "per_scene_miou": overall,
                   "per_class": per_class_results,
                   "overall_miou": float(np.mean(valid)) if valid else None}, f, indent=2)
    print(f"\n[3dovs] wrote {out_path}")

    # Also append a one-line CSV summary across runs.
    csv_path = args.out_dir / "miou_summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a") as f:
        if write_header:
            f.write("levels,mode,threshold,tau,scene,scene_miou,overall_miou\n")
        for s, m in overall.items():
            f.write(f"{'+'.join(str(l) for l in sorted(args.levels))},"
                    f"{args.mode},{args.threshold},{args.tau},{s},{m},"
                    f"{float(np.mean(valid)) if valid else ''}\n")
    print(f"[3dovs] appended summary row(s) to {csv_path}")


if __name__ == "__main__":
    main()
