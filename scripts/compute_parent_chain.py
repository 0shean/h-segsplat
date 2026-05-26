#!/usr/bin/env python3
"""
Compute v2 containment parent dict for H-SegSplat (PROJECT_PLAN.md §6.5.3).

For each view in the scene, computes three containment maps directly from the
full-resolution binary masks:

    6 -> 3      level-6 mask m's level-3 ancestor (argmax |m ∩ p| / |m|, accept if >= 0.9)
    6 -> 1      level-6 mask m's level-1 ancestor (computed directly, NOT chained)
    3 -> 1      level-3 mask m's level-1 ancestor

Output: <output_dir>/per_mask/parents.json with the schema described in §6.5.3,
keyed by the same global_id ordering that build_hsegsplat_inputs.py uses for
mask_features.npy / mask_directory.json.

Run AFTER build_hsegsplat_inputs.py so the mask_directory.json files exist (we
key parents by those global_ids for the server-side join).

Usage:
    python scripts/compute_parent_chain.py \
        --scene_dir  data/<scene> \
        --output_dir data/<scene>
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CONTAINMENT_THRESHOLD = 0.9


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", type=str, required=True,
                   help="Same --scene_dir passed to build_hsegsplat_inputs.py")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Same --output_dir passed to build_hsegsplat_inputs.py")
    p.add_argument("--threshold", type=float, default=CONTAINMENT_THRESHOLD,
                   help="Containment threshold |m ∩ p| / |m| (default 0.9).")
    return p.parse_args()


def load_level_masks_for_frame(mask_dir: Path):
    """Returns:
        masks: (N, H, W) bool — in metadata.json order
        Empty (0, H, W) array if no masks.
    """
    meta_path = mask_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    with open(meta_path) as f:
        meta = json.load(f)

    out = []
    for entry in meta:
        m_path = mask_dir / entry["mask_file"]
        m = cv2.imread(str(m_path), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(m_path)
        if m.ndim == 3:
            m = m[..., 0]
        out.append(m > 0)
    if not out:
        return np.zeros((0, 0, 0), dtype=bool)
    return np.stack(out, axis=0)


def best_containment_parent(child_masks: np.ndarray,
                            parent_masks: np.ndarray,
                            threshold: float) -> list:
    """For each child mask, return the parent mask's local index (within the
    parent array) maximizing |child ∩ parent| / |child|, or None if the best
    score is below threshold or there are no parents.

    Implementation: flatten and use boolean matmul. Cheap at our scale
    (N ~ 100 per level per view, HxW ~ 600,000 pixels).
    """
    N_c = child_masks.shape[0]
    N_p = parent_masks.shape[0]
    if N_c == 0:
        return []
    if N_p == 0:
        return [None] * N_c

    # Flatten to (N, H*W) for fast matmul.
    c_flat = child_masks.reshape(N_c, -1).astype(np.uint32)
    p_flat = parent_masks.reshape(N_p, -1).astype(np.uint32)

    # Intersection sizes: (N_c, N_p). uint32 matmul is fine for ~600k pixels.
    inter = c_flat @ p_flat.T  # (N_c, N_p)
    child_areas = c_flat.sum(axis=1).clip(min=1)  # (N_c,) — clip avoids div-by-0 on empties

    # |c ∩ p| / |c|, broadcast over parents.
    ratios = inter.astype(np.float32) / child_areas[:, None].astype(np.float32)

    best_parent = ratios.argmax(axis=1)
    best_score = ratios[np.arange(N_c), best_parent]

    out: list = []
    for i in range(N_c):
        if best_score[i] >= threshold:
            out.append(int(best_parent[i]))
        else:
            out.append(None)
    return out


def build_global_id_lookup(output_dir: Path, level: int) -> dict:
    """Read mask_directory.json for a level; return {(view_idx, local_mask_id) -> global_id}."""
    dir_path = output_dir / f"level_{level}" / "mask_directory.json"
    with open(dir_path) as f:
        entries = json.load(f)
    lut = {}
    for entry in entries:
        lut[(int(entry["view_idx"]), int(entry["local_mask_id"]))] = int(entry["global_id"])
    return lut


def main():
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    threshold = float(args.threshold)

    # Read frame order from build_hsegsplat_inputs.py's top-level meta.
    meta_path = output_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} missing — run build_hsegsplat_inputs.py first."
        )
    with open(meta_path) as f:
        top_meta = json.load(f)
    frame_order = top_meta["frame_order"]
    levels = [int(l) for l in top_meta["levels"]]
    needed = {1, 3, 6}
    if not needed.issubset(set(levels)):
        raise RuntimeError(f"Scene meta levels {levels} missing some of {needed}")

    print(f"Scene: {top_meta['scene_key']}  V={len(frame_order)}  threshold={threshold}")

    # Build global_id lookups once per level.
    gid_lut = {lvl: build_global_id_lookup(output_dir, lvl) for lvl in (1, 3, 6)}

    parents_out = {"level_6": {}, "level_3": {}}

    for v_idx, fname in enumerate(frame_order):
        stem = Path(fname).stem
        masks = {}
        for lvl in (1, 3, 6):
            mdir = scene_dir / f"masks_lvl_{lvl}" / stem
            masks[lvl] = load_level_masks_for_frame(mdir)
        H_W = None
        for lvl in (1, 3, 6):
            if masks[lvl].shape[0] > 0:
                H_W = masks[lvl].shape[1:]
                break
        if H_W is None:
            print(f"  [view {v_idx}] {fname}: no masks at any level — skipping")
            continue
        # Cross-level shape consistency check.
        for lvl in (1, 3, 6):
            if masks[lvl].shape[0] > 0 and masks[lvl].shape[1:] != H_W:
                raise ValueError(
                    f"shape mismatch in {fname}: lvl {lvl} masks are {masks[lvl].shape[1:]}, "
                    f"expected {H_W}"
                )

        # Containment maps for this view.
        parent_6_to_3 = best_containment_parent(masks[6], masks[3], threshold)
        parent_6_to_1 = best_containment_parent(masks[6], masks[1], threshold)
        parent_3_to_1 = best_containment_parent(masks[3], masks[1], threshold)

        n_link_6_3 = sum(1 for p in parent_6_to_3 if p is not None)
        n_link_6_1 = sum(1 for p in parent_6_to_1 if p is not None)
        n_link_3_1 = sum(1 for p in parent_3_to_1 if p is not None)
        print(f"  [view {v_idx}] {fname}: "
              f"N1={masks[1].shape[0]} N3={masks[3].shape[0]} N6={masks[6].shape[0]}  "
              f"links 6->3: {n_link_6_3}/{len(parent_6_to_3)}  "
              f"6->1: {n_link_6_1}/{len(parent_6_to_1)}  "
              f"3->1: {n_link_3_1}/{len(parent_3_to_1)}")

        # Stamp into parents_out using global ids.
        for local_id_6, p3_local in enumerate(parent_6_to_3):
            gid_6 = gid_lut[6][(v_idx, local_id_6)]
            entry = parents_out["level_6"].setdefault(str(gid_6),
                                                      {"level_3": None, "level_1": None})
            if p3_local is not None:
                entry["level_3"] = gid_lut[3][(v_idx, p3_local)]
        for local_id_6, p1_local in enumerate(parent_6_to_1):
            gid_6 = gid_lut[6][(v_idx, local_id_6)]
            entry = parents_out["level_6"].setdefault(str(gid_6),
                                                      {"level_3": None, "level_1": None})
            if p1_local is not None:
                entry["level_1"] = gid_lut[1][(v_idx, p1_local)]
        for local_id_3, p1_local in enumerate(parent_3_to_1):
            gid_3 = gid_lut[3][(v_idx, local_id_3)]
            entry = parents_out["level_3"].setdefault(str(gid_3), {"level_1": None})
            if p1_local is not None:
                entry["level_1"] = gid_lut[1][(v_idx, p1_local)]

    out_dir = output_dir / "per_mask"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parents.json"
    with open(out_path, "w") as f:
        json.dump({
            "scene_key": top_meta["scene_key"],
            "threshold": threshold,
            "parents": parents_out,
            "schema_version": "v2",
            "notes": (
                "Per-view containment computed at full resolution. "
                "6->1 is computed directly, NOT as transitive closure of 6->3 -> 3->1. "
                "global_ids align with level_<L>/mask_features.npy row order and "
                "mask_directory.json entries."
            ),
        }, f, indent=2)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
