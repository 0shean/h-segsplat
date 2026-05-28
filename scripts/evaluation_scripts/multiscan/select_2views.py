#!/usr/bin/env python3
"""
Select 2 context views from a MultiScan ingested scene by projected-vertex overlap.

Reads the transforms.json produced by multiscan_ingest.py + the matching PLY.
For each pair of frames, computes:
    - overlap_fwd = |verts_A_visible ∩ verts_B_visible| / |verts_A_visible|
    - baseline = ||cam_A - cam_B|| (translation distance)

Picks the pair that has ≥ MIN_OVERLAP in BOTH directions AND maximizes
overlap * sigmoid((baseline - BASELINE_TARGET) ) ... actually we just want
"≥60% overlap" and "non-trivial baseline" -- among all qualifying pairs, pick the one
with baseline closest to a target (default 0.4 m) so the two views aren't almost
identical, which gives DepthSplat depth cues.

Outputs:
    <ingested_dir>/view_selection.json    {"context": [i, j], "target": [i, j]}
    <ingested_dir>/_verify/view_pair_<i>_<j>.png  (side-by-side preview)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from verify_projection import read_ply_vertices_with_color, project  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ingested_dir", type=Path, required=True)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--sample", type=int, default=10000)
    p.add_argument("--min_overlap", type=float, default=0.6)
    p.add_argument("--baseline_target", type=float, default=0.4,
                   help="Preferred translation distance between the two views (meters).")
    p.add_argument("--max_baseline", type=float, default=1.5)
    p.add_argument("--candidate_stride", type=int, default=1,
                   help="Subsample candidate frames to speed up pair search.")
    return p.parse_args()


def main():
    args = parse_args()
    ingested_dir = args.ingested_dir
    with open(ingested_dir / "dslr" / "nerfstudio" / "transforms.json") as f:
        transforms = json.load(f)
    W, H = transforms["w"], transforms["h"]
    fx, fy = transforms["fl_x"], transforms["fl_y"]
    cx, cy = transforms["cx"], transforms["cy"]
    frames = transforms["frames"]

    print(f"Loading PLY ...", end="", flush=True)
    xyz, _ = read_ply_vertices_with_color(args.ply)
    if len(xyz) > args.sample:
        idx = np.linspace(0, len(xyz)-1, args.sample).astype(np.int64)
        xyz = xyz[idx]
    print(f" {len(xyz)} sampled verts")

    cand = list(range(0, len(frames), args.candidate_stride))
    print(f"Computing per-frame visibility for {len(cand)} candidate frames ...")

    visibility = []  # list of bool arrays (sample,)
    centers = []    # list of cam centers in world
    for fi in cand:
        c2w = np.array(frames[fi]["transform_matrix"], dtype=np.float64)
        _, _, _, inb = project(xyz, c2w, fx, fy, cx, cy, W, H,
                                opencv_convention=False)
        visibility.append(inb)
        centers.append(c2w[:3, 3])
    visibility = np.stack(visibility, axis=0)
    centers = np.stack(centers, axis=0)
    print(f"  per-frame |visible|: mean={visibility.sum(1).mean():.0f}  "
          f"min={visibility.sum(1).min()}  max={visibility.sum(1).max()}")

    n = len(cand)
    print(f"Scoring {n*(n-1)//2} pairs ...")
    best_pairs = []
    for ai in range(n):
        for bi in range(ai+1, n):
            va, vb = visibility[ai], visibility[bi]
            inter = (va & vb).sum()
            if inter == 0:
                continue
            ov_a = inter / max(va.sum(), 1)
            ov_b = inter / max(vb.sum(), 1)
            min_ov = min(ov_a, ov_b)
            if min_ov < args.min_overlap:
                continue
            baseline = float(np.linalg.norm(centers[ai] - centers[bi]))
            if baseline > args.max_baseline:
                continue
            # score: how close to baseline_target, with a small penalty for low overlap
            score = -abs(baseline - args.baseline_target) + 0.5 * min_ov
            best_pairs.append((score, cand[ai], cand[bi], min_ov, baseline,
                               int(va.sum()), int(vb.sum()), int(inter)))
    if not best_pairs:
        # relax overlap progressively
        for relax in (0.5, 0.4, 0.3):
            for ai in range(n):
                for bi in range(ai+1, n):
                    va, vb = visibility[ai], visibility[bi]
                    inter = (va & vb).sum()
                    if inter == 0: continue
                    ov_a = inter / max(va.sum(),1)
                    ov_b = inter / max(vb.sum(),1)
                    min_ov = min(ov_a, ov_b)
                    if min_ov < relax: continue
                    baseline = float(np.linalg.norm(centers[ai]-centers[bi]))
                    if baseline > args.max_baseline: continue
                    score = -abs(baseline - args.baseline_target) + 0.5*min_ov
                    best_pairs.append((score, cand[ai], cand[bi], min_ov, baseline,
                                       int(va.sum()), int(vb.sum()), int(inter)))
            if best_pairs:
                print(f"  No pair at {args.min_overlap:.2f}, relaxed to {relax:.2f}")
                break
    if not best_pairs:
        raise RuntimeError("No suitable pair found even at 30% overlap.")
    best_pairs.sort(key=lambda x: -x[0])

    print("Top 10 candidate pairs:")
    for sc, ai, bi, mo, bl, va, vb, it in best_pairs[:10]:
        print(f"  ({ai:3d}, {bi:3d})  min_overlap={mo:.2f}  baseline={bl:.3f}m  "
              f"|va|={va} |vb|={vb} |inter|={it}  score={sc:.3f}")

    _, ai, bi, mo, bl, *_ = best_pairs[0]
    print(f"Chosen pair: ({ai}, {bi})  overlap={mo:.2f}  baseline={bl:.3f}m")

    out = {
        "context": [ai, bi],
        "target": [ai, bi],
        "overlap": mo,
        "baseline_m": bl,
        "file_paths": [frames[ai]["file_path"], frames[bi]["file_path"]],
    }
    out_path = ingested_dir / "view_selection.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")

    # side-by-side preview
    images_dir = ingested_dir / "dslr" / "resized_images"
    img_a = cv2.imread(str(images_dir / frames[ai]["file_path"]))
    img_b = cv2.imread(str(images_dir / frames[bi]["file_path"]))
    pad = 4
    pair = np.zeros((H, 2*W + pad, 3), dtype=np.uint8)
    pair[:, :W] = img_a; pair[:, W+pad:] = img_b
    cv2.putText(pair, f"frame {ai}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)
    cv2.putText(pair, f"frame {bi}  ov={mo:.2f} base={bl:.2f}m",
                (W+pad+8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    (ingested_dir / "_verify").mkdir(exist_ok=True)
    cv2.imwrite(str(ingested_dir / "_verify" / f"view_pair_{ai}_{bi}.png"), pair)


if __name__ == "__main__":
    main()
