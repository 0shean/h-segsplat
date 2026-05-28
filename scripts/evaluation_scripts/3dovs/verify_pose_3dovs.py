#!/usr/bin/env python3
"""
Project COLMAP sparse 3D points into the chosen 2-context-view RGBs to
verify the pose convention we wrote into transforms.json is correct.

Pose convention check:
- transforms.json's transform_matrix is in BLENDER convention (cam looks -Z, +X right, +Y up).
- build_hsegsplat_inputs.py applies diag(1,-1,-1,1) to convert Blender -> OpenCV.
- For verify: we have c2w_blender; to project, convert to OpenCV via diag(1,-1,-1,1)
  multiplication on the right; then standard pinhole projection.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_points3D_binary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ingested_dir", type=Path, required=True,
                   help="data/3D-OVS/ingested/<scene>")
    p.add_argument("--scene_dir", type=Path, required=True,
                   help="Original data/3D-OVS/<scene>")
    p.add_argument("--n_sample", type=int, default=20000)
    return p.parse_args()


def main():
    args = parse_args()
    pts = read_points3D_binary(args.scene_dir / "sparse" / "0" / "points3D.bin")
    xyz = np.stack([p.xyz for p in pts.values()], axis=0)
    rgb = np.stack([p.rgb for p in pts.values()], axis=0)
    if len(xyz) > args.n_sample:
        idx = np.linspace(0, len(xyz)-1, args.n_sample).astype(np.int64)
        xyz = xyz[idx]; rgb = rgb[idx]
    print(f"Loaded {len(xyz)} sparse points; bbox: min={xyz.min(0)} max={xyz.max(0)}")

    with open(args.ingested_dir / "dslr" / "nerfstudio" / "transforms.json") as f:
        t = json.load(f)
    W, H = t["w"], t["h"]
    fx, fy, cx, cy = t["fl_x"], t["fl_y"], t["cx"], t["cy"]

    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # Blender <-> OpenCV (self-inverse)
    out_dir = args.ingested_dir / "_verify"
    out_dir.mkdir(exist_ok=True)
    for fr in t["frames"]:
        c2w_b = np.array(fr["transform_matrix"], dtype=np.float64)
        c2w_o = c2w_b @ flip
        w2c = np.linalg.inv(c2w_o)
        cam = (w2c @ np.concatenate([xyz, np.ones((len(xyz), 1))], -1).T).T[:, :3]
        z = cam[:, 2]
        in_front = z > 0.01
        u = (fx * cam[:, 0] + cx * cam[:, 2]) / (cam[:, 2] + 1e-12)
        v = (fy * cam[:, 1] + cy * cam[:, 2]) / (cam[:, 2] + 1e-12)
        inb = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        img_path = args.ingested_dir / "dslr" / "resized_images" / fr["file_path"]
        img = cv2.imread(str(img_path))
        overlay = img.copy()
        for uu, vv, color in zip(u[inb], v[inb], rgb[inb]):
            cv2.circle(overlay, (int(uu), int(vv)), 1, [int(c) for c in color[::-1]], -1)
        out = cv2.addWeighted(img, 0.4, overlay, 0.6, 0.0)
        out_path = out_dir / f"projection_{Path(fr['file_path']).stem}.png"
        cv2.imwrite(str(out_path), out)
        print(f"  {fr['file_path']}: in_front={(z>0).sum()}/{len(z)}, "
              f"in_bounds={inb.sum()}/{len(u)}  ->  {out_path}")


if __name__ == "__main__":
    main()
