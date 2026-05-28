#!/usr/bin/env python3
"""
Re-project PLY vertices into selected RGB frames to verify the
(c2w, intrinsics) we wrote in transforms.json are correct.

Reads:
    --ingested_dir : output of multiscan_ingest.py (has dslr/{nerfstudio,resized_images}/)
    --ply          : path to scene .ply (e.g. multiscan_test_plys_only/scene_*.ply)

For each --frames frame_idx, dots the projected vertices over the RGB and writes
a PNG to <ingested_dir>/_verify/projection_<frame_idx>.png. Visual check: dots
should fall on geometry, not on empty space.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ingested_dir", type=Path, required=True)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--frames", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20, 40, 80, 160],
                   help="Indices into transforms.json['frames'] to verify.")
    p.add_argument("--sample", type=int, default=20000,
                   help="Sub-sample vertices for plotting (uniform).")
    return p.parse_args()


def read_ply_vertices_with_color(ply_path: Path):
    """Minimal PLY parser for the MultiScan format:
       binary_little_endian; vertex properties: x,y,z (double), nx,ny,nz (double),
       r,g,b,a (uchar). Returns Nx3 float64 xyz and Nx3 uint8 rgb."""
    with open(ply_path, "rb") as f:
        # Header
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            if line.strip() == b"end_header":
                break
        # Now read vertex data. We know layout from the audit.
        # Locate "element vertex N"
        n_verts = None
        for hl in header_bytes.split(b"\n"):
            if hl.startswith(b"element vertex"):
                n_verts = int(hl.split()[-1])
                break
        if n_verts is None:
            raise RuntimeError("Missing element vertex header")
        # Per-vertex size: 3*8 (xyz) + 3*8 (nxyz) + 4*1 (rgba) = 52
        dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                          ("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8"),
                          ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")])
        raw = np.frombuffer(f.read(n_verts * dtype.itemsize), dtype=dtype)
        xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=-1).astype(np.float64)
        rgb = np.stack([raw["r"], raw["g"], raw["b"]], axis=-1)
    return xyz, rgb


def project(xyz_world: np.ndarray, c2w: np.ndarray, fx, fy, cx, cy, W, H,
            opencv_convention: bool):
    """Project Nx3 world points into a camera. If opencv_convention=True, expects
    c2w to be in OpenCV (cam looks +Z). If False (Blender / GL), flip y,z of cam axes."""
    # Build w2c
    w2c = np.linalg.inv(c2w)
    pts_h = np.concatenate([xyz_world, np.ones((len(xyz_world), 1))], axis=-1)  # N,4
    cam = (w2c @ pts_h.T).T[:, :3]  # N,3 in camera frame
    if not opencv_convention:
        # Blender / GL camera looks down -z. To project, flip y and z so cam looks +z.
        cam = cam * np.array([1.0, -1.0, -1.0])
    z = cam[:, 2]
    in_front = z > 0.01
    u = (fx * cam[:, 0] + cx * cam[:, 2]) / (cam[:, 2] + 1e-12)
    v = (fy * cam[:, 1] + cy * cam[:, 2]) / (cam[:, 2] + 1e-12)
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & in_front
    return u, v, z, in_bounds


def main():
    args = parse_args()
    ingested_dir = args.ingested_dir
    with open(ingested_dir / "dslr" / "nerfstudio" / "transforms.json") as f:
        transforms = json.load(f)
    W, H = transforms["w"], transforms["h"]
    fx, fy = transforms["fl_x"], transforms["fl_y"]
    cx, cy = transforms["cx"], transforms["cy"]
    frames = transforms["frames"]
    print(f"Loaded {len(frames)} frames at {W}x{H}, fx={fx:.2f} cx={cx:.2f}")

    print(f"Reading PLY ... ", end="", flush=True)
    xyz, rgb = read_ply_vertices_with_color(args.ply)
    print(f"{len(xyz)} verts, bbox min={xyz.min(0)}, max={xyz.max(0)}")

    if len(xyz) > args.sample:
        idx = np.linspace(0, len(xyz)-1, args.sample).astype(np.int64)
        xyz = xyz[idx]; rgb = rgb[idx]

    out_dir = ingested_dir / "_verify"
    out_dir.mkdir(exist_ok=True)
    images_dir = ingested_dir / "dslr" / "resized_images"

    for fi in args.frames:
        if fi >= len(frames):
            continue
        fr = frames[fi]
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        img_path = images_dir / fr["file_path"]
        img = cv2.imread(str(img_path))
        # we expect Blender convention (cam looks -z) in transforms.json
        u, v, z, inb = project(xyz, c2w, fx, fy, cx, cy, W, H, opencv_convention=False)
        print(f"  frame {fi:3d} ({Path(fr['file_path']).name}): "
              f"in_front={(z>0).sum()}/{len(z)}, in_bounds={inb.sum()}/{len(u)}")
        overlay = img.copy()
        for uu, vv, ii in zip(u[inb], v[inb], np.where(inb)[0]):
            color = tuple(int(c) for c in rgb[ii][::-1])  # BGR
            cv2.circle(overlay, (int(uu), int(vv)), 1, color, -1)
        # blend
        out = cv2.addWeighted(img, 0.4, overlay, 0.6, 0.0)
        out_path = out_dir / f"projection_f{fi:03d}.png"
        cv2.imwrite(str(out_path), out)
    print(f"Wrote verification overlays -> {out_dir}/")


if __name__ == "__main__":
    main()
