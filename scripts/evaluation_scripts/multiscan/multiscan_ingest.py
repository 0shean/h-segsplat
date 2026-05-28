#!/usr/bin/env python3
"""
MultiScan -> H-SegSplat ingest.

Takes a raw MultiScan scan folder (containing <scan>.mp4, <scan>.jsonl,
<scan>.json, <scan>.align.json) and produces the dslr/ layout that
second_stage/scripts/build_hsegsplat_inputs.py expects:

    <out>/dslr/nerfstudio/transforms.json     (Blender convention c2w)
    <out>/dslr/resized_images/<frame>.JPG     (target resolution)

The native MultiScan camera is portrait 1440x1920 with intrinsics
(fx=fy~=1588, cx~=944, cy~=726). The encoded video is rotated to
landscape 1920x1440. We rotate frames 90 CW back to landscape-aligned
orientation that matches the rotated intrinsics, then resize to
the H-SegSplat target W=960, H=640.

Intrinsics rotation (image rotated 90 CW, new W=oldH=1920, new H=oldW=1440):
    fx_new, fy_new = fy_old, fx_old
    cx_new = oldH - 1 - cy_old      (after 90 CW: x_new = oldH - 1 - y_old)
    cy_new = cx_old

Pose convention. The .jsonl ``transform`` field is a flat 16-list which when
reshape'd as (4,4) row-major has the translation in the LAST ROW
(row index 3), and determinant +1 in the top-left 3x3. That is the
column-major layout for a column-vector convention c2w matrix in
ARKit/right-handed-Y-up coordinates. So:

    c2w_arkit = np.array(transform).reshape(4,4).T

ARKit world is Y-up, looking down -Z (OpenGL/Blender convention).
H-SegSplat's downstream code applies diag(1,-1,-1,1) to map Blender ->
OpenCV. So we write transforms.json in *Blender* convention without
further flipping.

We also apply scan.align.json (a 4x4 alignment matrix) so the camera
poses live in the *same* world frame as the PLY mesh vertices.
align.json's coordinate_transform is a flat 16-list in the same
column-major layout. The relationship is:

    c2w_aligned = align @ c2w_arkit

(verified by reprojecting PLY vertices into the resulting view --- see
verify_projection.py).
"""

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


# H-SegSplat target resolution (matches existing test_scene).
TARGET_W = 960
TARGET_H = 640


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scan_dir", type=Path, required=True,
                   help="MultiScan scan folder e.g. data/Multiscan/scenes/scene_00005_00")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Output scene root (will create dslr/ subtree)")
    p.add_argument("--stride", type=int, default=60,
                   help="Take 1 frame every <stride> jsonl rows (60 = 1 fps).")
    p.add_argument("--max_frames", type=int, default=200,
                   help="Cap on number of frames extracted.")
    p.add_argument("--manual_pair", type=int, nargs=2, default=None,
                   metavar=("FRAME_A", "FRAME_B"),
                   help="Skip the stride + auto-select machinery; extract ONLY these two "
                        "jsonl frame indices as the context pair. Useful with "
                        "browse_frames.py for picking visually-good views by hand.")
    return p.parse_args()


def load_align(scan_dir: Path, stem: str) -> np.ndarray:
    """Load align.json. coordinate_transform is flat-16, column-major layout
    (same convention as the .jsonl transform).
    """
    path = scan_dir / f"{stem}.align.json"
    with open(path) as f:
        data = json.load(f)
    flat = data["coordinate_transform"]
    M = np.array(flat, dtype=np.float64).reshape(4, 4).T  # column-major -> standard
    return M


def parse_jsonl(scan_dir: Path, stem: str):
    """Yield (frame_index, c2w_arkit_4x4, intrinsics_dict, timestamp) for every row."""
    path = scan_dir / f"{stem}.jsonl"
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            T = np.array(d["transform"], dtype=np.float64).reshape(4, 4).T
            K_flat = d["intrinsics"]
            K = np.array(K_flat, dtype=np.float64).reshape(3, 3).T
            yield i, T, K, d["timestamp"]


def rotate_intrinsics_90cw(K: np.ndarray, old_W: int, old_H: int) -> tuple:
    """If we rotate an image 90 CW, new_W=old_H, new_H=old_W, and
       x_new = old_H - 1 - y_old
       y_new = x_old
       => fx_new=fy_old, fy_new=fx_old, cx_new=old_H-1-cy_old, cy_new=cx_old.
    """
    fx_old, fy_old = K[0, 0], K[1, 1]
    cx_old, cy_old = K[0, 2], K[1, 2]
    fx_new = fy_old
    fy_new = fx_old
    cx_new = (old_H - 1) - cy_old
    cy_new = cx_old
    new_W, new_H = old_H, old_W
    return fx_new, fy_new, cx_new, cy_new, new_W, new_H


def rotation_in_camera_90cw():
    """When we rotate the IMAGE 90 CW, the camera's +x axis (right) becomes
    -y (down) and +y axis (down) becomes +x (right) in the new image plane.
    The new camera frame R_new such that points_new = R_new @ points_old.
    R_new is rotation by +90 CW about z (which is the camera optical axis).
    Pixel (u,v) rotation: (u',v') = (oldH-1-v, u).
    The 3D camera-axis rotation that produces this pixel mapping is
    rotation about +z by -90 deg (i.e. 90 CW when looking along +z).
    => R = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    """
    return np.array([[0., 1., 0.],
                     [-1., 0., 0.],
                     [0., 0., 1.]], dtype=np.float64)


def arkit_to_blender(c2w_arkit: np.ndarray) -> np.ndarray:
    """ARKit world is Y-up, looking -Z. Blender world is also Y-up, looking
    -Z. So they're the same coordinate convention -- no change needed."""
    return c2w_arkit


def extract_frames(mp4_path: Path, frame_indices: list, tmp_dir: Path, stride: int):
    """Extract specified frame indices. Uses ffmpeg's modulo select if frames are
    a simple stride (cheap); falls back to per-frame select otherwise.
    The frame_indices list must be sorted in ascending order; the output file
    raw_NNNNN.jpg with NNNNN starting at 1 corresponds to frame_indices[NNNNN-1].
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    is_uniform = all(frame_indices[i] == i * stride for i in range(len(frame_indices)))
    if is_uniform:
        # Take every <stride>th frame, capped by -frames:v.
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(mp4_path),
            "-vf", f"select='not(mod(n\\,{stride}))'",
            "-vsync", "vfr",
            "-frames:v", str(len(frame_indices)),
            "-q:v", "2",
            str(tmp_dir / "raw_%05d.jpg"),
        ]
    else:
        # Per-frame equality is too long for ffmpeg's expression parser past
        # ~30 frames. Use between() over the union of singletons.
        sel_parts = [f"between(n\\,{i}\\,{i})" for i in frame_indices]
        # Split into chunks of <=30 to keep expression length sane.
        chunks = [sel_parts[i:i+30] for i in range(0, len(sel_parts), 30)]
        full = "+".join("(" + "+".join(c) + ")" for c in chunks)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(mp4_path),
            "-vf", f"select='{full}'",
            "-vsync", "vfr",
            "-q:v", "2",
            str(tmp_dir / "raw_%05d.jpg"),
        ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    scan_dir = args.scan_dir
    stem = scan_dir.name
    out_dir = args.out_dir
    images_out = out_dir / "dslr" / "resized_images"
    images_out.mkdir(parents=True, exist_ok=True)
    nerfstudio_out = out_dir / "dslr" / "nerfstudio"
    nerfstudio_out.mkdir(parents=True, exist_ok=True)

    # 1. Load align matrix
    align = load_align(scan_dir, stem)

    # 2. Walk jsonl: collect per-frame poses + intrinsics
    rows = list(parse_jsonl(scan_dir, stem))
    n_total = len(rows)
    if args.manual_pair is not None:
        sel_idx = list(args.manual_pair)
        for v in sel_idx:
            if not (0 <= v < n_total):
                raise ValueError(f"--manual_pair index {v} out of [0, {n_total})")
        print(f"[{stem}] MANUAL pair selected: jsonl indices {sel_idx}")
    else:
        sel_idx = list(range(0, n_total, args.stride))[: args.max_frames]
        print(f"[{stem}] jsonl has {n_total} frames; selecting {len(sel_idx)} with stride={args.stride}")

    # 3. Extract those frames as raw JPEGs in a tmp dir
    tmp_dir = out_dir / "_tmp_frames"
    if tmp_dir.exists():
        for p in tmp_dir.iterdir():
            p.unlink()
    extract_frames(scan_dir / f"{stem}.mp4", sel_idx, tmp_dir, args.stride)
    raw_jpegs = sorted(tmp_dir.glob("raw_*.jpg"))
    if len(raw_jpegs) != len(sel_idx):
        raise RuntimeError(f"ffmpeg extracted {len(raw_jpegs)} frames, expected {len(sel_idx)}")

    # 4. For each selected frame: rotate 90 CW + resize to target + write JPG;
    #    correspondingly rotate intrinsics; convert pose; append to transforms.
    R_imgrot = rotation_in_camera_90cw()
    R_imgrot_4x4 = np.eye(4); R_imgrot_4x4[:3, :3] = R_imgrot

    # MultiScan's encoded mp4 is landscape 1920x1440 with no rotation
    # metadata. The ARKit intrinsics in .jsonl (fx=fy ~ 1588, cx ~ 944,
    # cy ~ 726) match landscape 1920x1440 (cx near W/2=960, cy near H/2=720).
    # The scene-level .json's "resolution": [1440, 1920] is the device's
    # native portrait orientation but is irrelevant for the encoded video.
    #
    # So: NO rotation, intrinsics used as-is for landscape 1920x1440,
    # then resized to target.
    _, _, K0, _ = rows[sel_idx[0]]
    fx_src, fy_src = K0[0, 0], K0[1, 1]
    cx_src, cy_src = K0[0, 2], K0[1, 2]
    W_src, H_src = 1920, 1440

    target_W = TARGET_W   # 960
    target_H = TARGET_H   # 640
    print(f"[{stem}] target resolution {target_W}x{target_H} (landscape, matches landscape intrinsics)")

    # If aspect ratios differ (source 1920/1440 = 1.333, target 960/640 = 1.5),
    # we'd squash y vs x slightly. To avoid, do a content-preserving center
    # crop to the target aspect before resize.
    src_aspect = W_src / H_src
    tgt_aspect = target_W / target_H
    if abs(src_aspect - tgt_aspect) < 1e-3:
        crop_x0, crop_y0, crop_W, crop_H = 0, 0, W_src, H_src
    elif src_aspect < tgt_aspect:
        # source is taller than target -> crop top/bottom
        crop_W = W_src
        crop_H = int(round(W_src / tgt_aspect))
        crop_x0 = 0
        crop_y0 = (H_src - crop_H) // 2
    else:
        # source is wider than target -> crop left/right
        crop_H = H_src
        crop_W = int(round(H_src * tgt_aspect))
        crop_x0 = (W_src - crop_W) // 2
        crop_y0 = 0
    print(f"[{stem}] center-crop {W_src}x{H_src} -> {crop_W}x{crop_H} (offset {crop_x0},{crop_y0})")

    # Intrinsics after crop: cx,cy shift by -crop_x0,-crop_y0. After resize:
    # scale by target/crop.
    scale_x = target_W / crop_W
    scale_y = target_H / crop_H
    fx_final = fx_src * scale_x
    fy_final = fy_src * scale_y
    cx_final = (cx_src - crop_x0) * scale_x
    cy_final = (cy_src - crop_y0) * scale_y

    frames_list = []
    for k, (vidx, raw_jpeg) in enumerate(zip(sel_idx, raw_jpegs)):
        img = cv2.imread(str(raw_jpeg))
        if img is None:
            raise RuntimeError(f"cv2 failed on {raw_jpeg}")
        # crop to target aspect then resize
        img = img[crop_y0:crop_y0 + crop_H, crop_x0:crop_x0 + crop_W]
        img = cv2.resize(img, (target_W, target_H), interpolation=cv2.INTER_AREA)
        out_name = f"{stem}_f{vidx:06d}.jpg"
        cv2.imwrite(str(images_out / out_name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        _, T_arkit, _, _ = rows[vidx]
        c2w_final = align @ T_arkit   # no camera-frame rotation

        frames_list.append({
            "file_path": out_name,
            "transform_matrix": c2w_final.tolist(),
            "_jsonl_frame_index": vidx,
        })

    # Clean up tmp
    for p in tmp_dir.iterdir():
        p.unlink()
    tmp_dir.rmdir()

    transforms = {
        "camera_model": "pinhole",
        "w": target_W,
        "h": target_H,
        "fl_x": fx_final,
        "fl_y": fy_final,
        "cx": cx_final,
        "cy": cy_final,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "frames": frames_list,
    }
    out_transforms = nerfstudio_out / "transforms.json"
    with open(out_transforms, "w") as f:
        json.dump(transforms, f, indent=2)
    print(f"[{stem}] wrote {len(frames_list)} frames -> {out_transforms}")
    print(f"           intrinsics fx={fx_final:.2f} fy={fy_final:.2f} cx={cx_final:.2f} cy={cy_final:.2f}")

    # When the user supplied --manual_pair, also write the view_selection.json
    # that stage_for_pipeline.py expects, so we can skip select_2views.py.
    if args.manual_pair is not None and len(frames_list) == 2:
        c_a = np.array(frames_list[0]["transform_matrix"])[:3, 3]
        c_b = np.array(frames_list[1]["transform_matrix"])[:3, 3]
        baseline = float(np.linalg.norm(c_a - c_b))
        sel = {
            "context": [0, 1],
            "target": [0, 1],
            "manual": True,
            "baseline_m": baseline,
            "file_paths": [frames_list[0]["file_path"], frames_list[1]["file_path"]],
            "jsonl_frame_indices": list(args.manual_pair),
        }
        sel_path = out_dir / "view_selection.json"
        with open(sel_path, "w") as f:
            json.dump(sel, f, indent=2)
        print(f"[{stem}] manual: wrote {sel_path}  (baseline {baseline:.3f} m)")


if __name__ == "__main__":
    main()
