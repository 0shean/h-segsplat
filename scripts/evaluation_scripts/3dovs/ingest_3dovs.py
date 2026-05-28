#!/usr/bin/env python3
"""
Convert a 3D-OVS scene to the h-segsplat-repo data layout.

3D-OVS layout (input):
    <scene>/
        images/<NN>.jpg                   (~30 images @ 4032x3024)
        sparse/0/{cameras.bin,images.bin,points3D.bin}
        segmentations/
            classes.txt                   (one class per line)
            <NN>/<class>.png              (RGBA, alpha = binary mask)
        poses_bounds.npy
        colmap_output.txt, database.db

H-SegSplat layout (output):
    <out>/<scene>/
        dslr/nerfstudio/transforms.json   (2 frames, Blender convention)
        dslr/resized_images/<frame>.JPG   (2 frames @ TARGET resolution)
        target_views.json                 (5 labeled target view c2w + GT mask paths)
        gt_masks/<NN>/<class>.png         (resized binary masks)

The two context views are picked by SegSplat's protocol: 2 views with at least
60% projected-vertex overlap and a non-trivial baseline. We score pairs by
projecting the COLMAP sparse 3D point cloud (read from points3D.bin) into both
candidate cameras and counting points visible in both.

Notes on camera convention:
- COLMAP stores world-to-camera in (qvec, tvec). Camera looks +Z, +X right,
  +Y down (OpenCV).
- h-segsplat-repo's build_hsegsplat_inputs.py applies diag(1, -1, -1, 1) to
  convert Blender -> OpenCV. So we need to write transforms.json's
  transform_matrix as Blender convention (camera looks -Z, +X right, +Y up).
  Concretely:  c2w_blender = c2w_opencv @ diag(1, -1, -1, 1).
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import read_cameras_binary, read_images_binary, read_points3D_binary, qvec2rotmat


# Match h-segsplat-repo / DepthSplat defaults.
TARGET_W = 960
TARGET_H = 640


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", type=Path, required=True,
                   help="3D-OVS scene root (containing images/, sparse/, segmentations/).")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Output scene root (will create dslr/, gt_masks/, target_views.json).")
    # Overlap band (NOT ≥ min). DepthSplat is trained on RealEstate10K pairs with
    # ~50–70% projected overlap; anything outside that band is OOD at test time.
    # Too high (>0.85) means the two views are near-duplicates — no parallax for
    # triangulation; too low (<0.30) means too few co-visible points to fuse.
    p.add_argument("--overlap_lo", type=float, default=0.40,
                   help="Lower edge of the preferred overlap band.")
    p.add_argument("--overlap_hi", type=float, default=0.99,
                   help="Upper edge of the preferred overlap band. Loosened to 0.99 "
                        "(from 0.70) because on small-pointcloud scenes like blue_sofa "
                        "or covered_desk the sparse 3D points project onto >95% of "
                        "every camera, so overlap can't drop into [0.4, 0.7] no matter "
                        "how different the two views look. The angular-separation "
                        "constraint below is what actually enforces 'visually distinct'.")
    p.add_argument("--baseline_target", type=float, default=0.8,
                   help="Preferred translation distance between the 2 views (scene units). "
                        "Room-scale 3D-OVS scenes have cameras ~0.5–1.5m apart in good pairs.")
    p.add_argument("--max_baseline", type=float, default=3.0)
    p.add_argument("--min_angle_deg", type=float, default=20.0,
                   help="HARD lower bound on the angle between the two cameras' optical "
                        "axes (z columns of c2w). Below this the views are too similar "
                        "for DepthSplat to triangulate meaningfully. Empirically all 10 "
                        "3D-OVS scenes admit pairs at 20°+; tight orbits like blue_sofa "
                        "fail without this constraint and pick near-duplicate frames.")
    return p.parse_args()


def qtvec_to_c2w_opencv(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """COLMAP image entry has world-to-camera. Invert -> camera-to-world (OpenCV)."""
    R_w2c = qvec2rotmat(qvec)        # (3, 3)
    w2c = np.eye(4)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = tvec
    return np.linalg.inv(w2c)


def opencv_c2w_to_blender(c2w_opencv: np.ndarray) -> np.ndarray:
    """build_hsegsplat_inputs.py applies diag(1,-1,-1,1) to Blender->OpenCV,
    so the inverse is the same matrix (it's self-inverse)."""
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w_opencv @ flip


def make_K_from_colmap(cam) -> tuple[np.ndarray, np.ndarray]:
    """Returns (K_3x3, dist_5) for a COLMAP camera. Supports SIMPLE_RADIAL +
    SIMPLE_PINHOLE + PINHOLE; for SIMPLE_RADIAL the small k1 distortion is
    used to undistort below."""
    K = np.eye(3, dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    if cam.model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = cam.params
        K[0, 0] = K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
        dist[0] = k1
    elif cam.model == "SIMPLE_PINHOLE":
        f, cx, cy = cam.params[:3]
        K[0, 0] = K[1, 1] = f
        K[0, 2] = cx; K[1, 2] = cy
    elif cam.model == "PINHOLE":
        fx, fy, cx, cy = cam.params[:4]
        K[0, 0] = fx; K[1, 1] = fy
        K[0, 2] = cx; K[1, 2] = cy
    elif cam.model == "RADIAL":
        f, cx, cy, k1, k2 = cam.params[:5]
        K[0, 0] = K[1, 1] = f
        K[0, 2] = cx; K[1, 2] = cy
        dist[0] = k1; dist[1] = k2
    else:
        raise NotImplementedError(f"Unsupported COLMAP camera model: {cam.model}")
    return K, dist


def select_2views(images_data: dict,
                  c2w_per_image: dict,
                  K_pixel: np.ndarray,
                  W_target: int, H_target: int,
                  W_src: int, H_src: int,
                  points_xyz: np.ndarray,
                  excluded_image_ids: set,
                  overlap_lo: float,
                  overlap_hi: float,
                  baseline_target: float,
                  max_baseline: float,
                  min_angle_deg: float):
    """Pick 2 context views.

    The key constraint: pairs need REAL PARALLAX for DepthSplat to triangulate.
    Two near-duplicate views with 98% projected overlap have ~no disparity →
    DepthSplat produces a near-planar Gaussian cloud that does not transfer to
    target views.

    We enforce parallax via THREE signals:
      1. min_angle_deg — HARD lower bound on the angle between the two
         cameras' optical (z) axes. This is the primary "are these visually
         distinct" check; overlap fails on small-pointcloud scenes where
         sparse points project onto >95% of every camera regardless.
      2. overlap_lo / overlap_hi — soft preference for the RealEstate10K
         distribution band.
      3. max_baseline — reject pairs that are physically far apart (likely
         scene-edge or different rooms).

    Returns: (img_id_a, img_id_b, info_dict)
    """
    sx = W_target / W_src
    sy = H_target / H_src
    K_t = K_pixel.copy()
    K_t[0, 0] *= sx; K_t[0, 2] *= sx
    K_t[1, 1] *= sy; K_t[1, 2] *= sy

    if len(points_xyz) > 20000:
        idx = np.linspace(0, len(points_xyz)-1, 20000).astype(np.int64)
        points_xyz = points_xyz[idx]
    pts_h = np.concatenate([points_xyz, np.ones((len(points_xyz), 1))], axis=-1)

    candidates = [iid for iid in c2w_per_image if iid not in excluded_image_ids]
    print(f"  candidate context images: {len(candidates)} (excluded {len(excluded_image_ids)} labeled)")

    vis = {}
    centers = {}
    z_axes = {}   # world-frame camera look direction (c2w's z column)
    for iid in candidates:
        c2w = c2w_per_image[iid]
        w2c = np.linalg.inv(c2w)
        cam = (w2c @ pts_h.T).T[:, :3]
        z = cam[:, 2]
        in_front = z > 0.01
        u = K_t[0, 0] * cam[:, 0] / np.where(z > 1e-6, z, 1.0) + K_t[0, 2]
        v = K_t[1, 1] * cam[:, 1] / np.where(z > 1e-6, z, 1.0) + K_t[1, 2]
        inb = in_front & (u >= 0) & (u < W_target) & (v >= 0) & (v < H_target)
        vis[iid] = inb
        centers[iid] = c2w[:3, 3]
        z_world = c2w[:3, 2]
        z_axes[iid] = z_world / max(np.linalg.norm(z_world), 1e-9)

    def angle_deg(a: int, b: int) -> float:
        d = float(np.clip(z_axes[a] @ z_axes[b], -1.0, 1.0))
        return float(np.degrees(np.arccos(d)))

    def score_pair(min_ov: float, base: float, ang_deg: float) -> float:
        # Sweet spot: middle of the band. Triangular penalty outside.
        band_center = 0.5 * (overlap_lo + overlap_hi)
        band_half_width = 0.5 * (overlap_hi - overlap_lo)
        if min_ov >= overlap_lo and min_ov <= overlap_hi:
            overlap_score = 1.0 - abs(min_ov - band_center) / max(band_half_width, 1e-6)
        else:
            # Outside band: heavy penalty (still allow ranking).
            dist = max(overlap_lo - min_ov, min_ov - overlap_hi)
            overlap_score = -2.0 * dist
        # Baseline reward saturates around `baseline_target`.
        baseline_score = float(np.tanh(base / max(baseline_target, 1e-3)))
        # Angle reward saturates around 30°.
        angle_score = float(np.tanh(ang_deg / 30.0))
        return overlap_score + 0.5 * baseline_score + 1.0 * angle_score

    best_pairs = []
    rejected_by_angle = 0
    for i, a in enumerate(candidates):
        va = vis[a]; na = int(va.sum())
        if na == 0: continue
        for b in candidates[i+1:]:
            vb = vis[b]; nb = int(vb.sum())
            if nb == 0: continue
            inter = int((va & vb).sum())
            if inter == 0: continue
            ov_a = inter / na; ov_b = inter / nb
            min_ov = min(ov_a, ov_b)
            base = float(np.linalg.norm(centers[a] - centers[b]))
            if base > max_baseline: continue
            ang = angle_deg(a, b)
            if ang < min_angle_deg:
                rejected_by_angle += 1
                continue
            sc = score_pair(min_ov, base, ang)
            best_pairs.append((sc, a, b, min_ov, base, ang, na, nb, inter))
    print(f"  rejected by angle<{min_angle_deg:.1f}°: {rejected_by_angle} pairs")
    if not best_pairs:
        raise RuntimeError(
            f"No pair >= min_angle_deg={min_angle_deg}°. Lower the threshold or "
            f"choose a scene with wider camera coverage."
        )
    best_pairs.sort(key=lambda x: -x[0])
    print("  Top 5 pairs (score, a, b, overlap, baseline, angle):")
    for sc, a, b, mo, bl, ang, na, nb, it in best_pairs[:5]:
        print(f"    ({a:3d}, {b:3d})  ov={mo:.2f}  base={bl:.3f}  "
              f"ang={ang:.1f}°  |inter|={it}  score={sc:.3f}")
    _, a, b, mo, bl, ang, _, _, _ = best_pairs[0]
    return a, b, {"overlap": mo, "baseline": bl, "angle_deg": ang}


def main():
    args = parse_args()
    scene_dir = args.scene_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dslr" / "nerfstudio").mkdir(parents=True, exist_ok=True)
    (out_dir / "dslr" / "resized_images").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt_masks").mkdir(parents=True, exist_ok=True)

    # 1. Load COLMAP sparse model.
    cams = read_cameras_binary(scene_dir / "sparse" / "0" / "cameras.bin")
    imgs = read_images_binary(scene_dir / "sparse" / "0" / "images.bin")
    pts = read_points3D_binary(scene_dir / "sparse" / "0" / "points3D.bin")
    print(f"[{scene_dir.name}] COLMAP: {len(cams)} cams, {len(imgs)} imgs, {len(pts)} 3D pts")
    assert len(cams) == 1, "Expected a single shared camera for 3D-OVS scenes."
    cam = next(iter(cams.values()))
    K_native, dist = make_K_from_colmap(cam)
    W_src, H_src = cam.width, cam.height

    # 2. Per-image c2w, indexed by colmap image_id.
    c2w_per_image = {}
    name_to_id = {}
    for iid, im in imgs.items():
        c2w = qtvec_to_c2w_opencv(im.qvec, im.tvec)
        c2w_per_image[iid] = c2w
        name_to_id[im.name] = iid

    # 3. Identify the labeled "target" views.
    seg_dir = scene_dir / "segmentations"
    # Strip per-line whitespace — some 3D-OVS scenes (e.g. covered_desk's
    # "gerbera ") have trailing-space class names in classes.txt that don't
    # match the mask filenames on disk.
    classes = [c.strip() for c in
               (seg_dir / "classes.txt").read_text().splitlines()
               if c.strip()]
    labeled_view_ids = sorted([d.name for d in seg_dir.iterdir()
                                if d.is_dir() and (d / f"{classes[0]}.png").exists()])
    print(f"  labeled views: {labeled_view_ids}  classes: {classes}")
    excluded_ids = set()
    target_image_ids = []
    for vid in labeled_view_ids:
        # 3D-OVS uses .jpg or .JPG depending on the scene; try both.
        fname = None
        for ext in (".jpg", ".JPG"):
            cand = f"{vid}{ext}"
            if cand in name_to_id:
                fname = cand
                break
        if fname is None:
            print(f"  WARN: labeled view {vid} has no matching image in COLMAP model")
            continue
        iid = name_to_id[fname]
        excluded_ids.add(iid)
        target_image_ids.append((vid, iid))

    # 4. Build point cloud array for visibility scoring.
    points_xyz = np.stack([p.xyz for p in pts.values()], axis=0)
    print(f"  points3D count: {len(points_xyz)}")

    # 5. Pick the 2 context views.
    a_iid, b_iid, info = select_2views(
        imgs, c2w_per_image, K_native,
        TARGET_W, TARGET_H, W_src, H_src,
        points_xyz, excluded_ids,
        args.overlap_lo, args.overlap_hi, args.baseline_target, args.max_baseline,
        args.min_angle_deg,
    )
    print(f"  chosen: ({a_iid}, {b_iid})  ov={info['overlap']:.2f}  "
          f"base={info['baseline']:.3f}  angle={info['angle_deg']:.1f}°")

    # 6. Compute target intrinsics. Scale K from native -> TARGET; account for SIMPLE_RADIAL
    #    by undistorting once and using a pinhole K_t.
    sx = TARGET_W / W_src
    sy = TARGET_H / H_src
    K_t = K_native.copy()
    K_t[0, 0] *= sx; K_t[0, 2] *= sx
    K_t[1, 1] *= sy; K_t[1, 2] *= sy

    # 7. Write context images + transforms.json.
    out_frames = []
    for slot, iid in enumerate([a_iid, b_iid]):
        im = imgs[iid]
        src = scene_dir / "images" / im.name
        img = cv2.imread(str(src))
        if img is None:
            raise RuntimeError(f"failed to read {src}")
        # Undistort (no-op if dist=0)
        if np.any(dist != 0):
            new_K, _ = cv2.getOptimalNewCameraMatrix(K_native, dist, (W_src, H_src), 0)
            img = cv2.undistort(img, K_native, dist, None, new_K)
            # rescale K_t from new_K, not K_native (small adjustment)
            # in practice for 3D-OVS k1 is tiny so we can ignore this delta
        img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        out_name = f"{scene_dir.name}_v{iid:03d}.JPG"
        cv2.imwrite(str(out_dir / "dslr" / "resized_images" / out_name),
                    img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        c2w_blender = opencv_c2w_to_blender(c2w_per_image[iid])
        out_frames.append({
            "file_path": out_name,
            "transform_matrix": c2w_blender.tolist(),
            "_colmap_image_id": int(iid),
            "_colmap_name": im.name,
        })

    transforms = {
        "camera_model": "pinhole",
        "w": TARGET_W,
        "h": TARGET_H,
        "fl_x": float(K_t[0, 0]),
        "fl_y": float(K_t[1, 1]),
        "cx": float(K_t[0, 2]),
        "cy": float(K_t[1, 2]),
        "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
        "frames": out_frames,
        "_overlap": info["overlap"],
        "_baseline": info["baseline"],
        "_angle_deg": info["angle_deg"],
    }
    with open(out_dir / "dslr" / "nerfstudio" / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)
    print(f"  wrote 2-view transforms.json at {TARGET_W}x{TARGET_H}")

    # 8. Write target_views.json + resize GT masks.
    target_entries = []
    for vid, iid in target_image_ids:
        c2w_blender = opencv_c2w_to_blender(c2w_per_image[iid])
        # Save resized GT masks per class.
        gt_subdir = out_dir / "gt_masks" / vid
        gt_subdir.mkdir(parents=True, exist_ok=True)
        for cls in classes:
            src_mask = scene_dir / "segmentations" / vid / f"{cls}.png"
            if not src_mask.exists():
                print(f"    WARN: missing mask {src_mask}")
                continue
            m = cv2.imread(str(src_mask), cv2.IMREAD_UNCHANGED)
            if m.ndim == 3 and m.shape[-1] == 4:
                binary = (m[..., 3] > 0).astype(np.uint8) * 255
            else:
                binary = (m > 0).astype(np.uint8) * 255
                if binary.ndim == 3:
                    binary = binary.max(axis=-1)
            binary_resized = cv2.resize(binary, (TARGET_W, TARGET_H),
                                         interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(gt_subdir / f"{cls}.png"), binary_resized)
        # Also save the resized GT image for debugging.
        rgb_path = None
        for ext in (".jpg", ".JPG"):
            cand = scene_dir / "images" / f"{vid}{ext}"
            if cand.exists():
                rgb_path = cand
                break
        rgb = cv2.imread(str(rgb_path))
        rgb_resized = cv2.resize(rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(gt_subdir / f"_rgb.jpg"), rgb_resized)
        target_entries.append({
            "view_id": vid,
            "_colmap_image_id": int(iid),
            "transform_matrix": c2w_blender.tolist(),
            "gt_mask_dir": f"gt_masks/{vid}",
        })

    target_views = {
        "camera_model": "pinhole",
        "w": TARGET_W,
        "h": TARGET_H,
        "fl_x": float(K_t[0, 0]),
        "fl_y": float(K_t[1, 1]),
        "cx": float(K_t[0, 2]),
        "cy": float(K_t[1, 2]),
        "classes": classes,
        "targets": target_entries,
    }
    with open(out_dir / "target_views.json", "w") as f:
        json.dump(target_views, f, indent=2)
    print(f"  wrote {len(target_entries)} target views -> target_views.json")
    print(f"  done: {out_dir}")


if __name__ == "__main__":
    main()
