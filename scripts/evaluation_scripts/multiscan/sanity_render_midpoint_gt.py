#!/usr/bin/env python3
"""Sanity check: render the MultiScan GT mesh from the midpoint of two input
views and save a colorized PNG. The goal is to confirm that:

  - the midpoint pose is computed correctly in the PLY coordinate frame;
  - trimesh raycasting from that pose hits the mesh;
  - the resulting per-pixel objectId / partId maps look plausible (mostly
    non-(-1), instance IDs reasonably distributed).

Output (under --out_dir, default results/<scene>/midpoint_gt/):
  midpoint_objectId.png   colored by object class
  midpoint_partId.png     colored by part class
  midpoint_objectId.npy   (H, W) int32 per-pixel objectId, -1 where ray misses
  midpoint_partId.npy     (H, W) int32 per-pixel partId, -1 where ray misses
  midpoint_pose.json      the PLY-frame pose used (for later reuse)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from plyfile import PlyData


def quat_from_rotmat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rotmat_from_quat_wxyz(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def slerp(q1, q2, t):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.dot(q1, q2))
    if dot < 0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        out = q1 + t * (q2 - q1)
        return out / np.linalg.norm(out)
    th = np.arccos(dot)
    s = np.sin(th)
    return (np.sin((1 - t) * th) / s) * q1 + (np.sin(t * th) / s) * q2


def palette_color(idx: int):
    if idx < 0:
        return np.array([40, 40, 40], dtype=np.uint8)
    # Deterministic distinct colors via golden-ratio hue stepping.
    rng = np.random.default_rng(seed=int(idx) * 9973 + 13)
    rgb = rng.integers(40, 255, size=3, dtype=np.int32)
    return rgb.astype(np.uint8)


def colorize(label_map: np.ndarray) -> np.ndarray:
    """(H, W) int -> (H, W, 3) BGR."""
    H, W = label_map.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for v in np.unique(label_map):
        col = palette_color(int(v))
        out[label_map == v] = col[::-1]  # BGR
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=Path, required=True,
                    help="data/Multiscan/scenes/<scene> — has .ply, .jsonl, .align.json, .annotations.json")
    ap.add_argument("--ingested_dir", type=Path, required=True,
                    help="data/Multiscan/ingested/<scene> — has dslr/nerfstudio/transforms.json")
    ap.add_argument("--mesh", type=Path, default=None,
                    help="Override mesh path. Default: scene_dir/<stem>.ply")
    ap.add_argument("--gaussians_pt", type=Path, default=None,
                    help="Optional: also emit the midpoint pose in H-SegSplat frame "
                         "(reads input extrinsics from gaussians.pt). Used by the "
                         "Gaussian-render side of the pipeline. "
                         "Default: results/<scene>/gaussians.pt if it exists.")
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    scene_stem = args.scene_dir.name
    if args.mesh is None:
        args.mesh = args.scene_dir / f"{scene_stem}.ply"
    if args.out_dir is None:
        args.out_dir = Path("results") / scene_stem / "midpoint_gt"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.gaussians_pt is None:
        cand = Path("results") / scene_stem / "gaussians.pt"
        if cand.exists():
            args.gaussians_pt = cand

    # --- 1. Load input camera intrinsics from transforms.json (one set, shared)
    tj = json.load(open(args.ingested_dir / "dslr/nerfstudio/transforms.json"))
    W, H = tj["w"], tj["h"]
    K = np.array([
        [tj["fl_x"], 0,         tj["cx"]],
        [0,         tj["fl_y"], tj["cy"]],
        [0,         0,         1]
    ], dtype=np.float64)
    if len(tj["frames"]) != 2:
        raise ValueError(f"expected 2 frames, got {len(tj['frames'])}")
    frame_indices = [tj["frames"][0]["_jsonl_frame_index"],
                     tj["frames"][1]["_jsonl_frame_index"]]
    print(f"[mid] scene={scene_stem}  jsonl frames={frame_indices}  W={W} H={H}")

    # --- 2. Read ARKit frames + align matrix; PLY-frame pose = T_arkit @ align
    align = np.array(json.load(open(args.scene_dir / f"{scene_stem}.align.json"))[
        "coordinate_transform"]).reshape(4, 4).T
    with open(args.scene_dir / f"{scene_stem}.jsonl") as f:
        rows = [json.loads(l) for l in f]

    # Correct convention for these MultiScan PLYs (verified by single-ray
    # raycast tests against the mesh from each candidate pose):
    #   c2w_ply_opencv = (align^-1 @ T_arkit) @ diag(1, -1, -1, 1)
    # Two corrections:
    #   1. align^-1 @ T_arkit : ARKit-world -> PLY-world transform.
    #      render_gt_mesh.py's `T_arkit @ align` puts the camera 13m outside
    #      the mesh bbox, so its convention is wrong (this is task #20).
    #   2. diag(1, -1, -1, 1) right-multiply : OpenGL camera frame
    #      (-Z forward, +Y up) -> OpenCV (+Z forward, +Y down).
    #      Without this flip the camera looks at the ceiling and rays miss
    #      the mesh almost entirely (1.3% coverage).
    align_inv = np.linalg.inv(align)
    CAM_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
    poses_ply = []
    for fi in frame_indices:
        T_arkit = np.array(rows[fi]["transform"]).reshape(4, 4).T
        c2w_ply = (align_inv @ T_arkit) @ CAM_GL_TO_CV
        poses_ply.append(c2w_ply)
        print(f"[mid] frame {fi}: PLY-frame center = {c2w_ply[:3, 3]}  "
              f"forward = {c2w_ply[:3, 2]}")

    # --- 3. Midpoint pose in PLY frame: midpoint center + SLERP rotation
    C_mid = (poses_ply[0][:3, 3] + poses_ply[1][:3, 3]) * 0.5
    q1 = quat_from_rotmat(poses_ply[0][:3, :3])
    q2 = quat_from_rotmat(poses_ply[1][:3, :3])
    q_mid = slerp(q1, q2, 0.5)
    R_mid = rotmat_from_quat_wxyz(q_mid)
    T_mid = np.eye(4)
    T_mid[:3, :3] = R_mid
    T_mid[:3, 3] = C_mid
    print(f"[mid] midpoint center  = {C_mid}")
    print(f"[mid] midpoint forward = {R_mid[:, 2]}  (camera z-axis in world)")

    # H-SegSplat-frame midpoint: computed directly from gaussians.pt's own
    # extrinsics (which are in OpenCV-camera convention already and live in
    # the frame DepthSplat used during reconstruction). This is what the
    # Gaussian renderer needs at step 2.
    T_mid_hseg = None
    if args.gaussians_pt is not None and args.gaussians_pt.exists():
        import torch
        g = torch.load(args.gaussians_pt, map_location="cpu", weights_only=False)
        ext_hseg = g["extrinsics"].numpy()  # (2, 4, 4) c2w OpenCV in H-SegSplat frame
        Ch_mid = (ext_hseg[0, :3, 3] + ext_hseg[1, :3, 3]) * 0.5
        qh1 = quat_from_rotmat(ext_hseg[0, :3, :3])
        qh2 = quat_from_rotmat(ext_hseg[1, :3, :3])
        qh_mid = slerp(qh1, qh2, 0.5)
        Rh_mid = rotmat_from_quat_wxyz(qh_mid)
        T_mid_hseg = np.eye(4)
        T_mid_hseg[:3, :3] = Rh_mid
        T_mid_hseg[:3, 3] = Ch_mid
        K_hseg = g["intrinsics_pixel_K"].numpy()  # (2, 3, 3) - inputs share K
        K_hseg_use = K_hseg[0]
        print(f"[mid] H-SegSplat-frame midpoint center = {Ch_mid}")
        print(f"      Use this with gsplat.rasterization for the Gaussian render.")
    else:
        K_hseg_use = None

    # Persist all midpoint info for later steps.
    payload = {
        "scene": scene_stem,
        "jsonl_frame_indices": frame_indices,
        "c2w_ply_frame": T_mid.tolist(),
        "c2w_input_0_ply": poses_ply[0].tolist(),
        "c2w_input_1_ply": poses_ply[1].tolist(),
        "intrinsics_K": K.tolist(),
        "image_HW": [H, W],
        "align_matrix": align.tolist(),
    }
    if T_mid_hseg is not None:
        payload["c2w_hsegsplat_frame"] = T_mid_hseg.tolist()
        payload["intrinsics_K_hsegsplat"] = K_hseg_use.tolist()
    with open(args.out_dir / "midpoint_pose.json", "w") as f:
        json.dump(payload, f, indent=2)

    # --- 4. Load mesh + per-face labels
    print(f"[mid] loading mesh {args.mesh}")
    ply = PlyData.read(str(args.mesh))
    vtx = ply["vertex"].data
    faces = ply["face"].data
    V = np.stack([vtx["x"], vtx["y"], vtx["z"]], axis=1).astype(np.float64)
    F = np.stack([np.array(f["vertex_indices"], dtype=np.int64)
                  for f in faces], axis=0)
    face_objectId = np.array(faces["objectId"], dtype=np.int32)
    face_partId = np.array(faces["partId"], dtype=np.int32)
    print(f"[mid] mesh: {len(V)} verts, {len(F)} faces, "
          f"objectId range [{face_objectId.min()}..{face_objectId.max()}], "
          f"partId range [{face_partId.min()}..{face_partId.max()}]")

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

    # --- 5. Build per-pixel rays from the midpoint pose, raycast
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dirs_cam = np.stack([
        (xs - cx) / fx,
        (ys - cy) / fy,
        np.ones_like(xs, dtype=np.float64)
    ], axis=-1)
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_world = (R_mid @ dirs_cam.reshape(-1, 3).T).T
    origins = np.broadcast_to(C_mid, dirs_world.shape).copy()
    print(f"[mid] raycasting {len(origins)} rays into the mesh ...")
    hits = mesh.ray.intersects_first(origins, dirs_world)  # (HW,) face ids, -1 miss

    objectId_map = np.full((H, W), -1, dtype=np.int32)
    partId_map = np.full((H, W), -1, dtype=np.int32)
    mask = hits >= 0
    valid_hits = hits[mask]
    objectId_map.reshape(-1)[mask] = face_objectId[valid_hits]
    partId_map.reshape(-1)[mask] = face_partId[valid_hits]
    coverage = float(mask.sum()) / (H * W)
    print(f"[mid] ray hit coverage: {coverage:.2%}  "
          f"({len(np.unique(objectId_map)) - 1} unique objectIds, "
          f"{len(np.unique(partId_map)) - 1} unique partIds)")

    # --- 6. Save artifacts
    np.save(args.out_dir / "midpoint_objectId.npy", objectId_map)
    np.save(args.out_dir / "midpoint_partId.npy", partId_map)
    cv2.imwrite(str(args.out_dir / "midpoint_objectId.png"), colorize(objectId_map))
    cv2.imwrite(str(args.out_dir / "midpoint_partId.png"), colorize(partId_map))
    print(f"[mid] wrote artifacts under {args.out_dir}")

    # --- 7. Lookup names per ID via annotations.json (so user can sanity-check)
    ann = json.load(open(args.scene_dir / f"{scene_stem}.annotations.json"))
    obj_names = {o["objectId"]: o["label"] for o in ann["objects"]}
    part_names = {p["partId"]: p["label"] for p in ann["parts"]}
    visible_obj = sorted(set(int(v) for v in np.unique(objectId_map) if v >= 0))
    visible_part = sorted(set(int(v) for v in np.unique(partId_map) if v >= 0))
    print("[mid] visible objects:")
    for oid in visible_obj:
        n_pix = int((objectId_map == oid).sum())
        print(f"  objectId={oid:<3}  {obj_names.get(oid, '?'):<25}  {n_pix} pixels")
    print("[mid] visible parts:")
    for pid in visible_part:
        n_pix = int((partId_map == pid).sum())
        print(f"  partId={pid:<3}  {part_names.get(pid, '?'):<25}  {n_pix} pixels")


if __name__ == "__main__":
    main()
