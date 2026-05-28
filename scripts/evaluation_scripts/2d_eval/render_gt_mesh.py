#!/usr/bin/env python3
"""
Rasterize a semantically-labelled mesh at one or more camera poses to produce
per-pixel semantic ID maps that align with H-SegSplat's rendered feature maps.

Inputs:
    --mesh           PLY file with per-vertex semantic labels (encoded as
                     the integer in --vertex_label_txt, or read from a
                     per-vertex 'semantic' attribute if available).
    --vertex_labels  Optional: a text file with one integer per line,
                     length must match mesh vertex count. If given, this
                     overrides the PLY's per-vertex semantic attribute.
    --gaussians_pt   Path to gaussians.pt; we read 'extrinsics' (V,4,4 c2w
                     OpenCV), 'intrinsics_pixel_K' (V,3,3), 'image_HW'
                     (H,W) so the rendered class map aligns pixel-for-pixel
                     with H-SegSplat's rendered feature maps.
    --views          Which views to render. Default: all from gaussians.pt.

Output:
    <out_dir>/gt_class_map.npy   shape (V, H, W) int32, -1 = no GT hit
    <out_dir>/gt_overlay_v<v>.png   colored overlay for visual sanity

Method:
    For each view's camera, cast one ray per pixel via trimesh's BVH ray
    intersector. The first triangle hit per ray gives the face index;
    we look up the face's first vertex's semantic label as the pixel class.
    Background pixels (no triangle hit) are written as -1.

Why trimesh-raycast instead of Open3D's OffscreenRenderer:
    Open3D's Filament backend doesn't support headless rendering on macOS
    (EGL is Linux/Windows only there). trimesh's ray.RayMeshIntersector
    works on Mac with pure CPU and gives deterministic per-pixel face IDs
    that we then map back to vertex semantic labels.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh


# Re-use the existing palette so colors are consistent with other viz scripts.
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
    p.add_argument("--mesh", type=Path, required=True,
                   help="Path to the semantically-labelled PLY mesh.")
    p.add_argument("--vertex_labels", type=Path, default=None,
                   help="Optional text file with one int per line "
                        "(sem * 1000 + inst for MultiScan, or "
                        "scannet++ segIndices). Overrides face/vertex PLY attrs.")
    p.add_argument("--face_label_attr", type=str, default=None,
                   help="If set, read this per-face attribute from the PLY as the "
                        "instance label (e.g. 'objectId' for MultiScan PLYs). For a "
                        "single channel only; use --face_label_attrs for multiple.")
    p.add_argument("--face_label_attrs", type=str, nargs="+", default=None,
                   help="Like --face_label_attr but multi-channel. E.g. "
                        "--face_label_attrs objectId partId emits both gt_objectId_map.npy "
                        "and gt_partId_map.npy. The first attr is also written as "
                        "gt_class_map.npy for back-compat.")
    p.add_argument("--label_map_json", type=Path, default=None,
                   help="Optional MultiScan annotations.json (e.g. "
                        "scene_xxxxx.annotations.json) to map objectId -> "
                        "class-name string. If not given, output labels are "
                        "raw integer IDs.")
    p.add_argument("--gaussians_pt", type=Path, required=True,
                   help="Path to gaussians.pt for camera matrices.")
    p.add_argument("--views", type=int, nargs="*", default=None,
                   help="Which view indices to render. Default: all in gaussians.pt.")
    p.add_argument("--multiscan_scan_dir", type=Path, default=None,
                   help="If set, override gaussians.pt extrinsics with PLY-frame poses "
                        "computed as T_arkit @ align from the MultiScan scan's "
                        ".jsonl + .align.json. This is needed because the H-SegSplat "
                        "ingest currently puts gaussians in a 'frame X' that differs "
                        "from the PLY's 'frame Y' by left-vs-right multiplication of "
                        "the align matrix. The DepthSplat reconstruction is consistent "
                        "in frame X, but the PLY is in frame Y, so we need to raycast "
                        "the mesh in frame Y to get pixel-aligned GT.")
    p.add_argument("--jsonl_frame_indices", type=int, nargs="+", default=None,
                   help="If --multiscan_scan_dir is set, the jsonl frame indices "
                        "corresponding to each view in gaussians.pt. Must match length.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--pixel_stride", type=int, default=1,
                   help="Cast every Nth pixel (1 = full res, 2 = half-res, etc.).")
    return p.parse_args()


def vertex_label_to_color(vid: int) -> tuple:
    """For overlay debug: deterministic color per vertex label."""
    col = PALETTE_BGR[1 + (vid % (len(PALETTE_BGR) - 1))]
    return tuple(int(c) for c in col)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    print(f"[gt] loading mesh: {args.mesh}")
    mesh = trimesh.load(args.mesh, process=False)
    print(f"     verts: {len(mesh.vertices)}, faces: {len(mesh.faces)}")

    # Labels can be per-face OR per-vertex. Resolve which we have. We support
    # MULTIPLE per-face channels (e.g. objectId + partId) so the hierarchical
    # eval can read both in a single mesh-raycast pass.
    face_labels = None                          # primary channel (back-compat)
    face_labels_extra: dict[str, np.ndarray] = {}  # any additional channels
    primary_attr: str | None = None
    vertex_labels = None

    # Resolve which per-face attrs to read. --face_label_attrs (plural) takes
    # priority; --face_label_attr (singular) is the legacy form.
    requested_attrs: list[str] = []
    if args.face_label_attrs:
        requested_attrs = list(args.face_label_attrs)
    elif args.face_label_attr:
        requested_attrs = [args.face_label_attr]

    if requested_attrs:
        face_attrs = mesh.metadata.get("_ply_raw", {}).get("face", {}).get("data")
        if face_attrs is None:
            raise RuntimeError(
                f"PLY has no per-face attribute data; cannot read {requested_attrs}"
            )
        for attr in requested_attrs:
            if attr not in face_attrs.dtype.names:
                raise RuntimeError(
                    f"PLY has no per-face attr '{attr}'. Available: {face_attrs.dtype.names}"
                )
            arr = np.asarray(face_attrs[attr], dtype=np.int64)
            if len(arr) != len(mesh.faces):
                raise ValueError(
                    f"face labels '{attr}' ({len(arr)}) != mesh faces ({len(mesh.faces)})"
                )
            print(f"[gt] PLY per-face attr '{attr}': "
                  f"range {arr.min()}..{arr.max()}, unique={len(set(arr.tolist()))}")
            if primary_attr is None:
                primary_attr = attr
                face_labels = arr
            else:
                face_labels_extra[attr] = arr
    elif args.vertex_labels:
        print(f"[gt] loading per-vertex labels from {args.vertex_labels}")
        vertex_labels = np.loadtxt(args.vertex_labels, dtype=np.int64)
        if len(vertex_labels) != len(mesh.vertices):
            raise ValueError(
                f"vertex labels ({len(vertex_labels)}) != mesh vertices ({len(mesh.vertices)})"
            )
    else:
        # Fall back to the PLY's per-vertex attributes if any exist.
        attrs = getattr(mesh, "metadata", {}).get("_ply_raw", {}).get("vertex", {}).get("data", {})
        if attrs is None:
            raise RuntimeError(
                "No labels: pass --face_label_attr, --vertex_labels, or use a PLY with "
                "vertex semantic attributes."
            )
        for key in ("objectId", "partId", "label", "class"):
            if key in attrs.dtype.names:
                vertex_labels = np.asarray(attrs[key], dtype=np.int64)
                print(f"[gt] using PLY vertex attribute '{key}', range "
                      f"{vertex_labels.min()}..{vertex_labels.max()}")
                break
        else:
            raise RuntimeError(
                f"No recognized vertex label attribute. Available: {attrs.dtype.names}"
            )

    # Optional: load objectId -> class-name mapping from annotations.json
    label_map = None
    if args.label_map_json:
        ann = json.loads(args.label_map_json.read_text())
        objs = ann.get("objects", [])
        # Strip the instance suffix (".1", ".2") to get the class name.
        label_map = {o["objectId"]: o["label"].split(".")[0] for o in objs}
        print(f"[gt] label_map: {len(label_map)} entries from {args.label_map_json}")

    # Load gaussians.pt for cameras
    g = torch.load(args.gaussians_pt, map_location="cpu", weights_only=False)
    ext = g["extrinsics"].numpy()              # (V, 4, 4) c2w OpenCV
    K_all = g["intrinsics_pixel_K"].numpy()    # (V, 3, 3)
    H, W = g["image_HW"]
    V_total = ext.shape[0]
    print(f"[gt] gaussians.pt: V={V_total}, H={H}, W={W}")

    # MultiScan override: compute PLY-frame cam poses from ARKit + align.
    if args.multiscan_scan_dir is not None:
        if args.jsonl_frame_indices is None or len(args.jsonl_frame_indices) != V_total:
            raise ValueError(
                f"--multiscan_scan_dir requires --jsonl_frame_indices with one entry "
                f"per gaussians-view (got {len(args.jsonl_frame_indices) if args.jsonl_frame_indices else 0}, "
                f"need {V_total})"
            )
        stem = args.multiscan_scan_dir.name
        align = np.array(json.load(open(args.multiscan_scan_dir / f"{stem}.align.json"))[
            "coordinate_transform"]).reshape(4, 4).T
        with open(args.multiscan_scan_dir / f"{stem}.jsonl") as f:
            rows = [json.loads(l) for l in f]
        new_ext = np.zeros_like(ext)
        for v, fi in enumerate(args.jsonl_frame_indices):
            T_arkit = np.array(rows[fi]["transform"]).reshape(4, 4).T
            # PLY-frame c2w (frame Y, the one the PLY lives in)
            new_ext[v] = T_arkit @ align
            print(f"[gt] view {v} (jsonl frame {fi}): "
                  f"orig cam center {ext[v, :3, 3]} -> PLY-frame {new_ext[v, :3, 3]}")
        ext = new_ext

    views = list(range(V_total)) if not args.views else args.views

    # Build trimesh BVH once
    print(f"[gt] building ray intersector ...")
    ray = mesh.ray

    # Output arrays: (V, H, W) int32. -1 = no GT hit.
    gt_class_map = np.full((V_total, H, W), -1, dtype=np.int32)
    gt_extra_maps: dict[str, np.ndarray] = {
        attr: np.full((V_total, H, W), -1, dtype=np.int32)
        for attr in face_labels_extra
    }

    s = args.pixel_stride
    H_s = H // s
    W_s = W // s
    print(f"[gt] rendering at stride {s} -> {H_s}x{W_s} effective rays per view")

    for v in views:
        c2w = ext[v]
        K = K_all[v]
        # Camera center in world = c2w[:3, 3]
        cam_origin = c2w[:3, 3]
        # Pixel grid (subsampled by stride)
        ys = np.arange(0, H, s)
        xs = np.arange(0, W, s)
        uu, vv = np.meshgrid(xs, ys)  # both (H_s, W_s) — note: uu = x, vv = y
        # Camera-frame ray directions: (x - cx)/fx, (y - cy)/fy, 1
        dx = (uu - K[0, 2]) / K[0, 0]
        dy = (vv - K[1, 2]) / K[1, 1]
        dz = np.ones_like(dx)
        # Stack and rotate by c2w[:3, :3] to get world-frame directions
        dirs_cam = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)
        dirs_world = (c2w[:3, :3] @ dirs_cam.T).T
        # Normalize
        dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)
        # Origins (all same)
        origins = np.broadcast_to(cam_origin, dirs_world.shape)

        print(f"  view {v}: casting {len(dirs_world)} rays ...")
        # intersects_first returns the FIRST face hit per ray (or -1 if no hit)
        face_idx = ray.intersects_first(ray_origins=origins, ray_directions=dirs_world)
        # face_idx shape: (N_rays,) int, -1 = no intersection
        n_hit = int((face_idx >= 0).sum())
        print(f"    {n_hit}/{len(face_idx)} rays hit the mesh "
              f"({100*n_hit/max(len(face_idx),1):.1f}%)")

        # Map face_idx -> label. Either directly via per-face attr, or via
        # first-vertex lookup if we only have per-vertex labels.
        hit_mask = face_idx >= 0

        def gather_channel(face_attr_arr: np.ndarray) -> np.ndarray:
            out = np.full(face_idx.shape, -1, dtype=np.int64)
            if hit_mask.any():
                out[hit_mask] = face_attr_arr[face_idx[hit_mask]]
            return out

        # Primary channel
        if face_labels is not None:
            pix_labels = gather_channel(face_labels)
        else:
            pix_labels = np.full(face_idx.shape, -1, dtype=np.int64)
            if hit_mask.any():
                face_v0 = mesh.faces[face_idx[hit_mask], 0]
                pix_labels[hit_mask] = vertex_labels[face_v0]

        def write_to_view(arr_2d: np.ndarray, dest_map: np.ndarray):
            if s == 1:
                dest_map[v] = arr_2d.astype(np.int32)
            else:
                dest_map[v] = cv2.resize(arr_2d.astype(np.int32),
                                          (W, H), interpolation=cv2.INTER_NEAREST)

        write_to_view(pix_labels.reshape(H_s, W_s), gt_class_map)

        # Extra channels (e.g. partId alongside objectId)
        for attr, arr in face_labels_extra.items():
            pix_extra = gather_channel(arr).reshape(H_s, W_s)
            write_to_view(pix_extra, gt_extra_maps[attr])

        # Make a debug overlay: color each unique label with the palette
        rgb_path_candidates = [
            args.gaussians_pt.parent / f"render_view{v}.png",
        ]
        rgb_bg = None
        for cand in rgb_path_candidates:
            if cand.exists():
                rgb_bg = cv2.imread(str(cand))
                break
        if rgb_bg is None:
            rgb_bg = np.full((H, W, 3), 50, dtype=np.uint8)

        # Color the GT map
        gt_v = gt_class_map[v]
        unique_labels = sorted(set(gt_v[gt_v >= 0].tolist()))
        col_img = np.full((H, W, 3), 220, dtype=np.uint8)
        for li, lab in enumerate(unique_labels):
            col_img[gt_v == lab] = PALETTE_BGR[1 + (li % (len(PALETTE_BGR) - 1))]
        overlay = cv2.addWeighted(rgb_bg, 0.4, col_img, 0.6, 0.0)
        out_png = args.out_dir / f"gt_overlay_v{v}.png"
        cv2.imwrite(str(out_png), overlay)
        print(f"    wrote {out_png}  ({len(unique_labels)} unique labels in view {v})")

    np.save(args.out_dir / "gt_class_map.npy", gt_class_map)
    print(f"[gt] wrote {args.out_dir / 'gt_class_map.npy'}  shape={gt_class_map.shape}")
    # Also emit gt_<attr>_map.npy for each named channel (primary + extras),
    # so the hierarchical eval can ask for objectId + partId by name.
    channel_files = {}
    if primary_attr is not None:
        primary_path = args.out_dir / f"gt_{primary_attr}_map.npy"
        np.save(primary_path, gt_class_map)
        channel_files[primary_attr] = primary_path.name
    for attr, m in gt_extra_maps.items():
        p = args.out_dir / f"gt_{attr}_map.npy"
        np.save(p, m)
        channel_files[attr] = p.name
        print(f"[gt] wrote {p}  shape={m.shape}")
    if channel_files:
        with open(args.out_dir / "channels.json", "w") as f:
            json.dump({"primary": primary_attr, "files": channel_files}, f, indent=2)

    # Persist the label_map so the eval driver can map raw IDs -> class names.
    if label_map is not None:
        with open(args.out_dir / "label_map.json", "w") as f:
            json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
        print(f"[gt] wrote {args.out_dir / 'label_map.json'}  ({len(label_map)} entries)")


if __name__ == "__main__":
    main()
