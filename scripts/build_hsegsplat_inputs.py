#!/usr/bin/env python3
"""
Build H-SegSplat semantic inputs from a scene with per-level SAM masks + SigLIP features.

Three-level (1, 3, 6) extension of SegSplat (Siegel et al. 2025) §3.1.
Each level is an independent flat SegSplat — independent K-means, independent bank,
independent per-pixel cluster-ID map. No hierarchy tree (v1 design, PROJECT_PLAN.md §3.5).

Expected scene layout (under --scene_dir):
    dslr/
      nerfstudio/transforms.json     # pinhole, w=W, h=H
      resized_images/<frame>.JPG
    masks_lvl_1/<frame_stem>/{mask_*.png, metadata.json, siglip_embeddings.npy}
    masks_lvl_3/<frame_stem>/...
    masks_lvl_6/<frame_stem>/...

Outputs (under --output_dir):
    level_1/{bank.npy, index_maps.npy, mask_id_maps.npy, mask_to_cluster.json, meta.json}
    level_3/...
    level_6/...
    datasets/<scene_key>_2view/test/000000.torch     # level-independent (images + poses)
    datasets/<scene_key>_2view/test/index.json
    assets/<scene_key>_2view_eval.json
    <scene_key>_hsegsplat_colab.zip                  # everything above, bundled for Colab upload
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans


# SegSplat's clustering hyperparameter; kept fixed across all three levels for v1.
LAMBDA = 1.2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", type=str, required=True,
                   help="Scene dir with dslr/{nerfstudio,resized_images}/ and masks_lvl_{1,3,6}/")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--scene_key", type=str, default=None,
                   help="Scene key written into the .torch (defaults to output_dir basename).")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 3, 6],
                   help="Granularity levels to process. Each one needs a masks_lvl_<N>/ folder.")
    return p.parse_args()


def blender_to_opencv_c2w(blender_c2w: np.ndarray) -> np.ndarray:
    """Same convention as fix_data.py / prepare_2_images_for_depthsplat_colab.py."""
    blender2opencv = np.diag([1.0, -1.0, -1.0, 1.0])
    return blender_c2w @ blender2opencv


def load_frame_masks(frame_mask_dir: Path):
    """Returns:
        masks: (N, H, W) bool — per-mask binary maps in metadata order
        areas: (N,)      int  — per-mask pixel count
        feats: (N, D)    f32  — per-mask SigLIP features (raw, unnormalized)
        names: list[str]      — mask filenames in order
    """
    meta_path = frame_mask_dir / "metadata.json"
    feats_path = frame_mask_dir / "siglip_embeddings.npy"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feats = np.load(feats_path)
    if feats.shape[0] != len(meta):
        raise ValueError(
            f"{frame_mask_dir}: {len(meta)} masks in metadata.json but {feats.shape[0]} siglip embeddings"
        )

    masks = []
    areas = []
    names = []
    for entry in meta:
        m_path = frame_mask_dir / entry["mask_file"]
        m = cv2.imread(str(m_path), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(m_path)
        if m.ndim == 3:
            m = m[..., 0]
        masks.append(m > 0)
        areas.append(int(entry["area"]))
        names.append(entry["mask_file"])
    return np.stack(masks, axis=0), np.array(areas, dtype=np.int64), feats.astype(np.float32), names


def resolve_overlap_smallest_area(masks_bool: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """For each pixel covered by 1+ masks, pick the smallest-area mask.
    Returns (H, W) int32 with 0 = bg, 1..N = mask_id+1 (so 0 stays reserved for background)."""
    N, H, W = masks_bool.shape
    out = np.zeros((H, W), dtype=np.int32)
    if N == 0:
        return out
    # Paint masks largest -> smallest so the smallest mask wins each contested pixel.
    order = np.argsort(-areas)
    for k in order:
        out[masks_bool[k]] = k + 1
    return out


def process_level(level: int,
                  scene_dir: Path,
                  output_dir: Path,
                  frame_names: list,
                  H: int,
                  W: int,
                  scene_key: str) -> dict:
    """Run SegSplat's §3.1 once for one granularity level. Writes the level's assets
    under <output_dir>/level_<N>/. Returns the meta dict for the level."""

    masks_root = scene_dir / f"masks_lvl_{level}"
    if not masks_root.exists():
        raise FileNotFoundError(f"missing masks dir: {masks_root}")

    level_out = output_dir / f"level_{level}"
    level_out.mkdir(parents=True, exist_ok=True)

    per_frame_masks = []
    per_frame_areas = []
    per_frame_feats = []
    for fname in frame_names:
        mdir = masks_root / Path(fname).stem
        if not mdir.exists():
            raise FileNotFoundError(f"masks dir missing for level {level}: {mdir}")
        m, a, f, _names = load_frame_masks(mdir)
        if m.shape[1:] != (H, W):
            raise ValueError(f"lvl {level} / {fname}: masks are {m.shape[1:]} but image is {(H, W)}")
        per_frame_masks.append(m)
        per_frame_areas.append(a)
        per_frame_feats.append(f)

    # Pool features, L2-normalize, run K-means independently for this level.
    all_feats = np.concatenate(per_frame_feats, axis=0)  # (N_total, D)
    N_total, D = all_feats.shape

    norms = np.linalg.norm(all_feats, axis=1, keepdims=True).clip(min=1e-12)
    feats_unit = all_feats / norms

    V = len(frame_names)
    M = int(np.ceil(LAMBDA * N_total / V))
    M = max(M, 1)
    M = min(M, N_total)

    print(f"  [lvl {level}] V={V} N_total={N_total} D={D} -> M={M} clusters (lambda={LAMBDA})")

    kmeans = KMeans(n_clusters=M, n_init=10, random_state=0)
    labels = kmeans.fit_predict(feats_unit)

    centroids_unit = kmeans.cluster_centers_.astype(np.float32)
    centroids_unit /= np.linalg.norm(centroids_unit, axis=1, keepdims=True).clip(min=1e-12)

    # Bank: row 0 = zeros (background), rows 1..M = cluster centroids.
    bank = np.zeros((M + 1, D), dtype=np.float32)
    bank[1:] = centroids_unit

    index_maps = np.zeros((V, H, W), dtype=np.int16)
    mask_id_maps = np.zeros((V, H, W), dtype=np.int16)
    mask_to_cluster = {}

    cursor = 0
    for v_idx, fname in enumerate(frame_names):
        masks_bool = per_frame_masks[v_idx]
        areas = per_frame_areas[v_idx]
        N_v = masks_bool.shape[0]
        frame_labels = labels[cursor:cursor + N_v]
        cursor += N_v

        mask_id_map = resolve_overlap_smallest_area(masks_bool, areas)

        cluster_lookup = np.zeros(N_v + 1, dtype=np.int16)  # mask_id 0 (bg) -> cluster 0
        cluster_lookup[1:] = frame_labels.astype(np.int16) + 1  # cluster ids shifted by +1
        index_map = cluster_lookup[mask_id_map]

        index_maps[v_idx] = index_map
        mask_id_maps[v_idx] = mask_id_map.astype(np.int16)
        mask_to_cluster[fname] = frame_labels.tolist()

        bg = int((index_map == 0).sum())
        fg = int(index_map.size - bg)
        print(f"    [{v_idx}] {fname}: {N_v} masks  bg={bg} fg={fg} ({100*fg/index_map.size:.1f}%)")

    # v2: per-mask side table. Flat-order across views, same order as feats_unit /
    # the K-means input. Each row of mask_features.npy aligns with mask_directory[i].
    mask_directory = []
    cursor2 = 0
    for v_idx, fname in enumerate(frame_names):
        N_v = per_frame_masks[v_idx].shape[0]
        areas = per_frame_areas[v_idx]
        for local_id in range(N_v):
            mask_directory.append({
                "global_id": cursor2 + local_id,
                "view_idx": v_idx,
                "frame_name": fname,
                "local_mask_id": local_id,
                "area_full_res": int(areas[local_id]),
            })
        cursor2 += N_v

    np.save(level_out / "bank.npy", bank)
    np.save(level_out / "index_maps.npy", index_maps)
    np.save(level_out / "mask_id_maps.npy", mask_id_maps)
    np.save(level_out / "mask_features.npy", feats_unit.astype(np.float32))
    with open(level_out / "mask_to_cluster.json", "w") as f:
        json.dump(mask_to_cluster, f, indent=2)
    with open(level_out / "mask_directory.json", "w") as f:
        json.dump(mask_directory, f, indent=2)

    meta = {
        "scene_key": scene_key,
        "level": level,
        "V": V,
        "H": H,
        "W": W,
        "M": M,
        "D": D,
        "N_total": int(N_total),
        "lambda": LAMBDA,
        "frame_order": frame_names,
        "kmeans": {"n_init": 10, "random_state": 0},
        "feature_norm": "L2 (unit)",
        "bank_shape": list(bank.shape),
        "bank_row_0_is_background": True,
        "index_map_dtype": "int16",
        "index_map_convention": "0 = background, 1..M = cluster id",
        "mask_id_map_convention": "0 = background, 1..N_v = mask_id + 1 (per-frame namespace)",
        "overlap_resolution": "smallest_area_wins",
        # v2 outputs
        "mask_features_shape": [int(N_total), int(D)],
        "mask_features_norm": "L2 (unit)",
        "mask_directory_entries": int(N_total),
    }
    with open(level_out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def write_depthsplat_chunk(scene_dir: Path,
                           output_dir: Path,
                           scene_key: str,
                           frames_meta: list,
                           H: int,
                           W: int,
                           fx_norm: float,
                           fy_norm: float,
                           cx_norm: float,
                           cy_norm: float) -> tuple:
    """Pack the same scene into a DepthSplat .torch chunk + eval JSON. Level-independent
    (geometry/RGB only). Returns (torch_rel_path, eval_rel_path) inside output_dir."""

    image_dir = scene_dir / "dslr" / "resized_images"
    cameras = []
    images_raw = []
    timestamps = []
    for v_idx, fr_meta in enumerate(frames_meta):
        fname = Path(fr_meta["file_path"]).name
        img_path = image_dir / fname
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"cv2.imread failed: {img_path}")
        ih, iw = img.shape[:2]
        if (iw, ih) != (W, H):
            raise ValueError(f"{fname}: image is {iw}x{ih} but transforms says {W}x{H}")

        _, encoded = cv2.imencode(".jpg", img)
        images_raw.append(torch.tensor(np.array(encoded), dtype=torch.uint8))

        blender_c2w = np.array(fr_meta["transform_matrix"], dtype=np.float64)
        opencv_c2w = blender_to_opencv_c2w(blender_c2w)
        opencv_w2c = np.linalg.inv(opencv_c2w)
        camera = [fx_norm, fy_norm, cx_norm, cy_norm, 0.0, 0.0]
        camera.extend(opencv_w2c[:3].flatten().tolist())
        cameras.append(camera)
        timestamps.append(v_idx)

    example = {
        "key": scene_key,
        "url": scene_key,
        "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        "cameras": torch.tensor(cameras, dtype=torch.float32),
        "images": images_raw,
    }

    datasets_dir = output_dir / "datasets" / f"{scene_key}_2view" / "test"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    torch.save([example], datasets_dir / "000000.torch")
    with open(datasets_dir / "index.json", "w") as f:
        json.dump({scene_key: "000000.torch"}, f)

    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    eval_path = assets_dir / f"{scene_key}_2view_eval.json"
    with open(eval_path, "w") as f:
        json.dump({scene_key: {"context": [0, 1], "target": [0, 1]}}, f)

    return (Path("datasets") / f"{scene_key}_2view" / "test" / "000000.torch",
            Path("assets") / f"{scene_key}_2view_eval.json",
            Path("datasets") / f"{scene_key}_2view" / "test" / "index.json")


def main():
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_key = args.scene_key if args.scene_key else output_dir.name
    levels = sorted(set(args.levels))

    transforms_path = scene_dir / "dslr" / "nerfstudio" / "transforms.json"
    with open(transforms_path, "r") as f:
        transforms = json.load(f)
    if transforms.get("camera_model") != "pinhole":
        raise ValueError(f"camera_model is {transforms.get('camera_model')!r}; expected pinhole.")

    W = transforms["w"]
    H = transforms["h"]
    fx_norm = transforms["fl_x"] / W
    fy_norm = transforms["fl_y"] / H
    cx_norm = transforms["cx"] / W
    cy_norm = transforms["cy"] / H

    frames_meta = transforms["frames"]
    frame_names = [Path(fr["file_path"]).name for fr in frames_meta]

    print(f"Scene: {scene_key}    V={len(frame_names)}  H={H} W={W}  levels={levels}")

    level_metas = {}
    for lvl in levels:
        level_metas[lvl] = process_level(lvl, scene_dir, output_dir, frame_names, H, W, scene_key)

    torch_rel, eval_rel, index_rel = write_depthsplat_chunk(
        scene_dir, output_dir, scene_key, frames_meta, H, W,
        fx_norm, fy_norm, cx_norm, cy_norm,
    )

    # Top-level scene meta: intrinsics + the per-level summary so callers can introspect
    # without opening every level_<N>/meta.json.
    top_meta = {
        "scene_key": scene_key,
        "V": len(frame_names),
        "H": H,
        "W": W,
        "frame_order": frame_names,
        "intrinsics_normalized": {"fx": fx_norm, "fy": fy_norm, "cx": cx_norm, "cy": cy_norm},
        "levels": levels,
        "level_summary": {
            str(lvl): {"M": level_metas[lvl]["M"], "N_total": level_metas[lvl]["N_total"]}
            for lvl in levels
        },
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(top_meta, f, indent=2)

    zip_path = output_dir / f"{scene_key}_hsegsplat_colab.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # DepthSplat dataset (level-independent)
        for rel in (torch_rel, index_rel, eval_rel):
            zf.writestr(str(rel), (output_dir / rel).read_bytes())
        # Top-level scene meta
        zf.writestr("segsplat/meta.json", (output_dir / "meta.json").read_text())
        # Per-level semantic inputs (v1 + v2 files)
        for lvl in levels:
            level_out = output_dir / f"level_{lvl}"
            for fn in ("bank.npy", "index_maps.npy", "mask_id_maps.npy",
                       "mask_features.npy"):
                zf.writestr(f"segsplat/{scene_key}/level_{lvl}/{fn}",
                            (level_out / fn).read_bytes())
            for fn in ("meta.json", "mask_to_cluster.json", "mask_directory.json"):
                zf.writestr(f"segsplat/{scene_key}/level_{lvl}/{fn}",
                            (level_out / fn).read_text())
        # v2 cross-level parent dict (optional — present only if compute_parent_chain.py ran)
        parents_path = output_dir / "per_mask" / "parents.json"
        if parents_path.exists():
            zf.writestr(f"segsplat/{scene_key}/per_mask/parents.json",
                        parents_path.read_text())
        else:
            print(f"  [warn] {parents_path} not found — run compute_parent_chain.py "
                  f"after this script to produce v2 parent dict.")

    print(f"\nWrote outputs to: {output_dir}")
    print(f"Wrote colab zip:  {zip_path}")
    print("\nColab usage:")
    print(f"  !unzip -o {zip_path.name}")
    print(f"  dataset.roots=[datasets/{scene_key}_2view]")
    print(f"  dataset.view_sampler.index_path=assets/{scene_key}_2view_eval.json")
    print(f"  segsplat assets at: segsplat/{scene_key}/level_{{1,3,6}}/")


if __name__ == "__main__":
    main()
