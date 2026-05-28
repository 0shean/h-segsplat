#!/usr/bin/env python3
"""
Local experiment: does re-extracting SigLIP for lvl 1 with
(a) mask-zeroed-only crop (drop the bg-kept average), and
(b) NMS-dropped overlapping masks
actually move the per-class IoU on the sofa scene?

We re-use the existing rendered_feature_map_targets_lvl1.npy (alpha-blended
per-cluster mass) and only swap out the SigLIP features. This isolates the
SigLIP-feature-quality effect from the K-means cluster-boundary effect.

For each *original* cluster c, we compute its per-class LERF score by
averaging the new per-mask LERF scores of all masks the build script
assigned to that cluster. Then for each target-view pixel:

    per_pixel_class_score = E_real(pixel) @ new_cluster_class_score

argmax over classes -> predicted class. Compare to GT masks.

Crop strategies tested:
    A. baseline (stored features)            = unchanged
    B. mask-zeroed only                       = SegSplat's CLIP recipe
    C. mask-zeroed only + NMS on masks        = (B) + drop overlapping duplicates
    D. mask-zeroed only + NMS + tile-sample   = same as C, but for masks >= 50% of image
                                                use random 224x224 tiles inside the mask

The output is a small CSV comparing per-class IoUs.
"""

from pathlib import Path
import json
import sys
import argparse

import numpy as np
import cv2
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "scripts_multiscan" / "eval"))
from hsegsplat_offline_state import SigLIPTextEncoder  # noqa: E402


SCENE_DIR = REPO_DIR / "3dovs_sofa"
INGESTED_DIR = REPO_DIR / "3D-OVS" / "ingested" / "sofa"
COLAB_DIR = REPO_DIR / "3D-OVS" / "sofa" / "colab_output"

CLASSES = ['Pikachu', 'a stack of UNO cards', 'a red Nintendo Switch joy-con controller',
           'Gundam', 'Xbox wireless controller', 'grey sofa']
NMS_IOU_THRESHOLD = 0.5   # SAM-style: drop the lower-confidence mask if IoU >= this
LARGE_MASK_AREA_FRAC = 0.50  # for strategy D: when mask covers >= 50% of image
N_TILES = 8


def load_image_rgb(p: Path) -> np.ndarray:
    img = cv2.imread(str(p))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_masks(frame_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Returns (masks_bool[N, H, W], metadata list)."""
    meta = json.load(open(frame_dir / "metadata.json"))
    masks = []
    for entry in meta:
        m = cv2.imread(str(frame_dir / entry["mask_file"]), cv2.IMREAD_UNCHANGED)
        if m.ndim == 3:
            m = m[..., 0]
        masks.append(m > 0)
    return np.stack(masks, 0), meta


def nms_masks(masks: np.ndarray, meta: list[dict],
              iou_thresh: float = NMS_IOU_THRESHOLD,
              iom_thresh: float = 0.85) -> tuple[np.ndarray, list[int]]:
    """SAM-style NMS by Intersection-over-Min. Returns (kept_masks, kept_idx_in_original).
    Score = predicted_iou * stability_score (the SAM filter score)."""
    N = len(masks)
    scores = np.array([float(m.get("predicted_iou", 1.0)) *
                       float(m.get("stability_score", 1.0)) for m in meta])
    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(N, dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        a = masks[i]
        for j in order:
            if j == i or suppressed[j]:
                continue
            b = masks[j]
            inter = int((a & b).sum())
            if inter == 0:
                continue
            iou = inter / int((a | b).sum())
            iom = inter / min(int(a.sum()), int(b.sum()))
            if iou >= iou_thresh or iom >= iom_thresh:
                suppressed[j] = True
    keep.sort()
    return masks[keep], keep


def setup_clip():
    import open_clip
    import torchvision
    model, _, _ = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP',
                                                         pretrained='webli', device='cpu')
    model.eval()
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return model, preprocess


def pad_square_and_encode(model, preprocess, img_uint8_HWC: np.ndarray) -> np.ndarray:
    """Encode a single uint8 RGB crop. img_uint8_HWC must already be RGB."""
    h, w = img_uint8_HWC.shape[:2]
    L = max(h, w)
    pad = np.zeros((L, L, 3), dtype=np.uint8)
    if h > w:
        pad[:, (L - w) // 2:(L - w) // 2 + w] = img_uint8_HWC
    else:
        pad[(L - h) // 2:(L - h) // 2 + h, :] = img_uint8_HWC
    t = torch.from_numpy(pad.transpose(2, 0, 1).copy()).float() / 255.0
    inp = preprocess(t).unsqueeze(0)
    with torch.no_grad():
        feat = model.encode_image(inp)[0].cpu().numpy()
    return feat


def tight_bbox(mask: np.ndarray):
    rows = mask.any(axis=1); cols = mask.any(axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(y1), int(y2), int(x1), int(x2)


def encode_mask(model, preprocess, img_rgb: np.ndarray, mask: np.ndarray,
                strategy: str) -> np.ndarray:
    """Encodes a single mask with the given strategy. Returns L2-normalized SigLIP feature."""
    bbox = tight_bbox(mask)
    if bbox is None:
        return np.zeros(1152, dtype=np.float32)
    y1, y2, x1, x2 = bbox

    if strategy == "mask_zeroed":
        crop = img_rgb.copy()
        crop[~mask] = 0
        crop = crop[y1:y2+1, x1:x2+1]
        feat = pad_square_and_encode(model, preprocess, crop)

    elif strategy == "mask_zeroed_large_uses_tiles":
        area_frac = mask.sum() / float(mask.size)
        if area_frac >= LARGE_MASK_AREA_FRAC:
            # Tile-sample: collect tiles wholly inside mask
            tiles = []
            H, W = img_rgb.shape[:2]
            tile = 224
            rng = np.random.default_rng(seed=int(mask.sum()) & 0xffff)
            tries = 0
            while len(tiles) < N_TILES and tries < 1000:
                tries += 1
                yy = int(rng.integers(0, H - tile + 1))
                xx = int(rng.integers(0, W - tile + 1))
                if mask[yy:yy+tile, xx:xx+tile].mean() >= 0.95:
                    tiles.append(img_rgb[yy:yy+tile, xx:xx+tile])
            if len(tiles) == 0:
                # Fallback: mask-zeroed bbox.
                crop = img_rgb.copy(); crop[~mask] = 0
                crop = crop[y1:y2+1, x1:x2+1]
                feat = pad_square_and_encode(model, preprocess, crop)
            else:
                fs = [pad_square_and_encode(model, preprocess, t) for t in tiles]
                feat = np.mean(fs, axis=0)
        else:
            crop = img_rgb.copy(); crop[~mask] = 0
            crop = crop[y1:y2+1, x1:x2+1]
            feat = pad_square_and_encode(model, preprocess, crop)

    elif strategy == "stored":
        raise NotImplementedError("'stored' is read directly from disk, not encoded here")

    else:
        raise ValueError(f"unknown strategy: {strategy}")

    n = float(np.linalg.norm(feat))
    if n > 1e-9:
        feat = feat / n
    return feat.astype(np.float32)


def compute_iou(p: np.ndarray, g: np.ndarray) -> float:
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return float("nan")
    return inter / union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+",
                    default=["stored", "mask_zeroed", "mask_zeroed_nms",
                             "mask_zeroed_nms_tiles"])
    ap.add_argument("--tau", type=float, default=100.0)
    args = ap.parse_args()

    encoder = SigLIPTextEncoder(device="cpu")
    class_embs = np.stack([encoder(c) for c in CLASSES], 0)
    phi_canon = encoder.phi_canonical

    # Build per-mask features and per-mask LERF for each strategy.
    # The build pipeline pools masks across views into a single global flat list
    # ordered by view then by within-view local id. We follow the same order so
    # mask_to_cluster.json indexes line up.
    frames = ["sofa_v015", "sofa_v024"]
    mask_to_cluster = json.load(open(SCENE_DIR / "level_1" / "mask_to_cluster.json"))

    feat_paths_by_strategy = {}
    for strat in args.strategies:
        if strat == "stored":
            feats_flat = []
            for fname in frames:
                f = np.load(SCENE_DIR / "masks_lvl_1" / fname / "siglip_embeddings.npy")
                feats_flat.append(f)
            feats_flat = np.concatenate(feats_flat, axis=0)
            norms = np.linalg.norm(feats_flat, axis=-1, keepdims=True)
            feats_flat = feats_flat / np.maximum(norms, 1e-9)
            feat_paths_by_strategy[strat] = (feats_flat, None)  # None = no NMS applied
            continue

        print(f"Encoding strategy {strat!r} ...")
        model, preprocess = setup_clip()
        per_frame_feats = []
        per_frame_kept = []   # list of (frame_local_idx -> bool keep) for NMS variants
        for fname in frames:
            frame_dir = SCENE_DIR / "masks_lvl_1" / fname
            masks, meta = load_masks(frame_dir)
            img_rgb = load_image_rgb(SCENE_DIR / "dslr" / "resized_images" / f"{fname}.JPG")
            if "nms" in strat:
                kept_masks, kept_idx = nms_masks(masks, meta)
            else:
                kept_masks, kept_idx = masks, list(range(len(masks)))
            sub_strat = ("mask_zeroed_large_uses_tiles" if "tiles" in strat
                         else "mask_zeroed")
            feats = []
            for i, m in enumerate(kept_masks):
                feats.append(encode_mask(model, preprocess, img_rgb, m, sub_strat))
            per_frame_feats.append(np.stack(feats, 0))
            per_frame_kept.append(kept_idx)
            print(f"  {fname}: encoded {len(kept_idx)}/{len(masks)} masks "
                  f"(kept indices: {kept_idx})")
        feats_flat = np.concatenate(per_frame_feats, axis=0)
        feat_paths_by_strategy[strat] = (feats_flat, per_frame_kept)

    # For each strategy: per-mask LERF, then per-original-cluster aggregation.
    # We always re-use the ORIGINAL mask_to_cluster.json so the cluster pooling
    # structure is unchanged --- we're only testing the SigLIP-feature swap.
    # Important: when NMS dropped a mask, that mask's slot in the original cluster
    # gets no contribution; the cluster's score becomes the mean over the kept masks.
    n_classes = len(CLASSES)
    M_clusters = int(json.load(open(SCENE_DIR / "level_1" / "meta.json"))["M"])

    results_per_strategy = {}
    for strat, (feats_flat, kept_per_frame) in feat_paths_by_strategy.items():
        # build "flat_idx_in_original -> feat_idx_in_kept_flat" map.
        # The mask_to_cluster.json is per-frame indexed by within-frame local id.
        # Original flat indexing: view 0 masks first, then view 1.
        orig_to_kept_global = {}  # (frame_idx, local_id) -> kept_flat_idx, or None if dropped
        if kept_per_frame is None:
            # 'stored' strategy: feats_flat is already in original flat order
            cursor = 0
            for fi, fname in enumerate(frames):
                meta = json.load(open(SCENE_DIR / "masks_lvl_1" / fname / "metadata.json"))
                for li in range(len(meta)):
                    orig_to_kept_global[(fi, li)] = cursor
                    cursor += 1
        else:
            cursor = 0
            for fi, kept in enumerate(kept_per_frame):
                meta = json.load(open(SCENE_DIR / "masks_lvl_1" / frames[fi] / "metadata.json"))
                kept_set = set(kept)
                for li in range(len(meta)):
                    if li in kept_set:
                        orig_to_kept_global[(fi, li)] = cursor
                        cursor += 1
                    else:
                        orig_to_kept_global[(fi, li)] = None

        # Per-mask LERF: (N_kept, n_classes)
        sim_q = feats_flat @ class_embs.T
        sim_c = feats_flat @ phi_canon.T
        diff = args.tau * (sim_q[:, :, None] - sim_c[:, None, :])
        mask_lerf = (1.0 / (1.0 + np.exp(-diff))).min(axis=-1)

        # Per-cluster aggregation: average over the surviving masks of that cluster.
        cluster_class_score = np.zeros((M_clusters, n_classes), dtype=np.float32)
        for fname, frame_clusters in mask_to_cluster.items():
            fi = frames.index(fname.replace(".JPG", ""))
            for li, c in enumerate(frame_clusters):
                kept_idx = orig_to_kept_global.get((fi, li))
                if kept_idx is None:
                    continue
                cluster_class_score[c] += mask_lerf[kept_idx]
        # Count survivors per cluster and divide.
        cluster_count = np.zeros(M_clusters, dtype=np.int64)
        for fname, frame_clusters in mask_to_cluster.items():
            fi = frames.index(fname.replace(".JPG", ""))
            for li, c in enumerate(frame_clusters):
                if orig_to_kept_global.get((fi, li)) is not None:
                    cluster_count[c] += 1
        nonzero = cluster_count > 0
        cluster_class_score[nonzero] = (cluster_class_score[nonzero]
                                         / cluster_count[nonzero, None])

        # Evaluate per target view.
        rendered = np.load(COLAB_DIR / "rendered_feature_map_targets_lvl1.npy")  # (T, H, W, M+1)
        targets = json.load(open(INGESTED_DIR / "target_views.json"))["targets"]
        gt_root = INGESTED_DIR / "gt_masks"
        T, H, W, _ = rendered.shape

        per_class_iou = {c: [] for c in CLASSES}
        for ti, tgt in enumerate(targets):
            E_real = rendered[ti, ..., 1:]
            fg_mass = E_real.sum(-1)
            # Per-pixel per-class score
            per_class = (E_real.reshape(-1, M_clusters)
                         @ cluster_class_score).reshape(H, W, n_classes)
            # mass-weight + argmax
            per_class = per_class * fg_mass[..., None]
            pred = per_class.argmax(-1)
            valid = fg_mass >= 0.05
            pred[~valid] = -1
            for ci, cls in enumerate(CLASSES):
                gt_path = gt_root / tgt["view_id"] / f"{cls}.png"
                if not gt_path.exists():
                    continue
                g = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 0
                p = (pred == ci)
                per_class_iou[cls].append(compute_iou(p, g))

        scene_iou = {c: float(np.nanmean(per_class_iou[c]))
                      if per_class_iou[c] else float("nan") for c in CLASSES}
        valid_ious = [v for v in scene_iou.values() if not np.isnan(v)]
        scene_miou = float(np.mean(valid_ious))
        results_per_strategy[strat] = {"per_class": scene_iou, "miou": scene_miou}

    # Print comparison table.
    print("\n=== Per-class IoU on sofa (lvl 1 only, target views combined) ===")
    print(f"  {'class':<42s} " + " ".join(f"{s:>22s}" for s in args.strategies))
    for cls in CLASSES:
        row = f"  {cls:<42s} "
        for s in args.strategies:
            row += f"{results_per_strategy[s]['per_class'][cls]:>22.3f} "
        print(row)
    print(f"  {'mIoU':<42s} " +
          " ".join(f"{results_per_strategy[s]['miou']:>22.4f}" for s in args.strategies))


if __name__ == "__main__":
    main()
