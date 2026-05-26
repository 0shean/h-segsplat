#!/usr/bin/env python3
"""
H-SegSplat queries on rendered feature maps. Three modes:

  Flat (1 term):
      --child "chair" --child-level 1
  Parent (2 terms):
      --child "leg" --parent "chair"
  Grandparent (3 terms):
      --child "leg" --parent "arm" --grandparent "chair"

For each provided term, compute LERF relevancy (SegSplat eq. 4 / LERF Kerr et al. 2023)
against the level's bank, threshold at TAU, then AND across all provided levels.

Default levels when not overridden:
    --child-level       6     (finest)
    --parent-level      3
    --grandparent-level 1     (coarsest)

Inputs (under --scene_dir):
    rendered_rgb.npy                       (V, H, W, 3)         float32
    rendered_feature_map_lvl{1,3,6}.npy    (V, H, W, M_l + 1)   float32
    level_{1,3,6}/bank.npy                 (M_l + 1, D)         float32
    gaussians.pt                            (used only as bank fallback if level dirs missing)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import open_clip


CANONICAL_PHRASES = ["object", "things", "stuff"]
DEFAULT_LEVEL_CHILD = 6
DEFAULT_LEVEL_PARENT = 3
DEFAULT_LEVEL_GRANDPARENT = 1
TAU_LERF_TEMPERATURE = 100.0  # LERF paper / SegSplat: large temperature on cosine sim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", type=str, required=True,
                   help="Scene dir with rendered_*.npy + level_<N>/ subfolders.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Defaults to <scene_dir>/queries/.")

    p.add_argument("--child", type=str, required=True,
                   help="Required: finest-level term, e.g. 'leg'.")
    p.add_argument("--parent", type=str, default=None,
                   help="Optional: middle-level term, e.g. 'chair' (or 'arm' if grandparent given).")
    p.add_argument("--grandparent", type=str, default=None,
                   help="Optional: coarsest-level term, e.g. 'chair'.")

    p.add_argument("--child-level", type=int, default=DEFAULT_LEVEL_CHILD)
    p.add_argument("--parent-level", type=int, default=DEFAULT_LEVEL_PARENT)
    p.add_argument("--grandparent-level", type=int, default=DEFAULT_LEVEL_GRANDPARENT)

    p.add_argument("--threshold", type=float, default=0.5,
                   help="LERF relevancy threshold (SegSplat default 0.5).")
    p.add_argument("--mass-threshold", type=float, default=0.05,
                   help="Pixels with accumulated foreground mass below this -> 0 relevancy.")

    p.add_argument("--model-name", type=str, default="ViT-SO400M-14-SigLIP")
    p.add_argument("--pretrained", type=str, default="webli")

    return p.parse_args()


# ----------------------------------------------------------------------------
# Per-level loading + LERF
# ----------------------------------------------------------------------------

def load_level_inputs(scene_dir: Path, level: int):
    """Returns (feat_map, bank) as numpy arrays."""
    feat_path = scene_dir / f"rendered_feature_map_lvl{level}.npy"
    bank_path = scene_dir / f"level_{level}" / "bank.npy"
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)
    if not bank_path.exists():
        raise FileNotFoundError(bank_path)
    feat = np.load(feat_path)   # (V, H, W, M+1)
    bank = np.load(bank_path)   # (M+1, D)
    if bank.shape[0] != feat.shape[-1]:
        raise ValueError(
            f"level {level}: bank rows {bank.shape[0]} != feat channels {feat.shape[-1]}"
        )
    return feat, bank


def compute_F_unit_and_fg(feat_map: np.ndarray, bank: np.ndarray, mass_thresh: float):
    """F(v) = E(v) @ B, L2-normalized to unit norm. Returns (F_unit, fg_mask).

    Mass = sum over real cluster channels (skipping slot 0 = background) — this is
    the accumulated *semantic* alpha mass on real clusters at this pixel."""
    F = feat_map @ bank                              # (V, H, W, D)
    F_norm = np.linalg.norm(F, axis=-1, keepdims=True)
    F_unit = F / np.maximum(F_norm, 1e-12)
    fg_mass = feat_map[..., 1:].sum(axis=-1)         # (V, H, W)
    fg_mask = fg_mass >= mass_thresh
    return F_unit, fg_mask


def lerf_relevancy(F_unit: np.ndarray,
                   phi_query: np.ndarray,
                   phi_canonical: np.ndarray,
                   temperature: float = TAU_LERF_TEMPERATURE) -> np.ndarray:
    """LERF / SegSplat eq. 4. F_unit: (V, H, W, D); phi_query: (D,); phi_canonical: (C, D).

    relevancy = min over canonical c of
                exp(τ * sim_q) / (exp(τ * sim_q) + exp(τ * sim_c))
              = sigmoid(τ * (sim_q - sim_c)).

    Returns (V, H, W) in (0, 1)."""
    sim_q = (F_unit * phi_query).sum(axis=-1)                 # (V, H, W)
    sim_c = F_unit @ phi_canonical.T                          # (V, H, W, C)
    # Pairwise scores: sigmoid(τ * (sim_q - sim_c[..., c])) for each canonical c, take min.
    diff = temperature * (sim_q[..., None] - sim_c)           # (V, H, W, C)
    pair_score = 1.0 / (1.0 + np.exp(-diff))
    return pair_score.min(axis=-1)                            # (V, H, W)


# ----------------------------------------------------------------------------
# SigLIP text encoding
# ----------------------------------------------------------------------------

def encode_text(model, tokenizer, texts):
    with torch.no_grad():
        tokens = tokenizer(texts)
        emb = model.encode_text(tokens).float()
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return emb.cpu().numpy()


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------

def make_panel(rgb: np.ndarray,
               rel_per_level: dict,
               combined_rel: np.ndarray,
               threshold: float,
               title: str) -> np.ndarray:
    """Side-by-side panel: RGB | per-level heatmaps | combined-AND highlight.

    rgb: (H, W, 3) float in [0,1]
    rel_per_level: {level: (H, W) in [0,1]}
    combined_rel: (H, W) in [0,1] — final AND result (relevancy where all pass, else 0)
    """
    H, W = rgb.shape[:2]
    rgb_bgr = cv2.cvtColor((rgb * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    panels = [("RGB", rgb_bgr)]
    for lvl in sorted(rel_per_level.keys()):
        h = rel_per_level[lvl]
        heat = cv2.applyColorMap((np.clip(h, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        panels.append((f"lvl {lvl}", heat))

    # Combined view: only show pixels passing the AND, blended on RGB.
    mask = combined_rel >= threshold
    heat_combined = cv2.applyColorMap((np.clip(combined_rel, 0, 1) * 255).astype(np.uint8),
                                      cv2.COLORMAP_JET)
    overlay = rgb_bgr.copy()
    overlay[mask] = (0.4 * rgb_bgr[mask] + 0.6 * heat_combined[mask]).astype(np.uint8)
    panels.append(("AND", overlay))

    label_h = 32
    canvas = np.full((H + label_h, W * len(panels), 3), 30, dtype=np.uint8)
    for i, (label, img) in enumerate(panels):
        x0 = i * W
        canvas[label_h:label_h + H, x0:x0 + W] = img
        cv2.putText(canvas, label, (x0 + 10, label_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

    # Title strip on top.
    title_strip = np.full((label_h, canvas.shape[1], 3), 50, dtype=np.uint8)
    cv2.putText(title_strip, title, (10, label_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    return np.concatenate([title_strip, canvas], axis=0)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    out_dir = Path(args.output_dir) if args.output_dir else scene_dir / "queries"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the term list from explicit flags. Skip levels with no term.
    terms = {}  # level -> term
    if args.child is not None:
        terms[args.child_level] = args.child
    if args.parent is not None:
        if args.parent_level in terms:
            raise ValueError(f"--child-level and --parent-level both = {args.parent_level}")
        terms[args.parent_level] = args.parent
    if args.grandparent is not None:
        if args.grandparent_level in terms:
            raise ValueError(f"--grandparent-level collides with another level: {args.grandparent_level}")
        terms[args.grandparent_level] = args.grandparent

    print(f"Query terms by level: {terms}")

    # Load each level's rendered feature map + bank, build F_unit + fg mask.
    rgb = np.load(scene_dir / "rendered_rgb.npy")  # (V, H, W, 3)
    V, H, W, _ = rgb.shape
    print(f"Scene: V={V} H={H} W={W}")

    per_level_inputs = {}
    for lvl in terms:
        feat, bank = load_level_inputs(scene_dir, lvl)
        if feat.shape[:3] != (V, H, W):
            raise RuntimeError(f"level {lvl} feature shape {feat.shape[:3]} != ({V},{H},{W})")
        F_unit, fg_mask = compute_F_unit_and_fg(feat, bank, args.mass_threshold)
        per_level_inputs[lvl] = {"F_unit": F_unit, "fg_mask": fg_mask, "D": bank.shape[1]}
        print(f"  level {lvl}: feat_channels={feat.shape[-1]}  D={bank.shape[1]}  "
              f"fg_pixels={int(fg_mask.sum())}/{fg_mask.size}")

    # All levels share the same SigLIP D (we use one model). Sanity-check.
    Ds = {lvl: per_level_inputs[lvl]["D"] for lvl in per_level_inputs}
    if len(set(Ds.values())) != 1:
        raise RuntimeError(f"levels disagree on SigLIP D: {Ds}")
    D = next(iter(Ds.values()))

    # Encode SigLIP for all terms + canonical phrases in one pass.
    print(f"Loading {args.model_name} ({args.pretrained}) ...")
    model, _, _ = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()
    all_text = list(terms.values()) + CANONICAL_PHRASES
    all_emb = encode_text(model, tokenizer, all_text)  # (num_terms + 3, D)
    term_embs = {lvl: all_emb[i] for i, lvl in enumerate(terms)}
    phi_canonical = all_emb[len(terms):]               # (3, D)
    if D != all_emb.shape[1]:
        raise RuntimeError(f"SigLIP D={all_emb.shape[1]} but banks have D={D}")

    # Per-level LERF relevancy.
    rel_per_level = {}  # level -> (V, H, W)
    for lvl, term in terms.items():
        F_unit = per_level_inputs[lvl]["F_unit"]
        fg = per_level_inputs[lvl]["fg_mask"]
        rel = lerf_relevancy(F_unit, term_embs[lvl], phi_canonical)
        rel = np.where(fg, rel, 0.0)
        rel_per_level[lvl] = rel
        peak = float(rel.max())
        peak_v = int(rel.reshape(V, -1).max(axis=1).argmax())
        n_pass = int((rel >= args.threshold).sum())
        print(f'  level {lvl}, term "{term}": peak={peak:.3f} (view {peak_v})  '
              f'pixels>=tau: {n_pass}')

    # Combine: AND across levels — keep relevancy as the min, mask out pixels failing any level.
    combined_rel = np.ones_like(rgb[..., 0])  # (V, H, W) start at 1.0
    for lvl, rel in rel_per_level.items():
        combined_rel = np.minimum(combined_rel, rel)
    # If only one level was queried, combined_rel == that single level (correct).

    n_combined = int((combined_rel >= args.threshold).sum())
    print(f"Combined (AND) pixels >= {args.threshold}: {n_combined}")

    # Filename: child[_of_parent[_of_grandparent]] (omits parts not provided).
    def safe(s):
        return s.replace(" ", "_")
    name_parts = [safe(args.child)]
    if args.parent:
        name_parts.append("of_" + safe(args.parent))
    if args.grandparent:
        name_parts.append("of_" + safe(args.grandparent))
    query_name = "_".join(name_parts)

    # Title for the panel: "leg of arm of chair  (lvl 6 / lvl 3 / lvl 1)".
    title_terms = [args.child]
    if args.parent:
        title_terms.append(args.parent)
    if args.grandparent:
        title_terms.append(args.grandparent)
    title_levels = sorted(terms.keys(), reverse=True)
    title = (" of ".join(title_terms)
             + "    levels: " + " / ".join(f"lvl {l}" for l in title_levels)
             + f"    tau={args.threshold}")

    for v in range(V):
        rel_per_v = {lvl: rel[v] for lvl, rel in rel_per_level.items()}
        panel = make_panel(rgb[v], rel_per_v, combined_rel[v], args.threshold, title)
        out_path = out_dir / f"query_{query_name}_view{v}.png"
        cv2.imwrite(str(out_path), panel)
        print(f"  wrote {out_path}")

    # Also save the raw combined relevancy array for downstream use.
    np.save(out_dir / f"query_{query_name}_relevancy.npy", combined_rel.astype(np.float32))
    print(f"\nDone. Outputs under: {out_dir}/")


if __name__ == "__main__":
    main()
