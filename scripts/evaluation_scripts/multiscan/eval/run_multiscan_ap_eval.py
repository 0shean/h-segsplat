#!/usr/bin/env python3
"""
MultiScan Search3D AP / AP50 / AP25 eval driver for H-SegSplat outputs.

Inputs (per scene):
    --gaussians_dir/<scene>/gaussians.pt        (from h-segsplat-repo Colab run)
    --ply_dir/<scene>.ply                       (multiscan_test_plys_only/*.ply)
    --gt_dir/<scene>_obj_part_inst.txt          (multiscan_annotations_search3d/ov_part_annotations/)
    --align_dir/<scene>/<scene>.align.json      (Multiscan/scenes/<scene>/.align.json)

What it does:
    1. Load gaussians.pt — extracts Gaussian means + v2 payload.
    2. Load PLY vertices in PLY (= scene) frame.
    3. Build KD-tree over Gaussian means; assign nearest Gaussian per vertex.
    4. For each of Search3D's 47 (obj, part) joint tuples, run v2 fluid-binding
       query (6→3, 6→1, 3→1; max) → per-Gaussian relevancy → per-vertex.
    5. Threshold each per-vertex score map → one binary instance per class.
    6. Call Search3D's `evaluate()` on the assembled preds + GT files.
    7. Print AP / AP50 / AP25 averaged across scenes.

Note: v1/v2 cluster-based queries produce ONE prediction per class per scene
(union of vertices passing threshold). This is conservative — multiple instances
of the same class collapse into one. Multi-instance separation is future work
(PROJECT_PLAN.md §6.5.7).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))                 # scripts_multiscan/eval/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))             # scripts_multiscan/
sys.path.insert(0, str(REPO_DIR / "Multiscan"))                          # for multiscan_search3d_constants

from gaussian_to_vertex import (
    build_kdtree, assign_gaussian_per_vertex, per_g_to_per_v,
    build_search3d_prediction,
)
from hsegsplat_offline_state import OfflineState, SigLIPTextEncoder, evaluate_v2_two_terms

# Import the joint-tuple constants from the local MultiScan dump.
from multiscan_search3d_constants import (   # type: ignore
    VALID_JOINT_TUPLE_NAMES, VALID_JOINT_SEMANTIC_IDS, VALID_JOINT_LABEL_LIST,
)

# Import the PLY reader from the sibling script.
from verify_projection import read_ply_vertices_with_color  # type: ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gaussians_root", type=Path, required=True,
                   help="Folder with <scene>/gaussians.pt sub-dirs (downloads from Colab).")
    p.add_argument("--ply_dir", type=Path, required=True,
                   help="data/Multiscan/multiscan_test_plys_only/")
    p.add_argument("--gt_dir", type=Path, required=True,
                   help="data/Multiscan/multiscan_annotations_search3d/ov_part_annotations/")
    p.add_argument("--align_dir", type=Path, required=True,
                   help="data/Multiscan/scenes/  (each scene has <scene>.align.json)")
    p.add_argument("--scenes", nargs="+", required=True,
                   help="Scene IDs (e.g. scene_00005_00).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="LERF relevancy threshold for binarizing per-vertex masks.")
    p.add_argument("--max_v2g_dist", type=float, default=0.3,
                   help="Vertices farther than this from their nearest Gaussian are dropped "
                        "(no prediction). Units: same as PLY (meters).")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for SigLIP encoding. 'cpu' is fine for 47 tuples.")
    p.add_argument("--out_dir", type=Path, default=Path("data/Multiscan/eval_results"))
    p.add_argument("--dry_run", action="store_true",
                   help="Skip the actual Search3D evaluate() call; only run the bridge "
                        "and write predictions.json (useful for debugging the bridge).")
    return p.parse_args()


def load_align(align_path: Path) -> np.ndarray:
    with open(align_path) as f:
        data = json.load(f)
    flat = data["coordinate_transform"]
    M = np.array(flat, dtype=np.float64).reshape(4, 4).T  # column-major -> standard
    return M


def transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 M to Nx3 points."""
    h = np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=-1)
    return (M @ h.T).T[:, :3]


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Build the (child, parent) query list -----
    # VALID_JOINT_TUPLE_NAMES uses underscores in some names; replace with spaces
    # so SigLIP gets natural English.
    queries = []
    for sem_id, tup, label in zip(VALID_JOINT_SEMANTIC_IDS,
                                   VALID_JOINT_TUPLE_NAMES,
                                   VALID_JOINT_LABEL_LIST):
        obj_name = tup[0].replace("_", " ")
        part_name = tup[1].replace("_", " ")
        queries.append({
            "sem_id": int(sem_id),
            "child": part_name,
            "parent": obj_name,
            "label": label,
        })
    print(f"[eval] Built {len(queries)} (part, object) queries from Search3D's "
          f"VALID_JOINT_TUPLE_NAMES")

    # ----- Shared SigLIP text encoder -----
    print("[eval] Loading SigLIP ...")
    encoder = SigLIPTextEncoder(device=args.device)

    # ----- Per-scene -----
    per_scene_preds: dict[str, dict] = {}
    per_scene_gt_files: dict[str, str] = {}
    for scene in args.scenes:
        print(f"\n[eval] === {scene} ===")
        g_pt = args.gaussians_root / scene / "gaussians.pt"
        ply_path = args.ply_dir / f"{scene}.ply"
        gt_path = args.gt_dir / f"{scene}_obj_part_inst.txt"
        align_path = args.align_dir / scene / f"{scene}.align.json"
        for p, label in [(g_pt, "gaussians.pt"), (ply_path, "ply"),
                         (gt_path, "gt"), (align_path, "align.json")]:
            if not p.exists():
                print(f"  [warn] missing {label}: {p} — skipping scene")
                break
        else:
            state = OfflineState(g_pt, device=args.device)

            # Load PLY vertices and transform from PLY frame to gaussian world frame.
            # gaussians.pt's means are in the same world as transforms.json frames,
            # which we built as: c2w = align @ T_arkit. So the world frame in
            # gaussians.pt = "align-applied ARKit world" = PLY frame.
            # So PLY vertices are already in the right frame.
            xyz_ply, _ = read_ply_vertices_with_color(ply_path)
            print(f"  PLY verts: {len(xyz_ply)}")

            # KD-tree over Gaussians.
            nn = build_kdtree(state.means)
            v2g, d_v2g = assign_gaussian_per_vertex(nn, xyz_ply.astype(np.float32))
            n_in_range = int((d_v2g <= args.max_v2g_dist).sum())
            print(f"  vertices in range (<= {args.max_v2g_dist}m of any Gaussian): "
                  f"{n_in_range}/{len(xyz_ply)} ({100*n_in_range/len(xyz_ply):.1f}%)")

            # Run all 47 queries.
            per_class_per_v: dict[int, np.ndarray] = {}
            for q in queries:
                per_g = evaluate_v2_two_terms(state, encoder, q["child"], q["parent"])
                per_v = per_g_to_per_v(per_g, v2g)
                per_class_per_v[q["sem_id"]] = per_v
            # Summary
            n_hit = sum(int((sc >= args.threshold).any()) for sc in per_class_per_v.values())
            print(f"  classes with ≥1 vertex above threshold {args.threshold}: "
                  f"{n_hit}/{len(queries)}")

            pred = build_search3d_prediction(per_class_per_v,
                                              score_threshold=args.threshold,
                                              max_distance=d_v2g,
                                              max_distance_cutoff=args.max_v2g_dist)
            print(f"  emitted predictions: {len(pred['pred_classes'])}")
            per_scene_preds[scene] = pred
            per_scene_gt_files[scene] = str(gt_path)

    # Persist predictions for later re-use.
    out_pred_dir = args.out_dir / "predictions"
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    for scene, pred in per_scene_preds.items():
        np.savez(out_pred_dir / f"{scene}.npz",
                 pred_classes=pred["pred_classes"],
                 pred_scores=pred["pred_scores"],
                 pred_masks=pred["pred_masks"])
    with open(args.out_dir / "gt_files.json", "w") as f:
        json.dump(per_scene_gt_files, f, indent=2)
    print(f"\n[eval] Wrote per-scene predictions to {out_pred_dir}/")

    if args.dry_run:
        print("[eval] --dry_run requested: skipping Search3D evaluate()")
        return

    # ----- Call Search3D evaluator -----
    # We dynamically import their util_3d + the OV part eval module. The user
    # must clone the search3d repo locally and have it on PYTHONPATH, e.g.:
    #     git clone https://github.com/aycatakmaz/search3d.git
    #     export PYTHONPATH=$PWD/search3d:$PYTHONPATH
    # Find search3d clone — default: <repo>/third_party/search3d
    s3d_root = REPO_DIR / "third_party" / "search3d"
    if not s3d_root.exists():
        print(f"[eval] search3d not found at {s3d_root}. Clone it first:\n"
              f"    git clone https://github.com/aycatakmaz/search3d.git {s3d_root}\n"
              f"or pass --dry_run.")
        return
    s3d_eval_dir = s3d_root / "search3d" / "benchmark" / "evaluation" / "multiscan"
    sys.path.insert(0, str(s3d_root))               # for 'search3d.benchmark.*' package
    sys.path.insert(0, str(s3d_eval_dir))           # for sibling 'import util', 'import util_3d'
    try:
        from search3d.benchmark.evaluation.multiscan.eval_semantic_instance_parts_OV import evaluate  # type: ignore
    except ImportError as e:
        print(f"[eval] failed to import search3d evaluator ({e}). Aborting.")
        return

    # Their evaluate(preds, gt_path, output_file, dataset="scannet") expects:
    #   - preds: {scene_id: {pred_classes, pred_scores, pred_masks}}
    #   - gt_path: DIRECTORY containing <scene_id>_obj_part_inst.txt files.
    output_file = str(args.out_dir / "search3d_ap_results.txt")
    print(f"[eval] calling Search3D evaluate() with gt_path={args.gt_dir} ...")
    avgs = evaluate(preds=per_scene_preds,
                    gt_path=str(args.gt_dir),
                    output_file=output_file,
                    dataset="multiscan")
    print(f"[eval] wrote results to {output_file}")
    # Persist averaged metrics as JSON too.
    metrics_json = args.out_dir / "search3d_ap_results.json"
    try:
        # avgs is typically a nested dict — make it JSON-friendly.
        import json as _json
        def _convert(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            return o
        with open(metrics_json, "w") as f:
            _json.dump(avgs, f, indent=2, default=_convert)
        print(f"[eval] also wrote {metrics_json}")
    except Exception as e:
        print(f"[eval] failed to dump avgs to JSON ({e}); raw text file is at {output_file}")


if __name__ == "__main__":
    main()
