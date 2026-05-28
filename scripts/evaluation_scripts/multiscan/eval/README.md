# MultiScan AP / AP50 / AP25 eval (Search3D protocol)

## What this does

Takes H-SegSplat `gaussians.pt` outputs (from Colab) and computes Search3D's
3D part-instance-segmentation AP, AP50, AP25 metrics on the MultiScan
benchmark. Comparable directly to Search3D's Table II numbers.

## One-time setup

```bash
# 1. Search3D's evaluator (needed for AP scoring).
git clone https://github.com/aycatakmaz/search3d.git ~/Desktop/SegSplat/third_party/search3d
export PYTHONPATH=$HOME/Desktop/SegSplat/third_party/search3d:$PYTHONPATH

# 2. Python deps (most should be present already).
pip install scikit-learn open_clip_torch transformers sentencepiece
```

## Per-scene workflow (after Colab finishes for a scene)

```bash
# Download the Colab outputs (the last cell already does this):
#   data/Multiscan/colab_outputs/<scene>/gaussians.pt

# Then evaluate:
python scripts_multiscan/eval/run_multiscan_ap_eval.py \
    --gaussians_root data/Multiscan/colab_outputs \
    --ply_dir        data/Multiscan/multiscan_test_plys_only \
    --gt_dir         data/Multiscan/multiscan_annotations_search3d/ov_part_annotations \
    --align_dir      data/Multiscan/scenes \
    --scenes scene_00005_00 scene_00005_01 scene_00006_00 scene_00006_01 \
    --threshold 0.5 \
    --max_v2g_dist 0.3 \
    --out_dir data/Multiscan/eval_results
```

Add `--dry_run` to skip the Search3D evaluator and just write per-scene
prediction `.npz` files (useful for debugging the Gaussian-to-vertex bridge
before relying on the AP numbers).

## What gets written

```
Multiscan/eval_results/
├── predictions/<scene>.npz       # pred_classes, pred_scores, pred_masks per scene
├── gt_files.json                  # which GT files map to which scene
├── search3d_ap_results.txt        # the evaluator's text report (per-class + averages)
└── search3d_ap_results.json       # same data, JSON
```

## Caveats

1. **One instance per class per scene.** v1/v2's cluster-based representation
   can't separate multiple physical instances of the same class. We emit the
   union of vertices passing threshold as a single instance per
   `(object, part)` tuple. AP-style metrics that reward instance-level
   recall will penalize this. This is the same limitation noted in
   `PROJECT_PLAN.md §6.5.7`.

2. **Vertex-to-Gaussian via nearest-neighbor.** A PLY vertex inherits the
   nearest Gaussian's score. Vertices farther than `--max_v2g_dist` (default
   0.3 m) from any Gaussian are dropped — H-SegSplat reconstructs only
   what's visible in the 2 context views, so back-of-scene vertices are
   simply not predicted on. Report what fraction was in-range alongside
   the AP number.

3. **Score threshold = 0.5** matches SegSplat/LERF convention. At
   `tau=100` LERF saturates near-binary, so the threshold's exact value
   matters less than the (child, parent) tuple having SigLIP responses.

## Quick comparison to expected baselines (Search3D Table II)

| Method | AP | AP50 | AP25 |
|---|---|---|---|
| OpenScene | 3.2 | 5.5 | 13.7 |
| OpenMask3D | 3.1 | 6.2 | 18.2 |
| GARField+Search3D | 3.5 | 8.9 | 20.5 |
| **Search3D** | **7.9** | **14.5** | **31.5** |
| **H-SegSplat (ours)** | ? | ? | ? |

These are computed over Search3D's 47 (obj, part) tuples on the full
MultiScan test set (~120 scenes). Our subset (4 scenes initially) is
not directly comparable but should sit in the same ballpark per-scene.
