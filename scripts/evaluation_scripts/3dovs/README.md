# 3D-OVS evaluation (SegSplat parity table)

Reproduces SegSplat's 3D-OVS mIoU on 8 scenes (bed, bench, blue_sofa, lawn,
office_desk, room, snacks, sofa). Comparable directly to:

- SegSplat paper Table on 3D-OVS
- N2F2 Table 3 (bed / bench / room / sofa / lawn)
- LangSplat / 3D-OVS / OV-Seg numbers in the same N2F2 table

## Workflow

### A. Local ingest (already done)

```bash
# For each scene:
python scripts_3dovs/ingest_3dovs.py \
    --scene_dir data/3D-OVS/<scene> \
    --out_dir data/3D-OVS/ingested/<scene>
```

Output:
- `data/3D-OVS/ingested/<scene>/dslr/{nerfstudio,resized_images}/`  (2 context views, 960x640)
- `data/3D-OVS/ingested/<scene>/target_views.json`                  (5 target poses + classes)
- `data/3D-OVS/ingested/<scene>/gt_masks/<view_id>/<class>.png`     (resized binary GT)

Staged for Colab upload:
- `data/3D-OVS/staged_zips/3dovs_<scene>.zip` — contains `dslr/` + `target_views.json`.

### B. Colab run (per scene)

In the Colab notebook:

```python
SCENE_NAME = '3dovs_bed'   # or 3dovs_bench / 3dovs_blue_sofa / ...
# upload 3dovs_<scene>.zip via files.upload(), extract to data/, then:
!bash pipeline/run_pipeline.sh data/$SCENE_NAME
```

The patched `run_hsegsplat_inference.py` detects `target_views.json` in the
scene dir and renders the 5 target views in addition to the 2 context views,
producing:

- `gaussians.pt`
- `rendered_rgb.npy`, `rendered_feature_map_lvl{1,3,6}.npy` (context views — unchanged)
- `rendered_rgb_targets.npy`, `rendered_feature_map_targets_lvl{1,3,6}.npy` (NEW — target views)
- `render_target_<view_id>.png`                              (NEW — for visual check)

Download `gaussians.pt` + `rendered_feature_map_targets_lvl1.npy` to
`data/3D-OVS/colab_outputs/<scene>/` (level 1 is the only one needed for 3D-OVS;
level 3/6 are useful for hierarchical extensions but not for this flat mIoU).

### C. Local eval

```bash
python scripts_3dovs/eval_3dovs_miou.py \
    --colab_root    data/3D-OVS/colab_outputs \
    --ingested_root data/3D-OVS/ingested \
    --scenes bed bench blue_sofa lawn office_desk room snacks sofa \
    --threshold 0.5 \
    --out_dir data/3D-OVS/eval_results
```

Outputs:
- `data/3D-OVS/eval_results/miou_per_scene.json` — per-class IoU + per-scene mIoU + overall.
- Console table matching the N2F2 Table 3 row format.

## Expected baseline (N2F2 Table 3, mIoU)

| Method | bed | bench | room | sofa | lawn | Overall |
|---|---|---|---|---|---|---|
| OV-Seg | 79.8 | 88.9 | 71.4 | 66.1 | 81.2 | 77.5 |
| LERF | 73.5 | 53.2 | 46.6 | 27.0 | 73.7 | 54.8 |
| 3D-OVS (Liu et al.) | 89.5 | 89.3 | 92.8 | 74.0 | 88.2 | 86.8 |
| LangSplat | 92.5 | 94.2 | 94.1 | 90.0 | 96.1 | 93.4 |
| N2F2 | 93.8 | 92.6 | 93.5 | 92.1 | 96.3 | 93.9 |
| **H-SegSplat (ours)** | ? | ? | ? | ? | ? | ? |

## Caveats

1. **Feed-forward setting:** all the above baselines either optimize per-scene
   (LERF, LangSplat, N2F2, 3D-OVS) or aggregate over the full image set
   (OV-Seg). Ours uses only 2 input views. Position our row honestly as the
   "feed-forward, 2-view" entry.
2. **2-view scene coverage:** target views may show scene regions not present
   in the context-view Gaussians; those pixels can't be predicted (will land
   in background/low-mass regions and contribute zeros to the predicted class
   map). The eval script masks them via `--mass_threshold`.
3. **Class set is per-scene:** 3D-OVS provides ~5-7 object classes per scene.
   The mIoU is computed over those classes only, not a global class set.
