# H-SegSplat evaluation scripts

All evaluation tooling we built for the semester project lives here. The
production pipeline (mask extraction → SigLIP → build → inference) is in
the parent `scripts/` directory; this folder is only for benchmarks and
result analysis.

Raw datasets and per-scene Colab outputs live under the repo's `data/`
directory, which is gitignored. The scripts here assume you run them from
the **repo root**:

```bash
cd ~/Desktop/h-segsplat-repo
python scripts/evaluation_scripts/3dovs/eval_3dovs_miou.py \
    --colab_root    data/3D-OVS/colab_outputs \
    --ingested_root data/3D-OVS/ingested \
    --scenes bed bench sofa
```

---

## Layout

```
scripts/evaluation_scripts/
├── 3dovs/                 3D-OVS open-vocab segmentation benchmark
│   ├── ingest_3dovs.py            COLMAP -> 2-view dslr/ layout
│   ├── eval_3dovs_miou.py         per-class LERF + mIoU
│   ├── eval_3dovs_miou_v2.py      per-mask-features variant
│   ├── visualize_levels.py        per-level overlay PNGs
│   ├── visualize_predictions.py   pred vs GT overlays
│   ├── verify_pose_3dovs.py       pose-convention sanity check
│   ├── experiment_lvl1_siglip.py  the bg-kept-crop ablation
│   ├── colmap_io.py               canonical COLMAP binary reader
│   ├── colab_download_snippet.py  paste-into-Colab cell for downloads
│   ├── colab_render_targets_cell.py  paste-into-Colab cell for target-view rendering
│   └── README.md
├── multiscan/             MultiScan dataset (Search3D protocol replacement)
│   ├── multiscan_ingest.py        mp4 + jsonl -> 2-view dslr/ layout
│   ├── browse_frames.py           thumbnail grid for manual view picking
│   ├── select_2views.py           auto 2-view selector
│   ├── stage_for_pipeline.py      stage chosen pair into repo's data/
│   ├── verify_projection.py       pose-alignment sanity check
│   ├── visualize_levels_multiscan.py  per-level overlay PNGs at context views
│   └── eval/
│       ├── hsegsplat_offline_state.py  shared OfflineState + SigLIP encoder
│       ├── gaussian_to_vertex.py        per-Gaussian -> per-vertex bridge
│       ├── run_multiscan_ap_eval.py     Search3D AP/AP50/AP25 driver (blocked on coverage)
│       └── README.md
└── 2d_eval/               2D mIoU vs rasterized GT mesh (works for MultiScan + ScanNet++)
    ├── render_gt_mesh.py          rasterizes labelled mesh at gaussians.pt's camera poses
    └── eval_2d_miou.py            per-class LERF + mIoU vs that GT map
```

---

## Three eval tracks

### 3D-OVS (`3dovs/`)

The headline open-vocab semantic-segmentation benchmark from the 3D-OVS
paper. Each scene has ~5–6 named classes (e.g. *red bag*, *banana*,
*camera*). We adopt SegSplat's evaluation protocol — 2-view feed-forward,
per-class LERF relevancy, argmax at the labelled target views, IoU vs
binary GT masks.

End-to-end:

```bash
# 1. Ingest the raw 3D-OVS scene (COLMAP -> 2-view dslr/ + 5 labelled targets):
python scripts/evaluation_scripts/3dovs/ingest_3dovs.py \
    --scene_dir data/3D-OVS/<scene> \
    --out_dir   data/3D-OVS/ingested/<scene>

# 2. Stage the resulting dslr/ into the pipeline's data/, zip, upload to Colab,
#    run pipeline/run_pipeline.sh, download gaussians.pt + rendered feature maps.
#    See 3dovs/README.md for the exact Colab cells.

# 3. Eval:
python scripts/evaluation_scripts/3dovs/eval_3dovs_miou.py \
    --colab_root    data/3D-OVS/colab_outputs \
    --ingested_root data/3D-OVS/ingested \
    --scenes bed bench lawn sofa \
    --levels 3 \
    --mode argmax
```

Final report numbers used `--levels 3 --mode argmax --tau 100`.

### MultiScan (`multiscan/`)

MultiScan provides per-vertex semantic + part labels on iPad LiDAR scans.
We use it both for the Search3D 47-tuple AP benchmark (blocked: 2-view
H-SegSplat doesn't see enough of the scene per scan) and as a source of
posed RGB-D frames for general 2D semantic evaluation.

The `browse_frames.py + multiscan_ingest.py --manual_pair` flow lets you
hand-pick two visually-good views from a 9000-frame iPad video, which is
necessary because most jsonl frames are motion-blurred or pointed at the
floor.

### 2D mIoU vs rasterized mesh (`2d_eval/`)

The honest 2D semantic eval: rasterize the GT mesh at H-SegSplat's camera
poses to get a dense per-pixel class map, then compare to argmax
predictions. Works on any dataset that ships a semantically-labelled mesh
(MultiScan via per-face `objectId`, ScanNet++ via `segments_anno.json`).

```bash
# 1. Rasterize the GT mesh at the gaussians.pt camera poses:
python scripts/evaluation_scripts/2d_eval/render_gt_mesh.py \
    --mesh data/Multiscan/multiscan_test_plys_only/<scan>.ply \
    --face_label_attr objectId \
    --label_map_json data/Multiscan/scenes/<scan>/<scan>.annotations.json \
    --gaussians_pt data/Multiscan/colab_outputs/<scan>/gaussians.pt \
    --multiscan_scan_dir data/Multiscan/scenes/<scan> \
    --jsonl_frame_indices <FRAME_A> <FRAME_B> \
    --out_dir data/Multiscan/colab_outputs/<scan>/gt_renders

# 2. Eval:
python scripts/evaluation_scripts/2d_eval/eval_2d_miou.py \
    --gt_dir       data/Multiscan/colab_outputs/<scan>/gt_renders \
    --colab_output data/Multiscan/colab_outputs/<scan> \
    --levels 3
```

Both unweighted mIoU (mean over visible classes) and area-weighted mIoU
(weighted by GT pixel count) are reported.

---

## Notes

- **Pose convention bug** (task #20 in our session log): `multiscan_ingest.py`
  uses `c2w = align @ T_arkit` to write `transforms.json`. The correct
  convention is `T_arkit @ align`. We work around this in
  `2d_eval/render_gt_mesh.py` by overriding the gaussians.pt extrinsics
  with PLY-frame poses via `--multiscan_scan_dir` and
  `--jsonl_frame_indices`. The fix is deferred so that existing
  gaussians.pt files stay valid.
