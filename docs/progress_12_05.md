# H-SegSplat Progress Report — 2026-05-12

End-to-end working pipeline for **Hierarchical SegSplat** (v1): feed-forward 3D Gaussian Splatting with three parallel granularity levels (1, 3, 6) and hierarchical text queries up to three levels deep ("child of parent of grandparent"). The full path — preprocess → DepthSplat inference → query — runs and produces sensible relevancy maps on a 2-view ScanNet++ scene.

This document is the comprehensive record of what was built today.

---

## 1. Where we started

Before this session the project had:

- A working **Phase-3 (flat SegSplat)** Colab pipeline. Single-level, single bank, single cluster-ID per Gaussian.
- A test scene (ScanNet++ DSLR, frames `54aed99f_DSC03788/3789.JPG`), fisheye-undistorted to pinhole 640×960, with Nerfstudio poses, already preprocessed upstream.
- SemanticSAM and SigLIP outputs run on the cluster — initially with all three levels jumbled together in one folder, then re-run with one folder per level.
- A `PROJECT_PLAN.md` whose original design was per-mask IDs + a 90% containment tree + tree-walk queries — too complex for v1.

The session goal: **finalize the plan to a simpler per-cluster v1 design, then implement it end-to-end** for hierarchical queries.

---

## 2. Plan revision

`PROJECT_PLAN.md` was rewritten to match a per-cluster-at-three-levels design:

- **What we removed from v1.** Per-mask IDs (and per-view-namespaced mask IDs), the 90% containment tree, the validated closer-level-wins parent rule, and tree-walk queries. These were the original §3.5 / §6.5 machinery.
- **What we put in instead.** Three independent SegSplats, one per granularity. Each Gaussian carries three integer **cluster IDs** (one per level). Three independent K-means runs, three independent banks, three independent per-pixel cluster maps. No tree, no parent links.
- **Hierarchical queries are spatial AND at the pixel.** "Leg of chair" highlights pixels where the level-6 feature matches "leg" AND the level-1 feature matches "chair" at the same rendered pixel. Parent-child relationship is implicit in co-occurrence, not encoded structurally.
- **What v1 cannot do.** Instance disambiguation: "leg of *the chair on the left*" merges into "leg of any chair" because all chair instances cluster into the same level-1 slot. This is acceptable and explicitly documented as future work.
- **Three-deep grandparent queries are supported.** "Leg of arm of chair" is just an AND with three terms across three levels. SemanticSAM's three granularities are a hard ceiling — deeper hierarchies are out of scope by design.

The revised plan is the canonical reference. Every script in `second_stage/scripts/` matches it.

---

## 3. Implementation overview

Everything in `second_stage/` is the v1 H-SegSplat implementation. The first-stage code in `first_stage/` is the Phase-3 reference, left untouched.

```
second_stage/
├── data/test_scene_3_lvl_masks_3_folders/
│   ├── dslr/{nerfstudio/transforms.json, resized_images/, colmap/}
│   ├── masks_lvl_1/<frame>/{mask_*.png, metadata.json, siglip_embeddings.npy}
│   ├── masks_lvl_3/<frame>/...
│   └── masks_lvl_6/<frame>/...
├── scripts/
│   ├── build_hsegsplat_inputs.py
│   ├── visualize_hsegsplat_inputs.py
│   ├── colab_hsegsplat_inference.py
│   ├── label_masks_with_siglip.py
│   └── query_hsegsplat.py
└── outputs/test_scene_hsegsplat/
    ├── level_{1,3,6}/{bank.npy, index_maps.npy, mask_id_maps.npy,
    │                  mask_to_cluster.json, meta.json, vis/}
    ├── datasets/test_scene_hsegsplat_2view/test/{000000.torch, index.json}
    ├── assets/test_scene_hsegsplat_2view_eval.json
    ├── meta.json
    ├── test_scene_hsegsplat_hsegsplat_colab.zip
    ├── vis/levels_panel_<frame>.png             (input-stage sanity panels)
    ├── rendered_rgb.npy                          (V, 640, 960, 3)
    ├── rendered_feature_map_lvl1.npy             (V, 640, 960, 10)
    ├── rendered_feature_map_lvl3.npy             (V, 640, 960, 37)
    ├── rendered_feature_map_lvl6.npy             (V, 640, 960, 60)
    ├── gaussians.pt                              (G=1228800)
    ├── mask_labels/labels_lvl{1,3,6}_<frame>.png
    └── queries/query_<terms>_view{0,1}.png + .npy
```

---

## 4. Stage 1 — `build_hsegsplat_inputs.py`

**Purpose:** turn the per-level SAM masks + SigLIP features into the data format H-SegSplat needs at inference time. Produces one set of SegSplat assets *per level* plus a DepthSplat `.torch` chunk for the same scene, all bundled into a single Colab-ready zip.

**Inputs:**
- `dslr/nerfstudio/transforms.json` — pinhole intrinsics, w/h, per-frame `transform_matrix` in Blender convention
- `dslr/resized_images/<frame>.JPG` — 640×960 RGB
- `masks_lvl_{1,3,6}/<frame_stem>/` — for each frame and each level:
  - `mask_<i>.png` — N binary masks
  - `metadata.json` — bbox, area, IoU, stability per mask
  - `siglip_embeddings.npy` — `(N, 1152)` per-mask SigLIP features (raw, unnormalized; the script normalizes them)

**Per-level loop (SegSplat §3.1 verbatim, run three times):**

1. Load each frame's masks + features. Assert mask shape matches `(H, W)`.
2. Pool features across all views → `(N_total, 1152)`. L2-normalize.
3. K-means with `M = ceil(λ · N_total / V)`, λ=1.2, `n_init=10, random_state=0`.
4. Re-normalize cluster centroids to unit norm.
5. Build `bank ∈ R^{(M+1) × 1152}` with **row 0 = zeros (background)**, rows 1..M = centroids.
6. For each frame, resolve mask overlaps by **smallest-area-wins** (paint masks largest→smallest so the smallest mask wins each contested pixel). Produces a `(H, W)` `mask_id_map` with 0 = bg, 1..N_v = mask_id+1.
7. Map each mask to its cluster label → `(H, W)` `index_map` with 0 = bg, 1..M = cluster.

**Outputs per level (under `outputs/<scene>/level_<N>/`):**
- `bank.npy` `(M+1, 1152)` float32
- `index_maps.npy` `(V, H, W)` int16 — per-pixel cluster ID per view
- `mask_id_maps.npy` `(V, H, W)` int16 — per-pixel mask ID per view (debug)
- `mask_to_cluster.json` — for each frame, which cluster each of its masks landed in
- `meta.json` — `{scene_key, level, V, H, W, M, D, N_total, lambda, frame_order, kmeans config, ...}`

**Level-independent outputs (under `outputs/<scene>/`):**
- `datasets/<scene>_2view/test/000000.torch` — DepthSplat dataset chunk: per-frame `{key, url, timestamps, cameras = [fx_norm, fy_norm, cx_norm, cy_norm, k1=0, k2=0, w2c[:3].flatten()], images = JPEG bytes}`. Poses are converted Blender→OpenCV via `diag(1, -1, -1, 1)` right-multiply and inverted to w2c.
- `datasets/<scene>_2view/test/index.json` — `{scene_key: "000000.torch"}`
- `assets/<scene>_2view_eval.json` — `{scene_key: {context: [0, 1], target: [0, 1]}}`
- `meta.json` — top-level scene meta with normalized intrinsics and per-level summary
- `<scene>_hsegsplat_colab.zip` — bundle of everything for Colab upload

**Run on the test scene:**

```
V=2  H=640 W=960  levels=[1, 3, 6]
  [lvl 1] N_total=14  → M=9  clusters
  [lvl 3] N_total=60  → M=36 clusters
  [lvl 6] N_total=98  → M=59 clusters
```

The level-1 bank is byte-identical to the original Phase-3 baseline (same seed, same masks, same K-means) — a useful regression check.

---

## 5. Stage 2 — `visualize_hsegsplat_inputs.py`

**Purpose:** sanity-check the per-pixel cluster maps before going to GPU. Catches K-means / mask-loading / overlap-resolution bugs cheaply.

For each level:
- `cluster_<frame>.png` — colorize the `index_map` (one HSV-spread color per cluster, black=bg)
- `cluster_overlay_<frame>.png` — alpha-blend (α=0.55) onto the input RGB on foreground pixels
- `mask_id_<frame>.png` — colorize the pre-cluster `mask_id_map` (debug)
- `legend_clusters.png`, `legend_masks.png`

Plus a single combined 4-up panel per frame at `outputs/<scene>/vis/levels_panel_<frame>.png` showing `RGB | lvl1 | lvl3 | lvl6` overlays side-by-side. This is the main visual sanity check — clusters should look like progressively finer segmentations.

Foreground coverage on the test scene:
- Level 1: ~74–82% foreground (most pixels belong to some "whole-object" mask)
- Level 3: ~88–90% foreground (intermediate)
- Level 6: ~21–25% foreground (fine parts are sparse — most pixels have no level-6 mask)

The low level-6 foreground is **expected**: fine SemanticSAM masks don't tile the image; they isolate specific parts. Pixels not in any level-6 mask become "background" at level 6 only.

---

## 6. Stage 3 — `colab_hsegsplat_inference.py` (Colab GPU)

**Purpose:** run DepthSplat once on the 2-view scene to get Gaussians, then render RGB once per view + a `(M_l+1)`-channel feature map per view per level.

**Setup on Colab:** clone `github.com/cvg/depthsplat`, install PyTorch 2.4 + CUDA 12.4 + DepthSplat requirements, download the pretrained checkpoint `depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth`, install `gsplat` + `imageio`, upload the H-SegSplat zip + the inference script.

**Hydra invocation** (same `+experiment=dl3dv` skeleton as Phase 3, two H-specific lines added):
```
++segsplat.assets_dir=segsplat/test_scene_hsegsplat
++segsplat.output_dir=outputs/test_scene_hsegsplat
++segsplat.levels=[1,3,6]
```

**Script pipeline:**

1. Build DepthSplat encoder + load checkpoint via the same `src/main.py` hydra path as Phase 3.
2. Get one batch from the test dataloader (one scene, V=2). Move to CUDA. Apply `data_shim` (patch padding).
3. Run encoder on `batch["context"]` → Gaussians: `means [G, 3]`, `covariances [G, 3, 3]`, `harmonics [G, 3, 9]`, `opacities [G]` where `G = V·H·W = 2·640·960 = 1,228,800`.
4. Assert post-shim `(H, W)` matches all level metas (640×960). Assert `G == V·H·W`.
5. For each level, load `bank.npy + index_maps.npy + meta.json` and build a `(G, M_l+1)` per-Gaussian one-hot via the flat-index ↔ (v, y, x) mapping `i = v·(H·W) + y·W + x` (identical to DepthSplat's flatten).
6. Decompose covariances → unit quaternions (xyzw) + scales. Reorder to gsplat's wxyz convention.
7. Denormalize intrinsics from `[0, 1]` to pixels. Invert `c2w` to gsplat-style `viewmats`.
8. For each input view:
   - **RGB render** with gsplat: `colors = (G, K_sh=9, 3)` SH coefficients, `sh_degree=int(round(sqrt(9))−1)=2`, `render_mode="RGB"`.
   - **Per-level feature render** (×3): `colors = (G, M_l+1)` one-hot, `sh_degree=None` (view-invariant blend), `render_mode="RGB"`. gsplat shares the geometry projection across passes, so total cost is much less than 4× a single render.

**Outputs:**
- `rendered_rgb.npy` `(2, 640, 960, 3)` float32 in [0,1]
- `rendered_feature_map_lvl1.npy` `(2, 640, 960, 10)`
- `rendered_feature_map_lvl3.npy` `(2, 640, 960, 37)`
- `rendered_feature_map_lvl6.npy` `(2, 640, 960, 60)`
- `render_view{0,1}.png` for visual check
- `gaussians.pt` — bundle of geometry (means/quats/scales/opacities/harmonics) + per-level dicts (`cluster_index[lvl]`, `banks[lvl]`, `M[lvl]`) + cameras + image shape

**Sanity checks after Colab download:**

- Per-pixel **accumulated alpha mass `Σ T_i α_i`** is identical across all three feature maps (mean 0.894, min 0.250, max 1.000). The mass depends only on geometry — three feature passes on the same Gaussians give the same per-pixel mass. Sharing this fact across levels is a strong correctness signal.
- Mass distribution across channels matches input-map foreground fractions: 22% on background slot at lvl 1, 10% at lvl 3, 76% at lvl 6. The level-6 dominance of the background slot reflects that most pixels in the rendered view come from Gaussians whose source pixel had no fine SAM mask — this is correct (§3.6).
- All per-level `cluster_index` tensors use the full `[0, M]` range; no dead clusters.

---

## 7. Stage 4 — `label_masks_with_siglip.py` (ground-truth eyeballing)

**Purpose:** before issuing queries, see what SigLIP actually thinks each mask is, so query terms can be chosen from the available vocabulary.

For each frame and each level:
1. Read all the mask PNGs and the precomputed `siglip_embeddings.npy`.
2. L2-normalize the mask features (the raw ones from `run_siglip2.py` have norms ~20).
3. Encode all 2878 candidate classes from `semantic_classes.txt` with SigLIP-text (batched).
4. For each mask, dot-product its feature against all class embeddings; pick top-1; print the class name and cosine score at the mask centroid.
5. Skip text for masks smaller than 400 pixels (outline only) to avoid level-6 clutter.

Outputs 6 PNGs (2 frames × 3 levels) at `outputs/<scene>/mask_labels/labels_lvl{N}_<frame>.png`. Each shows colored mask outlines on the original RGB with centroid labels like `"chair (0.34)"`. The cosine score in parentheses gives you instant confidence: 0.3+ is a real match for SigLIP (its scores run lower than CLIP — don't expect 0.9).

---

## 8. Stage 5 — `query_hsegsplat.py`

**Purpose:** run a hierarchical text query against the rendered feature maps and produce relevancy heatmaps + a final AND-combined highlight.

**CLI (explicit per-level flags, no string parsing):**
```
--child STR             required (finest level term)
--parent STR            optional
--grandparent STR       optional
--child-level INT       default 6
--parent-level INT      default 3
--grandparent-level INT default 1
--threshold FLOAT       default 0.5 (SegSplat)
--mass-threshold FLOAT  default 0.05
```

`--parent` and `--grandparent` are both optional. Omitting one just means no constraint at that level — that's how "grandparent: arbitrary" works.

**Pipeline (in order):**

1. **Build the level→term map.** From the explicit flags. Reject duplicate level assignments.
2. **Per-level feature recovery.** For each level `l` with a term:
   - Load `rendered_feature_map_lvl{l}.npy` `(V, H, W, M_l+1)` and `level_<l>/bank.npy` `(M_l+1, D)`.
   - `F_l(v) = E_l(v) @ bank_l` → `(V, H, W, D=1152)` (SegSplat eq. 3). This is the rendered SigLIP feature at each pixel — the bank lookup composed with the alpha-blended one-hot.
   - L2-normalize `F_l` per pixel.
   - Compute `fg_mass = sum of E_l[..., 1:]` over real cluster channels (excluding the bg slot). Pixels with `fg_mass < 0.05` are masked to relevancy 0 (avoids spurious matches in regions with little semantic mass).
3. **SigLIP encoding.** Load SigLIP `ViT-SO400M-14 / webli`. Encode all provided terms plus the three canonical phrases `"object", "things", "stuff"` in one pass. L2-normalize.
4. **Per-level LERF relevancy** (SegSplat eq. 4 / LERF Kerr et al. 2023):
   - `sim_q = F_l_unit · phi_query` — per-pixel cosine similarity to the term
   - `sim_c = F_l_unit @ phi_canonical^T` — per-pixel cosine to each canonical
   - `relevancy(v) = min_c sigmoid(τ · (sim_q − sim_c))` with `τ=100` (LERF temperature)
   - Mask to 0 on background.
5. **Combine across levels** by taking the **min** of the per-level relevancies (this implements a soft AND: a pixel passes only if all level conditions are satisfied with relevancy ≥ τ).
6. **Outputs:**
   - PNG panel per view: `RGB | lvl_child heatmap | lvl_parent heatmap | (lvl_grandparent) | AND overlay`. Pixels passing `≥ threshold` are highlighted on the RGB.
   - Raw combined relevancy as `.npy` for downstream use.

**Two real queries run today on the test scene:**

| Query | Levels | Per-level pixels ≥ τ | Combined AND |
|---|---|---|---|
| `glass pane` of `window` (grandparent arbitrary) | lvl 6: 277,809  lvl 3: 817,825 | — | 199,422 |
| `door` of `cupboard` of `cupboard` | lvl 6: 264,145  lvl 3: 174,924  lvl 1: 175,690 | | 81,124 |

User confirmed the output PNGs look right.

**Note on the LERF temperature τ=100.** With this temperature the relevancy nearly saturates to 0 or 1, so `peak=1.000` per level is expected behavior — it just means SigLIP scored the query meaningfully higher than the canonicals at the peak pixel. The interesting number is **how many pixels** pass the threshold, not the peak value. This matches SegSplat's paper (where ≥ 0.5 is a near-binary decision).

---

## 9. Key design decisions made during the session

These are the calls that shaped the final implementation. Each was a real choice with alternatives considered.

1. **Per-cluster, not per-mask, at three levels.** Simpler, smaller banks, no cross-view ID-matching problem. Cluster identity *is* the cross-view bridge (same SegSplat mechanism, repeated three times).
2. **No containment tree.** With per-cluster IDs a tree walks from centroid to centroid — not what containment buys us. Spatial AND at the pixel does the work instead.
3. **Three independent K-means.** Pooled-across-levels K-means would conflate granularities. Each level is run independently with the same λ=1.2 heuristic.
4. **Smallest-area-wins overlap resolution.** Inherited from SegSplat. At each level, when multiple masks overlap a pixel, the smallest mask claims it. Same rule applies at every level.
5. **One-hots include a background slot at row 0.** Pixels with no SAM mask at a given level get an all-zero one-hot, which produces zero semantic contribution at render. Color rendering is unaffected. Bank row 0 is reserved (all zeros) so the index-map convention (0 = bg, 1..M = cluster) maps cleanly to `feat[..., 0]` being the bg channel.
6. **Three parallel feature rasterizations, not concatenated.** Cleaner per-level relevancy. gsplat shares geometry projection across passes, so the cost penalty is modest. Plan §3.1 expectation; not yet benchmarked.
7. **LERF relevancy, not raw cosine.** The base `query_segsplat.py` used plain cosine; the new script implements LERF eq. 4 with canonical phrases. This matches SegSplat's paper and the τ=0.5 threshold convention.
8. **L2-normalize mask features on the fly in `label_masks_with_siglip.py`.** The raw features from `run_siglip2.py` (average of two crops) are not unit-norm. The build pipeline normalizes them before K-means; the labeling script needs to do the same.
9. **`gaussians.pt` schema.** One geometry payload (level-independent) plus dicts keyed by level for `cluster_index`, `banks`, `M`. Future-proof and avoids duplicating geometry.
10. **Three-deep query support via the same AND.** "Grandparent" doesn't require new machinery — it's just one more term and one more level in the AND. Skipping a level is supported by simply omitting its term.

---

## 10. What works end-to-end today

```
data/test_scene_3_lvl_masks_3_folders/
    ↓ build_hsegsplat_inputs.py    (CPU, local, ~2 minutes)
outputs/test_scene_hsegsplat/level_{1,3,6}/* + .torch chunk + zip
    ↓ visualize_hsegsplat_inputs.py    (CPU, local, sanity check)
outputs/test_scene_hsegsplat/vis/levels_panel_*.png
    ↓ upload zip + colab_hsegsplat_inference.py to Colab
    ↓ run +experiment=dl3dv inference (GPU, Colab, ~minutes)
rendered_rgb.npy + rendered_feature_map_lvl{1,3,6}.npy + gaussians.pt
    ↓ download to outputs/test_scene_hsegsplat/
    ↓ label_masks_with_siglip.py    (CPU, local, to pick query terms)
outputs/test_scene_hsegsplat/mask_labels/*.png
    ↓ query_hsegsplat.py --child ... --parent ... [--grandparent ...]
outputs/test_scene_hsegsplat/queries/query_*_view{0,1}.png
```

Confirmed runs:

- **Flat:** `--child "chair" --child-level 1` → peak 0.748 at lvl 1, 168k pixels ≥ τ.
- **Parent:** `--child "glass pane" --parent "window"` (grandparent arbitrary) → 199k px AND.
- **Grandparent:** `--child "door" --parent "cupboard" --grandparent "cupboard"` → 81k px AND.

---

## 11. Known limitations and future work

### Limitations of v1 (by design)

- **No instance disambiguation at level 1.** All chair instances share one cluster. "Leg of the chair on the left" is not expressible.
- **Three-level ceiling.** SemanticSAM's granularities {1, 3, 6} cap us at three-deep queries. "X of Y of Z of W" is out of scope.
- **Spatial AND only.** The relationship between "leg" and "chair" is only co-occurrence at the rendered pixel. If a leg mask happens to not be inside a chair mask at level 1 (e.g. an orphaned stool leg), the AND won't fire.
- **Level granularities are statistical, not structural.** Nothing guarantees a level-6 mask is nested inside a level-3 mask. Spatial AND mostly works because the granularities are usually consistent, but pathological cases will fail.

### Things still to do

- **Novel-view rendering script** — `render_novel_view_local.py` from `first_stage` needs adapting to load `gaussians.pt` and rasterize at arbitrary poses for all three levels. Lets us produce relevancy at custom camera angles, not just input views.
- **Benchmark the parallel-vs-concatenated rasterization cost.** Plan §3.1 expects ≤ 1.5× per pass; we haven't measured.
- **Threshold tuning per query / per level.** τ=0.5 is the SegSplat default. The LERF temperature τ=100 makes relevancy nearly binary; if that's too coarse for visual heatmaps, we can lower the temperature.
- **More scenes.** Today we have one ScanNet++ test scene. The pipeline should generalize but isn't tested.
- **Quantitative eval on 3D-OVS.** Phase 7 of the plan. Currently qualitative only.
- **Stretch: 3D viewer with in-viewer queries.** Discussed; deferred. Bake-to-PLY per query is the path of least resistance.
- **Stretch (future work, §6.5 of plan): per-mask IDs + containment tree + instance-aware tree-walk queries.** Defer until v1 results actually motivate it.

---

## 12. File-by-file summary

### `second_stage/scripts/build_hsegsplat_inputs.py`
Builds per-level SegSplat assets (bank + index_maps + mask_id_maps) from `masks_lvl_{N}/`. Also produces the DepthSplat `.torch` chunk and eval JSON for the same scene. Bundles everything into `<scene>_hsegsplat_colab.zip` for Colab upload. λ=1.2 fixed across levels. Smallest-area-wins overlap. Pinhole assumed (asserts `camera_model="pinhole"`). Blender→OpenCV pose conversion.

### `second_stage/scripts/visualize_hsegsplat_inputs.py`
Per-level cluster overlays + a combined 4-up panel (RGB | lvl1 | lvl3 | lvl6) per frame. Pure CPU, no model loaded.

### `second_stage/scripts/colab_hsegsplat_inference.py`
Hydra-configured DepthSplat inference script for Colab. Reads `++segsplat.levels=[1,3,6]`. Loads three sets of `(bank, index_maps, meta)`, builds three per-Gaussian one-hots via flat-index ↔ (v, y, x), runs 1 RGB + 3 feature rasterizations per view with gsplat. Writes RGB, three feature maps, and a `gaussians.pt` bundle.

### `second_stage/scripts/label_masks_with_siglip.py`
For each (frame, level), labels every mask with its top-1 SigLIP class from `semantic_classes.txt` (2878 candidates) at the mask centroid. Uses the precomputed `siglip_embeddings.npy`, L2-normalizes them in place. Drops text on masks below 400 pixels.

### `second_stage/scripts/query_hsegsplat.py`
Flat / parent / grandparent queries with explicit `--child / --parent / --grandparent` flags. Loads per-level rendered feature maps and banks, computes `F = E @ bank`, applies LERF relevancy against canonical phrases `"object", "things", "stuff"` with τ=100, mass-thresholds at 0.05, AND-combines across levels by min, thresholds at 0.5 for highlighting. Writes a labeled side-by-side panel per view plus the raw relevancy `.npy`.

---

## 13. Unity desktop viewer with in-viewer hierarchical queries

After the 2D query path was confirmed working, we built a Unity-based 3D viewer that loads the H-SegSplat scene and lets you type three text terms (child / parent / grandparent) directly in the Inspector to highlight matching Gaussians in **magenta**. All runs locally on Mac (M3 Pro) with no cloud, no Quest, no XR.

### 13.1 Architecture

```
[Unity desktop, Editor mode]              [Python FastAPI on 127.0.0.1:8000]
  HSegSplatClient  (Inspector fields)  ──► POST /query_combined
  + custom Editor button                    {child, parent, grandparent,
                                             child_level, parent_level, grandparent_level}
                                            │ load gaussians.pt once
                                            │ encode SigLIP text per query (cached)
                                            │ per-cluster LERF → scatter to per-Gaussian
                                            │ AND-combine across provided levels by per-Gaussian min
                                            ◄ binary float32 × N_gaussians
  GaussianSplatRenderer ◄ reflection-write into _SplatQuerySimilarities buffer
  + patched RenderGaussianSplats.shader  ── reads relevancy + threshold,
                                             tints splats magenta when relevancy ≥ threshold
```

Three machine pieces total:
- **Python server** (`second_stage/viewer/server/serve.py`) — FastAPI + OpenCLIP SigLIP. Three endpoints (`/info`, `/query_combined`, `/extrema`).
- **PLY converter** (`second_stage/viewer/gaussians_to_ply.py`) — turns the Colab `gaussians.pt` into a standard 3DGS `.ply` that the aras-p Unity package can bake into a `GaussianSplatAsset`.
- **Unity project** at `second_stage/viewer/unity_project/` — Unity 2023.1.14f1, built-in render pipeline, aras-p `com.aras-p.gaussian-splatting` package (copied from `FINAL_TEST/`).

### 13.2 Server (`second_stage/viewer/server/serve.py`)

On startup:
- Loads `gaussians.pt`. Pulls out `cluster_index[lvl]` (G int64 per Gaussian per level), `banks[lvl]` (M_l+1 × 1152 SigLIP centroids; row 0 = zeros for background), and the three M_l values.
- Loads SigLIP `ViT-SO400M-14 / webli` via OpenCLIP. Encodes canonical phrases `["object", "things", "stuff"]` once.

On `/query_combined`:
- Encodes each provided term with SigLIP (cached after first time).
- For each level: real_bank = bank[1:] (drop background row). Compute `sim_q = real_bank @ phi_q`, `sim_c = real_bank @ phi_canonical.T`, apply LERF eq. 4 with τ=100: `pair_score = sigmoid(τ · (sim_q − sim_c))`, take min over canonicals → per-cluster relevancy in (0, 1).
- Scatter to per-Gaussian via `cluster_index[lvl]` (background gets 0.0).
- AND-combine across provided levels by elementwise **min**.
- Return `float32[G]` binary stream.

**Per-Gaussian AND vs. per-pixel AND distinction:** The 2D `query_hsegsplat.py` AND's at the pixel after alpha-blending. The 3D viewer AND's in cluster space, per Gaussian. A Gaussian passes only if its level-1 cluster matches the grandparent term AND its level-3 cluster matches the parent term AND its level-6 cluster matches the child term — *simultaneously, as cluster memberships of the same Gaussian*. Different math, similar result.

### 13.3 PLY converter (`second_stage/viewer/gaussians_to_ply.py`)

Standard 3DGS PLY format with 41 fields per vertex: `x y z`, `nx ny nz` (zeros, ignored), `f_dc_{0..2}`, `f_rest_{0..23}` (channel-major for K_sh-1=8 remaining SH bands × 3 channels), `opacity` (logit, package applies sigmoid), `scale_{0..2}` (log), `rot_{0..3}` (wxyz quaternion). Reads our `gaussians.pt` (linear scales/opacities, wxyz quats, harmonics in `(G, 3, K_sh)` layout) and writes binary little-endian PLY. Test scene → 192 MB PLY, 1.23M Gaussians.

### 13.4 Unity scripts (`second_stage/viewer/unity_scripts/`)

**`HSegSplatClient.cs`** — attached to the Splats GameObject. Has `[ExecuteAlways]` so it works without Play mode. Inspector exposes editable fields: `Server Url`, `Child`, `Parent`, `Grandparent`, `Child Level`, `Parent Level`, `Grandparent Level`, `Threshold` (slider 0–1), `Last Status` (read-only multiline). Two public methods `SubmitQuery()` and `ClearQuery()`. POSTs JSON, receives `4×G` bytes, casts to `float[G]`, finds the package's `internal m_GpuQuerySimilarities` field via reflection, calls `SetData(relevancy)`. Also sets `_MinRelevancyScore` as a shader global. Calls `SceneView.RepaintAll()` so changes are visible immediately in Edit mode.

**`HSegSplatClientEditor.cs`** (in `Assets/Scripts/Editor/`) — custom Inspector that draws two buttons below the default fields: **`[ Submit Query ]`** and **`[ Clear Query ]`**. Clicking is one click instead of right-click → context menu.

**`SetNearCutoff.cs`** (in `Assets/Scripts/Editor/`) — diagnostic menu items (`Tools → HSegSplat → Cutoff X.X`) for changing `m_NearPlaneCutoff` at runtime when debugging Gaussian projection. Kept as a useful debugging utility.

**`RenderGaussianSplats.shader`** — drop-in replacement for the package's main render shader. Adds:
- `StructuredBuffer<float> _SplatQuerySimilarities` and `float _MinRelevancyScore`.
- One extra interpolator `relevancy : TEXCOORD1` on `v2f`. The vertex stage reads `_SplatQuerySimilarities[instID]` and passes it to fragment.
- In fragment: if `relevancy >= _MinRelevancyScore && _MinRelevancyScore > 0.0`, lerp the splat's RGB toward magenta `(1, 0, 1)` at strength 0.75.

The patched shader is byte-different from upstream but functionally identical for non-query rendering — when no query is active (`_MinRelevancyScore = 2.0`, set by `ClearQuery()` or default) the condition never fires.

### 13.5 Setup that was performed (one-time)

1. **Python venv built with Python 3.11** (Homebrew-installed). System Python 3.9.6 isn't new enough for current `open_clip_torch`. Venv path: `second_stage/viewer/server/venv/`.
2. **Server deps installed:** `fastapi, uvicorn, pydantic, torch, open_clip_torch, transformers, sentencepiece, numpy`. The first installation missed `transformers` (open_clip's SigLIP tokenizer is an HF BPE) — fixed in `requirements.txt`.
3. **PLY converter run once** on `test_scene_hsegsplat/gaussians.pt` → 192 MB `.ply`.
4. **Unity 2023.1.14f1 project created** at `second_stage/viewer/unity_project/`, Built-in render pipeline.
5. **Splat package copied** from `FINAL_TEST/Mixed_reality_gaussian_splatting/package/` to `unity_project/Packages/com.aras-p.gaussian-splatting/`.
6. **Patched render shader copied** over the package's `Shaders/RenderGaussianSplats.shader`.
7. **PLY baked** via `Tools → Gaussian Splats → Create GaussianSplatAsset` at Very High quality.
8. **Splats GameObject** created with `Gaussian Splat Renderer` component, asset assigned, all shader/compute references wired (Shader Splats, Composite, Debug Points, Debug Boxes, Occam Similarities = WombatShader, Occam Composite, CS Splat Utilities).
9. **HSegSplatClient.cs + HSegSplatClientEditor.cs** dropped into `Assets/Scripts/` (Editor script in `Assets/Scripts/Editor/`).
10. **HSegSplatClient component** added to the Splats GameObject.

### 13.6 Setup issues encountered (and resolutions)

These are recorded so future-us doesn't burn time rediscovering them:

- **Python 3.9 system Python too old for open_clip_torch.** Fix: build venv from `/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv venv` explicitly. `python3` alone points to system Python on macOS even with `brew install python@3.11`.
- **`open_clip` errored with `ModuleNotFoundError: 'transformers'`** when loading the SigLIP tokenizer. SigLIP uses an HF BPE tokenizer (not CLIP's). Fix: add `transformers, sentencepiece` to `requirements.txt`.
- **Persisting Plastic SCM `CS0006` compile errors** in Console after importing the splat package. Unrelated to our code — Unity ships a Plastic plugin that references binaries not present on macOS. Fix: remove the **Version Control** (`com.unity.collab-proxy`) package from Package Manager, then `Assets → Reimport All` to flush cached references. After this, the `Tools → Gaussian Splats → ...` menu also appears (compile errors elsewhere prevent the package's Editor scripts from loading).
- **"Resource references are missing, or platform does not support compute shaders"** at startup. The renderer's `resourcesAreSetUp` check requires *seven* shader/compute references including two Occam-mode ones (`m_ShaderOccamSimilarities = WombatShader.shader`, `m_ShaderOccamComposite = OccamComposite.shader`) that don't auto-assign. Fix: wire all seven manually.
- **Compute shaders not supported on macOS Editor.** The same error fires when Unity is using OpenGL Core instead of Metal. Fix: `Edit → Project Settings → Player → Other Settings → Graphics APIs for Mac` → uncheck `Auto Graphics API for Mac`, ensure `Metal` is first in the list. **Requires full editor quit-and-reopen** to take effect.
- **TMP `NullReferenceException` on `MaterialReference..ctor`** when clicking placeholder text. TMP Essentials weren't actually wired up properly. Fix path that worked: delete the entire Canvas hierarchy and recreate — newly created TMP elements pick up `LiberationSans SDF` automatically when TMP Essentials are present. Eventually we removed the UI entirely (see below).
- **UI canvas with three input fields was abandoned.** Initial design had `HSegSplatUI.cs` driving three TMP input fields + Submit/Clear buttons on a screen-space Canvas. Dropped in favor of an **Inspector-driven workflow**: edit the three string fields and threshold directly on the `HSegSplatClient` component, click the two buttons drawn by `HSegSplatClientEditor`. No Play mode, no canvas, no event system needed. The `HSegSplatUI.cs` file was deleted.
- **Near Plane Cutoff Inspector slider appears clamped to 0–1.** No `[Range]` attribute exists in the package C# code; it's a Unity Inspector quirk for this particular field. Setting values above 1 by typing or via `SetNearCutoff.cs` works — they're not actually clamped, just visually misleading.
- **"Gaussians fill the screen when zooming in" turned out to be Scene-view orthographic mode.** Pressing the gizmo at top-right of the Scene view toggles between Perspective and Isometric/Ortho. Once in Ortho, dollying changes apparent size without affecting clip-space depth, so individual splats appear to grow without bound and `_SplatNearPlaneCutoff` has no apparent effect. **Fix: ensure Scene view shows "Persp" at the top-right corner, not "Iso".** This was the single most time-consuming red herring of the session.

### 13.7 How it's used day-to-day

```bash
# Terminal — once per session
cd second_stage/viewer/server
source venv/bin/activate
python serve.py ../../outputs/test_scene_hsegsplat/gaussians.pt
```

In Unity (no need to enter Play mode):

1. Click `Splats` in Hierarchy.
2. In the Inspector, on `HSegSplatClient`:
   - Type into **Child**, **Parent**, **Grandparent** fields. Leave any empty to skip that level.
   - Adjust level dropdowns if needed (defaults 6 / 3 / 1).
   - Adjust **Threshold** slider (default 0.5).
3. Click the **[ Submit Query ]** button drawn below the fields.
4. After ~1 second, splats matching the AND tint magenta in the Scene view.
5. Click **[ Clear Query ]** to remove the highlight.

`Last Status` shows the result inline: number of splats highlighted, min/max relevancy, threshold used.

### 13.8 Confirmed queries on the test scene

Same queries as the 2D pipeline, now in 3D:
- **Flat:** `child="chair", child-level=1` → highlights chair Gaussians.
- **Parent:** `child="glass pane", parent="window"` (no grandparent) → highlights glass/window Gaussians.
- **Grandparent:** `child="door", parent="cupboard", grandparent="cupboard"` → highlights cupboard-door Gaussians.

### 13.9 What's not in the viewer (future work)

- **No novel-view capture from the viewer.** Right now the viewer is interactive; if you want to save a relevancy render at a specific pose, you'd screenshot the Scene view. A scripted "render this scene to PNG from camera pose X" could be added in 50 lines.
- **No per-level inspection.** The AND is computed server-side, so you only see the combined result. If we wanted to toggle which level contributes, the server would need to return three buffers or expose three endpoints, and the shader would need three highlight modes. Not hard, but not implemented.
- **Single scene at a time.** The server loads one `gaussians.pt` at startup. Switching scenes requires restarting the server. A multi-scene mode would let you swap via an endpoint.
- **No way to dim non-matching splats.** Current design: matching = magenta tint, non-matching = original color. An "isolate" mode (hide all non-matching) would be useful for visual focus.

### 13.10 Files added to the repo

```
second_stage/viewer/
├── README.md                                full step-by-step setup + usage
├── gaussians_to_ply.py                      gaussians.pt → standard 3DGS .ply
├── server/
│   ├── serve.py                             FastAPI + SigLIP + per-Gaussian LERF AND
│   └── requirements.txt
└── unity_scripts/
    ├── HSegSplatClient.cs                   Inspector-driven query client
    ├── HSegSplatClientEditor.cs             Submit/Clear buttons in Inspector
    ├── SetNearCutoff.cs                     debug menu items for cutoff tuning
    └── RenderGaussianSplats.shader          patched main shader (magenta tint)
```

The Unity project itself (`second_stage/viewer/unity_project/`) is set up locally but not part of the git-tracked source — the package, baked asset, and Library are large and machine-specific. To recreate it on another machine, follow `second_stage/viewer/README.md`.

---

*End of progress report.*
