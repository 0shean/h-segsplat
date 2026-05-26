# SegSplat with Hierarchical Open-Set Segmentation — Project Plan

This document is the complete design plan for our hierarchical extension of SegSplat. It is meant to be self-contained: a reader (or future-us) should be able to pick it up cold and understand what we're building, why, and how every piece fits.

---

## 1. High-level goal

Build a **feed-forward 3D Gaussian Splatting pipeline** that:

1. Takes a small set of multi-view input images (typically 2 views).
2. Produces a 3D Gaussian scene in a single feed-forward pass.
3. Embeds **open-set semantic features** in each Gaussian using SAM + SigLIP — without any per-scene optimization.
4. Supports **hierarchical text queries** like *"leg of chair"* or *"child of sofa"* that exploit a 3-level mask hierarchy (granularities 1, 3, 6 from semantic-SAM).
5. Renders novel views with semantic relevancy maps highlighting query matches.

We base the design on **SegSplat** (Siegel et al., 2025) and extend its single-level cluster-based representation to **three parallel cluster-based levels** at semantic-SAM granularities 1, 3, 6.

The geometric backbone is **DepthSplat** (Xu et al., 2024), used frozen.

---

## 2. Why this design

### 2.1 What SegSplat does (baseline)

SegSplat's key insight: DepthSplat already predicts **one Gaussian per input pixel per input view**. So if you have a per-pixel semantic index `S_k(u, v) ∈ {0..M}`, you get a per-Gaussian semantic index for free — no extra learning, no per-scene optimization.

SegSplat then:
- Runs SAM on each input image → masks.
- Runs CLIP on each mask crop → per-mask embeddings.
- K-means clusters all mask embeddings across input views → `M` cluster centroids = **semantic memory bank** `B ∈ R^{M × D_clip}`.
- Each pixel gets a one-hot vector `e_j ∈ {0,1}^M` indicating its cluster.
- At render time, alpha-blend the one-hots with the standard 3DGS equation.
- Recover dense CLIP feature map at novel view via `F(v) = E(v)^T B`.
- Text query: SigLIP/CLIP-encode query, compute relevancy per pixel (LERF formula), threshold → highlight mask.

### 2.2 What we change

Three modifications:

1. **Three granularity levels (1, 3, 6) from semantic-SAM** instead of one.
2. **Three parallel cluster-ID one-hots per Gaussian** — same SegSplat trick (per-cluster, not per-mask), repeated independently at each level. Each Gaussian carries three integer cluster IDs.
3. **Hierarchical query support** via per-level independent relevancy combined spatially at the pixel — no tree walk, no instance identity needed.

We use **SigLIP instead of CLIP** for embeddings (better semantic features).

The DepthSplat backbone is unchanged. Our work is entirely in the **side pipeline** that produces masks/banks and the **render+query path**.

**v1 deliberately drops instance awareness.** Cluster IDs at level 1 merge all chair instances into the same slot, so "leg of *the chair on the left*" is not expressible. The v1 query "leg of chair" works because it only asks "is there a leg here AND a chair here" — spatial co-occurrence, not instance membership. Per-mask IDs and a containment tree are listed as **future work** (§6.5) when instance disambiguation becomes empirically motivated.

---

## 3. Architectural decisions (locked in)

These are the design calls we made during planning. Each was a real choice with alternatives considered.

### 3.1 Three parallel cluster-ID one-hot vectors per Gaussian

Each Gaussian carries:
- `e^(1) ∈ {0,1}^{M_1}` — level-1 (coarsest) cluster one-hot
- `e^(3) ∈ {0,1}^{M_3}` — level-3 cluster one-hot
- `e^(6) ∈ {0,1}^{M_6}` — level-6 (finest) cluster one-hot

Stored on disk as **three integer cluster IDs** per Gaussian (4 bytes × 3 = 12 bytes). One-hots are materialized at render time.

This is SegSplat's exact trick, run three times in parallel — one per granularity. Each level is an independent flat SegSplat with its own SAM masks, its own SigLIP features, its own K-means bank.

**Why per-cluster (not per-mask):** simpler, smaller banks, no cross-view ID-matching problem. The cost is losing instance identity — all chair instances collapse into the same level-1 cluster. We accept that for v1 (see §2.2); per-mask IDs are deferred to future work (§6.5).

**Why parallel rather than concatenated:** cleaner per-level relevancy interpretation. Geometry projection is shared across the three feature passes by gsplat, so we expect total render cost ≤ ~1.5× a single pass (not 3×). **This is an expectation, not a measurement** — benchmark in Phase 5 before locking in the design. If concatenation turns out to be faster or simpler with no quality loss, switch.

### 3.2 Per-cluster rendering at all three levels

We rasterize **cluster IDs**, not mask IDs. Same as SegSplat, repeated per level. K-means groups SigLIP features at each level independently into `M_level` clusters, giving three banks `B^(1), B^(3), B^(6)` of cluster centroids.

Cost: `M_level` is per-scene, set by the same heuristic as SegSplat (`M = λ · N_total / K` with λ=1.2, where `N_total` is total masks across input views at that level and `K` is the number of input views).
- Expected ranges (2 context views): `M_1 ≈ 5–15`, `M_3 ≈ 20–50`, `M_6 ≈ 60–150`.
- gsplat handles this comfortably.

### 3.3 Cross-view consistency comes for free from clustering

Each Gaussian belongs to exactly one source view by DepthSplat's construction. With per-cluster IDs, the same chair seen in two views ends up in the same level-1 cluster slot because K-means groups their SigLIP features together — exactly SegSplat's mechanism. No global mask matching, no per-view namespacing.

This is the main simplification over a per-mask design: we don't have a cross-view identity problem because cluster identity *is* the cross-view bridge.

### 3.4 Argmax on rendered cluster-ID maps (only if needed)

After alpha-blending, `E^(level)(v)` is a *soft* distribution over cluster IDs at that level. For text queries we recover `F^(level)(v) = E^(level)(v)^T B^(level)` (SegSplat eq. 3) and run LERF relevancy directly on the continuous feature — **no argmax needed for the v1 query path** (§6.3).

Argmax is only needed if we want to visualize per-level segmentation maps (color-by-cluster). Known failure mode there: silhouette edges where two Gaussians contribute roughly equally → speckle. Fallback options (in order of preference):

1. **Confidence threshold:** if `max(E^(level)(v)) / sum(E^(level)(v)) < 0.5`, mark uncertain → skip that pixel in the visualization.
2. **Spatial smoothing:** 3×3 majority vote on the argmax map.

Don't pre-implement these. Add them only if needed.

> **Note on `E^(level)(v)` mass.** Standard 3DGS rasterization yields `E(v) = Σ T_i α_i e_i` where `T_i = Π_{j<i}(1 − α_j)`. The total accumulated mass `Σ T_i α_i` is `≤ 1` and is often substantially less than 1 in low-density regions. So any confidence threshold must be applied to `max / sum` (fraction of accumulated mass on the top cluster), **not** to raw `max(E^(level)(v))`. Easy to get wrong while debugging.

### 3.5 No hierarchy tree in v1

The plan previously included a parent tree over masks (90% containment, validated closer-level-wins) to support tree-walk queries from finest level up to coarsest. **v1 drops this entirely.**

Reason: with per-cluster IDs, a tree walk would walk from a cluster centroid (at level 6) to a cluster centroid (at level 1), which is not what containment buys us — containment is a mask-instance property. And the v1 query semantics (§6.3) get the "is X part of Y" answer from **spatial co-occurrence at the rendered pixel**, not from a tree edge.

What this drops compared to a per-mask + tree design:
- We cannot ask "leg of *the chair on the left*" (no instance identity at level 1).
- We cannot enforce that "leg" and "chair" actually belong to each other in 3D — only that they co-occur in this rendered view.

Both are acceptable for v1. See §6.5 for the future-work version that adds them back.

### 3.6 Background Gaussians: render normally, not queryable

If a Gaussian's source pixel has no SAM mask at level `k`, its `e^(k)` is **all zeros** (no slot active).

Consequences:
- **Color rendering** is unaffected — uses SH and opacity normally.
- **Semantic rendering** receives zero contribution from background Gaussians → at novel-view pixels dominated by background, `‖E^(level)(v)‖ ≈ 0` → relevancy is near-zero → not highlighted. Correct behavior.

A pixel can be background at one level and not another (e.g., level-1 mask exists but no level-6 sub-mask covers it). Each level is independent in this regard; no inheritance enforced.

### 3.7 SAM+SigLIP at full resolution; nearest-neighbor downsample to DepthSplat resolution

DepthSplat operates on resized inputs (e.g., 256×256). SAM produces better masks at full resolution. So:

1. Run **SAM at full resolution** for all 3 granularities.
2. Run **SigLIP at full resolution** on each mask's crop (one feature per mask).
3. K-means per level → 3 banks `B^(1), B^(3), B^(6)`. Each mask is assigned its closest cluster centroid → per-pixel cluster-ID map at each level.
4. Drop masks that would have <2 pixels after downsampling (they contribute nothing to any Gaussian).
5. **Nearest-neighbor downsample** the per-pixel cluster-ID maps to DepthSplat resolution.

Bilinear/area downsampling is **forbidden** — these are integer label maps; averaging them is meaningless.

---

## 4. The DepthSplat side: minimal changes

DepthSplat is treated as a **frozen geometric module**. The only "modification" is documenting and exposing the deterministic pixel-to-Gaussian mapping that already exists.

### 4.1 The 1-to-1 mapping already exists

In `depthsplat/src/model/encoder/encoder_depthsplat.py`:

- `num_surfaces = 1` and `gaussians_per_pixel = 1` are the defaults (config files: `config/model/encoder/depthsplat.yaml`, `config/experiment/re10k.yaml`, etc.). **Do not change.**
- The depth predictor produces `[B, V, H, W]` per-pixel depths (line 162).
- The gaussian head produces `[BV, C, H, W]` per-pixel Gaussian parameters (line 242).
- `sample_image_grid((h, w))` generates one ray per pixel (line 282).
- `GaussianAdapter.forward(...)` lifts each pixel's `(xy_ray, depth, raw_params)` into a 3D Gaussian (line 318).
- Final flatten (lines 346–362):
  ```
  "b v r srf spp xyz -> b (v r srf spp) xyz"
  ```
  with `r = H*W`, `srf = 1`, `spp = 1`.

So **Gaussian flat index `i` → pixel `(v, y, x)`**:
```python
v = i // (H * W)
rem = i % (H * W)
y = rem // W
x = rem % W
```

Total Gaussians per scene = `V × H × W`.

### 4.2 What we need on the DepthSplat side

Essentially nothing structural. Two pieces of glue:

1. **A documented index-recovery helper** (~10 lines) that converts between `(v, y, x)` and flat Gaussian index. We can either return the unflattened structure `[B, V, H, W, ...]` or expose a utility function.

2. **Resolution agreement check.** Whatever `(H, W)` the encoder ends up using (after `apply_crop_shim` and `apply_patch_shim`) is the resolution our downsampled cluster-ID maps must match. Add an explicit assertion.

That's it for DepthSplat. No new layers, no retraining, no architectural changes.

---

## 5. The standalone inference entry point

We're not using DepthSplat's `.torch` chunk dataset. We're building a **standalone inference script**:

```
input: a directory of images + camera poses
output: rendered novel-view RGB + query relevancy maps
```

This avoids touching the dataset/`view_sampler` machinery and lets the semantic side run as plain Python.

### 5.1 Pipeline overview

```
[ Input images (full-res) ]
        │
        ├─→ [ Resize to DepthSplat resolution ] ──→ [ DepthSplat encoder ] ──→ Gaussians (V·H·W)
        │
        └─→ [ Semantic side pipeline ]
                  │
                  ├─ SAM at granularities 1, 3, 6   (full-res)
                  ├─ SigLIP per-mask features       (full-res crops)
                  ├─ K-means per level              → 3 banks B^(1), B^(3), B^(6) (cluster centroids)
                  ├─ Per-pixel cluster-ID maps      (full-res, one per level)
                  ├─ Drop tiny masks (<2 pixels post-downsample)
                  └─ NN-downsample cluster-ID maps  → 3 maps S^(level) at DepthSplat (H, W)
                                                       └─→ Per-Gaussian (cluster_id_1, cluster_id_3, cluster_id_6)
                                                             via flat-index ↔ (v, y, x) mapping

[ Novel view request: camera pose + optional text query ]
        │
        ├─ Render RGB              (gsplat with SH)
        ├─ Render feature maps     (gsplat with feature dim M_level, SH degree 0)
        │     └─ E^(1)(v), E^(3)(v), E^(6)(v)
        │
        └─ If text query:
             ├─ SigLIP-encode query terms
             ├─ Compute F^(level)(v) = E^(level)(v)^T B^(level)   for the queried levels
             ├─ LERF relevancy at each queried level
             ├─ Combine per-level relevancies spatially (AND at pixel — §6.3)
             └─ Threshold → relevancy mask → highlight on rendered RGB
```

### 5.2 Why standalone

- No need to extend `apply_crop_shim` / `apply_augmentation_shim` for semantic data.
- Resolution decision is a single `Resize(...)` call before everything.
- The semantic pipeline is plain torch/numpy, easy to debug and iterate on.
- DepthSplat runs as a black-box `gaussians = model(images, poses)`.

---

## 6. Hierarchical query semantics

### 6.1 Data structures

Per scene (post-preprocessing):

- **Three banks of cluster centroids:** `B^(1) ∈ R^{M_1 × D_siglip}`, `B^(3) ∈ R^{M_3 × D_siglip}`, `B^(6) ∈ R^{M_6 × D_siglip}`. Each row is the SigLIP centroid for one cluster of masks at that level (K-means output).
- **Three cluster-ID maps per view:** `S^(level)_v ∈ {0..M_level}^{H × W}` (0 = background).
- **Per-Gaussian:** three integer cluster IDs (one per level), recoverable from the flat-index ↔ pixel mapping plus the corresponding `S^(level)_v`.
- **No tree.** No parent links. v1 query semantics don't need them.

### 6.2 Simple query: "chair"

- SigLIP-encode "chair" → `φ_qry`.
- Compute `F^(1)(v) = E^(1)(v)^T B^(1)` (rendered SigLIP-feature map at level 1).
- LERF relevancy at each pixel (using canonical phrases "object", "things", "stuff" with SigLIP):
  ```
  relevancy(v) = min over canonical phrases of softmax-relative score
  ```
- Threshold at 0.5 → highlight mask.

This is just SegSplat at level 1.

### 6.3 Hierarchical query: "leg of chair" — per-level independent relevancy (v1)

Parsed as `(child_term="leg", parent_term="chair")` with target levels `(level_6, level_1)` (or `level_3` if appropriate; the level mapping is a UX decision — see §6.4).

For each pixel `v` in the rendered novel view:

1. Compute level-1 relevancy of `F^(1)(v)` against "chair" → `r_chair(v)`.
2. Compute level-6 relevancy of `F^(6)(v)` against "leg" → `r_leg(v)`.
3. Pixel is in the highlight mask iff both `r_chair(v) ≥ τ` AND `r_leg(v) ≥ τ`.

The "is this leg part of the chair I see?" constraint is enforced **spatially** — same pixel — not by tree edges. With per-cluster IDs, all chair instances share the level-1 `chair` slot anyway, so a tree wouldn't disambiguate them either.

**Tradeoff:** loses the ability to disambiguate "leg of *this specific* chair instance" when multiple chair instances are visible. Both legs and both chairs light up; the AND mask covers all of them. For v1 we accept this — instance disambiguation is future work (§6.5).

### 6.4 Open design questions for the query layer

- **Level inference:** "leg of chair" maps to `(6, 1)`, but "wheel of car" might be `(3, 1)`. We may need a small heuristic or LLM call to pick target levels per query.
- **K per level:** how many clusters at each level? SegSplat uses `M = λ · N_total / K` with λ=1.2 — same heuristic per level is the obvious starting point, but levels 3 and 6 have many more masks and may want different λ.
- **Multi-token queries:** "chair" is one token; "leg of red chair" needs more parsing. Out of scope for v1.

### 6.5 v2 design: per-mask IDs + containment-derived parent chain + fluid level binding

v1 has two limitations: (a) parent-child relationship is inferred from spatial co-occurrence at the rendered pixel only, not from any structural link, and (b) the AND clips silhouette regions — a level-6 "leg" mask that sticks 10% outside the level-3 "chair" mask loses those border pixels, exactly the pixels you most want highlighted.

v2 fixes both by promoting containment from a per-pixel test to a **per-mask structural link**. Implementation lives entirely in the build script + server; Unity client and shader are unchanged.

#### 6.5.1 Per-Gaussian payload

Replace "three cluster IDs" with **one finest-mask reference plus three SigLIP-feature references**:

- `finest_level[G]` ∈ {1, 3, 6, 0=bg} — the level of the finest mask covering this Gaussian's source pixel. Determined at build time: prefer level 6, fall back to 3, fall back to 1, else background.
- `finest_local_mask_id[G]` ∈ {0..N_v_lvl} — its mask index *within that level for that view*.

The (view, level, local_mask_id) triple uniquely identifies the Gaussian's "home" mask.

#### 6.5.2 Per-mask side table

For each level `L`, write at build time:

- `level_<L>/mask_features.npy` — `(N_L_total, D=1152)` L2-normalized SigLIP features, in a fixed flat order spanning all views.
- `level_<L>/mask_directory.json` — list of `{view_idx, frame_name, local_mask_id, area_full_res, global_id}` in the same flat order.

#### 6.5.3 Containment parent dict

A separate script (`compute_parent_chain.py`) runs at build time on the full-resolution binary masks. For each view, it computes three containment maps:

- **6 → 3:** for each level-6 mask `m`, find the level-3 mask `p` maximizing `|m ∩ p| / |m|`. Link if `≥ 0.9`.
- **6 → 1:** for each level-6 mask `m`, find the level-1 mask `p`. Link if `≥ 0.9`. **Computed directly — not derived as the transitive closure of 6→3 → 3→1.** This is what enables the 2-term query's "6→1 binding" (§6.5.5) to work even when no qualifying level-3 ancestor exists.
- **3 → 1:** same rule.

Output: `per_mask/parents.json`:
```json
{
  "level_6": {
    "<global_id>": {"level_3": <gid> | null, "level_1": <gid> | null}
  },
  "level_3": {
    "<global_id>": {"level_1": <gid> | null}
  }
}
```

Containment is O(N²) numpy at our scale (~100 masks per level per view) — runs in seconds.

#### 6.5.4 Query semantics — "Scheme B"

The parent constraint is evaluated against the *parent mask's own SigLIP feature*, not against the parent-level cluster at the Gaussian's source pixel. So a level-6 leg-Gaussian whose pixel is *outside* the level-3 chair-mask still gets highlighted — its level-6 mask was linked to that chair mask at build time and the link survives.

This is the "Scheme B" choice (over "Scheme A" — per-pixel AND of two cluster matches). Scheme A would systematically clip silhouettes; Scheme B treats containment as a one-time structural assertion.

#### 6.5.5 Fluid level binding by query shape

The number of provided terms determines which level bindings are valid. A Gaussian is highlighted if **any** valid binding passes; the returned relevancy is the max over bindings (each binding's score is the min of its per-term relevancies — soft AND within a binding).

**1 term — `child` only:**
- Match `child` against every Gaussian's finest mask, whatever level it landed on (1, 3, or 6).
- The child can be coarse or fine — the query is fully unconstrained on hierarchy depth.

**2 terms — `child` + `parent` (three sub-bindings, take max):**

| Binding | Child level | Parent level | Notes |
|---|---|---|---|
| 6→3 | 6 | 3 | Standard "fine part of object" case |
| 6→1 | 6 | 1 | Lets fine parts pair with a coarse parent even when level-3 ancestor missing |
| 3→1 | 3 | 1 | "Mid-grain part of object" |

For each Gaussian, evaluate all three bindings; take the max.

**3 terms — `child` + `parent` + `grandparent` (locked, no fallback):**
- Child mask must be at lvl 6 and match `child`.
- Its lvl 3 ancestor must exist and match `parent`.
- Its lvl 1 ancestor must exist and match `grandparent`.
- No sub-bindings, no fallbacks — the user has fully specified the hierarchy.

#### 6.5.6 Server-side evaluation

On startup, load per-mask features + parent dict + per-Gaussian finest-mask payload. On query:

1. SigLIP-encode each provided term (cached).
2. For each term `T_x` and each relevant level `L`, compute LERF relevancy `R_x_L[N_L]` against `mask_features[L]` (per-mask, not per-Gaussian; tiny).
3. For each binding required by the query shape, gather per-Gaussian relevancies:
   - Child: `R_child_L[finest_local_mask_id[G]]` if `finest_level[G] == L`, else fail this binding.
   - Parent: look up parent mask via the parent dict, gather its relevancy.
   - Grandparent: same.
4. Per-binding score = min over its terms. Output = max over bindings. Background Gaussians get 0.
5. Return `float32[G]` — same format as v1, shader unchanged.

All gather operations are vectorized numpy; no Python per-Gaussian loop.

#### 6.5.7 What this still does *not* give us

- **Instance disambiguation** — multiple chair instances still aren't separable. The level-1 chair mask containing a leg's level-3 ancestor doesn't carry instance identity. If you want "leg of *the chair on the left*," you need a different system (multi-view instance matching + per-instance IDs). Out of scope for v2.
- **Cross-view containment** — parent dict is per-view. A level-6 mask in view 0 with no level-3 parent in view 0 cannot inherit a level-3 parent from view 1, even if the same physical part is segmented at level 3 in the other view. Acceptable for v2.

### 6.6 Search3D §III-B bbox expansion for SigLIP context — **implemented**

Search3D (Takmaz et al. 2025) reports that expanding the tight 2D bbox around a mask before SigLIP encoding gives better features for small parts. The paper splits the trick across two stages:

- **Object-level (their step 4):** multi-scale cropping — 3 nested crops at expansion ratios `k_exp = 0.2` per step, all encoded and average-pooled per view.
- **Part-level (their step 5):** single 10% expansion (`k_exp = 0.1`) around the tight bbox. One crop per part-mask.

#### What we implement (`scripts/run_siglip.py`)

Per-level encoding strategy:

| Level | Strategy | Crops per mask | Rationale |
|---|---|---|---|
| 1 | original two-crop average (mask-zeroed + bbox-with-bg) | 2 | whole-object masks already include plenty of context |
| 3 | single 10% expanded bbox, background kept | 1 | mid-grain masks benefit from a bit of surrounding context |
| 6 | single 10% expanded bbox, background kept | 1 | Search3D's part-level recipe — the level where it matters most |

The expansion is configured via `MaskCropEncoder.DEFAULT_EXPANSION = {1: 0.0, 3: 0.1, 6: 0.1}` and threaded through `embed_all(..., level=L)`.

We extend Search3D's part-only recommendation by also expanding level 3. Defensible because level 3 masks are mid-grain; whether it helps in practice is an empirical question for the eval phase.

#### Critical invariant — expansion is SigLIP-input only

The expanded bbox exists for ~50 ms during stage 2's forward pass, then is discarded. **Nothing downstream knows or cares.** Specifically unaffected:

- `mask_*.png` files on disk (SAM output is unchanged)
- per-pixel mask_id_maps, full-res containment computation
- the per-Gaussian finest-mask reference (`finest_local_mask_id`, `finest_global_mask_id`)
- `parents.json` (computed on the original binary masks)
- the 1-to-1 DepthSplat pixel↔Gaussian mapping

The only quantity affected is **the SigLIP feature** of each mask, which feeds K-means → banks → per-mask LERF scores at query time.

#### Cost

1× the SigLIP forward passes at levels 3 and 6 (instead of 2× from the dropped two-crop average), so actually *slightly faster* at preprocess time. Level 1 keeps the two-crop method.

#### Future variant (not implemented)

Search3D's full object-level recipe: 3 nested crops at `k_exp = 0.2` averaged. Costs 3× the forward passes per mask. Try as an ablation if eval shows the single-expansion isn't enough.

---

## 7. Implementation phases

The overarching strategy: **reproduce SegSplat first (flat, single-level), then add hierarchy**. This keeps every phase debuggable in isolation. If level-1 flat queries don't work, no amount of hierarchical sophistication will save us — and conversely, if they do work, the hierarchical extension is incremental.

### Phase 1: DepthSplat sanity check (smallest possible)
- Standalone script that loads a pretrained DepthSplat checkpoint, takes 2 images + poses, outputs Gaussians.
- Verify the flat-index ↔ pixel mapping by reconstructing a "color-by-source-pixel" Gaussian cloud and rendering it.
- Confirm post-shim `(H, W)` resolution.
- **OOD smoke test:** the pretrained DepthSplat checkpoint is trained on RealEstate10K (indoor real-estate scenes). 3D-OVS scenes are object-centric and may be out-of-distribution. Render a few 3D-OVS scenes with the bundled rasterizer first and check that geometry/PSNR is reasonable (SegSplat reports ~17 PSNR on 3D-OVS vs. ~26 on RE10K — a real drop, but it does work). If geometry is broken on object-centric scenes, no amount of semantic work will help; pause and reassess.

### Phase 2: Semantic side pipeline (2D only, independent of DepthSplat)
- Wrapper around our existing semantic-SAM + SigLIP pipeline that produces:
  - Per-image, per-level mask maps at full res for granularities 1, 3, 6.
  - Per-mask SigLIP features.
- K-means per level → three banks `B^(1), B^(3), B^(6)` of cluster centroids. Each mask gets assigned to its closest centroid → per-pixel cluster-ID map at each level.
- Drop tiny masks (<2 pixels post-downsample).
- Nearest-neighbor downsample cluster-ID maps to DepthSplat's `(H, W)`.

### Phase 2.5: 2D query sanity check (de-risk the query semantics before 3D)
**This is the most important phase to do early.** Before touching DepthSplat, run the full SAM+SigLIP+K-means pipeline on plain 2D images and validate:

- For a few test images, do queries like "chair" (flat level-1) actually pick out chairs?
- Does "leg of chair" (per-level independent AND at levels 1+6) correctly highlight chair legs?
- Visualize: cluster-ID maps at all 3 levels, cluster colors consistent within a scene.

If queries don't work in 2D, they won't work in 3D — fix the 2D pipeline first.

### Phase 3: Reproduce SegSplat baseline (flat, level-1 only, cluster bank)
*Already done in the existing colab pipeline.* This is the starting point.
- Gaussians ↔ level-1 cluster IDs (SegSplat exactly).
- Replace DepthSplat's bundled CUDA rasterizer with **gsplat** (the bundled one is only 3-channel).
- Render: RGB + level-1 one-hot semantic feature map (SH degree 0).
- Compute `F^(1)(v) = E^(1)(v)^T B^(1)`, run LERF relevancy for a single text query, threshold, highlight.
- **Goal already achieved:** end-to-end flat query path is working.

### Phase 4: Multi-level cluster-ID per-Gaussian data
- Extend the per-Gaussian semantic data from one cluster ID to three: `(cluster_id_1, cluster_id_3, cluster_id_6)`.
- Three banks `B^(1), B^(3), B^(6)` produced by the Phase 2 pipeline.
- Sanity check: visualize per-level cluster-ID maps (color-by-cluster) on input images; check they look like progressively finer segmentations.

### Phase 5: Three parallel rasterizations
- Three parallel feature renders at novel view → `E^(1), E^(3), E^(6)`.
- **Benchmark total render cost** vs Phase 3 single-level. If it's >2× a single pass, consider concatenated rasterization (§3.1) — measurement decides.
- Visualize each level's rendered cluster map (via `argmax` of `E^(level)(v)` for visualization only, not for queries) to verify alignment with input images.

### Phase 6: Hierarchical queries (per-level independent AND, v1)
- Implement query parsing: text → `(child_term, child_level, parent_term, parent_level)`.
- Compute LERF relevancy independently at each queried level.
- Highlight pixels passing both relevancy thresholds (§6.3).
- Test queries: "leg of chair", "wheel of car", "child of sofa", etc.

### Phase 7: Evaluation
- **Quantitative (flat queries only):** 3D-OVS mIoU at level 1, compare to SegSplat. This validates Phases 3–4 (we should match SegSplat at level 1 since level-1 is just SegSplat).
- **Qualitative (hierarchical queries):** render novel views with hierarchical queries on a few test scenes. Hierarchy is the main contribution, but no standard benchmark exists — qualitative renders are the primary evidence.
- **Optional:** small hand-labeled hierarchical eval set (e.g., 5–10 scenes × 5 hierarchical queries) if time permits. Decide upfront whether this is in scope.

---

## 8. Things to watch for / known risks

### 8.1 Resolution mismatch
The single biggest practical risk. Mitigation: every tensor that has spatial dims gets an explicit `(H, W)` assertion before it touches Gaussian-aligned data.

### 8.2 Cache invalidation across re-runs
SAM mask indices are **list positions**, arbitrary. If we ever regenerate masks (different SAM version, different prompts/granularity, different seed), all caches keyed by mask IDs become silently stale.

Mitigation: hash `(image_bytes, sam_config)` and key all caches (banks, hierarchies, downsampled maps) by that hash. Invalidate together.

If we generate masks once and never regenerate, this is moot.

### 8.3 Argmax speckle at silhouettes (visualization only)
Only relevant if we render per-level color-by-cluster visualizations. The v1 query path uses continuous features and does not argmax. If we add the visualization and see speckle, apply mitigations from §3.4.

### 8.4 Tiny mask loss at downsampling
Step 4 of §3.7 — drop masks <2 pixels at DepthSplat resolution. Otherwise we have rasterization slots that are always zero. Not broken, just wasteful and confusing during debugging.

### 8.5 Background pixels at multiple levels
A pixel with no SAM mask at any level → all-zero one-hots at all levels → contributes nothing to semantic rendering. Color rendering unaffected. This is correct behavior; just be aware that "background" is implicit, not a special slot.

### 8.6 Instance collapse at level 1 (by design, but worth flagging)
Two visually similar chairs end up in the same level-1 cluster. The v1 query "leg of chair" lights up legs of *all* chairs in view. This is a deliberate v1 simplification (§6.5), not a bug — but it's the first thing to check if results look "too inclusive" on a multi-instance scene.

---

## 9. What we are explicitly NOT doing (in v1)

- **Not** training or fine-tuning DepthSplat.
- **Not** training a per-scene autoencoder (that's what makes LangSplat slow; SegSplat's whole point is to skip it).
- **Not** using per-mask IDs. Cluster IDs at three levels, same as SegSplat repeated three times. Per-mask is future work (§6.5).
- **Not** building any hierarchy tree. v1 queries are spatial AND at the pixel, not tree walks. Containment trees are future work (§6.5).
- **Not** doing instance-aware queries. "Leg of *this* chair" is out of scope.
- **Not** modifying the dataset / view_sampler / `.torch` chunk pipeline. Standalone inference only.
- **Not** trying to handle dynamic scenes. Static only, like SegSplat.

---

## 10. Glossary / cheat sheet

- **DepthSplat:** frozen feed-forward 3DGS backbone. Predicts per-pixel Gaussians.
- **SegSplat:** baseline paper. Adds open-set semantics on top of DepthSplat via SAM+CLIP+K-means.
- **SAM / semantic-SAM:** Segment Anything; semantic-SAM gives multi-granularity masks.
- **SigLIP:** image-text contrastive model; we use it instead of CLIP for embeddings.
- **One-hot per Gaussian (`e_j`):** indicator vector for which cluster a Gaussian belongs to (at a given level).
- **Memory bank (`B`):** matrix where each row is a SigLIP centroid for one cluster.
- **Alpha-blending in feature space:** standard 3DGS rasterization, but the per-Gaussian quantity is a feature vector instead of RGB. Linear → commutes with bank lookup → enables the "render one-hot, then multiply by `B`" trick.
- **LERF relevancy:** the formula from `LERF` (Kerr et al. 2023) that scores a pixel's feature against a query in the presence of canonical reference phrases ("object", "things", "stuff").
- **Granularity levels 1/3/6:** semantic-SAM's three coarseness settings. 1 = coarse (whole objects), 6 = fine (parts).

---

## 11. Unity desktop viewer (production interface for v1, used for v2)

v1 is delivered as an **interactive Unity desktop viewer** at `second_stage/viewer/`. Inference runs on Colab; results are downloaded and loaded into the viewer for interactive querying. Three pieces:

### 11.1 Three components

- **Python server** (`second_stage/viewer/server/serve.py`) — FastAPI + OpenCLIP SigLIP. Loads `gaussians.pt` once at startup. Endpoints: `/info`, `/query_combined`, `/extrema`. On query, encodes terms with SigLIP, computes per-cluster LERF relevancy, scatters to per-Gaussian via `cluster_index[lvl]`, AND-combines across provided levels by per-Gaussian min, returns `float32[G]` binary stream.
- **PLY converter** (`second_stage/viewer/gaussians_to_ply.py`) — turns `gaussians.pt` into a standard 3DGS `.ply` for the aras-p Unity package to bake into a `GaussianSplatAsset`.
- **Unity project** at `second_stage/viewer/unity_project/` (Unity 2023.1.14f1, Built-in render pipeline, aras-p `com.aras-p.gaussian-splatting` package). Local-only; not git-tracked.

### 11.2 Per-Gaussian AND vs. per-pixel AND

The 2D `query_hsegsplat.py` script AND's at the rendered pixel after alpha-blending. The viewer AND's in cluster space, per Gaussian — a Gaussian passes only if its level-1 cluster matches the grandparent term AND its level-3 cluster matches the parent term AND its level-6 cluster matches the child term, simultaneously, as cluster memberships of the same Gaussian. Different math, similar visual result on practical scenes.

### 11.3 Unity-side glue

- **`HSegSplatClient.cs`** — `[ExecuteAlways]` MonoBehaviour on the Splats GameObject. Inspector exposes `Child / Parent / Grandparent` text fields, `Child Level / Parent Level / Grandparent Level` ints, `Threshold` slider. POSTs JSON to the server, receives `4·G` bytes, casts to `float[G]`, writes into the package's `internal m_GpuQuerySimilarities` field via reflection, calls `SetData`, sets `_MinRelevancyScore` shader global, repaints the Scene view.
- **`HSegSplatClientEditor.cs`** — custom Inspector adds two buttons: **[ Submit Query ]** / **[ Clear Query ]**.
- **`RenderGaussianSplats.shader`** — drop-in replacement for the package's main render shader. Adds `StructuredBuffer<float> _SplatQuerySimilarities` + `float _MinRelevancyScore`; vertex stage reads `_SplatQuerySimilarities[instID]`; fragment lerps splat color toward magenta `(1, 0, 1)` at strength 0.75 when `relevancy ≥ _MinRelevancyScore`.

When no query is active (`_MinRelevancyScore = 2.0` or unset), the shader behaves identically to the upstream package.

### 11.4 Colab workflow

```
1. !git clone depthsplat + install torch 2.4 + cu124 + reqs
2. Download pretrained depthsplat checkpoint + RE10K test subset
3. files.upload(<scene>_hsegsplat_colab.zip) + unzip
4. files.upload(colab_hsegsplat_inference.py)
5. pip install gsplat imageio
6. python colab_hsegsplat_inference.py +experiment=dl3dv ... ++segsplat.levels=[1,3,6]
7. Display render_view*.png inline
8. files.download(rendered_rgb.npy, rendered_feature_map_lvl{1,3,6}.npy, gaussians.pt)
```

The viewer loads `gaussians.pt` directly; the feature maps are only used by the 2D `query_hsegsplat.py` script and are not needed for the 3D viewer.

### 11.5 Day-to-day use

```bash
cd second_stage/viewer/server && source venv/bin/activate
python serve.py ../../outputs/<scene>/gaussians.pt
```

In Unity (no Play mode needed): select Splats → type Child/Parent/Grandparent in Inspector → click [ Submit Query ]. Matching Gaussians tint magenta.

---

## 12. v2 implementation plan (next deliverable)

This section is the contract for the v2 upgrade. §6.5 describes the design; this section lists the concrete edits.

### 12.1 Goals

1. Replace per-Gaussian three-cluster payload with per-mask reference + parent dict (§6.5.1–6.5.3).
2. Implement Scheme B query semantics with fluid level binding by query shape (§6.5.4–6.5.5).
3. Adopt Search3D §III-B bbox expansion at levels 3 and 6 (10% single expansion, see §6.6).

v2 ships as a **side-by-side `/query_combined_v2` endpoint** — v1 stays alive for direct comparison.

### 12.2 Concrete edits

| File | Change |
|---|---|
| `second_stage/scripts/build_hsegsplat_inputs.py` | Emit `level_<L>/mask_features.npy` + `level_<L>/mask_directory.json` per level. Update Colab zip bundling. |
| `second_stage/scripts/compute_parent_chain.py` | **New.** Numpy containment over full-res masks. Writes `per_mask/parents.json` (6→3, 6→1, 3→1 direct). |
| `second_stage/scripts/colab_hsegsplat_inference.py` | Add `finest_level[G]` + `finest_local_mask_id[G]` to `gaussians.pt`. Pack `mask_features`, `mask_directory`, `parents` into `gaussians.pt` so the viewer is one-file. |
| `second_stage/viewer/server/serve.py` | Add `/query_combined_v2` implementing fluid-binding evaluator. Keep `/query_combined` untouched. |
| `second_stage/viewer/unity_scripts/HSegSplatClient.cs` | Inspector checkbox **Use v2 (per-mask)** switching endpoint. |
| `scripts/run_siglip.py` | Per-level crop strategy: level 1 keeps two-crop, levels 3+6 use single 10%-expanded bbox (Search3D §III-B). See §6.6. |

### 12.3 Order of operations

1. Build script + parent-chain script (local).
2. Spot-check parent dict on the test scene before going to Colab.
3. Update Colab inference script. Re-run cells 4 → 6 → 8 of the Colab workflow.
4. Add v2 endpoint to server. Hot-reload — same `gaussians.pt`.
5. Add Unity v2 toggle.
6. Compare v1 vs v2 highlights on the existing queries ("chair", "glass pane of window", "door of cupboard of cupboard"). Eyeball whether silhouette regions look more complete.
7. Search3D bbox expansion (§6.6) now built into `scripts/run_siglip.py`. Re-run stage 2 onward to apply.

### 12.4 What v2 unlocks vs v1

- "Leg of chair" now highlights the *entire* level-6 leg mask wherever its containing level-3 mask (or fallback level-1 mask) matches "chair" — not just the geometric intersection of the two mask regions at the rendered pixel.
- Fluid 1/2/3-term semantics: "chair" works at any level, "leg of chair" tries 6→3 + 6→1 + 3→1, "panel of door of cupboard" locks to 6→3→1.
- Containment is computed once at build time; query time stays a few numpy gather ops.

### 12.5 What v2 still doesn't give us

- No instance disambiguation (§6.5.7). Multi-chair scenes still highlight all instances.
- No cross-view parent inheritance (§6.5.7).
- Same three-level ceiling (semantic-SAM gives us {1, 3, 6}).

---

## 13. Files / locations

- `depthsplat/src/model/encoder/encoder_depthsplat.py` — the file with the per-pixel Gaussian construction. Lines 282–362 are the relevant range.
- `depthsplat/src/model/decoder/decoder_splatting_cuda.py` — bundled rasterizer. Replaced with gsplat in v1.
- `depthsplat/config/model/encoder/depthsplat.yaml` — has `num_surfaces: 1` and `gaussians_per_pixel: 1`. Don't change.
- `SegSplat.pdf`, `depthsplat.pdf`, `search3d.pdf` — reference papers in the repo root.
- `second_stage/scripts/` — build + inference + 2D query scripts (v1 in place, v2 incoming).
- `second_stage/viewer/` — production interface: Python query server + PLY converter + Unity project.
- `first_stage/` — Phase-3 (flat SegSplat) reference; left untouched.

---

*End of plan.*
