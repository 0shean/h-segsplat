# H-SegSplat Unity Viewer

A minimal desktop Unity viewer for H-SegSplat scenes. Renders a 3DGS scene reconstructed by `colab_hsegsplat_inference.py` and lets you type three text terms (child / parent / grandparent) to highlight matching Gaussians in **magenta**.

Architecture:

```
[Unity desktop app]              [Python FastAPI on localhost:8000]
  HSegSplatUI  ────────────────►  /query_combined  (child, parent, grandparent, levels)
  HSegSplatClient                 │
                                  │ load gaussians.pt once
  GaussianSplatRenderer  ◄────────┘ encode SigLIP text per query
  (aras-p splat package,          combine AND across levels in cluster space
   shader patched to              return binary float32 × N_gaussians
   tint magenta on relevancy ≥ τ)
```

Non-matching Gaussians render with their normal SH-evaluated color. Matching Gaussians (relevancy ≥ threshold) are tinted magenta in the fragment shader.

---

## What's in this folder

```
viewer/
├── README.md                           ← this file
├── gaussians_to_ply.py                 converts gaussians.pt → standard 3DGS .ply
├── server/
│   ├── serve.py                        FastAPI + SigLIP
│   └── requirements.txt
└── unity_scripts/
    ├── HSegSplatClient.cs              Unity ↔ server
    ├── HSegSplatUI.cs                  three input fields + buttons
    └── RenderGaussianSplats.shader     PATCHED main render shader (drop into package)
```

---

## One-time setup

### 1) Convert `gaussians.pt` → `.ply`

From the repo root:

```bash
python3 second_stage/viewer/gaussians_to_ply.py \
  --gaussians second_stage/outputs/test_scene_hsegsplat/gaussians.pt \
  --out       second_stage/outputs/test_scene_hsegsplat/test_scene_hsegsplat.ply
```

Expected: ~192 MB file, "Loaded 1228800 Gaussians, K_sh=9".

### 2) Set up the Python server (one virtualenv, ~6 GB on disk for torch+open_clip)

```bash
cd second_stage/viewer/server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Smoke-test it loads:

```bash
python serve.py ../../outputs/test_scene_hsegsplat/gaussians.pt --port 8000
```

Expected logs (first run downloads SigLIP weights):

```
[serve] Loading gaussians.pt: ../../outputs/test_scene_hsegsplat/gaussians.pt
[serve] Loaded G=1228800, levels=[1, 3, 6], M={1: 9, 3: 36, 6: 59}, D=1152
[serve] Loading ViT-SO400M-14-SigLIP (webli)
[serve] Encoding canonical phrases
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Quick sanity test from another terminal:

```bash
curl -X POST http://127.0.0.1:8000/extrema \
  -H "Content-Type: application/json" \
  -d '{"child":"chair","child_level":1}'
# {"min":0.0,"max":...,"n_gaussians":1228800,"n_above_0_5":...}
```

Keep this terminal running while you use Unity. Stop with Ctrl-C.

### 3) Create a new Unity project (2023.1.14f1)

Open Unity Hub → New project → **3D (built-in render pipeline or URP — either works)** → location: `second_stage/viewer/unity_project/`.

### 4) Import the splat package into the new project

The aras-p Gaussian Splatting package lives at `FINAL_TEST/Mixed_reality_gaussian_splatting/package/`. Two options:

- **Copy:** `cp -r FINAL_TEST/Mixed_reality_gaussian_splatting/package second_stage/viewer/unity_project/Packages/com.aras-p.gaussian-splatting`. Unity auto-imports on next focus.
- **Add from disk:** in Unity, `Window → Package Manager → + → Add package from disk` → select `package.json` inside that folder.

Wait for Unity to compile.

### 5) Patch the package's render shader (one file)

The stock shader doesn't tint splats by relevancy. Overwrite the package's `RenderGaussianSplats.shader` with our patched version:

```bash
cp second_stage/viewer/unity_scripts/RenderGaussianSplats.shader \
   second_stage/viewer/unity_project/Packages/com.aras-p.gaussian-splatting/Shaders/RenderGaussianSplats.shader
```

(Or wherever the package landed in `Packages/`.) Switch back to Unity; it reimports the shader.

The patch adds three things only:
- reads `_SplatQuerySimilarities[instID]` per-splat
- reads `_MinRelevancyScore` (set as a shader global by our C# script)
- if `relevancy ≥ _MinRelevancyScore`, lerps the splat color toward `(1, 0, 1)` (magenta) by 75%.

If you ever update the package and need to re-apply, the diff is small — the file is annotated.

### 6) Import the `.ply` as a splat asset

In Unity:

1. `Tools → Gaussian Splats → Create GaussianSplatAsset` (the package adds this menu).
2. Input file: `second_stage/outputs/test_scene_hsegsplat/test_scene_hsegsplat.ply`.
3. Quality: **Very High** (preserves all 1.23M splats; lower quantization).
4. Output folder: somewhere under `Assets/`, e.g. `Assets/GaussianAssets/`.
5. Wait for the bake. Result: a `GaussianSplatAsset` ScriptableObject + a few binary blobs in `Assets/GaussianAssets/`.

### 7) Build the scene

1. New empty GameObject, name it `Splats`.
2. Add a `Gaussian Splat Renderer` component.
3. Drag the baked `GaussianSplatAsset` from `Assets/GaussianAssets/` into the renderer's `Asset` slot.
4. You should see the splats in the Scene view. Frame them with `F`. If not, double-check that the asset is non-empty and that the `Renderer` has it assigned.

### 8) Drop in our two scripts

Copy `HSegSplatClient.cs` and `HSegSplatUI.cs` into `Assets/Scripts/` (create the folder if needed):

```bash
mkdir -p second_stage/viewer/unity_project/Assets/Scripts
cp second_stage/viewer/unity_scripts/HSegSplatClient.cs second_stage/viewer/unity_project/Assets/Scripts/
cp second_stage/viewer/unity_scripts/HSegSplatUI.cs     second_stage/viewer/unity_project/Assets/Scripts/
```

Unity compiles. If you see "missing TMP" errors, the editor will prompt to import TMP Essentials — accept.

### 9) Wire up the components

On the `Splats` GameObject:

1. **Add Component → `HSegSplatClient`.** Inspector fields:
   - `Server Url`: `http://127.0.0.1:8000` (default).
   - `Splat Renderer`: leave empty (the script finds the local `GaussianSplatRenderer` automatically).
   - `Child Level / Parent Level / Grandparent Level`: 6 / 3 / 1 (defaults match the Python CLI).
   - `Threshold`: 0.5 (SegSplat default).

Build a tiny UI in the Hierarchy:

2. **GameObject → UI → Canvas.** A Canvas + EventSystem appear.
3. Right-click the Canvas → **UI → Input Field - TextMeshPro**. Rename it `ChildInput`. Anchor top-left, position it visibly. Repeat for `ParentInput` and `GrandparentInput`. Stack them vertically.
4. Right-click the Canvas → **UI → Button - TextMeshPro**. Rename `SubmitButton`. Set its child Text to "Query". Repeat for `ClearButton` (text "Clear").
5. Right-click the Canvas → **UI → Text - TextMeshPro**. Rename `StatusText`. Place it below the buttons.
6. Add Component to the Canvas → **`HSegSplatUI`**. Drag each of the five UI elements into the matching inspector slot. Drag the `Splats` GameObject into the `Client` slot.

### 10) Camera

The default scene has a `Main Camera` but no orbit / WASD controls. Either:

- Use Unity's "Cinemachine" Free Look (Package Manager → Cinemachine → install → add a Free Look Camera).
- Or grab a tiny WASD/orbit script from the asset store / web. For a first test, position the camera manually in the Scene view and copy its transform onto the Main Camera (right-click camera → Align With View).

The splat package ships a sample scene with a fly camera at `Packages/com.aras-p.gaussian-splatting/Samples/...` — easiest path is to copy the camera GameObject from there.

---

## Day-to-day usage

```bash
# Terminal 1
cd second_stage/viewer/server
source venv/bin/activate
python serve.py ../../outputs/test_scene_hsegsplat/gaussians.pt
```

Open Unity, press **Play**, type queries in the three input fields:

| Field | Examples |
|---|---|
| Child (level 6, finest) | `"leg"`, `"glass pane"`, `"door"`, `"keyboard"` |
| Parent (level 3) | `"chair"`, `"window"`, `"cupboard"` |
| Grandparent (level 1, coarsest) | `"furniture"`, `"wall"`, `""` (leave empty to skip) |

Click **Query** → magenta-tinted splats appear over the matching region. Click **Clear** to remove.

Watch the Unity Console for `[HSegSplat]` log lines (min/max relevancy, number of matches). Watch the server terminal for `[query]` lines.

---

## How it works (under the hood)

**Server.** On startup, loads `gaussians.pt` and pulls out three things: `cluster_index[lvl]` of shape `(G=1.23M,)` (one cluster ID per Gaussian per level), `banks[lvl]` of shape `(M_l+1, 1152)` (one SigLIP cluster centroid per row, row 0 = background), and the three `M_l` values. Then loads SigLIP `ViT-SO400M-14 / webli` and encodes the canonical phrases `["object", "things", "stuff"]` once.

For each query, for each provided level:
1. SigLIP-encode the term (cached after first time).
2. Compute per-cluster cosine to the term and to each canonical (small matmul, `M_l × 1152` × `1152`).
3. Apply LERF: `relevancy_cluster = min_canon sigmoid(τ · (sim_q − sim_canon))` with τ=100. This gives a single relevancy per cluster.
4. Scatter to per-Gaussian using `cluster_index[lvl]`: row 0 (background) → 0.0, rows 1..M → cluster's relevancy.

Combine across provided levels by **per-Gaussian min** (the spatial-AND becomes a cluster-AND: a Gaussian passes only if all of its (level, cluster) memberships pass).

Return the resulting `float32[G]` as a binary stream.

**Unity client.** POSTs the three terms + level integers as JSON to `/query_combined`. Receives `4 × G` bytes, casts to `float[G]`, finds the internal `m_GpuQuerySimilarities` `GraphicsBuffer` on the `GaussianSplatRenderer` via reflection (the field is `internal`), calls `SetData(relevancy)`. Also sets `_MinRelevancyScore` as a shader global via `Shader.SetGlobalFloat`.

**Patched render shader.** Reads `_SplatQuerySimilarities[instID]` and `_MinRelevancyScore`. In the fragment stage, after computing the normal splat color, if `relevancy ≥ threshold`, lerps RGB toward magenta `(1, 0, 1)` at 0.75 strength.

---

## Why we picked these defaults

- **Threshold = 0.5.** Matches SegSplat eq. 4 / `query_hsegsplat.py`. LERF with τ=100 produces nearly-binary relevancy in (0, 1), so 0.5 is a natural cut.
- **LERF temperature τ = 100.** LERF paper / SegSplat default. Sharpens the soft AND so that "query closer than canonicals" maps to ~1 and the opposite maps to ~0.
- **Magenta tint via 0.75 lerp** (not full replace). Keeps a hint of underlying color so you can still see structural detail through the highlight.

---

## Files patched in the package

Just one: `RenderGaussianSplats.shader`. Everything else is read via reflection or via the existing public API. If you re-import the package from upstream, re-copy the shader to keep the highlight.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `[ClipClient] failed` or "Connection refused" in Unity console | Server not running, or wrong port in `HSegSplatClient.Server Url`. |
| Server logs `received` but Unity logs `relevancy length X != splatCount Y` | You imported a different `.ply` than the one matching `gaussians.pt`. Re-bake the asset from the same `.ply`. |
| Splats render correctly but nothing tints magenta | Patched shader not in place. Check that `Packages/.../Shaders/RenderGaussianSplats.shader` has the H-SegSplat comment at the top. |
| Unity says `Could not find internal m_GpuQuerySimilarities via reflection` | The splat package was updated and renamed the field. Open `GaussianSplatRenderer.cs` and grep for `QuerySimilarities` — update the field name in `HSegSplatClient.Start()`. |
| Server log says `(open_clip download failed)` on first run | SigLIP `webli` weights need internet on first run (~1.6 GB). After that they're cached under `~/.cache/huggingface/hub/`. |
| Unity hangs on `m_GpuQuerySimilarities.SetData` for several seconds | 1.23M floats = 4.7 MB upload — should be near-instant on M3. Likely the server is the slow one (SigLIP encode + matmul). Check server log timing. |
