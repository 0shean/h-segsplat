# H-SegSplat

A feed-forward 3D Gaussian Splatting pipeline with **hierarchical open-set semantic queries**.
Built on top of [DepthSplat](https://github.com/cvg/depthsplat) (frozen geometric backbone) and
[SegSplat](https://arxiv.org/abs/...) (semantic side pipeline), extended with three parallel
granularity levels (1, 3, 6) from semantic-SAM and Scheme B containment-based parent linking
for hierarchical queries like *"leg of chair"*.

See `docs/PROJECT_PLAN.md` for the complete design rationale.

---

## What you get

From 2 input images + camera poses → a 3D Gaussian scene with per-Gaussian semantic identity
at three levels of granularity, queryable in real-time via an interactive Unity desktop viewer.

A user types `child = "leg"`, `parent = "chair"` in the Inspector → matching Gaussians tint
magenta in 3D.

---

## Quickstart on Colab

Open `colab/run_in_colab.ipynb` in Colab. The notebook walks through:

1. Clone repo + download checkpoints.
2. Build three Python venvs (SAM, SigLIP, H-SegSplat).
3. Upload your scene under `data/<scene>/`.
4. Run `pipeline/run_pipeline.sh data/<scene>`.
5. Download `gaussians.pt` and feed it to the Unity viewer.

---

## Quickstart on a Linux box / ETH Euler

```bash
git clone https://github.com/0shean/h-segsplat.git
cd h-segsplat

# Build the three venvs (each ~5–15 min the first time).
bash envs/sam/setup.sh
bash envs/siglip/setup.sh
bash envs/hsegsplat/setup.sh

# Place checkpoints.
mkdir -p depthsplat/pretrained
wget -O depthsplat/pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth \
    https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth
wget -O swinl_only_sam_many2many.pth \
    https://huggingface.co/UX-Decoder/Semantic-SAM/resolve/main/swinl_only_sam_many2many.pth

# Place your scene under data/<scene>/dslr/{nerfstudio/transforms.json, resized_images/}/
# Then run the full pipeline:
bash pipeline/run_pipeline.sh data/<scene>
```

On ETH Euler, the `envs/*/setup.sh` scripts auto-detect `module load` and use the
configured Python 3.10 / CUDA 12.1 toolchain. Wrap the pipeline in a SLURM job like:

```bash
sbatch --gpus=rtx_3090:1 --time=12:00:00 --mem-per-cpu=8G --cpus-per-task=4 \
       --wrap='bash pipeline/run_pipeline.sh data/<scene>'
```

---

## Pipeline stages

| Stage | Script | Venv | What it produces |
|---|---|---|---|
| 1 | `scripts/run_semantic_sam.py` | `envs/sam` | `data/<scene>/masks_lvl_{1,3,6}/<frame>/mask_*.png` + `metadata.json` |
| 2 | `scripts/run_siglip.py` | `envs/siglip` | `data/<scene>/masks_lvl_{1,3,6}/<frame>/siglip_embeddings.npy` |
| 3 | `scripts/build_hsegsplat_inputs.py` | `envs/hsegsplat` | `data/<scene>/level_{1,3,6}/{bank,index_maps,mask_features,...}.npy`, `datasets/`, `assets/` |
| 4 | `scripts/compute_parent_chain.py` | `envs/hsegsplat` | `data/<scene>/per_mask/parents.json` |
| 5 | `scripts/run_hsegsplat_inference.py` | `envs/hsegsplat` | `data/<scene>/gaussians.pt` + `rendered_*.npy` |

Each stage is **idempotent** — re-running skips work that's already done.

### Optional: SAM ViT-H at level 1 (matches SegSplat's protocol)

Semantic-SAM at granularity 1 sometimes produces overlapping "object + nearby
context" masks (e.g. a Pikachu plush plus a slab of surrounding sofa). When
those masks are SigLIP-encoded the salient object dominates the feature, and
the surrounding sofa pixels get the Pikachu cluster index → spillover at
render time. SegSplat (§3.1 step 1) avoids this by using Meta's original SAM
ViT-H with NMS at "whole object" prompting.

To enable that protocol set two env vars before invoking the pipeline:

```bash
export USE_SAM_VITH_LVL1=1
export SAM_VITH_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth  # download from
   # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

bash pipeline/run_pipeline.sh data/<scene>
```

When set, the orchestrator runs `stage_01a_sam_vith.sh` (SAM ViT-H → lvl-1
masks) before `stage_01_masks.sh`, and Semantic-SAM runs only at levels 3
and 6. The downstream stages and the gaussians.pt schema are unchanged.

The `segment_anything` Python package is installed alongside Semantic-SAM
in `envs/sam/venv` — no extra venv needed.

---

## Project layout

```
h-segsplat/
├── pipeline/                 # orchestrator + per-stage wrappers (bash)
├── envs/                     # three venv setup scripts
│   ├── sam/        setup.sh  # Python 3.10, torch 2.1.2 + cu121, detectron2, custom CUDA op
│   ├── siglip/     setup.sh  # Python 3.10, torch 2.1.2, open_clip_torch (SigLIP)
│   └── hsegsplat/  setup.sh  # Python 3.10, torch 2.4.0 + cu124, DepthSplat reqs, gsplat
├── scripts/                  # all stage Python scripts in one place
├── depthsplat/               # vendored upstream (cvg/depthsplat @ 2dad25a)
├── Semantic-SAM/             # vendored upstream (UX-Decoder/Semantic-SAM)
├── viewer/                   # Unity desktop viewer + FastAPI query server
│   ├── server/               # serve.py — loads gaussians.pt, serves /query_combined_v2
│   ├── gaussians_to_ply.py
│   ├── unity_scripts/        # canonical source; the unity_project/ is gitignored
│   └── README.md
├── data/                     # empty; user uploads scenes here
├── docs/                     # PROJECT_PLAN.md, PDFs, semantic class list
├── colab/                    # run_in_colab.ipynb
└── README.md
```

---

## Why three venvs

The three stacks have incompatible dependency requirements that nothing can reconcile:

- **SAM**: torch 2.1.2 + cu12.1 + detectron2 + a custom CUDA op compiled against this exact
  torch/CUDA combination. NumPy < 2.
- **SigLIP**: torch 2.1.2, open_clip_torch (won't load with the torch 2.4 wheels used by DepthSplat).
- **H-SegSplat**: torch 2.4.0 + cu12.4 (DepthSplat's pinned requirement) + gsplat.

Co-locating them in one venv breaks at least the SAM CUDA op. So we keep three.
The pipeline orchestrator activates the right venv for each stage; you don't need to think about it.

---

## Querying with the viewer

Once `gaussians.pt` is on your local machine:

```bash
cd viewer/server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python serve.py /path/to/gaussians.pt
# server listens on http://127.0.0.1:8000
```

Then open the Unity project (see `viewer/README.md` for one-time setup), select the
`Splats` GameObject, enable the **Use V2** checkbox in the Inspector, type your query, and
click **[ Submit Query ]**. Matching Gaussians tint magenta.

Three query shapes are supported (see `docs/PROJECT_PLAN.md` §6.5.5):

- `child` only — match at any level (1, 3, or 6).
- `child + parent` — three sub-bindings tried in parallel: 6→3, 6→1, 3→1.
- `child + parent + grandparent` — locked binding 6→3→1.

---

## Vendored components

See `depthsplat/UPSTREAM.md` and `Semantic-SAM/UPSTREAM.md` for source URLs, pinned SHAs, and
re-vendoring instructions. Neither directory has been modified — both are clean snapshots.
