# Semantic-SAM — vendored upstream snapshot

This directory is an unmodified copy of [UX-Decoder/Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM).

| Field | Value |
|---|---|
| Source | https://github.com/UX-Decoder/Semantic-SAM |
| Vendored on | 2026-05-26 |
| Modifications | **None.** The repo is installed editable (`pip install -e .`) during `envs/sam/setup.sh`; that step builds the CUDA op `MultiScaleDeformableAttention` in `semantic_sam/body/encoder/ops/`. |
| Excluded | `examples/` (30 MB of demo images, not used by the pipeline). |

## Why vendored

SemanticSAM is run as the stage-1 mask generator. Vendoring keeps everything in a single
`git clone` (no submodule init) and pins the version we tested against.

## Re-vendoring (if upstream updates and we want to pick them up)

```bash
rm -rf Semantic-SAM
git clone https://github.com/UX-Decoder/Semantic-SAM.git
rm -rf Semantic-SAM/.git Semantic-SAM/examples
# Update the date above.
```

## Build artifacts (ignored by .gitignore at repo root)

Running `envs/sam/setup.sh` generates the following inside this directory:

- `Semantic_SAM.egg-info/` — editable-install metadata
- `**/__pycache__/` — bytecode caches
- `semantic_sam/body/encoder/ops/build/` — CUDA op build tree
- `semantic_sam/body/encoder/ops/*.so` — compiled CUDA extension

These are .gitignored. Never commit them.

## Do not edit files in this directory

The pipeline calls SemanticSAM via the upstream `SemanticSamAutomaticMaskGenerator` API
(see `scripts/run_semantic_sam.py`). Keep the vendored tree diffable against upstream.
