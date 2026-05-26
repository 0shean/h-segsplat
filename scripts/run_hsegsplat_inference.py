#!/usr/bin/env python3
"""
H-SegSplat inference on Colab — three parallel cluster SegSplats (levels 1, 3, 6).

Designed to be invoked exactly like Phase-3 colab_segsplat_inference.py, with
++segsplat.assets_dir pointed at the H-SegSplat scene root (which contains
level_1/, level_3/, level_6/ subfolders produced by build_hsegsplat_inputs.py):

    !python colab_hsegsplat_inference.py \
        +experiment=dl3dv \
        dataset.test_chunk_interval=1 \
        dataset.roots=[datasets/<scene_key>_2view] \
        dataset.image_shape=[640,960] \
        dataset.ori_image_shape=[640,960] \
        dataset.max_fov=130.0 \
        model.encoder.num_scales=2 \
        model.encoder.upsample_factor=4 \
        model.encoder.lowest_feature_resolution=8 \
        model.encoder.monodepth_vit_type=vitb \
        model.encoder.gaussian_adapter.gaussian_scale_max=0.2 \
        checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth \
        mode=test \
        dataset/view_sampler=evaluation \
        dataset.view_sampler.index_path=assets/<scene_key>_2view_eval.json \
        data_loader.test.num_workers=0 \
        ++segsplat.assets_dir=segsplat/<scene_key> \
        ++segsplat.output_dir=outputs/<scene_key> \
        ++segsplat.levels=[1,3,6]

Pipeline:
  1. Load DepthSplat encoder + checkpoint (same as Phase 3).
  2. Pull one batch from the test dataloader -> Gaussians (V*H*W per scene).
  3. For each level k:
       - Load level_k/{bank.npy, index_maps.npy, meta.json}.
       - Build per-Gaussian one-hot e^(k) via flat-index <-> (v, y, x) mapping.
       - For each input view, render an (M_k + 1)-channel feature map E^(k).
  4. Render RGB once per input view (SH).
  5. Save:
       rendered_rgb.npy              (V, H, W, 3)
       rendered_feature_map_lvl1.npy (V, H, W, M_1 + 1)
       rendered_feature_map_lvl3.npy (V, H, W, M_3 + 1)
       rendered_feature_map_lvl6.npy (V, H, W, M_6 + 1)
       gaussians.pt   geometry + per-level cluster_index + per-level bank + cameras
"""

import json
import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

with __import__("jaxtyping").install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.misc.step_tracker import StepTracker
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder
    from src.model.model_wrapper import ModelWrapper
    from src.loss import get_losses


# ----------------------------------------------------------------------------
# Math helpers (identical to colab_segsplat_inference.py)
# ----------------------------------------------------------------------------

def covariance_to_quat_scale(cov: torch.Tensor):
    """Decompose 3x3 covariance Sigma = R diag(s^2) R^T -> (quat (xyzw), scale)."""
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=1e-12)
    scales = eigvals.sqrt()

    det = torch.linalg.det(eigvecs)
    flip = (det < 0).unsqueeze(-1).unsqueeze(-1)
    eigvecs = torch.where(flip, eigvecs * torch.tensor([-1.0, 1.0, 1.0], device=cov.device), eigvecs)

    R = eigvecs
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22

    eps = 1e-8
    cond1 = trace > 0
    s1 = torch.sqrt(trace.clamp(min=eps) + 1.0) * 2.0
    qw1 = 0.25 * s1
    qx1 = (m21 - m12) / s1
    qy1 = (m02 - m20) / s1
    qz1 = (m10 - m01) / s1

    cond2 = (m00 > m11) & (m00 > m22)
    s2 = torch.sqrt((1.0 + m00 - m11 - m22).clamp(min=eps)) * 2.0
    qw2 = (m21 - m12) / s2
    qx2 = 0.25 * s2
    qy2 = (m01 + m10) / s2
    qz2 = (m02 + m20) / s2

    cond3 = m11 > m22
    s3 = torch.sqrt((1.0 + m11 - m00 - m22).clamp(min=eps)) * 2.0
    qw3 = (m02 - m20) / s3
    qx3 = (m01 + m10) / s3
    qy3 = 0.25 * s3
    qz3 = (m12 + m21) / s3

    s4 = torch.sqrt((1.0 + m22 - m00 - m11).clamp(min=eps)) * 2.0
    qw4 = (m10 - m01) / s4
    qx4 = (m02 + m20) / s4
    qy4 = (m12 + m21) / s4
    qz4 = 0.25 * s4

    qw = torch.where(cond1, qw1, torch.where(cond2, qw2, torch.where(cond3, qw3, qw4)))
    qx = torch.where(cond1, qx1, torch.where(cond2, qx2, torch.where(cond3, qx3, qx4)))
    qy = torch.where(cond1, qy1, torch.where(cond2, qy2, torch.where(cond3, qy3, qy4)))
    qz = torch.where(cond1, qz1, torch.where(cond2, qz2, torch.where(cond3, qz3, qz4)))

    quat_xyzw = torch.stack([qx, qy, qz, qw], dim=-1)
    quat_xyzw = quat_xyzw / quat_xyzw.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return quat_xyzw, scales


def c2w_to_w2c_for_gsplat(c2w: torch.Tensor) -> torch.Tensor:
    """gsplat wants viewmats = world-to-camera 4x4."""
    return torch.linalg.inv(c2w)


# ----------------------------------------------------------------------------
# Per-level asset loading
# ----------------------------------------------------------------------------

def load_level_assets(level_dir: Path, device: torch.device):
    """Returns (bank_t, index_maps_t, mask_id_maps_t, mask_features_t, mask_directory, meta)
    for one granularity level. The v2 outputs (mask_id_maps, mask_features, mask_directory)
    are optional and return as None if absent — keeps the script back-compat with v1-only assets."""
    bank = np.load(level_dir / "bank.npy")              # (M+1, D) float32
    index_maps = np.load(level_dir / "index_maps.npy")  # (V, H, W) int16
    with open(level_dir / "meta.json") as f:
        meta = json.load(f)
    bank_t = torch.from_numpy(bank).to(device)
    index_maps_t = torch.from_numpy(index_maps.astype(np.int64)).to(device)

    # v2 inputs (optional for back-compat).
    mask_id_maps_t = None
    mid_path = level_dir / "mask_id_maps.npy"
    if mid_path.exists():
        mid = np.load(mid_path)
        mask_id_maps_t = torch.from_numpy(mid.astype(np.int64)).to(device)

    mask_features_t = None
    mf_path = level_dir / "mask_features.npy"
    if mf_path.exists():
        mf = np.load(mf_path)
        mask_features_t = torch.from_numpy(mf.astype(np.float32))  # CPU; small

    mask_directory = None
    md_path = level_dir / "mask_directory.json"
    if md_path.exists():
        with open(md_path) as f:
            mask_directory = json.load(f)

    return bank_t, index_maps_t, mask_id_maps_t, mask_features_t, mask_directory, meta


# ----------------------------------------------------------------------------
# Hydra entry
# ----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="config", config_name="main")
def run(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    seg_assets_dir = Path(cfg_dict.segsplat.assets_dir)
    seg_output_dir = Path(cfg_dict.segsplat.output_dir)
    seg_output_dir.mkdir(parents=True, exist_ok=True)

    # Levels can come from segsplat config; default to [1, 3, 6].
    levels = list(cfg_dict.segsplat.get("levels", [1, 3, 6]))
    print(f"[hsegsplat] levels={levels}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build encoder + load checkpoint ---
    step_tracker = StepTracker()
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    model_wrapper = ModelWrapper(
        cfg.optimizer, cfg.test, cfg.train,
        encoder, encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
        eval_data_cfg=None,
    )
    if cfg.checkpointing.pretrained_model is not None:
        ckpt = torch.load(cfg.checkpointing.pretrained_model, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        strict = not cfg.checkpointing.no_strict_load
        model_wrapper.load_state_dict(ckpt, strict=strict)
        print(f"[hsegsplat] Loaded checkpoint: {cfg.checkpointing.pretrained_model}")

    model_wrapper.eval().to(device)

    # --- Build data module + iterate one batch ---
    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker, global_rank=0)
    test_loader = data_module.test_dataloader()

    # --- Load semantic assets for all requested levels ---
    level_data = {}  # level (int) -> {"bank", "index_maps", "mask_id_maps", "mask_features",
                     #                  "mask_directory", "meta"}
    H_seg = W_seg = V_seg = None
    for lvl in levels:
        level_dir = seg_assets_dir / f"level_{lvl}"
        if not level_dir.exists():
            raise FileNotFoundError(f"missing level dir: {level_dir}")
        (bank_t, index_maps_t, mask_id_maps_t,
         mask_features_t, mask_directory, meta) = load_level_assets(level_dir, device)
        if H_seg is None:
            H_seg, W_seg, V_seg = meta["H"], meta["W"], meta["V"]
        elif (meta["H"], meta["W"], meta["V"]) != (H_seg, W_seg, V_seg):
            raise RuntimeError(
                f"level {lvl} meta disagrees: ({meta['H']},{meta['W']},{meta['V']}) vs "
                f"({H_seg},{W_seg},{V_seg})"
            )
        print(f"[hsegsplat] level {lvl}: M={meta['M']}  bank={tuple(bank_t.shape)}  "
              f"index_maps={tuple(index_maps_t.shape)}  "
              f"mask_id_maps={'yes' if mask_id_maps_t is not None else 'no'}  "
              f"mask_features={'yes' if mask_features_t is not None else 'no'}")
        level_data[lvl] = {
            "bank": bank_t,
            "index_maps": index_maps_t,
            "mask_id_maps": mask_id_maps_t,
            "mask_features": mask_features_t,
            "mask_directory": mask_directory,
            "meta": meta,
        }

    # --- v2: load cross-level parent dict if present ---
    parents_v2 = None
    parents_path = seg_assets_dir / "per_mask" / "parents.json"
    if parents_path.exists():
        with open(parents_path) as f:
            parents_v2 = json.load(f)
        print(f"[hsegsplat] loaded v2 parents.json: "
              f"{len(parents_v2.get('parents', {}).get('level_6', {}))} lvl6 entries, "
              f"{len(parents_v2.get('parents', {}).get('level_3', {}))} lvl3 entries")
    else:
        print(f"[hsegsplat] no v2 parents.json — v2 query path will be unavailable in viewer")

    for batch_idx, batch in enumerate(test_loader):
        # Move batch to device.
        batch_dev = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                batch_dev[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                                for kk, vv in v.items()}
            elif isinstance(v, torch.Tensor):
                batch_dev[k] = v.to(device)
            else:
                batch_dev[k] = v

        batch_dev = model_wrapper.data_shim(batch_dev)

        with torch.no_grad():
            enc_out = model_wrapper.encoder(
                batch_dev["context"],
                model_wrapper.global_step,
                False,
                scene_names=batch_dev.get("scene"),
            )

        gaussians = enc_out["gaussians"] if isinstance(enc_out, dict) else enc_out

        b, V, _, H, W = batch_dev["context"]["image"].shape
        if (H, W) != (H_seg, W_seg):
            raise RuntimeError(
                f"DepthSplat input is {H}x{W} but hsegsplat assets are at {H_seg}x{W_seg}."
            )
        if V != V_seg:
            raise RuntimeError(f"DepthSplat V={V} but hsegsplat assets have V={V_seg}.")

        means = gaussians.means[0]              # (G, 3)
        covs = gaussians.covariances[0]         # (G, 3, 3)
        harmonics = gaussians.harmonics[0]      # (G, 3, K_sh)
        opacities = gaussians.opacities[0]      # (G,)
        G = means.shape[0]
        if G != V * H * W:
            raise RuntimeError(f"expected V*H*W={V*H*W} Gaussians, got {G}.")

        quats_xyzw, scales = covariance_to_quat_scale(covs)
        quats_wxyz = torch.stack([quats_xyzw[..., 3], quats_xyzw[..., 0],
                                  quats_xyzw[..., 1], quats_xyzw[..., 2]], dim=-1)

        # Per-Gaussian one-hot per level via flat-index <-> (v, y, x) mapping.
        # Order: i = v*(H*W) + y*W + x  (matches DepthSplat's flatten).
        cluster_indices = {}   # level -> (G,) int64
        one_hots = {}          # level -> (G, M+1) float32
        for lvl in levels:
            idx_flat = level_data[lvl]["index_maps"].reshape(V * H * W)  # (G,)
            cluster_indices[lvl] = idx_flat
            M_l = int(level_data[lvl]["meta"]["M"])
            one_hot = torch.zeros((G, M_l + 1), dtype=torch.float32, device=device)
            one_hot[torch.arange(G, device=device), idx_flat] = 1.0
            one_hots[lvl] = one_hot

        # --- v2: per-Gaussian finest-mask reference ---
        # For each Gaussian, find the finest level whose mask_id_map has a non-bg entry
        # at the Gaussian's source pixel. Convention: try 6, 3, 1 in order.
        # Stored as:
        #   finest_level[G]            int8   -- 0 = bg, else the level
        #   finest_local_mask_id[G]    int32  -- local_mask_id within that view+level
        #   finest_global_mask_id[G]   int32  -- row index into level_<L>/mask_features.npy
        finest_level = None
        finest_local = None
        finest_global = None
        have_v2 = all(level_data[lvl]["mask_id_maps"] is not None for lvl in levels) \
                  and all(level_data[lvl]["mask_directory"] is not None for lvl in levels)
        if have_v2:
            finest_level = torch.zeros(G, dtype=torch.int8, device=device)
            finest_local = torch.full((G,), -1, dtype=torch.int32, device=device)
            finest_global = torch.full((G,), -1, dtype=torch.int32, device=device)

            # Pre-build per-level (view_idx, local_mask_id) -> global_id LUTs as tensors,
            # so we can gather per Gaussian once finest_level/finest_local are populated.
            # gid_table[lvl] shape: (V, max_local_id_in_level + 1), int32, -1 = none.
            gid_tables = {}
            for lvl in levels:
                dir_entries = level_data[lvl]["mask_directory"]
                max_local = 0
                for e in dir_entries:
                    if e["local_mask_id"] > max_local:
                        max_local = int(e["local_mask_id"])
                gt = torch.full((V_seg, max_local + 1), -1, dtype=torch.int32, device=device)
                for e in dir_entries:
                    gt[int(e["view_idx"]), int(e["local_mask_id"])] = int(e["global_id"])
                gid_tables[lvl] = gt

            # Walk from finest level to coarsest. Once a Gaussian is claimed, leave it alone.
            # `claimed` is the set of Gaussians whose finest_level has already been set.
            claimed = torch.zeros(G, dtype=torch.bool, device=device)
            view_idx_per_g = torch.arange(G, device=device) // (H * W)  # (G,)

            for lvl in sorted(levels, reverse=True):  # 6, 3, 1
                mid_flat = level_data[lvl]["mask_id_maps"].reshape(V * H * W)  # (G,) int64
                # Convention: 0 = bg, 1..N_v = local_mask_id + 1.
                has_mask = (mid_flat > 0) & ~claimed
                if not bool(has_mask.any()):
                    continue
                local_ids = (mid_flat - 1).to(torch.int32)  # invalid where mid_flat == 0

                # Gather global ids for Gaussians in has_mask.
                gt = gid_tables[lvl]  # (V, max_local+1)
                # Clamp so out-of-range stays in bounds; we only use values where has_mask.
                local_ids_safe = local_ids.clamp(min=0)
                gids = gt[view_idx_per_g.to(torch.long), local_ids_safe.to(torch.long)]

                finest_level[has_mask] = int(lvl)
                finest_local[has_mask] = local_ids[has_mask]
                finest_global[has_mask] = gids[has_mask]
                claimed = claimed | has_mask

            n_bg = int((finest_level == 0).sum())
            for lvl in sorted(levels, reverse=True):
                n_l = int((finest_level == lvl).sum())
                print(f"[hsegsplat] finest_level == {lvl}: {n_l} Gaussians "
                      f"({100*n_l/G:.1f}%)")
            print(f"[hsegsplat] finest_level == bg: {n_bg} Gaussians ({100*n_bg/G:.1f}%)")

        # --- Render at each input view with gsplat ---
        from gsplat import rasterization

        c2w = batch_dev["context"]["extrinsics"][0]  # (V, 4, 4)
        intrinsics_norm = batch_dev["context"]["intrinsics"][0]  # (V, 3, 3) normalized [0,1]
        Ks = intrinsics_norm.clone()
        Ks[:, 0, 0] *= W
        Ks[:, 0, 2] *= W
        Ks[:, 1, 1] *= H
        Ks[:, 1, 2] *= H

        viewmats = c2w_to_w2c_for_gsplat(c2w)  # (V, 4, 4)

        K_sh = harmonics.shape[-1]
        sh_degree = int(round(np.sqrt(K_sh)) - 1)
        sh_coeffs = harmonics.permute(0, 2, 1).contiguous()  # (G, K_sh, 3)

        per_view_rgb = []
        per_view_feat = {lvl: [] for lvl in levels}

        for v in range(V):
            # RGB (SH).
            rgb, _alpha, _info = rasterization(
                means=means,
                quats=quats_wxyz,
                scales=scales,
                opacities=opacities,
                colors=sh_coeffs,
                viewmats=viewmats[v:v + 1],
                Ks=Ks[v:v + 1],
                width=W,
                height=H,
                sh_degree=sh_degree,
                render_mode="RGB",
            )
            per_view_rgb.append(rgb[0].clamp(0, 1).cpu())

            # Per-level (M+1)-channel feature, SH degree 0 (view-invariant).
            for lvl in levels:
                feat, _a, _i = rasterization(
                    means=means,
                    quats=quats_wxyz,
                    scales=scales,
                    opacities=opacities,
                    colors=one_hots[lvl],
                    viewmats=viewmats[v:v + 1],
                    Ks=Ks[v:v + 1],
                    width=W,
                    height=H,
                    sh_degree=None,
                    render_mode="RGB",
                )
                per_view_feat[lvl].append(feat[0].cpu())

        # --- Save outputs ---
        # Use scene_key from any level's meta (they all agree).
        scene_key = level_data[levels[0]]["meta"]["scene_key"]
        out_dir = seg_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        rgb_arr = torch.stack(per_view_rgb, 0).numpy()  # (V, H, W, 3)
        np.save(out_dir / "rendered_rgb.npy", rgb_arr.astype(np.float32))

        import imageio.v2 as imageio
        for v in range(V):
            imageio.imwrite(out_dir / f"render_view{v}.png",
                            (rgb_arr[v] * 255).clip(0, 255).astype(np.uint8))

        for lvl in levels:
            feat_arr = torch.stack(per_view_feat[lvl], 0).numpy()  # (V, H, W, M_l+1)
            np.save(out_dir / f"rendered_feature_map_lvl{lvl}.npy", feat_arr.astype(np.float32))
            print(f"[hsegsplat] rendered_feature_map_lvl{lvl}.npy: {feat_arr.shape}")

        # gaussians.pt: one geometry copy + per-level cluster_index/bank + v2 payload.
        payload = {
            "means": means.cpu(),
            "quats_wxyz": quats_wxyz.cpu(),
            "scales": scales.cpu(),
            "opacities": opacities.cpu(),
            "harmonics": harmonics.cpu(),
            "cluster_index": {lvl: cluster_indices[lvl].cpu() for lvl in levels},
            "banks": {lvl: level_data[lvl]["bank"].cpu() for lvl in levels},
            "M": {lvl: int(level_data[lvl]["meta"]["M"]) for lvl in levels},
            "extrinsics": c2w.cpu(),
            "intrinsics_pixel_K": Ks.cpu(),
            "image_HW": (H, W),
            "levels": levels,
        }
        # v2 additions: per-Gaussian finest-mask + per-level mask features + parents dict.
        if have_v2:
            payload["v2"] = {
                "finest_level": finest_level.cpu(),                # (G,) int8
                "finest_local_mask_id": finest_local.cpu(),        # (G,) int32, -1 = bg
                "finest_global_mask_id": finest_global.cpu(),      # (G,) int32, -1 = bg
                "mask_features": {                                  # per-level (N_L, D) f32
                    lvl: level_data[lvl]["mask_features"]
                    for lvl in levels
                },
                "mask_directory": {                                 # per-level list of dicts
                    lvl: level_data[lvl]["mask_directory"]
                    for lvl in levels
                },
                "parents": parents_v2,                              # the parents.json dict, or None
                "schema_version": "v2",
            }
        torch.save(payload, out_dir / "gaussians.pt")

        print(f"[hsegsplat] Wrote outputs to {out_dir}/  (G={G})")

        # Single-batch run — one scene per invocation.
        break


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
    run()
