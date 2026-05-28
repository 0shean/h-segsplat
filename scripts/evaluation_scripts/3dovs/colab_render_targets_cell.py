# Colab cell — paste this AFTER `run_pipeline.sh` succeeds for a 3D-OVS scene.
#
# Re-loads gaussians.pt and renders the 5 target views from target_views.json.
# Writes the same artifacts the patched run_hsegsplat_inference.py would have:
#
#   data/<scene>/rendered_rgb_targets.npy
#   data/<scene>/rendered_feature_map_targets_lvl1.npy   (and lvl3, lvl6)
#   data/<scene>/render_target_<view_id>.png
#
# This is needed because the cloud repo on Colab does NOT yet have the
# target-rendering patch; this cell injects the equivalent code inline.

import json
import torch
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from gsplat import rasterization

scene_dir = Path(f"data/{SCENE_NAME}")    # SCENE_NAME e.g. '3dovs_bed'
g = torch.load(scene_dir / "gaussians.pt", map_location="cuda", weights_only=False)
means = g["means"].cuda()
quats_wxyz = g["quats_wxyz"].cuda()
scales = g["scales"].cuda()
opacities = g["opacities"].cuda()
harmonics = g["harmonics"].cuda()
levels = g["levels"]
H, W = g["image_HW"]
K_sh = harmonics.shape[-1]
sh_degree = int(round(np.sqrt(K_sh)) - 1)
sh_coeffs = harmonics.permute(0, 2, 1).contiguous()  # (G, K_sh, 3)

# Per-level one-hots from cluster_index + M.
cluster_index = {int(l): g["cluster_index"][l].cuda() for l in levels}
M = {int(l): int(g["M"][l]) for l in levels}
G_count = means.shape[0]
one_hots = {}
for lvl in levels:
    M_l = M[lvl]
    oh = torch.zeros((G_count, M_l + 1), dtype=torch.float32, device="cuda")
    oh[torch.arange(G_count, device="cuda"), cluster_index[lvl]] = 1.0
    one_hots[lvl] = oh

# Load target_views.json
with open(scene_dir / "target_views.json") as f:
    tv = json.load(f)
t_W, t_H = int(tv["w"]), int(tv["h"])
assert (t_H, t_W) == (H, W), f"target_views {t_W}x{t_H} != gaussians {W}x{H}"
t_K = torch.tensor([[tv["fl_x"], 0.0, tv["cx"]],
                    [0.0, tv["fl_y"], tv["cy"]],
                    [0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda")
blender2opencv = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0],
                                         dtype=torch.float32, device="cuda"))

t_rgb = []
t_feat = {lvl: [] for lvl in levels}
for tgt in tv["targets"]:
    c2w_b = torch.tensor(tgt["transform_matrix"], dtype=torch.float32, device="cuda")
    c2w_o = c2w_b @ blender2opencv
    viewmat = torch.linalg.inv(c2w_o).unsqueeze(0)
    rgb_t, _, _ = rasterization(
        means=means, quats=quats_wxyz, scales=scales, opacities=opacities,
        colors=sh_coeffs, viewmats=viewmat, Ks=t_K.unsqueeze(0),
        width=W, height=H, sh_degree=sh_degree, render_mode="RGB",
    )
    t_rgb.append(rgb_t[0].clamp(0, 1).cpu())
    for lvl in levels:
        feat_t, _, _ = rasterization(
            means=means, quats=quats_wxyz, scales=scales, opacities=opacities,
            colors=one_hots[lvl], viewmats=viewmat, Ks=t_K.unsqueeze(0),
            width=W, height=H, sh_degree=None, render_mode="RGB",
        )
        t_feat[lvl].append(feat_t[0].cpu())

t_rgb_arr = torch.stack(t_rgb, 0).numpy()
np.save(scene_dir / "rendered_rgb_targets.npy", t_rgb_arr.astype(np.float32))
for t_idx, tgt in enumerate(tv["targets"]):
    imageio.imwrite(scene_dir / f"render_target_{tgt['view_id']}.png",
                    (t_rgb_arr[t_idx] * 255).clip(0, 255).astype(np.uint8))
for lvl in levels:
    arr = torch.stack(t_feat[lvl], 0).numpy()
    np.save(scene_dir / f"rendered_feature_map_targets_lvl{lvl}.npy",
            arr.astype(np.float32))
    print(f"saved rendered_feature_map_targets_lvl{lvl}.npy: {arr.shape}")
print(f"rendered {len(tv['targets'])} target views for {SCENE_NAME}")

# Download the new artifacts
from google.colab import files
files.download(str(scene_dir / "rendered_feature_map_targets_lvl1.npy"))
# files.download(str(scene_dir / "rendered_feature_map_targets_lvl3.npy"))
# files.download(str(scene_dir / "rendered_feature_map_targets_lvl6.npy"))
# files.download(str(scene_dir / "rendered_rgb_targets.npy"))
