#!/usr/bin/env python3
"""Render H-SegSplat from a single midpoint pose using gsplat.

Reads gaussians.pt + midpoint_pose.json for one scene, calls
gsplat.rasterization() once for RGB and once per level for the (M+1)-channel
cluster distribution. Saves:

    <out_dir>/midpoint_rgb.npy                 (H, W, 3) float32 in [0,1]
    <out_dir>/midpoint_rgb.png                 visual sanity
    <out_dir>/midpoint_feature_map_lvl{1,3,6}.npy
                                               (H, W, M_l+1) float32

Designed to run on the cluster (CUDA required) — the hsegsplat venv has
gsplat. CPU paths exist in gsplat but compile is slow; only --device cuda
is supported by this script for simplicity.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def c2w_to_w2c_for_gsplat(c2w: torch.Tensor) -> torch.Tensor:
    """gsplat wants viewmats = world-to-camera 4x4."""
    return torch.linalg.inv(c2w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaussians_pt", type=Path, required=True)
    ap.add_argument("--midpoint_pose_json", type=Path, default=None,
                    help="Optional. Output of sanity_render_midpoint_gt.py with "
                         "'c2w_hsegsplat_frame' and 'intrinsics_K_hsegsplat' "
                         "fields. If omitted, the script computes the midpoint "
                         "pose directly from gaussians.pt's extrinsics.")
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 3, 6])
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Default: dirname(midpoint_pose_json).")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = args.midpoint_pose_json.parent if args.midpoint_pose_json else Path(".")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load pose + intrinsics. If midpoint_pose.json is missing or doesn't have
    # the H-SegSplat frame, fall back to computing the midpoint pose directly
    # from gaussians.pt's extrinsics (works for any 2-view input, no MultiScan
    # mesh-side data required).
    pose_payload = {}
    if args.midpoint_pose_json and args.midpoint_pose_json.exists():
        pose_payload = json.load(open(args.midpoint_pose_json))

    g_preview = torch.load(args.gaussians_pt, map_location="cpu", weights_only=False)
    H, W = pose_payload.get("image_HW", g_preview["image_HW"])

    if "c2w_hsegsplat_frame" in pose_payload:
        c2w_mid_np = np.array(pose_payload["c2w_hsegsplat_frame"], dtype=np.float32)
        K_mid_np = np.array(pose_payload["intrinsics_K_hsegsplat"], dtype=np.float32)
        print(f"[render] using midpoint pose from {args.midpoint_pose_json}")
    else:
        ext = g_preview["extrinsics"].numpy()        # (V, 4, 4) c2w OpenCV
        K_all = g_preview["intrinsics_pixel_K"].numpy()
        if ext.shape[0] < 2:
            raise RuntimeError(f"need at least 2 input views, got {ext.shape[0]}")
        # Midpoint center + SLERP rotation between the first two input views.
        # Pure-numpy implementation; no scipy dependency on the cluster.
        def _q_from_R(M):
            tr = M[0, 0] + M[1, 1] + M[2, 2]
            if tr > 0:
                S = 2 * np.sqrt(tr + 1)
                return np.array([0.25 * S, (M[2, 1] - M[1, 2]) / S,
                                 (M[0, 2] - M[2, 0]) / S, (M[1, 0] - M[0, 1]) / S])
            if M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
                S = 2 * np.sqrt(1 + M[0, 0] - M[1, 1] - M[2, 2])
                return np.array([(M[2, 1] - M[1, 2]) / S, 0.25 * S,
                                 (M[0, 1] + M[1, 0]) / S, (M[0, 2] + M[2, 0]) / S])
            if M[1, 1] > M[2, 2]:
                S = 2 * np.sqrt(1 + M[1, 1] - M[0, 0] - M[2, 2])
                return np.array([(M[0, 2] - M[2, 0]) / S, (M[0, 1] + M[1, 0]) / S,
                                 0.25 * S, (M[1, 2] + M[2, 1]) / S])
            S = 2 * np.sqrt(1 + M[2, 2] - M[0, 0] - M[1, 1])
            return np.array([(M[1, 0] - M[0, 1]) / S, (M[0, 2] + M[2, 0]) / S,
                             (M[1, 2] + M[2, 1]) / S, 0.25 * S])
        def _R_from_q(q):
            q = q / np.linalg.norm(q)
            w, x, y, z = q
            return np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
            ])
        def _slerp(q1, q2, t):
            q1 = q1 / np.linalg.norm(q1); q2 = q2 / np.linalg.norm(q2)
            d = float(np.dot(q1, q2))
            if d < 0:
                q2 = -q2; d = -d
            if d > 0.9995:
                out = q1 + t * (q2 - q1)
                return out / np.linalg.norm(out)
            th = np.arccos(d); s = np.sin(th)
            return (np.sin((1 - t) * th) / s) * q1 + (np.sin(t * th) / s) * q2
        C_mid = (ext[0, :3, 3] + ext[1, :3, 3]) * 0.5
        q1 = _q_from_R(ext[0, :3, :3]); q2 = _q_from_R(ext[1, :3, :3])
        q_mid = _slerp(q1, q2, 0.5)
        R_mid = _R_from_q(q_mid)
        c2w_mid_np = np.eye(4, dtype=np.float32)
        c2w_mid_np[:3, :3] = R_mid
        c2w_mid_np[:3, 3] = C_mid
        K_mid_np = K_all[0].astype(np.float32)
        print(f"[render] no usable midpoint_pose.json — derived midpoint pose "
              f"from gaussians.pt extrinsics directly")

    c2w_mid = torch.tensor(c2w_mid_np, dtype=torch.float32, device=args.device)[None]
    K_mid = torch.tensor(K_mid_np, dtype=torch.float32, device=args.device)[None]
    print(f"[render] H={H} W={W} device={args.device}")
    print(f"[render] midpoint center = {c2w_mid[0, :3, 3].cpu().numpy()}")
    del g_preview

    # Load gaussians
    print(f"[render] loading {args.gaussians_pt}")
    g = torch.load(args.gaussians_pt, map_location=args.device, weights_only=False)
    means = g["means"].to(args.device)                # (G, 3)
    quats_wxyz = g["quats_wxyz"].to(args.device)      # (G, 4)
    scales = g["scales"].to(args.device)              # (G, 3)
    opacities = g["opacities"].to(args.device)        # (G,)
    harmonics = g["harmonics"].to(args.device)        # (G, 3, K_sh)
    G_count = means.shape[0]
    K_sh = harmonics.shape[-1]
    sh_degree = int(round(np.sqrt(K_sh)) - 1)
    sh_coeffs = harmonics.permute(0, 2, 1).contiguous()  # (G, K_sh, 3)
    print(f"[render] G={G_count} K_sh={K_sh} sh_degree={sh_degree}")

    # Per-level one-hot cluster vectors (G, M_l+1). Channel 0 = background slot.
    cluster_indices = g["cluster_index"]              # dict: lvl -> (G,) int64
    one_hots = {}
    for lvl in args.levels:
        if lvl not in cluster_indices:
            raise KeyError(f"cluster_index missing lvl {lvl}")
        idx = cluster_indices[lvl].to(args.device)
        M_l = int(g["M"][lvl])
        # Background slot is index 0; cluster indices are 1..M
        oh = torch.zeros((G_count, M_l + 1), dtype=torch.float32, device=args.device)
        oh[torch.arange(G_count, device=args.device), idx] = 1.0
        one_hots[lvl] = oh
        print(f"[render] lvl{lvl}: M={M_l}  one_hot shape={tuple(oh.shape)}")

    # Build viewmats
    viewmats = c2w_to_w2c_for_gsplat(c2w_mid)

    # Import gsplat here so the failure surface is clear if it's not installed.
    from gsplat import rasterization

    # --- RGB render ---
    print("[render] rasterizing RGB ...")
    rgb, _alpha, _info = rasterization(
        means=means,
        quats=quats_wxyz,
        scales=scales,
        opacities=opacities,
        colors=sh_coeffs,
        viewmats=viewmats,
        Ks=K_mid,
        width=W,
        height=H,
        sh_degree=sh_degree,
        render_mode="RGB",
    )
    rgb_np = rgb[0].clamp(0, 1).cpu().numpy().astype(np.float32)
    np.save(args.out_dir / "midpoint_rgb.npy", rgb_np)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(str(args.out_dir / "midpoint_rgb.png"),
                        (rgb_np * 255).clip(0, 255).astype(np.uint8))
    except Exception:
        pass
    print(f"[render] wrote midpoint_rgb.npy ({rgb_np.shape})")

    # --- Per-level feature maps ---
    for lvl in args.levels:
        print(f"[render] rasterizing lvl{lvl} cluster distribution ...")
        feat, _a, _i = rasterization(
            means=means,
            quats=quats_wxyz,
            scales=scales,
            opacities=opacities,
            colors=one_hots[lvl],
            viewmats=viewmats,
            Ks=K_mid,
            width=W,
            height=H,
            sh_degree=None,
            render_mode="RGB",
        )
        feat_np = feat[0].cpu().numpy().astype(np.float32)
        out_path = args.out_dir / f"midpoint_feature_map_lvl{lvl}.npy"
        np.save(out_path, feat_np)
        print(f"[render] wrote {out_path.name} ({feat_np.shape})")

    print("[render] done.")


if __name__ == "__main__":
    main()
