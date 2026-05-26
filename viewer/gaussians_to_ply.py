#!/usr/bin/env python3
"""
Convert our gaussians.pt (output of colab_hsegsplat_inference.py) into a standard
3DGS .ply file readable by the aras-p Unity GaussianSplatting package.

Standard 3DGS PLY layout (per vertex, all float32):
    x, y, z
    nx, ny, nz                      (normals — written as 0, package ignores them)
    f_dc_0, f_dc_1, f_dc_2          (DC SH band, RGB)
    f_rest_0..f_rest_{3*(K_sh-1)-1} (remaining SH bands, channel-major: R then G then B per band)
    opacity                         (logit; the package applies sigmoid)
    scale_0, scale_1, scale_2       (log scales)
    rot_0..rot_3                    (quaternion in w, x, y, z; the package normalizes)

Our gaussians.pt stores:
    means: (G, 3)                   linear positions
    quats_wxyz: (G, 4)              unit quaternions, gsplat (w, x, y, z) order
    scales: (G, 3)                  *linear* scales (output of eigh().sqrt() in inference)
    opacities: (G,)                 *linear* opacities in [0, 1]
    harmonics: (G, 3, K_sh)         channel-major: harmonics[g, c, k]  with c in (R,G,B)
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch


def to_log_scale(linear_scales: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(linear_scales, eps, None))


def to_logit_opacity(linear_op: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(linear_op, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def write_ply(out_path: Path, attrs: dict[str, np.ndarray], field_order: list[str]):
    """attrs: name -> (G,) float32. Writes a binary little-endian PLY."""
    n = next(iter(attrs.values())).shape[0]
    for name, a in attrs.items():
        if a.shape != (n,):
            raise ValueError(f"attr {name!r} has shape {a.shape}, expected ({n},)")
        if a.dtype != np.float32:
            attrs[name] = a.astype(np.float32)

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    for fname in field_order:
        header.append(f"property float {fname}")
    header.append("end_header\n")
    header_bytes = ("\n".join(header)).encode("ascii")

    # Stack columns in field_order
    cols = np.stack([attrs[f] for f in field_order], axis=1)  # (n, F) float32

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header_bytes)
        f.write(cols.tobytes(order="C"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaussians", required=True, help="Path to gaussians.pt")
    ap.add_argument("--out", required=True, help="Output .ply path")
    args = ap.parse_args()

    g = torch.load(args.gaussians, map_location="cpu", weights_only=False)
    means = g["means"].numpy().astype(np.float32)           # (G, 3)
    quats_wxyz = g["quats_wxyz"].numpy().astype(np.float32) # (G, 4)
    scales_lin = g["scales"].numpy().astype(np.float32)     # (G, 3)
    op_lin = g["opacities"].numpy().astype(np.float32)      # (G,)
    sh = g["harmonics"].numpy().astype(np.float32)          # (G, 3, K_sh)
    G, _, K_sh = sh.shape
    assert means.shape == (G, 3), means.shape
    assert quats_wxyz.shape == (G, 4), quats_wxyz.shape
    assert scales_lin.shape == (G, 3), scales_lin.shape
    assert op_lin.shape == (G,), op_lin.shape
    print(f"Loaded {G} Gaussians, K_sh={K_sh}")

    # DC band = SH coefficient 0 for R, G, B.
    f_dc = sh[:, :, 0]                                       # (G, 3)
    # Remaining bands flattened channel-major: for k=1..K_sh-1, write R, G, B in turn.
    # Standard 3DGS PLY orders f_rest as: for each non-DC k, [R_k, G_k, B_k].
    rest = sh[:, :, 1:]                                      # (G, 3, K_sh-1)
    rest_flat = rest.transpose(0, 2, 1).reshape(G, -1)       # (G, 3*(K_sh-1))

    scales_log = to_log_scale(scales_lin)
    op_logit = to_logit_opacity(op_lin)

    attrs = {
        "x": means[:, 0], "y": means[:, 1], "z": means[:, 2],
        "nx": np.zeros(G, np.float32), "ny": np.zeros(G, np.float32), "nz": np.zeros(G, np.float32),
        "f_dc_0": f_dc[:, 0], "f_dc_1": f_dc[:, 1], "f_dc_2": f_dc[:, 2],
        "opacity": op_logit,
        "scale_0": scales_log[:, 0], "scale_1": scales_log[:, 1], "scale_2": scales_log[:, 2],
        "rot_0": quats_wxyz[:, 0], "rot_1": quats_wxyz[:, 1],
        "rot_2": quats_wxyz[:, 2], "rot_3": quats_wxyz[:, 3],
    }
    for i in range(rest_flat.shape[1]):
        attrs[f"f_rest_{i}"] = rest_flat[:, i]

    field_order = ["x", "y", "z", "nx", "ny", "nz",
                   "f_dc_0", "f_dc_1", "f_dc_2"]
    field_order += [f"f_rest_{i}" for i in range(rest_flat.shape[1])]
    field_order += ["opacity", "scale_0", "scale_1", "scale_2",
                    "rot_0", "rot_1", "rot_2", "rot_3"]

    out_path = Path(args.out)
    write_ply(out_path, attrs, field_order)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path}  ({size_mb:.1f} MB, {len(field_order)} fields per vertex)")


if __name__ == "__main__":
    main()
