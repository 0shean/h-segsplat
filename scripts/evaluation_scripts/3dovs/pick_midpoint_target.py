#!/usr/bin/env python3
"""For each 3D-OVS scene, find the labelled target view whose camera pose
is closest to the midpoint of the two input views.

Midpoint pose is defined as:
  position = (C1 + C2) / 2           (Euclidean midpoint of camera centers)
  rotation = SLERP(R1, R2, t=0.5)    (half-way slerp of the two rotations)

For each candidate target view, we report:
  - position_dist: || target_center - midpoint_center ||
  - angle_deg: angular distance between target rotation and midpoint rotation
               (axis-angle magnitude of R_target * R_midpoint^T, in degrees)
  - score = position_dist / median(position_dist) + angle_deg / median(angle_deg)
            (z-score-ish so the two components are comparable across scenes)

The target with the smallest score is the recommended pick.

Reads transforms.json (input poses) + target_views.json (candidate poses)
from data/3D-OVS/ingested/<scene>/.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def rotmat_from_quat_wxyz(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def quat_from_rotmat(R):
    """Return wxyz unit quaternion from a 3x3 rotation matrix (Shepperd's method)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def slerp(q1, q2, t):
    """Spherical linear interpolation of unit quaternions. q1, q2 in wxyz."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.dot(q1, q2))
    if dot < 0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        out = q1 + t * (q2 - q1)
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(dot)
    sin_0 = np.sin(theta_0)
    a = np.sin((1 - t) * theta_0) / sin_0
    b = np.sin(t * theta_0) / sin_0
    return a * q1 + b * q2


def rotation_angle_deg(R_a, R_b):
    """Angle (degrees) of the rotation R_a @ R_b^T, via axis-angle."""
    R = R_a @ R_b.T
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, float(cos_theta)))
    return float(np.degrees(np.arccos(cos_theta)))


def load_pose(matrix):
    M = np.array(matrix, dtype=np.float64)  # 4x4 c2w
    return M[:3, :3], M[:3, 3]


def pick_for_scene(scene_dir: Path):
    tj = json.load(open(scene_dir / "dslr/nerfstudio/transforms.json"))
    tv = json.load(open(scene_dir / "target_views.json"))
    if len(tj["frames"]) != 2:
        raise ValueError(f"expected 2 input frames, got {len(tj['frames'])}")

    R1, C1 = load_pose(tj["frames"][0]["transform_matrix"])
    R2, C2 = load_pose(tj["frames"][1]["transform_matrix"])
    C_mid = (C1 + C2) * 0.5
    q1 = quat_from_rotmat(R1)
    q2 = quat_from_rotmat(R2)
    q_mid = slerp(q1, q2, 0.5)
    R_mid = rotmat_from_quat_wxyz(q_mid)

    rows = []
    for tgt in tv["targets"]:
        Rt, Ct = load_pose(tgt["transform_matrix"])
        pos_d = float(np.linalg.norm(Ct - C_mid))
        ang_d = rotation_angle_deg(Rt, R_mid)
        rows.append({"view_id": tgt["view_id"], "pos_d": pos_d, "ang_d": ang_d})

    # Combined score: position relative to its median + angle relative to its median.
    # This makes the two components comparable when the scene's pose units differ.
    pos_med = np.median([r["pos_d"] for r in rows]) or 1.0
    ang_med = np.median([r["ang_d"] for r in rows]) or 1.0
    for r in rows:
        r["score"] = r["pos_d"] / pos_med + r["ang_d"] / ang_med
    rows.sort(key=lambda r: r["score"])
    return C1, C2, C_mid, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingested_root", type=Path,
                    default=Path("data/3D-OVS/ingested"))
    ap.add_argument("--scenes", nargs="+",
                    default=["bed", "bench", "blue_sofa", "covered_desk", "lawn",
                             "office_desk", "room", "snacks", "sofa", "table"])
    args = ap.parse_args()

    picks = {}
    print(f"{'scene':<14} {'view':>5} {'pos_d':>8} {'ang_d':>8} {'score':>7}   "
          f"(others ranked)")
    for scene in args.scenes:
        scene_dir = args.ingested_root / scene
        if not (scene_dir / "target_views.json").exists():
            print(f"{scene:<14} [skip] no target_views.json")
            continue
        _, _, _, rows = pick_for_scene(scene_dir)
        best = rows[0]
        picks[scene] = best["view_id"]
        other = "  ".join(f"{r['view_id']}:{r['score']:.2f}" for r in rows[1:])
        print(f"{scene:<14} {best['view_id']:>5} {best['pos_d']:>8.3f} "
              f"{best['ang_d']:>8.2f} {best['score']:>7.2f}   {other}")

    cmd_args = " ".join(f"{s}:{v}" for s, v in picks.items())
    print(f"\nSuggested --target_view arguments:\n  --target_view {cmd_args}")


if __name__ == "__main__":
    main()
