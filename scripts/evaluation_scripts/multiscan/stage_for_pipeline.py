#!/usr/bin/env python3
"""
Stage a MultiScan ingested scene for the h-segsplat-repo pipeline.

Reads:
    <ingested>/dslr/nerfstudio/transforms.json   (all candidate frames)
    <ingested>/dslr/resized_images/<frame>.jpg   (all candidate frames)
    <ingested>/view_selection.json               (chosen 2-view pair indices)

Writes (under <out>/<scene>/):
    dslr/nerfstudio/transforms.json   ONLY the 2 chosen frames, indexed [0, 1]
    dslr/resized_images/<frame>.JPG   ONLY the 2 chosen frames

This format matches what build_hsegsplat_inputs.py expects: exactly 2 frames,
with the .torch chunk's view-sampler index hardcoded to {"context":[0,1],"target":[0,1]}.

The chosen indices map (orig_jsonl_idx -> new 0, 1) is recorded in
<out>/<scene>/dslr/nerfstudio/transforms.json as an extra ``source_frames`` key
for traceability.
"""

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ingested_dir", type=Path, required=True,
                   help="Output of multiscan_ingest.py + select_2views.py.")
    p.add_argument("--scene_name", type=str, required=True,
                   help="Name used inside the repo's data/ dir.")
    p.add_argument("--out_root", type=Path, required=True,
                   help="Repo's data/ folder (e.g. h-segsplat-repo/data).")
    return p.parse_args()


def main():
    args = parse_args()
    src_t = args.ingested_dir / "dslr" / "nerfstudio" / "transforms.json"
    src_imgs = args.ingested_dir / "dslr" / "resized_images"
    src_sel = args.ingested_dir / "view_selection.json"

    with open(src_t) as f:
        t = json.load(f)
    with open(src_sel) as f:
        sel = json.load(f)
    a_idx, b_idx = sel["context"]
    frame_a = t["frames"][a_idx]
    frame_b = t["frames"][b_idx]

    out_scene = args.out_root / args.scene_name
    out_nerf = out_scene / "dslr" / "nerfstudio"
    out_imgs = out_scene / "dslr" / "resized_images"
    out_nerf.mkdir(parents=True, exist_ok=True)
    out_imgs.mkdir(parents=True, exist_ok=True)

    # Copy the two images. Keep original filenames so transforms.file_path
    # still resolves.
    src_a = src_imgs / frame_a["file_path"]
    src_b = src_imgs / frame_b["file_path"]
    # The h-segsplat-repo pipeline expects .JPG (uppercase) in resized_images.
    # Keep extension uppercase to be safe (matches ScanNet++ test scene).
    dst_a_name = Path(frame_a["file_path"]).stem + ".JPG"
    dst_b_name = Path(frame_b["file_path"]).stem + ".JPG"
    shutil.copy2(src_a, out_imgs / dst_a_name)
    shutil.copy2(src_b, out_imgs / dst_b_name)

    out_t = {
        "camera_model": t["camera_model"],
        "w": t["w"],
        "h": t["h"],
        "fl_x": t["fl_x"],
        "fl_y": t["fl_y"],
        "cx": t["cx"],
        "cy": t["cy"],
        "k1": t.get("k1", 0.0),
        "k2": t.get("k2", 0.0),
        "k3": t.get("k3", 0.0),
        "k4": t.get("k4", 0.0),
        "frames": [
            {"file_path": dst_a_name, "transform_matrix": frame_a["transform_matrix"]},
            {"file_path": dst_b_name, "transform_matrix": frame_b["transform_matrix"]},
        ],
        "source_frames": [
            {"slot": 0, "orig_index_in_ingest": a_idx,
             "_jsonl_frame_index": frame_a.get("_jsonl_frame_index")},
            {"slot": 1, "orig_index_in_ingest": b_idx,
             "_jsonl_frame_index": frame_b.get("_jsonl_frame_index")},
        ],
        "_view_selection_overlap": sel.get("overlap"),
        "_view_selection_baseline_m": sel.get("baseline_m"),
    }
    with open(out_nerf / "transforms.json", "w") as f:
        json.dump(out_t, f, indent=2)
    print(f"[stage] {args.scene_name}: wrote 2-frame transforms ({t['w']}x{t['h']}) at {out_scene}")
    print(f"        frame 0: {dst_a_name}  (ingest idx {a_idx})")
    print(f"        frame 1: {dst_b_name}  (ingest idx {b_idx})")


if __name__ == "__main__":
    main()
