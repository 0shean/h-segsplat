#!/usr/bin/env python3
"""
Build a labelled thumbnail grid from a MultiScan .mp4 so you can hand-pick
a 2-view context pair for h-segsplat.

The MultiScan video is the iPad's 60 fps LiDAR capture (often 2-3 minutes long).
Most frames are motion-blurred, near-duplicates, or pointed at the floor while
the user walked between views. Auto-selection by projected-vertex overlap gives
geometrically reasonable but visually random pairs. The thumbnail grid lets you
eyeball the whole scan and pick a pair that actually shows the objects you care
about, with real parallax between them.

Workflow:
    python scripts_multiscan/browse_frames.py --scan scene_00006_00 --stride 30
        -> writes data/Multiscan/ingested/scene_00006_00/_thumbnails.png
    Open the PNG, find two frames you like, note their indices (drawn on each tile)
    python scripts_multiscan/multiscan_ingest.py \\
        --scan_dir data/Multiscan/scenes/scene_00006_00 \\
        --out_dir  data/Multiscan/ingested/scene_00006_00 \\
        --manual_pair 1200 5400

The thumbnails are pulled from <root>/Multiscan/scenes/<scan>/<scan>.mp4. The
script ALSO extracts ARKit poses from the matching .jsonl so each thumbnail can
be labelled with the camera's translation from the scan start --- that helps
spot pairs with real baseline rather than duplicates.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", type=str, required=True,
                   help="Scan name, e.g. scene_00006_00. Looks under data/Multiscan/scenes/<scan>/.")
    p.add_argument("--root", type=Path, default=Path("Multiscan"),
                   help="Multiscan root dir (default: Multiscan).")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Where to write _thumbnails.png. Default: <root>/ingested/<scan>/.")
    p.add_argument("--stride", type=int, default=30,
                   help="Take every Nth frame for the grid. 30 = 0.5 sec at 60 fps "
                        "(typically ~300 thumbnails over a 2.5-min scan). Smaller = more "
                        "tiles to scroll through; larger = risk missing the best frame.")
    p.add_argument("--max_frames", type=int, default=500,
                   help="Cap on thumbnails extracted (prevents enormous grids).")
    p.add_argument("--tile_w", type=int, default=240, help="Thumbnail width in pixels.")
    p.add_argument("--cols", type=int, default=8, help="Number of columns in the grid.")
    p.add_argument("--label_height", type=int, default=22,
                   help="Pixel rows reserved under each thumbnail for the frame-index label.")
    return p.parse_args()


def extract_thumbnails(mp4: Path, frame_indices: list[int], stride: int,
                       tmp_dir: Path, tile_w: int) -> list[Path]:
    """Use ffmpeg to extract the selected frame indices into tmp_dir, resized
    to width `tile_w` (keeping aspect ratio)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    is_uniform = all(frame_indices[i] == i * stride for i in range(len(frame_indices)))
    if is_uniform:
        sel_filter = f"select='not(mod(n\\,{stride}))'"
    else:
        # Per-frame select for non-uniform indexing (rare here).
        parts = [f"between(n\\,{i}\\,{i})" for i in frame_indices]
        chunks = [parts[k:k+30] for k in range(0, len(parts), 30)]
        sel_filter = "select='" + "+".join("(" + "+".join(c) + ")" for c in chunks) + "'"
    vf = f"{sel_filter},scale={tile_w}:-2"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(mp4),
           "-vf", vf,
           "-vsync", "vfr",
           "-frames:v", str(len(frame_indices)),
           "-q:v", "5",
           str(tmp_dir / "tile_%05d.jpg")]
    subprocess.run(cmd, check=True)
    return sorted(tmp_dir.glob("tile_*.jpg"))


def read_jsonl_translations(jsonl: Path) -> np.ndarray:
    """Returns (N, 3) ARKit translations, indexed by frame number."""
    ts = []
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            T = np.array(d["transform"], dtype=np.float64).reshape(4, 4).T
            ts.append(T[:3, 3])
    return np.stack(ts, 0)


def main():
    args = parse_args()
    scan_dir = args.root / "scenes" / args.scan
    mp4 = scan_dir / f"{args.scan}.mp4"
    jsonl = scan_dir / f"{args.scan}.jsonl"
    if not mp4.exists():
        raise SystemExit(f"missing mp4: {mp4}")
    if not jsonl.exists():
        raise SystemExit(f"missing jsonl: {jsonl}")

    # Frame count from jsonl
    with open(jsonl) as f:
        n_total = sum(1 for _ in f)
    sel = list(range(0, n_total, args.stride))[: args.max_frames]
    print(f"[{args.scan}] jsonl has {n_total} frames; extracting {len(sel)} thumbnails "
          f"(stride={args.stride}, tile_w={args.tile_w})")

    # Translations for labels
    translations = read_jsonl_translations(jsonl)
    t0 = translations[0]
    # Relative position from start, so labels are scene-scale.

    # Output dir
    out_dir = args.out_dir or (args.root / "ingested" / args.scan)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_thumbs_tmp"
    if tmp_dir.exists():
        for p in tmp_dir.iterdir():
            p.unlink()

    paths = extract_thumbnails(mp4, sel, args.stride, tmp_dir, args.tile_w)
    if len(paths) != len(sel):
        print(f"  WARN: requested {len(sel)} thumbnails, got {len(paths)} from ffmpeg")
        sel = sel[: len(paths)]

    # Read all thumbnails, normalise to (tile_h, tile_w) by padding with grey
    tiles = []
    tile_h = None
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            raise RuntimeError(f"cv2 failed to read {p}")
        if tile_h is None:
            tile_h = im.shape[0]
        if im.shape[0] != tile_h or im.shape[1] != args.tile_w:
            # Pad/crop to (tile_h, tile_w) — should be a no-op given ffmpeg scaling
            new_im = np.full((tile_h, args.tile_w, 3), 100, dtype=np.uint8)
            h, w = min(im.shape[0], tile_h), min(im.shape[1], args.tile_w)
            new_im[:h, :w] = im[:h, :w]
            im = new_im
        tiles.append(im)

    # Lay out the grid
    cols = args.cols
    rows = (len(tiles) + cols - 1) // cols
    cell_w = args.tile_w
    cell_h = tile_h + args.label_height
    grid = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)

    for k, (idx, tile) in enumerate(zip(sel, tiles)):
        r = k // cols
        c = k % cols
        y0 = r * cell_h
        x0 = c * cell_w
        grid[y0:y0 + tile_h, x0:x0 + cell_w] = tile
        # Label
        rel_t = translations[idx] - t0
        label_top = y0 + tile_h
        cv2.rectangle(grid, (x0, label_top), (x0 + cell_w, label_top + args.label_height),
                      (240, 240, 240), -1)
        # Frame index in big black; ARKit translation in small grey
        cv2.putText(grid, f"{idx}",
                    (x0 + 5, label_top + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        dist_label = f"dx={rel_t[0]:+.1f} dy={rel_t[1]:+.1f} dz={rel_t[2]:+.1f}"
        cv2.putText(grid, dist_label,
                    (x0 + 55, label_top + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (90, 90, 90), 1, cv2.LINE_AA)

    out_png = out_dir / "_thumbnails.png"
    cv2.imwrite(str(out_png), grid)
    print(f"[{args.scan}] wrote {out_png}  ({grid.shape[1]}x{grid.shape[0]}, {len(tiles)} tiles)")

    # Clean up the per-tile JPEGs
    for p in paths:
        p.unlink()
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    print(f"\nNext step:")
    print(f"  python scripts_multiscan/multiscan_ingest.py \\")
    print(f"      --scan_dir {args.root}/scenes/{args.scan} \\")
    print(f"      --out_dir  {args.root}/ingested/{args.scan} \\")
    print(f"      --manual_pair <frame_A> <frame_B>")


if __name__ == "__main__":
    main()
