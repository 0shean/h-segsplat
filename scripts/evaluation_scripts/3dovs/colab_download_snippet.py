# Colab cell — drop this AT THE END after `run_pipeline.sh` succeeds for a 3D-OVS scene.
# Downloads what the local mIoU eval needs.

from pathlib import Path
from IPython.display import display, Image as IPImage
from google.colab import files

scene_dir = Path(f"data/{SCENE_NAME}")  # SCENE_NAME e.g. '3dovs_bed'

# Show the 2 input context views.
print("=== Input context views ===")
for img in sorted((scene_dir / "dslr" / "resized_images").iterdir()):
    print(img.name); display(IPImage(filename=str(img)))

# Show the 2 rendered context views.
print("=== Rendered context views ===")
for img in sorted(scene_dir.glob("render_view*.png")):
    print(img.name); display(IPImage(filename=str(img)))

# Show all rendered target views (NEW for 3D-OVS).
print("=== Rendered target views ===")
for img in sorted(scene_dir.glob("render_target_*.png")):
    print(img.name); display(IPImage(filename=str(img)))

# Show per-stage timings for this scene.
print("\n=== Per-stage timings ===")
import csv as _csv
timings_csv = scene_dir / "pipeline_timings.csv"
if timings_csv.exists():
    with open(timings_csv) as f:
        rows = list(_csv.reader(f))
    if rows:
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        for r in rows:
            print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
else:
    print("  (no pipeline_timings.csv — using an old run_pipeline.sh?)")

# Download artifacts:
files.download(str(scene_dir / "gaussians.pt"))
files.download(str(scene_dir / "rendered_feature_map_targets_lvl1.npy"))
files.download(str(scene_dir / "pipeline_timings.csv"))
# Optional for hierarchical / debugging:
# files.download(str(scene_dir / "rendered_feature_map_targets_lvl3.npy"))
# files.download(str(scene_dir / "rendered_feature_map_targets_lvl6.npy"))
# files.download(str(scene_dir / "rendered_rgb_targets.npy"))
# Aggregate timings (one file with all scenes you've run this session):
# files.download("data/pipeline_timings.csv")
