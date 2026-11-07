from pathlib import Path
from contextlib import ExitStack
import csv

import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw" / "mumbai_scenes"
out = root / "outputs" / "approved_patches"
out.mkdir(parents=True, exist_ok=True)

approved = [
    "P02-03",
    "P02-05",
    "P03-04",
    "P03-05",
    "P04-05",
    "P05-05",
    "P07-05",
    "P08-05"
]

csv_path = root / "outputs" / "training_candidates" / "april_patch_candidates.csv"

with open(csv_path, newline="") as file:
    records = {
        row["patch_id"]: row
        for row in csv.DictReader(file)
    }

matches = list((raw / "2025-04-16").rglob("BAND2.tif"))
if len(matches) != 1:
    raise ValueError("Expected one April scene")

scene_dir = matches[0].parent

review_paths = []

with ExitStack() as stack:
    datasets = [
        stack.enter_context(rasterio.open(scene_dir / name))
        for name in ["BAND4.tif", "BAND3.tif", "BAND2.tif"]
    ]

    reference = datasets[0]

    preview_bands = [
        ds.read(1, out_shape=(1200, 1200), masked=False)
        for ds in datasets
    ]
    preview_stack = np.stack(preview_bands, axis=-1).astype(np.float32)
    preview_valid = np.all(preview_stack != 0, axis=-1)

    limits = []
    for i in range(3):
        vals = preview_stack[:, :, i][preview_valid]
        limits.append(np.percentile(vals, [2, 98]))

    manifest_rows = []

    for patch_id in approved:
        row = records[patch_id]
        col = int(row["col"])
        r = int(row["row"])

        window = Window(col, r, 256, 256)

        image = np.stack(
            [ds.read(1, window=window) for ds in datasets],
            axis=-1
        ).astype(np.float32)

        valid = np.all(image != 0, axis=-1)

        np.save(out / f"{patch_id}.npy", image)

        rgb = np.zeros_like(image, dtype=np.float32)
        for c, (low, high) in enumerate(limits):
            if high > low:
                rgb[:, :, c] = np.clip(
                    (image[:, :, c] - low) / (high - low),
                    0, 1
                )

        rgb[~valid] = 0
        png_path = out / f"{patch_id}.png"
        plt.imsave(png_path, rgb)

        manifest_rows.append({
            "patch_id": patch_id,
            "row": r,
            "col": col,
            "valid_fraction": float(valid.mean()),
            "npy_path": str(out / f"{patch_id}.npy"),
            "png_path": str(png_path)
        })

with open(out / "approved_patch_manifest.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "patch_id",
            "row",
            "col",
            "valid_fraction",
            "npy_path",
            "png_path"
        ]
    )
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"Approved patches saved: {len(approved)}")
print(f"Manifest: {out / 'approved_patch_manifest.csv'}")
print(f"Output folder: {out}")