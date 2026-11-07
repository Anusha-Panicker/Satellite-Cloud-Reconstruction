
from pathlib import Path
from contextlib import ExitStack
import csv

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw" / "mumbai_scenes"
out = root / "outputs" / "training_candidates"

selected = [
    "P01-03", "P02-03", "P02-05", "P03-04",
    "P03-05", "P04-04", "P04-05", "P05-05",
    "P06-05", "P07-04", "P07-05", "P08-05"
]

with open(
    out / "april_patch_candidates.csv",
    newline=""
) as file:
    records = {
        row["patch_id"]: row
        for row in csv.DictReader(file)
    }

matches = list(
    (raw / "2025-04-16").rglob("BAND2.tif")
)

if len(matches) != 1:
    raise ValueError("Expected one April scene")

scene_dir = matches[0].parent

fig, axes = plt.subplots(3, 4, figsize=(16, 13))

with ExitStack() as stack:
    datasets = [
        stack.enter_context(
            rasterio.open(scene_dir / filename)
        )
        for filename in [
            "BAND4.tif", "BAND3.tif", "BAND2.tif"
        ]
    ]

    reference = datasets[0]

    for ds in datasets[1:]:
        if (
            ds.crs != reference.crs
            or ds.transform != reference.transform
            or ds.width != reference.width
            or ds.height != reference.height
        ):
            raise ValueError("Band grid mismatch")

    scale = min(
        1.0,
        1200 / max(reference.width, reference.height)
    )

    preview_width = round(reference.width * scale)
    preview_height = round(reference.height * scale)

    samples = [
        ds.read(
            1,
            out_shape=(preview_height, preview_width),
            resampling=Resampling.nearest
        )
        for ds in datasets
    ]

    sample_valid = np.all(
        np.stack(samples) != 0, axis=0
    )

    limits = [
        np.percentile(band[sample_valid], [2, 98])
        for band in samples
    ]

    for ax, patch_id in zip(axes.flat, selected):
        record = records[patch_id]

        window = Window(
            int(record["col"]),
            int(record["row"]),
            256,
            256
        )

        image = np.stack([
            ds.read(1, window=window)
            for ds in datasets
        ], axis=-1).astype(np.float32)

        valid = np.all(image != 0, axis=-1)
        preview = np.zeros_like(image)

        for channel, (low, high) in enumerate(limits):
            if high > low:
                preview[:, :, channel] = np.clip(
                    (image[:, :, channel] - low)
                    / (high - low),
                    0, 1
                )

        preview[~valid] = 0

        ax.imshow(preview)
        ax.set_title(patch_id)
        ax.axis("off")

fig.suptitle(
    "April 2025 - Detailed Land Patch Review",
    fontsize=16
)
fig.tight_layout()

output_path = out / "april_detailed_review.png"
fig.savefig(output_path, dpi=140)
plt.close(fig)

print(f"Reviewed patches: {len(selected)}")
print(f"Saved: {output_path}")
