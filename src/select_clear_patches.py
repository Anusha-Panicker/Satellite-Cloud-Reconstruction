
from pathlib import Path
from contextlib import ExitStack
import csv

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

root = Path(__file__).resolve().parents[1]
raw_dir = root / "data" / "raw" / "mumbai_scenes"
out_dir = root / "outputs" / "training_candidates"
out_dir.mkdir(parents=True, exist_ok=True)

matches = list(
    (raw_dir / "2025-04-16").rglob("BAND2.tif")
)

if len(matches) != 1:
    raise ValueError("Expected one April scene")

scene_dir = matches[0].parent

patch_size = 256
grid_size = 8
records = []

fig, axes = plt.subplots(
    grid_size, grid_size, figsize=(16, 16)
)

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

    x_positions = np.linspace(
        patch_size // 2,
        reference.width - patch_size // 2 - 1,
        grid_size
    ).astype(int)

    y_positions = np.linspace(
        patch_size // 2,
        reference.height - patch_size // 2 - 1,
        grid_size
    ).astype(int)

    for iy, cy in enumerate(y_positions):
        for ix, cx in enumerate(x_positions):
            x = int(cx - patch_size // 2)
            y = int(cy - patch_size // 2)

            window = Window(
                x, y, patch_size, patch_size
            )

            image = np.stack([
                ds.read(1, window=window)
                for ds in datasets
            ], axis=-1).astype(np.float32)

            valid = np.any(image != 0, axis=-1)
            valid_fraction = valid.mean()

            ax = axes[iy, ix]
            ax.axis("off")

            patch_id = f"P{iy+1:02d}-{ix+1:02d}"

            if valid_fraction < 0.99:
                ax.text(
                    0.5, 0.5, "Excluded",
                    ha="center", va="center",
                    transform=ax.transAxes
                )
                continue

            preview = np.zeros_like(image)

            for channel in range(3):
                values = image[:, :, channel][valid]
                low, high = np.percentile(
                    values, [2, 98]
                )

                if high > low:
                    preview[:, :, channel] = np.clip(
                        (image[:, :, channel] - low)
                        / (high - low),
                        0, 1
                    )

            preview[~valid] = 0

            ax.imshow(preview)
            ax.set_title(patch_id, fontsize=8)

            records.append([
                patch_id, x, y,
                round(float(valid_fraction), 4)
            ])

fig.suptitle(
    "16 April 2025 - Clear Patch Candidates",
    fontsize=16
)
fig.tight_layout()

image_path = out_dir / "april_patch_contact_sheet.png"
fig.savefig(image_path, dpi=130)
plt.close(fig)

csv_path = out_dir / "april_patch_candidates.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "patch_id", "col", "row", "valid_fraction"
    ])
    writer.writerows(records)

print(f"Candidate patches: {len(records)}")
print(f"Patch dimensions: {patch_size} x {patch_size}")
print(f"Saved contact sheet: {image_path}")
print(f"Saved patch catalogue: {csv_path}")
