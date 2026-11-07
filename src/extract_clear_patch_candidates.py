
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw" / "mumbai_scenes"
out = root / "outputs" / "clear_patch_candidates"
out.mkdir(parents=True, exist_ok=True)

matches = list((raw / "2025-04-16").rglob("BAND2.tif"))

if len(matches) != 1:
    raise ValueError("Expected one April satellite scene")

scene_dir = matches[0].parent

patch_size = 256
target_count = 24
rng = np.random.default_rng(42)

patches = []
records = []

with rasterio.open(scene_dir / "BAND2.tif") as b2, \
     rasterio.open(scene_dir / "BAND3.tif") as b3, \
     rasterio.open(scene_dir / "BAND4.tif") as b4:

    grid = (b2.crs, b2.transform, b2.width, b2.height)

    for ds in [b3, b4]:
        if (ds.crs, ds.transform, ds.width, ds.height) != grid:
            raise ValueError("Band grid mismatch")

    positions = [
        (row, col)
        for row in range(0, b2.height - patch_size + 1, patch_size)
        for col in range(0, b2.width - patch_size + 1, patch_size)
    ]

    rng.shuffle(positions)

    for row, col in positions:
        window = Window(col, row, patch_size, patch_size)

        patch = np.stack([
            ds.read(1, window=window)
            for ds in [b2, b3, b4]
        ], axis=-1)

        valid = np.all(patch != 0, axis=-1)

        if valid.mean() < 0.995:
            continue

        index = len(patches) + 1
        name = f"patch_{index:03d}"

        np.save(out / f"{name}.npy", patch)

        patches.append(patch)
        records.append((name, row, col, valid.mean()))

        print(f"Saved {name}: row={row}, col={col}")

        if len(patches) >= target_count:
            break

if not patches:
    raise ValueError("No suitable nonzero patches found")

with open(out / "patch_index.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patch", "row", "col", "valid_fraction"])
    writer.writerows(records)

fig, axes = plt.subplots(4, 6, figsize=(15, 11))

for ax in axes.flat:
    ax.axis("off")

for i, patch in enumerate(patches):
    image = patch[:, :, [2, 1, 0]].astype(np.float32)
    preview = np.zeros_like(image)

    for channel in range(3):
        values = image[:, :, channel]
        low, high = np.percentile(values, [2, 98])

        if high > low:
            preview[:, :, channel] = np.clip(
                (values - low) / (high - low),
                0, 1
            )

    axes.flat[i].imshow(preview)
    axes.flat[i].set_title(f"Patch {i + 1}", fontsize=9)
    axes.flat[i].axis("off")

fig.tight_layout()
path = out / "clear_patch_contact_sheet.png"
fig.savefig(path, dpi=120)
plt.close(fig)

print(f"\nExtracted {len(patches)} candidate patches")
print(f"Contact sheet saved: {path}")
