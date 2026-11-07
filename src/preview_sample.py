from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

project_dir = Path(__file__).resolve().parents[1]
scene_dir = project_dir / "data" / "raw" / "221666021" / "221666021"
output_dir = project_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

bands = []
reference_grid = None

# NIR, red, green become the display's red, green, blue channels.
for filename in ["BAND4.tif", "BAND3.tif", "BAND2.tif"]:
    with rasterio.open(scene_dir / filename) as dataset:
        grid = (
            dataset.width,
            dataset.height,
            dataset.crs,
            dataset.transform,
        )

        if reference_grid is None:
            reference_grid = grid
        elif grid != reference_grid:
            raise ValueError(f"Image grid mismatch in {filename}")

        scale = min(1.0, 1200 / max(dataset.width, dataset.height))
        height = max(1, round(dataset.height * scale))
        width = max(1, round(dataset.width * scale))

        band = dataset.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.nearest,
        )
        bands.append(band)

image = np.stack(bands, axis=-1).astype(np.float32)

# Provisional exclusion for display only.
valid = np.any(image != 0, axis=-1)

if not valid.any():
    raise ValueError("The sampled image contains only zero values.")

preview = np.zeros_like(image)

for channel, name in enumerate(["NIR", "Red", "Green"]):
    values = image[:, :, channel][valid]
    lower, upper = np.percentile(values, [2, 98])

    print(
        f"{name}: sampled min={values.min():.0f}, "
        f"max={values.max():.0f}, "
        f"display range={lower:.1f} to {upper:.1f}"
    )

    if upper > lower:
        preview[:, :, channel] = np.clip(
            (image[:, :, channel] - lower) / (upper - lower),
            0,
            1,
        )

preview[~valid] = 0

output_path = output_dir / "liss4_false_colour_preview.png"
plt.imsave(output_path, preview)

print(f"\nPreview dimensions: {width} x {height}")
print(f"Sampled all-zero pixels: {(~valid).mean():.2%}")
print(f"Saved to: {output_path}")