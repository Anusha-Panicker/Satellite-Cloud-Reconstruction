
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
output_dir = project_dir / "outputs" / "mumbai_previews"
output_dir.mkdir(parents=True, exist_ok=True)

dates = ["2025-02-27", "2025-04-16", "2025-06-03"]

for date in dates:
    matches = [
        path for path in raw_dir.rglob("BAND2.tif")
        if date in path.parts
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one scene for {date}, found {len(matches)}"
        )

    scene_dir = matches[0].parent
    bands = []
    reference_grid = None

    for filename in ["BAND4.tif", "BAND3.tif", "BAND2.tif"]:
        with rasterio.open(scene_dir / filename) as dataset:
            grid = (
                dataset.width,
                dataset.height,
                dataset.crs,
                dataset.transform
            )

            if reference_grid is None:
                reference_grid = grid
                scale = min(
                    1.0, 1200 / max(dataset.width, dataset.height)
                )
                height = max(1, round(dataset.height * scale))
                width = max(1, round(dataset.width * scale))
            elif grid != reference_grid:
                raise ValueError(f"Grid mismatch: {date}")

            band = dataset.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.nearest
            )
            bands.append(band)

    image = np.stack(bands, axis=-1).astype(np.float32)
    valid = np.any(image != 0, axis=-1)

    if not valid.any():
        raise ValueError(f"No nonzero pixels: {date}")

    preview = np.zeros_like(image)

    for channel in range(3):
        values = image[:, :, channel][valid]
        lower, upper = np.percentile(values, [2, 98])

        if upper > lower:
            preview[:, :, channel] = np.clip(
                (image[:, :, channel] - lower)
                / (upper - lower),
                0,
                1
            )

    preview[~valid] = 0

    output_path = output_dir / f"{date}_false_colour.png"
    plt.imsave(output_path, preview)

    print(f"\nDate: {date}")
    print(f"Dimensions: {width} x {height}")
    print(f"All-zero pixels: {(~valid).mean():.2%}")
    print(f"Grid transform: {reference_grid[3]}")
    print(f"Saved: {output_path}")

print("\nAll three previews generated successfully.")
