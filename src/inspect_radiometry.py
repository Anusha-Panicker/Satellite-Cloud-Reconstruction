
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

root = Path(__file__).resolve().parents[1]
raw_dir = root / "data" / "raw" / "mumbai_scenes"

dates = ["2025-04-16", "2025-06-03"]
band_names = ["BAND2.tif", "BAND3.tif", "BAND4.tif"]

for date in dates:
    matches = list((raw_dir / date).rglob("BAND2.tif"))

    if len(matches) != 1:
        raise ValueError(f"Expected one scene for {date}")

    scene_dir = matches[0].parent
    bands = []

    print(f"\n{'=' * 45}")
    print(f"Scene: {date}")

    for filename in band_names:
        with rasterio.open(scene_dir / filename) as ds:
            scale = min(1.0, 1200 / max(ds.width, ds.height))
            height = max(1, round(ds.height * scale))
            width = max(1, round(ds.width * scale))

            band = ds.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.nearest
            )

            bands.append(band)

    image = np.stack(bands, axis=0)
    valid = np.any(image != 0, axis=0)

    print(f"Sampled valid area: {valid.mean():.2%}")

    for filename, band in zip(band_names, bands):
        values = band[valid]

        print(f"\n{filename}")
        print(f"Data type: {band.dtype}")
        print(f"Minimum: {values.min()}")
        print(f"Maximum: {values.max()}")
        print(f"Median: {np.median(values):.2f}")
        print(f"1st percentile: {np.percentile(values, 1):.2f}")
        print(f"99th percentile: {np.percentile(values, 99):.2f}")
        print(f"Values above 1023: {(values > 1023).mean():.4%}")
        print(f"Values equal to 1023: {(values == 1023).mean():.4%}")

print("\nRadiometric inspection completed.")
