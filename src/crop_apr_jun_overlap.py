
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw" / "mumbai_scenes"
output_dir = project_dir / "outputs" / "aligned_apr_jun"
output_dir.mkdir(parents=True, exist_ok=True)

dates = ["2025-04-16", "2025-06-03"]
scenes = {}

for date in dates:
    matches = list((raw_dir / date).rglob("BAND2.tif"))
    if len(matches) != 1:
        raise ValueError(f"Expected one scene for {date}")
    scenes[date] = matches[0].parent

with rasterio.open(scenes[dates[0]] / "BAND2.tif") as apr:
    apr_bounds = apr.bounds
    reference_crs = apr.crs
    reference_transform = apr.transform

with rasterio.open(scenes[dates[1]] / "BAND2.tif") as jun:
    jun_bounds = jun.bounds
    if jun.crs != reference_crs:
        raise ValueError("CRS mismatch")
    june_transform = jun.transform

left = max(apr_bounds.left, jun_bounds.left)
right = min(apr_bounds.right, jun_bounds.right)
bottom = max(apr_bounds.bottom, jun_bounds.bottom)
top = min(apr_bounds.top, jun_bounds.top)

if left >= right or bottom >= top:
    raise ValueError("No geographic overlap")

pixel_size = 5.0

def whole_pixel(value):
    rounded = round(value)
    if not np.isclose(value, rounded, atol=1e-5):
        raise ValueError("Scenes do not share the same pixel grid")
    return int(rounded)

width = whole_pixel((right - left) / pixel_size)
height = whole_pixel((top - bottom) / pixel_size)

preview_width = min(1200, width)
preview_height = max(1, round(height * preview_width / width))

print(f"Overlap pixels: {width} x {height}", flush=True)
print(
    f"Preview pixels: {preview_width} x {preview_height}",
    flush=True
)

for date in dates:
    print(f"\nProcessing {date}...", flush=True)

    scene_dir = scenes[date]
    bands = []
    grid = None

    for filename in ["BAND4.tif", "BAND3.tif", "BAND2.tif"]:
        with rasterio.open(scene_dir / filename) as ds:
            if ds.crs != reference_crs:
                raise ValueError(f"CRS mismatch: {filename}")
            if ds.res != (pixel_size, pixel_size):
                raise ValueError(f"Resolution mismatch: {filename}")

            if ds.transform.b != 0 or ds.transform.d != 0:
                raise ValueError("Rotated raster grid not supported")

            col = whole_pixel((left - ds.bounds.left) / pixel_size)
            row = whole_pixel((ds.bounds.top - top) / pixel_size)

            if col < 0 or row < 0:
                raise ValueError("Invalid window offset")
            if col + width > ds.width or row + height > ds.height:
                raise ValueError("Window exceeds image bounds")

            window = Window(col, row, width, height)

            band = ds.read(
                1,
                window=window,
                out_shape=(preview_height, preview_width),
                resampling=Resampling.nearest
            )

            current_grid = ds.window_transform(window)

            if grid is None:
                grid = current_grid
            elif not np.allclose(
                tuple(grid)[:6], tuple(current_grid)[:6]
            ):
                raise ValueError("Band grid mismatch")

            bands.append(band)

    image = np.stack(bands, axis=-1).astype(np.float32)
    valid = np.any(image != 0, axis=-1)

    if not valid.any():
        raise ValueError(f"No nonzero pixels for {date}")

    preview = np.zeros_like(image)

    for channel in range(3):
        values = image[:, :, channel][valid]
        low, high = np.percentile(values, [2, 98])

        if high > low:
            preview[:, :, channel] = np.clip(
                (image[:, :, channel] - low) / (high - low),
                0, 1
            )

    preview[~valid] = 0

    output_path = output_dir / f"{date}_overlap.png"
    plt.imsave(output_path, preview)

    print(f"All-zero pixels: {(~valid).mean():.2%}", flush=True)
    print(f"Saved: {output_path}", flush=True)

print("\nBoth previews generated successfully.", flush=True)
