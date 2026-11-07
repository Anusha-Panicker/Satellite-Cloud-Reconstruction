
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

root = Path(__file__).resolve().parents[1]
raw_dir = root / "data" / "raw" / "mumbai_scenes"
out_dir = root / "outputs" / "overlap_masks"
out_dir.mkdir(parents=True, exist_ok=True)

dates = ["2025-04-16", "2025-06-03"]
scenes = {}

for date in dates:
    matches = list((raw_dir / date).rglob("BAND2.tif"))
    if len(matches) != 1:
        raise ValueError(f"Expected one scene for {date}")
    scenes[date] = matches[0].parent

metadata = {}

for date, folder in scenes.items():
    with rasterio.open(folder / "BAND2.tif") as ds:
        metadata[date] = {
            "bounds": ds.bounds,
            "crs": ds.crs,
            "transform": ds.transform
        }

apr = metadata[dates[0]]
jun = metadata[dates[1]]

if apr["crs"] != jun["crs"]:
    raise ValueError("CRS mismatch")

for item in [apr, jun]:
    t = item["transform"]
    if not np.isclose(t.a, 5) or not np.isclose(t.e, -5):
        raise ValueError("Unexpected pixel resolution")
    if not np.isclose(t.b, 0) or not np.isclose(t.d, 0):
        raise ValueError("Rotated grid not supported")

left = max(apr["bounds"].left, jun["bounds"].left)
right = min(apr["bounds"].right, jun["bounds"].right)
bottom = max(apr["bounds"].bottom, jun["bounds"].bottom)
top = min(apr["bounds"].top, jun["bounds"].top)

if left >= right or bottom >= top:
    raise ValueError("No common geographic area")

def pixel_index(value):
    rounded = round(value)
    if not np.isclose(value, rounded, atol=1e-5):
        raise ValueError("Pixel grids are not aligned")
    return int(rounded)

width = pixel_index((right - left) / 5)
height = pixel_index((top - bottom) / 5)

preview_width = min(1200, width)
preview_height = max(1, round(height * preview_width / width))

def read_scene(folder):
    bands = []

    for filename in ["BAND4.tif", "BAND3.tif", "BAND2.tif"]:
        with rasterio.open(folder / filename) as ds:
            col = pixel_index((left - ds.bounds.left) / 5)
            row = pixel_index((ds.bounds.top - top) / 5)

            if (
                ds.crs != apr["crs"]
                or col < 0 or row < 0
                or col + width > ds.width
                or row + height > ds.height
            ):
                raise ValueError(f"Invalid grid for {filename}")

            band = ds.read(
                1,
                window=Window(col, row, width, height),
                out_shape=(preview_height, preview_width),
                resampling=Resampling.nearest
            ).astype(np.float32)

            bands.append(band)

    return np.stack(bands, axis=-1)

april = read_scene(scenes[dates[0]])
june = read_scene(scenes[dates[1]])

april_valid = np.any(april != 0, axis=-1)
june_valid = np.any(june != 0, axis=-1)
shared_valid = april_valid & june_valid

if not shared_valid.any():
    raise ValueError("No shared nonzero pixels found")

plt.imsave(
    out_dir / "shared_valid_mask.png",
    shared_valid.astype(float),
    cmap="gray",
    vmin=0,
    vmax=1
)

preview = np.zeros_like(june)

for channel in range(3):
    values = june[:, :, channel][shared_valid]
    low, high = np.percentile(values, [2, 98])

    if high > low:
        preview[:, :, channel] = np.clip(
            (june[:, :, channel] - low) / (high - low),
            0, 1
        )

preview[~shared_valid] = 0

visible = (june[:, :, 1] + june[:, :, 2]) / 2

for percentile in [85, 92]:
    threshold = np.percentile(
        visible[shared_valid], percentile
    )

    candidate = shared_valid & (visible >= threshold)

    overlay = preview.copy()
    overlay[candidate] = (
        0.3 * overlay[candidate]
        + 0.7 * np.array([1.0, 1.0, 0.0])
    )

    plt.imsave(
        out_dir / f"cloud_overlay_{percentile}.png",
        overlay
    )

    print(
        f"Threshold {percentile}: "
        f"{candidate.sum() / shared_valid.sum():.2%} "
        "of shared nonzero pixels"
    )

print(f"Preview size: {preview_width} x {preview_height}")
print(f"Shared nonzero area: {shared_valid.mean():.2%}")
print(f"Outputs saved to: {out_dir}")
