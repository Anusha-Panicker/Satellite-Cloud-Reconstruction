from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw" / "mumbai_scenes"
output_dir = project_dir / "outputs" / "cloud_mask_preview"
output_dir.mkdir(parents=True, exist_ok=True)

scene_date = "2025-06-03"
matches = list((raw_dir / scene_date).rglob("BAND2.tif"))
if len(matches) != 1:
    raise ValueError(f"Expected one scene for {scene_date}")

scene_dir = matches[0].parent

with rasterio.open(scene_dir / "BAND2.tif") as b2_ds, \
     rasterio.open(scene_dir / "BAND3.tif") as b3_ds, \
     rasterio.open(scene_dir / "BAND4.tif") as b4_ds:

    width = b2_ds.width
    height = b2_ds.height

    scale = min(1.0, 1200 / max(width, height))
    preview_height = max(1, round(height * scale))
    preview_width = max(1, round(width * scale))

    b2 = b2_ds.read(
        1,
        out_shape=(preview_height, preview_width)
    ).astype(np.float32)

    b3 = b3_ds.read(
        1,
        out_shape=(preview_height, preview_width)
    ).astype(np.float32)

    b4 = b4_ds.read(
        1,
        out_shape=(preview_height, preview_width)
    ).astype(np.float32)

image = np.stack([b4, b3, b2], axis=-1)
valid = np.any(image != 0, axis=-1)

if not valid.any():
    raise ValueError("No valid pixels found")

preview = np.zeros_like(image)

for channel in range(3):
    values = image[:, :, channel][valid]
    low, high = np.percentile(values, [2, 98])

    if high > low:
        preview[:, :, channel] = np.clip(
            (image[:, :, channel] - low) / (high - low),
            0,
            1
        )

# Simple cloud score:
# Clouds tend to be bright in visible bands and not extremely high in NIR-red difference.
visible_brightness = (b2 + b3) / 2.0
brightness_threshold = np.percentile(visible_brightness[valid], 92)

cloud_mask = (visible_brightness >= brightness_threshold) & valid

overlay = preview.copy()
overlay[cloud_mask] = [1.0, 1.0, 0.0]

plt.imsave(output_dir / "2025-06-03_false_colour.png", preview)
plt.imsave(output_dir / "2025-06-03_cloud_overlay.png", overlay)
plt.imsave(output_dir / "2025-06-03_cloud_mask.png", cloud_mask.astype(float), cmap="gray")

print(f"Preview size: {preview_width} x {preview_height}")
print(f"Valid pixels: {valid.mean():.2%}")
print(f"Cloud-mask pixels: {cloud_mask.mean():.2%}")
print(f"Saved: {output_dir / '2025-06-03_false_colour.png'}")
print(f"Saved: {output_dir / '2025-06-03_cloud_overlay.png'}")
print(f"Saved: {output_dir / '2025-06-03_cloud_mask.png'}")