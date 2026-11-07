
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw" / "mumbai_scenes"
out = root / "outputs" / "alignment_checks"
out.mkdir(parents=True, exist_ok=True)

dates = ["2025-04-16", "2025-06-03"]
scenes = {}

for date in dates:
    matches = list((raw / date).rglob("BAND2.tif"))
    if len(matches) != 1:
        raise ValueError(f"Expected one scene for {date}")
    scenes[date] = matches[0].parent

metadata = {}

for date in dates:
    with rasterio.open(scenes[date] / "BAND2.tif") as ds:
        metadata[date] = {
            "bounds": ds.bounds,
            "crs": ds.crs,
            "transform": ds.transform,
            "width": ds.width,
            "height": ds.height
        }

a = metadata[dates[0]]
j = metadata[dates[1]]

if a["crs"] != j["crs"]:
    raise ValueError("CRS mismatch")

for meta in [a, j]:
    t = meta["transform"]
    if (
        abs(t.a - 5) > 1e-6
        or abs(t.e + 5) > 1e-6
        or abs(t.b) > 1e-6
        or abs(t.d) > 1e-6
    ):
        raise ValueError("Unexpected raster grid")

left = max(a["bounds"].left, j["bounds"].left)
right = min(a["bounds"].right, j["bounds"].right)
top = min(a["bounds"].top, j["bounds"].top)
bottom = max(a["bounds"].bottom, j["bounds"].bottom)

if left >= right or bottom >= top:
    raise ValueError("No geographic overlap")

def exact_pixel(value):
    result = round(value)
    if abs(value - result) > 1e-4:
        raise ValueError("Non-integer pixel offset")
    return int(result)

width = exact_pixel((right - left) / 5)
height = exact_pixel((top - bottom) / 5)

size = 512


locations = [
    ("north_clouds", 0.40, 0.20),
    ("central_land", 0.60, 0.48),
    ("eastern_clouds", 0.82, 0.72),
    ("inland_upper", 0.63, 0.32),
    ("inland_middle", 0.67, 0.47),
    ("inland_lower", 0.58, 0.65)
]


def read_patch(date, x, y):
    folder = scenes[date]
    meta = metadata[date]

    base_col = exact_pixel(
        (left - meta["bounds"].left) / 5
    )
    base_row = exact_pixel(
        (meta["bounds"].top - top) / 5
    )

    window = Window(
        base_col + x,
        base_row + y,
        size,
        size
    )

    bands = []

    for filename in ["BAND4.tif", "BAND3.tif", "BAND2.tif"]:
        with rasterio.open(folder / filename) as ds:
            if (
                ds.crs != meta["crs"]
                or ds.transform != meta["transform"]
            ):
                raise ValueError(
                    f"Band grid mismatch: {date} {filename}"
                )

            band = ds.read(1, window=window)
            if band.shape != (size, size):
                raise ValueError("Incomplete patch")

            bands.append(band)

    return np.stack(bands, axis=-1).astype(np.float32)

for name, fx, fy in locations:
    x = max(0, min(width - size, int(fx * width) - size // 2))
    y = max(0, min(height - size, int(fy * height) - size // 2))

    april = read_patch(dates[0], x, y)
    june = read_patch(dates[1], x, y)

    
    
    if name in [
        "central_land",
        "north_clouds",
        "inland_upper",
        "inland_middle",
        "inland_lower"
    ]:


        
        regions = {
            "central_land": (80, 280, 100, 320),
            "north_clouds": (30, 190, 20, 185),
            "inland_upper": (96, 416, 96, 416),
            "inland_middle": (96, 416, 96, 416),
            "inland_lower": (96, 416, 96, 416)
        }


        y0, y1, x0, x1 = regions[name]

        a = april[:, :, 0]
        b = june[:, :, 0]

        a_edges = np.hypot(*np.gradient(a))
        b_edges = np.hypot(*np.gradient(b))

        reference = a_edges[y0:y1, x0:x1]

        def correlation(first, second):
            first = first.ravel().astype(np.float64)
            second = second.ravel().astype(np.float64)

            first -= first.mean()
            second -= second.mean()

            denominator = (
                np.linalg.norm(first)
                * np.linalg.norm(second)
            )

            if denominator < 1e-12:
                return 0.0

            return float(np.dot(first, second) / denominator)

        best_score = -1.0
        best_shift = (0, 0)
        zero_score = correlation(
            reference,
            b_edges[y0:y1, x0:x1]
        )

        for dy in range(-8, 9):
            for dx in range(-8, 9):
                candidate = b_edges[
                    y0 + dy:y1 + dy,
                    x0 + dx:x1 + dx
                ]

                score = correlation(reference, candidate)

                if score > best_score:
                    best_score = score
                    best_shift = (dx, dy)

        print(f"\nRegion: {name}")
        print(f"Zero-shift correlation: {zero_score:.4f}")
        print(f"Best correlation: {best_score:.4f}")
        print(f"Estimated shift (dx, dy): {best_shift}")


    valid = (
        np.any(april != 0, axis=-1)
        & np.any(june != 0, axis=-1)
    )

    if not valid.any():
        print(f"Skipping {name}: no shared nonzero pixels")
        continue

    images = [april, june]
    previews = [
        np.zeros_like(april),
        np.zeros_like(june)
    ]

    for channel in range(3):
        values = np.concatenate([
            april[:, :, channel][valid],
            june[:, :, channel][valid]
        ])

        low, high = np.percentile(values, [2, 98])

        if high > low:
            for idx, image in enumerate(images):
                previews[idx][:, :, channel] = np.clip(
                    (image[:, :, channel] - low)
                    / (high - low),
                    0, 1
                )

    for preview in previews:
        preview[~valid] = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, preview, title in zip(
        axes,
        previews,
        ["16 April 2025", "3 June 2025"]
    ):
        ax.imshow(preview)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(name.replace("_", " ").title())
    fig.tight_layout()

    path = out / f"{name}_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)

    
    yy, xx = np.indices((size, size))

    take_april = ((xx // 64 + yy // 64) % 2) == 0

    checkerboard = np.where(
        take_april[..., None],
        previews[0],
        previews[1]
    )

    checker_path = out / f"{name}_checkerboard.png"

    plt.imsave(checker_path, checkerboard)

    print(f"Saved checkerboard: {checker_path}")

    
    if name == "inland_upper":
        dx, dy = -1, 4

        shifted = np.zeros_like(previews[1])

        x0 = max(0, -dx)
        x1 = min(size, size - dx)
        y0 = max(0, -dy)
        y1 = min(size, size - dy)

        shifted[y0:y1, x0:x1] = previews[1][
            y0 + dy:y1 + dy,
            x0 + dx:x1 + dx
        ]

        yy, xx = np.indices((size, size))
        checker = ((xx // 32 + yy // 32) % 2) == 0

        before = np.where(
            checker[..., None],
            previews[0],
            previews[1]
        )

        after = np.where(
            checker[..., None],
            previews[0],
            shifted
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(before)
        axes[0].set_title("Before correction")
        axes[0].axis("off")

        axes[1].imshow(after)
        axes[1].set_title("After proposed correction")
        axes[1].axis("off")

        fig.tight_layout()

        result_path = out / "inland_upper_before_after_shift.png"
        fig.savefig(result_path, dpi=150)
        plt.close(fig)

        print(f"Saved: {result_path}")



    print(f"Saved: {path}")
    print(f"Shared nonzero pixels: {valid.mean():.2%}")

print("\nAlignment inspection patches completed.")
