
from pathlib import Path
from contextlib import ExitStack
from collections import Counter
import csv

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

root = Path(__file__).resolve().parents[1]
raw_dir = root / "data" / "raw" / "mumbai_scenes"
out_dir = root / "outputs" / "large_patch_dataset"
out_dir.mkdir(parents=True, exist_ok=True)

matches = list(
    (raw_dir / "2025-04-16").rglob("BAND2.tif")
)

if len(matches) != 1:
    raise ValueError("Expected exactly one April scene")

scene_dir = matches[0].parent

patch_size = 256
stride = 320
block_size = 2048
margin = 256
min_valid = 0.995

targets = {
    "train": 500,
    "val": 100,
    "test": 100
}

rng = np.random.default_rng(42)
pools = {split: [] for split in targets}

with ExitStack() as stack:
    datasets = [
        stack.enter_context(
            rasterio.open(scene_dir / name)
        )
        for name in [
            "BAND4.tif",
            "BAND3.tif",
            "BAND2.tif"
        ]
    ]

    ref = datasets[0]

    for ds in datasets[1:]:
        if (
            ds.crs != ref.crs
            or ds.transform != ref.transform
            or ds.width != ref.width
            or ds.height != ref.height
        ):
            raise ValueError("Band grid mismatch")

    for row in range(
        0, ref.height - patch_size + 1, stride
    ):
        for col in range(
            0, ref.width - patch_size + 1, stride
        ):

            local_x = col % block_size
            local_y = row % block_size

            if (
                local_x < margin
                or local_y < margin
                or local_x + patch_size > block_size - margin
                or local_y + patch_size > block_size - margin
            ):
                continue

            window = Window(
                col, row, patch_size, patch_size
            )

            image = np.stack([
                ds.read(1, window=window)
                for ds in datasets
            ])

            valid = np.all(image != 0, axis=0)
            valid_fraction = float(valid.mean())

            if valid_fraction < min_valid:
                continue

            nir = image[0][valid].astype(np.float32)
            red = image[1][valid].astype(np.float32)

            ndvi = (nir - red) / (nir + red + 1e-6)

            bx = col // block_size
            by = row // block_size
            bucket = (bx * 17 + by * 19 + 7) % 20

            if bucket < 14:
                split = "train"
            elif bucket < 17:
                split = "val"
            else:
                split = "test"

            pools[split].append({
                "col": col,
                "row": row,
                "block_x": bx,
                "block_y": by,
                "split": split,
                "valid_fraction": round(valid_fraction, 5),
                "median_ndvi": round(
                    float(np.median(ndvi)), 4
                )
            })

    print("Eligible candidates by split:")
    for split, pool in pools.items():
        print(f"{split}: {len(pool)}")

    selected = []

    for split, target in targets.items():
        pool = pools[split]
        rng.shuffle(pool)
        selected.extend(pool[:target])

    rng.shuffle(selected)

    for i, record in enumerate(selected, start=1):
        record["patch_id"] = f"C{i:04d}"
        record["review_status"] = "pending"

    csv_path = out_dir / "candidate_catalog.csv"

    fieldnames = [
        "patch_id", "col", "row",
        "block_x", "block_y", "split",
        "valid_fraction", "median_ndvi",
        "review_status"
    ]

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(selected)

    review_sample = selected.copy()
    rng.shuffle(review_sample)
    review_sample = review_sample[:100]

    fig, axes = plt.subplots(
        10, 10, figsize=(17, 18)
    )

    low = np.array([45, 50, 60], dtype=np.float32)
    high = np.array([280, 210, 200], dtype=np.float32)

    for ax in axes.flat:
        ax.axis("off")

    for ax, record in zip(axes.flat, review_sample):
        window = Window(
            record["col"],
            record["row"],
            patch_size,
            patch_size
        )

        image = np.stack([
            ds.read(1, window=window)
            for ds in datasets
        ], axis=-1).astype(np.float32)

        valid = np.all(image != 0, axis=-1)

        preview = np.clip(
            (image - low) / (high - low),
            0, 1
        )

        preview[~valid] = 0

        ax.imshow(preview)
        ax.set_title(
            record["patch_id"],
            fontsize=7
        )

    fig.suptitle(
        "April 2025 - Large Dataset Candidate Review",
        fontsize=16
    )
    fig.tight_layout()

    sheet_path = out_dir / "candidate_contact_sheet.png"
    fig.savefig(sheet_path, dpi=130)
    plt.close(fig)

counts = Counter(
    record["split"] for record in selected
)

print("\nSelected candidates:")
print(f"Train: {counts['train']}")
print(f"Validation: {counts['val']}")
print(f"Test: {counts['test']}")
print(f"Total: {len(selected)}")

print(f"\nReview sample: {len(review_sample)}")
print(f"Catalog: {csv_path}")
print(f"Contact sheet: {sheet_path}")
print("\nCandidate catalog generation completed.")
