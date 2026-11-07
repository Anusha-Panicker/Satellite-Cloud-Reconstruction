
from pathlib import Path
from contextlib import ExitStack
import csv

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw" / "mumbai_scenes"
out = root / "outputs" / "large_patch_dataset"
out.mkdir(parents=True, exist_ok=True)

catalog_path = out / "candidate_catalog.csv"

with open(catalog_path, newline="") as file:
    records = list(csv.DictReader(file))

if not records:
    raise ValueError("Candidate catalog is empty")

matches = list((raw / "2025-04-16").rglob("BAND2.tif"))

if len(matches) != 1:
    raise ValueError("Expected exactly one April scene")

scene_dir = matches[0].parent

patch_size = 256

low = np.array(
    [45, 50, 60], dtype=np.float32
)

high = np.array(
    [280, 210, 200], dtype=np.float32
)

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

    reference = datasets[0]

    for ds in datasets[1:]:
        if (
            ds.crs != reference.crs
            or ds.transform != reference.transform
            or ds.width != reference.width
            or ds.height != reference.height
        ):
            raise ValueError("Satellite band grid mismatch")

    def load_patch(record):

        col = int(record["col"])
        row = int(record["row"])

        window = Window(
            col,
            row,
            patch_size,
            patch_size
        )

        image = np.stack([
            ds.read(1, window=window)
            for ds in datasets
        ], axis=-1).astype(np.float32)

        if image.shape != (patch_size, patch_size, 3):
            raise ValueError(
                f"Incomplete patch: {record['patch_id']}"
            )

        return image

    print("Starting spectral audit...")

    for record in records:

        image = load_patch(record)

        valid = np.all(image != 0, axis=-1)

        if not valid.any():
            raise ValueError(
                f"Empty patch: {record['patch_id']}"
            )

        nir = image[:, :, 0][valid]
        green = image[:, :, 2][valid]

        contrast = (green - nir) / (
            green + nir + 1e-6
        )

        record["green_nir_contrast"] = round(
            float(np.median(contrast)), 4
        )

        record["nir_median_dn"] = round(
            float(np.median(nir)), 2
        )

    print(f"Audited candidates: {len(records)}")

    ranked = sorted(
        records,
        key=lambda item: float(
            item["green_nir_contrast"]
        )
    )

    groups = {
        "low_contrast": ranked[:25],
        "high_contrast": ranked[-25:]
    }

    for group_name, group in groups.items():

        print(f"\nGenerating {group_name} review...")

        fig, axes = plt.subplots(
            5, 5,
            figsize=(13, 14)
        )

        displayed_count = 0

        for idx, (ax, record) in enumerate(
            zip(axes.flat, group)
        ):

            image = load_patch(record)

            valid = np.all(image != 0, axis=-1)

            preview = np.clip(
                (image - low) / (high - low),
                0,
                1
            )

            preview[~valid] = 0

            preview_uint8 = (
                preview * 255
            ).astype(np.uint8)

            ax.imshow(
                preview_uint8,
                interpolation="nearest"
            )

            ax.set_title(
                f"{record['patch_id']} | "
                f"{float(record['green_nir_contrast']):.4f}",
                fontsize=8
            )

            ax.axis("off")

            displayed_count += 1

            if idx == 0:

                sample_path = (
                    out / f"{group_name}_sample.png"
                )

                plt.imsave(
                    sample_path,
                    preview_uint8
                )

                print(
                    f"Sample DN range: "
                    f"{image.min():.0f} to {image.max():.0f}"
                )

                print(
                    f"Saved sample: {sample_path}"
                )

        fig.suptitle(
            f"April 2025 - {group_name.replace('_', ' ').title()}",
            fontsize=16
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.96],
            h_pad=1.5,
            w_pad=0.5
        )

        review_path = (
            out / f"{group_name}_review.png"
        )

        fig.savefig(
            review_path,
            dpi=130,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"{group_name} displayed images: "
            f"{displayed_count}"
        )

        print(f"Saved review: {review_path}")

csv_path = out / "candidate_spectral_audit.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(records[0].keys())
    )

    writer.writeheader()
    writer.writerows(records)

print("\n===================================")
print("SPECTRAL AUDIT COMPLETED")
print("===================================")

print(f"Total candidates: {len(records)}")
print(f"Spectral CSV: {csv_path}")

print("\nGenerated files:")
print("1. low_contrast_review.png")
print("2. high_contrast_review.png")
print("3. low_contrast_sample.png")
print("4. high_contrast_sample.png")
print("5. candidate_spectral_audit.csv")

