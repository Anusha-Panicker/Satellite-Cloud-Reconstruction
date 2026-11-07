from pathlib import Path
from contextlib import ExitStack
from collections import Counter
import csv

import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
raw_dir = root / "data" / "raw" / "mumbai_scenes" / "2025-04-16"
catalog_dir = root / "outputs" / "large_patch_dataset"
balanced_path = catalog_dir / "balanced_candidate_catalog.csv"
qc_path = catalog_dir / "quality_review" / "patch_quality_audit.csv"
output_dir = root / "data" / "processed" / "cloud_reconstruction_v1"
patch_dir = output_dir / "patches"
mask_dir = output_dir / "valid_masks"
meta_dir = output_dir / "metadata"
preview_dir = output_dir / "previews"

for directory in [patch_dir, mask_dir, meta_dir, preview_dir]:
    directory.mkdir(parents=True, exist_ok=True)

with open(balanced_path, newline="", encoding="utf-8") as f:
    balanced = list(csv.DictReader(f))
with open(qc_path, newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

if len(balanced) != 650 or len(records) != 650:
    raise ValueError("Expected 650 records in both balanced and QC catalogues")

identity = lambda r: (r["patch_id"], r["col"], r["row"], r["split"])
if {identity(r) for r in balanced} != {identity(r) for r in records}:
    raise ValueError("Balanced and QC catalogues do not match")
if len({r["patch_id"] for r in records}) != 650:
    raise ValueError("Duplicate patch IDs found")

expected = {"train": 500, "val": 75, "test": 75}
counts = Counter(r["split"] for r in records)
if counts != expected:
    raise ValueError(f"Unexpected split sizes: {dict(counts)}")

matches = list(raw_dir.rglob("BAND2.tif"))
if len(matches) != 1:
    raise ValueError("Expected exactly one April 2025 BAND2.tif")
scene_dir = matches[0].parent
size = 256
manifest = []

with rasterio.Env(GDAL_CACHEMAX=64):
    with ExitStack() as stack:
        bands = [
            stack.enter_context(rasterio.open(scene_dir / name))
            for name in ("BAND4.tif", "BAND3.tif", "BAND2.tif")
        ]
        reference = bands[0]
        for ds in bands[1:]:
            if (ds.crs != reference.crs or ds.transform != reference.transform
                    or ds.width != reference.width or ds.height != reference.height):
                raise ValueError("Satellite band grids are not aligned")

        print("Extracting 650 provisional patches...", flush=True)
        for index, record in enumerate(records, 1):
            patch_id = record["patch_id"]
            split = record["split"]
            col = int(record["col"])
            row = int(record["row"])
            if not (0 <= col <= reference.width - size and
                    0 <= row <= reference.height - size):
                raise ValueError(f"Window outside satellite scene: {patch_id}")

            window = Window(col, row, size, size)
            image = np.stack(
                [ds.read(1, window=window) for ds in bands], axis=-1
            )
            mask_ok = np.logical_and.reduce(
                [ds.read_masks(1, window=window) > 0 for ds in bands]
            )
            valid = mask_ok & np.all(image > 0, axis=-1)
            if image.shape != (size, size, 3):
                raise ValueError(f"Unexpected image shape: {patch_id}")
            if image.dtype != np.uint16:
                raise ValueError(f"Unexpected image dtype: {image.dtype}")
            if image.max() > 1023:
                raise ValueError(f"DN exceeds 10-bit range: {patch_id}")
            if valid.mean() < 0.995:
                raise ValueError(f"Too many invalid pixels in {patch_id}")

            image_path = patch_dir / split / f"{patch_id}.npy"
            mask_path = mask_dir / split / f"{patch_id}_valid.npy"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(image_path, image, allow_pickle=False)
            np.save(mask_path, valid.astype(np.uint8), allow_pickle=False)
