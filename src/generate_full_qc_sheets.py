"""Generate visual QC contact sheets for every saved Mumbai multispectral patch.

Reads only the existing .npy image/mask files and the finalized manifest.
Does not change the satellite arrays, masks, manifest or split assignments.
"""
from pathlib import Path
from collections import Counter
import csv
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "cloud_reconstruction_v1"
MANIFEST = DATASET / "metadata" / "patch_manifest.csv"
DECISIONS = DATASET / "metadata" / "manual_qc_review.csv"
REVIEW_DIR = ROOT / "outputs" / "large_patch_dataset" / "full_qc_review"

SPLITS = ("train", "val", "test")
BATCH_SIZE = 25
COLUMNS = 5
TILE_SIZE = 240
CELL_WIDTH = 264
CELL_HEIGHT = 280
HEADER_HEIGHT = 55
MARGIN = 16
LOW = np.array([45, 50, 60], dtype=np.float32)
HIGH = np.array([280, 210, 200], dtype=np.float32)
RESAMPLE = (Image.Resampling.BILINEAR if hasattr(Image, "Resampling")
            else Image.BILINEAR)

if not MANIFEST.is_file():
    raise FileNotFoundError(f"Missing patch manifest: {MANIFEST}")

with MANIFEST.open(newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

if not records or len({r["patch_id"] for r in records}) != len(records):
    raise ValueError("Patch manifest is empty or contains duplicate IDs")

counts = Counter(r["split"] for r in records)
if counts != {"train": 500, "val": 75, "test": 75}:
    raise ValueError(f"Unexpected dataset split counts: {dict(counts)}")

# Keep a separate editable decision sheet. Do not overwrite earlier manual decisions.
if not DECISIONS.exists():
    fields = ["patch_id", "split", "qc_priority", "surface_proxy",
              "manual_status", "issue", "notes"]
    with DECISIONS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(records, key=lambda x: (SPLITS.index(x["split"]), x["patch_id"])):
            writer.writerow({
                "patch_id": r["patch_id"],
                "split": r["split"],
                "qc_priority": r.get("qc_priority", ""),
                "surface_proxy": r.get("surface_proxy", ""),
                "manual_status": "pending",
                "issue": "",
                "notes": ""
            })
    print(f"Created QC decision sheet: {DECISIONS}", flush=True)
else:
    with DECISIONS.open(newline="", encoding="utf-8") as f:
        existing = list(csv.DictReader(f))
    ids = {r["patch_id"] for r in existing}
    if ids != {r["patch_id"] for r in records}:
        raise ValueError("Existing manual QC CSV does not match the dataset manifest")
    print("Preserving existing manual QC decisions.", flush=True)

REVIEW_DIR.mkdir(parents=True, exist_ok=True)
index_rows = []
font = ImageFont.load_default()

for split in SPLITS:
    subset = sorted((r for r in records if r["split"] == split),
                    key=lambda r: r["patch_id"])
    folder = REVIEW_DIR / split
    folder.mkdir(parents=True, exist_ok=True)

    for offset in range(0, len(subset), BATCH_SIZE):
        batch = subset[offset:offset + BATCH_SIZE]
        sheet_number = offset // BATCH_SIZE + 1
        total_sheets = math.ceil(len(subset) / BATCH_SIZE)
        rows = math.ceil(len(batch) / COLUMNS)
        canvas = Image.new(
            "RGB",
            (MARGIN * 2 + COLUMNS * CELL_WIDTH,
             HEADER_HEIGHT + MARGIN + rows * CELL_HEIGHT),
            "white"
        )
        draw = ImageDraw.Draw(canvas)
        title = f"Mumbai QC | {split.upper()} | Sheet {sheet_number:02}/{total_sheets:02}"
        draw.text((MARGIN, 12), title, fill="black", font=font)
        draw.text((MARGIN, 29), "NIR / Red / Green false colour  |  REVIEW ONLY",
                  fill="black", font=font)

        sheet_name = f"{split}_qc_{sheet_number:02d}.png"
        for i, rec in enumerate(batch):
            image_path = ROOT / rec["image_path"]
            mask_path = ROOT / rec["valid_mask_path"]
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing patch or mask: {rec['patch_id']}")

            image = np.load(image_path, allow_pickle=False)
            valid = np.load(mask_path, allow_pickle=False).astype(bool)
            if image.shape != (256, 256, 3) or valid.shape != (256, 256):
                raise ValueError(f"Unexpected dimensions: {rec['patch_id']}")

            rgb = np.clip((image.astype(np.float32) - LOW) / (HIGH - LOW), 0, 1)
            rgb[~valid] = 0
            rgb = (rgb * 255).astype(np.uint8)
            thumb = Image.fromarray(rgb).resize((TILE_SIZE, TILE_SIZE), RESAMPLE)
            rr, cc = divmod(i, COLUMNS)
            x = MARGIN + cc * CELL_WIDTH
            y = HEADER_HEIGHT + rr * CELL_HEIGHT
            canvas.paste(thumb, (x, y))

            label = rec["patch_id"]
            if rec.get("qc_priority") == "high_brightness_review":
                label += "  [BRIGHT]"
            draw.text((x, y + TILE_SIZE + 3), label, fill="black", font=font)
            draw.text((x, y + TILE_SIZE + 18), rec.get("surface_proxy", ""),
                      fill="#555555", font=font)
            index_rows.append({"split": split, "sheet": sheet_name,
                               "patch_id": rec["patch_id"]})

        dest = folder / sheet_name
        canvas.save(dest, format="PNG", optimize=True)
        print(f"Saved {split}: {sheet_number}/{total_sheets} -> {dest.name}",
              flush=True)

index_path = REVIEW_DIR / "sheet_index.csv"
with index_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["split", "sheet", "patch_id"])
    writer.writeheader()
    writer.writerows(index_rows)

print("\nFULL VISUAL QC SHEETS GENERATED", flush=True)
print(f"Reviewed image inventory: {len(records)} patches", flush=True)
print("Sheets: 20 train + 3 val + 3 test = 26", flush=True)
print(f"Image folder: {REVIEW_DIR}", flush=True)
print(f"Sheet index: {index_path}", flush=True)
print(f"Manual QC decisions: {DECISIONS}", flush=True)
print("All manual statuses remain pending until explicitly reviewed.", flush=True)
