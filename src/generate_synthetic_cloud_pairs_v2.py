"""Generate synthetic cloudy/clear LISS-IV pairs from QC-approved Mumbai patches.

Uses exactly the same cloud model as the reviewed six-patch `synthetic_clouds_v2.py`
pilot. No pandas dependency. Does not alter original extracted imagery or manifests.

Inputs: data/processed/cloud_reconstruction_v1/metadata/approved_target_manifest.csv
Outputs: data/processed/cloud_reconstruction_v2_synthetic/ and a QC preview under outputs/

Requires: numpy, Pillow; synthetic_clouds_v2.py alongside this script in src/.
"""

from collections import Counter
from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageDraw

# Import the actual reviewed V2 simulator instead of replacing it with different logic.
from synthetic_clouds_v2 import synthesize, false_color

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "cloud_reconstruction_v1"
APPROVED_CSV = SOURCE / "metadata" / "approved_target_manifest.csv"
OUTPUT = ROOT / "data" / "processed" / "cloud_reconstruction_v2_synthetic"
PREVIEW_DIR = ROOT / "outputs" / "synthetic_cloud_dataset_v2"

EXPECTED_COUNTS = {"train": 499, "val": 75, "test": 75}
BASE_SEED = 33007
PREVIEW_PER_SPLIT = 3


def read_approved_rows():
    if not APPROVED_CSV.is_file():
        raise FileNotFoundError(f"Missing approved target manifest: {APPROVED_CSV}")

    with APPROVED_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    required = {"patch_id", "split", "manual_status", "image_path", "valid_mask_path"}
    if not required.issubset(columns):
        raise ValueError(f"Approved manifest is missing columns: {sorted(required - columns)}")
    if len(rows) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"Expected 649 approved records; found {len(rows)}")

    ids = [r["patch_id"].strip() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate patch IDs in approved manifest")
    if "C0137" in ids:
        raise ValueError("C0137 is uncertain and must not appear in the approved manifest")

    counts = Counter(r["split"].strip() for r in rows)
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected approved split counts: {dict(counts)}")
    for r in rows:
        if r["manual_status"].strip() != "approved":
            raise ValueError(f"Nonapproved record found: {r['patch_id']}")

    return sorted(rows, key=lambda r: (('train', 'val', 'test').index(r["split"].strip()), r["patch_id"]))


def resolve_source(relative_path):
    path = Path(relative_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing source array: {path}")
    return path


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rel(path):
    return path.relative_to(ROOT).as_posix()


def preview_panel(arr, mode="RGB"):
    return Image.fromarray(arr, mode=mode)


def make_preview(saved_rows):
    """Preview with the pilot's fixed, consistent display function."""
    selected = []
    sample_rng = np.random.default_rng(2026)
    for split in ("train", "val", "test"):
        group = [r for r in saved_rows if r["split"] == split]
        chosen = sample_rng.choice(len(group), size=min(PREVIEW_PER_SPLIT, len(group)), replace=False)
        selected.extend(group[int(i)] for i in chosen)

    tile = 212
    gap = 12
    caption_height = 36
    header = 58
    cols = 4
    width = gap + cols * (tile + gap)
    height = header + len(selected) * (tile + caption_height + gap)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 10), "Mumbai LISS-IV | Full synthetic dataset V2 | QC preview", fill="black")
    draw.text((gap, 29), "Original | Simulated cloudy | Cloud opacity | Binary cloud mask", fill="black")
    resize = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR

    for row_index, r in enumerate(selected):
        clear = np.load(ROOT / r["target_path"], allow_pickle=False)
        cloudy = np.load(ROOT / r["input_path"], allow_pickle=False)
        alpha = np.load(ROOT / r["alpha_path"], allow_pickle=False)
        mask = np.load(ROOT / r["mask_path"], allow_pickle=False)
        valid = np.load(ROOT / r["valid_mask_path"], allow_pickle=False).astype(bool)
        panels = [
            false_color(clear, valid),
            false_color(cloudy, valid),
            np.repeat(np.uint8(np.clip(alpha, 0, 1)[..., None] * 255), 3, axis=2),
            np.repeat(np.uint8(mask[..., None] * 255), 3, axis=2),
        ]
        captions = [
            f"{r['patch_id']} | {r['split']} | original",
            f"Cloudy | coverage: {float(r['cloud_fraction']):.1%}",
            "Greyscale = simulated opacity",
            "White = alpha >= 0.30",
        ]
        y = header + row_index * (tile + caption_height + gap)
        for j, (panel, caption) in enumerate(zip(panels, captions)):
            x = gap + j * (tile + gap)
            sheet.paste(Image.fromarray(panel).resize((tile, tile), resize), (x, y))
            draw.text((x, y + tile + 6), caption, fill="black")

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    result = PREVIEW_DIR / "synthetic_dataset_preview_v2.png"
    sheet.save(result)
    return result


def main():
    approved = read_approved_rows()
    print(f"QC-approved targets: {len(approved)}", flush=True)
    print("Using the EXACT synthetic-cloud model from synthetic_clouds_v2.py", flush=True)
    print("No pandas dependency; originals will not be modified.", flush=True)

    rows = []
    split_stats = {split: [] for split in EXPECTED_COUNTS}

    for i, record in enumerate(approved, 1):
        patch_id = record["patch_id"].strip()
        split = record["split"].strip()
        source_image = resolve_source(record["image_path"])
        source_mask = resolve_source(record["valid_mask_path"])
        clear = np.load(source_image, allow_pickle=False)
        valid = np.load(source_mask, allow_pickle=False).astype(bool)
        if clear.shape != (256, 256, 3) or valid.shape != (256, 256):
            raise ValueError(f"Unexpected saved patch/mask shape for {patch_id}")
        if clear.dtype != np.uint16:
            raise ValueError(f"Expected original uint16 DNs, got {clear.dtype} in {patch_id}")
        if valid.mean() < 0.995:
            raise ValueError(f"Insufficient valid pixels for {patch_id}")

        # Exactly the same deterministic seed used in the successful V2 pilot.
        seed = BASE_SEED + int(patch_id[1:])
        cloudy, alpha, binary, shadow, coverage = synthesize(
            clear, valid, np.random.default_rng(seed)
        )

        destinations = {
            "input_path": OUTPUT / split / "inputs" / f"{patch_id}.npy",
            "target_path": OUTPUT / split / "targets" / f"{patch_id}.npy",
            "alpha_path": OUTPUT / split / "alpha" / f"{patch_id}.npy",
            "mask_path": OUTPUT / split / "masks" / f"{patch_id}.npy",
            "shadow_path": OUTPUT / split / "shadows" / f"{patch_id}.npy",
        }
        for p in destinations.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        np.save(destinations["input_path"], cloudy.astype(np.float32))
        np.save(destinations["target_path"], clear)  # retains original uint16 DN values
        np.save(destinations["alpha_path"], alpha.astype(np.float32))
        np.save(destinations["mask_path"], binary.astype(np.uint8))
        np.save(destinations["shadow_path"], shadow.astype(np.float32))

        rows.append({
            "patch_id": patch_id,
            "split": split,
            "surface_proxy": record.get("surface_proxy", ""),
            "input_path": rel(destinations["input_path"]),
            "target_path": rel(destinations["target_path"]),
            "alpha_path": rel(destinations["alpha_path"]),
            "mask_path": rel(destinations["mask_path"]),
            "shadow_path": rel(destinations["shadow_path"]),
            "valid_mask_path": rel(source_mask),
            "source_patch_path": rel(source_image),
            "cloud_fraction": round(coverage, 6),
            "valid_fraction": round(float(valid.mean()), 6),
            "seed": seed,
            "bands": "BAND4_NIR,BAND3_Red,BAND2_Green",
            "dn_divisor": 1023,
            "generator": "synthetic_clouds_v2.synthesize",
        })
        split_stats[split].append(coverage)
        if i == 1 or i % 25 == 0 or i == len(approved):
            print(f"Generated {i}/{len(approved)} pairs", flush=True)

    manifest = OUTPUT / "metadata" / "synthetic_pairs_manifest_v2.csv"
    save_csv(manifest, rows, list(rows[0]))
    summary_rows = []
    for split, values in split_stats.items():
        summary_rows.append({
            "split": split, "count": len(values),
            "cloud_fraction_mean": round(float(np.mean(values)), 6),
            "cloud_fraction_min": round(float(np.min(values)), 6),
            "cloud_fraction_max": round(float(np.max(values)), 6),
        })
    summary = OUTPUT / "metadata" / "synthetic_pairs_summary_v2.csv"
    save_csv(summary, summary_rows, list(summary_rows[0]))
    preview = make_preview(rows)

    print("\nSYNTHETIC CLOUD DATASET V2 COMPLETE", flush=True)
    print("Counts:", dict(Counter(r["split"] for r in rows)), flush=True)
    print("Manifest:", manifest, flush=True)
    print("Summary: ", summary, flush=True)
    print("Preview: ", preview, flush=True)
    print("Simulation is uncalibrated DN-space augmentation; real-cloud performance unvalidated.")


if __name__ == "__main__":
    main()
