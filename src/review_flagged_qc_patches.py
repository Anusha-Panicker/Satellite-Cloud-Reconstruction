"""Review the 24 manually flagged Mumbai reconstruction targets.

Reads only existing 256x256 NIR/Red/Green NumPy patches and validity masks.
Produces one overview, 24 per-patch four-panel diagnostic images and a CSV.
Never edits image arrays, split assignments or manual_qc_review.csv.
"""
from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "cloud_reconstruction_v1"
MANIFEST = DATASET / "metadata" / "patch_manifest.csv"
DECISIONS = DATASET / "metadata" / "manual_qc_review.csv"
OUTPUT = ROOT / "outputs" / "large_patch_dataset" / "flagged_qc_review"
DETAILS = OUTPUT / "details"

# Flags from the first-pass reviews of all 26 contact sheets.
# Each is a request to inspect, NOT a rejection or confirmed cloud label.
FLAGS = {
    "train": {
        "C0023": "Bright rooftops", "C0043": "Bright exposed ground",
        "C0051": "Reflective structures", "C0137": "Bright edge feature",
        "C0171": "Industrial roofs", "C0278": "Dark patch / shadow check",
        "C0304": "Bright urban roofs", "C0305": "Bright rectangular roofs",
        "C0315": "Bright surface / haze check", "C0329": "Bright circular structures",
        "C0386": "Pale exposed terrain", "C0513": "Pale bright surfaces",
        "C0544": "Pale cyan terrain", "C0661": "Industrial roofs",
    },
    "val": {
        "C0160": "Pale exposed terrain", "C0330": "Reflective urban surface",
        "C0502": "Pale infrastructure", "C0576": "Pale terrain / wetland",
    },
    "test": {
        "C0119": "Pale terrain / roofs", "C0147": "Bright urban roofs",
        "C0415": "Pale branching terrain", "C0490": "Reflective structures",
        "C0662": "Dense bright urban", "C0678": "Pale exposed surface",
    },
}

LOW = np.array([45, 50, 60], dtype=np.float32)  # NIR, Red, Green
HIGH = np.array([280, 210, 200], dtype=np.float32)
FONT_PATHS = (
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def font(size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_pair(record):
    image_path = ROOT / record["image_path"]
    mask_path = ROOT / record["valid_mask_path"]
    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(
            f"Missing saved array/mask for {record['patch_id']}: "
            f"{image_path} | {mask_path}"
        )
    image = np.load(image_path, allow_pickle=False)
    valid = np.load(mask_path, allow_pickle=False).astype(bool)
    if image.shape != (256, 256, 3) or valid.shape != (256, 256):
        raise ValueError(f"Unexpected array shape for {record['patch_id']}")
    if image.dtype != np.uint16 or image.max() > 1023 or not valid.any():
        raise ValueError(f"Invalid sensor data in {record['patch_id']}")
    return image, valid


def false_colour(image, valid):
    display = np.clip((image.astype(np.float32) - LOW) / (HIGH - LOW), 0, 1)
    display[~valid] = 0
    return Image.fromarray(np.rint(display * 255).astype(np.uint8), "RGB")


def band_view(band, valid):
    # Per-patch stretch makes texture easier to see; it is NOT quantitative.
    pixels = band[valid].astype(np.float32)
    p2, p98 = np.percentile(pixels, [2, 98])
    if p98 <= p2:
        p98 = p2 + 1.0
    display = np.clip((band.astype(np.float32) - p2) / (p98 - p2), 0, 1)
    display[~valid] = 0
    return Image.fromarray(np.rint(display * 255).astype(np.uint8), "L").convert("RGB")


def make_detail(record, reason, image, valid, dest):
    margin, head, tile, gap, caption_height = 18, 96, 256, 24, 29
    cell_w, cell_h = tile, tile + caption_height
    width = margin * 2 + 2 * cell_w + gap
    height = head + margin + 2 * cell_h + gap + margin
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 12),
              f"{record['patch_id']}  |  {record['split'].upper()}  |  DETAIL QC",
              font=font(19), fill="black")
    draw.text((margin, 43), f"Check: {reason}", font=font(14), fill="#333333")
    draw.text((margin, 66), "Band views use local contrast stretch (visual check only)",
              font=font(12), fill="#555555")
    panels = [
        ("Fixed false colour (NIR/R/G)", false_colour(image, valid)),
        ("NIR | BAND4", band_view(image[:, :, 0], valid)),
        ("Red | BAND3", band_view(image[:, :, 1], valid)),
        ("Green | BAND2", band_view(image[:, :, 2], valid)),
    ]
    for i, (label, panel) in enumerate(panels):
        r, c = divmod(i, 2)
        x = margin + c * (cell_w + gap)
        y = head + r * (cell_h + gap)
        sheet.paste(panel, (x, y))
        draw.text((x, y + tile + 5), label, fill="black", font=font(12))
    sheet.save(dest, format="PNG")


def make_overview(items):
    cols, tile, cell_w, cell_h, margin, header = 4, 256, 288, 315, 14, 64
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (2 * margin + cols * cell_w,
                              header + rows * cell_h + margin), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 9), "Mumbai | 24 flagged patches | detailed QC overview",
              font=font(19), fill="black")
    draw.text((margin, 38), "Flags are uncertain cases, not rejected samples",
              font=font(12), fill="#444444")
    for index, (record, reason, image, valid) in enumerate(items):
        r, c = divmod(index, cols)
        x = margin + c * cell_w
        y = header + r * cell_h
        sheet.paste(false_colour(image, valid), (x, y))
        draw.text((x, y + tile + 5),
                  f"{record['patch_id']}  |  {record['split']}",
                  font=font(14), fill="black")
        draw.text((x, y + tile + 28), reason, font=font(11), fill="#555555")
    path = OUTPUT / "flagged_24_overview.png"
    sheet.save(path, format="PNG")
    return path


def main():
    manifest = read_csv(MANIFEST)
    if len(manifest) != 650:
        raise ValueError(f"Expected 650 manifest rows, got {len(manifest)}")
    lookup = {r["patch_id"]: r for r in manifest}
    if len(lookup) != len(manifest):
        raise ValueError("Duplicate patch IDs in manifest")

    if DECISIONS.is_file():
        manual = read_csv(DECISIONS)
        decisions = {r["patch_id"]: r for r in manual}
        if len(decisions) != len(manual):
            raise ValueError("Duplicate IDs in existing manual QC decisions")
    else:
        decisions = {}
        print(f"Manual decision CSV not found (left untouched): {DECISIONS}", flush=True)

    targets = [(split, patch_id, reason)
               for split, group in FLAGS.items()
               for patch_id, reason in group.items()]
    if len(targets) != 24 or len({t[1] for t in targets}) != 24:
        raise RuntimeError("The flagged patch list must contain 24 unique IDs")
    for split, patch_id, _ in targets:
        if patch_id not in lookup:
            raise ValueError(f"Flagged ID missing from manifest: {patch_id}")
        if lookup[patch_id]["split"] != split:
            raise ValueError(f"Split mismatch: {patch_id} expected {split}")

    DETAILS.mkdir(parents=True, exist_ok=True)
    overview_items = []
    index_rows = []
    for i, (split, patch_id, reason) in enumerate(targets, 1):
        record = lookup[patch_id]
        image, valid = load_pair(record)
        detail_file = DETAILS / f"{split}_{patch_id}_detail.png"
        make_detail(record, reason, image, valid, detail_file)
        nir, red, green = [image[:, :, k] for k in range(3)]
        bright = (nir >= 220) & (red >= 220) & (green >= 220) & valid
        index_rows.append({
            "patch_id": patch_id,
            "split": split,
            "reason_for_review": reason,
            "current_manual_status": decisions.get(patch_id, {}).get("manual_status", "unknown"),
            "valid_fraction": round(float(valid.mean()), 5),
            "bright_fraction": round(float(bright.mean()), 5),
            "nir_p99_dn": round(float(np.percentile(nir[valid], 99)), 1),
            "red_p99_dn": round(float(np.percentile(red[valid], 99)), 1),
            "green_p99_dn": round(float(np.percentile(green[valid], 99)), 1),
            "detail_image": detail_file.relative_to(ROOT).as_posix(),
        })
        overview_items.append((record, reason, image, valid))
        if i == 1 or i % 4 == 0 or i == len(targets):
            print(f"Prepared {i}/{len(targets)} flagged patch details", flush=True)

    overview = make_overview(overview_items)
    index = OUTPUT / "flagged_24_index.csv"
    with index.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    print("\nFLAGGED PATCH REVIEW GENERATED", flush=True)
    print(f"Overview: {overview}", flush=True)
    print(f"Individual detail images: {DETAILS} ({len(targets)} PNGs)", flush=True)
    print(f"Index: {index}", flush=True)
    print(f"Manual QC CSV unchanged: {DECISIONS}", flush=True)
    print("All flagged patches remain pending your final quality decision.", flush=True)


if __name__ == "__main__":
    main()

