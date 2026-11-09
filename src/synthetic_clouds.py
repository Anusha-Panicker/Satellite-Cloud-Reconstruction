"""Pilot synthetic cloud simulation for QC-approved Mumbai LISS-IV targets.

Inputs: saved uint16 [NIR, Red, Green] DN patches (0..1023).
Outputs: visual preview and a small .npz pilot sample, not a full training set.

Synthetic overlays are approximate, not calibrated atmospheric physics.
"""

from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "cloud_reconstruction_v1"
MANIFEST = DATASET / "metadata" / "approved_target_manifest.csv"
OUT = ROOT / "outputs" / "synthetic_cloud_pilot"
SEED = 2026

# Fixed DN stretches for false-color display only: [NIR, Red, Green].
DISPLAY_LOW = np.array([45, 50, 60], dtype=np.float32)
DISPLAY_HIGH = np.array([280, 210, 200], dtype=np.float32)


def smooth_noise(rng, size, cells, blur_radius=0):
    """Low-frequency noise from a small grid, without SciPy dependencies."""
    coarse = (rng.random((cells, cells)) * 255).astype(np.uint8)
    img = Image.fromarray(coarse, mode="L").resize(
        (size, size), Image.Resampling.BICUBIC
    )
    if blur_radius:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    return np.asarray(img, dtype=np.float32) / 255.0


def synthetic_clouds(clean_dn, valid_mask, rng):
    """Return (cloudy_dn, alpha_mask, binary_cloud_mask) for one DN patch.

    Cloud alpha and labels are generated from smooth multiscale blobs.
    Cloud brightness is a deliberately simple DN-space baseline.
    """
    if clean_dn.ndim != 3 or clean_dn.shape[2] != 3:
        raise ValueError("Expected image shape H x W x 3 in [NIR, Red, Green] order")
    height, width = clean_dn.shape[:2]
    if height != width:
        raise ValueError("Pilot expects square patches")
    if valid_mask.shape != (height, width):
        raise ValueError("Validity mask shape mismatch")
    if clean_dn.max() > 1023 or clean_dn.min() < 0:
        raise ValueError("Expected original 10-bit DN values in 0..1023")

    size = height
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    field = np.zeros((size, size), dtype=np.float32)

    for _ in range(int(rng.integers(3, 7))):
        cx, cy = rng.uniform(-0.1 * size, 1.1 * size, 2)
        rx = rng.uniform(0.12 * size, 0.40 * size)
        ry = rng.uniform(0.12 * size, 0.40 * size)
        angle = rng.uniform(0, np.pi)
        dx, dy = xx - cx, yy - cy
        u = np.cos(angle) * dx + np.sin(angle) * dy
        v = -np.sin(angle) * dx + np.cos(angle) * dy
        blob = np.exp(-0.5 * ((u / rx) ** 2 + (v / ry) ** 2))
        field = np.maximum(field, blob.astype(np.float32))

    field += 0.23 * (smooth_noise(rng, size, 9) - 0.5)
    field += 0.10 * (smooth_noise(rng, size, 23) - 0.5)
    lo, hi = np.percentile(field, [1, 99])
    field = np.clip((field - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    desired_cover = float(rng.uniform(0.18, 0.48))
    threshold = float(np.quantile(field, 1.0 - desired_cover))
    feather = 0.045
    t = np.clip((field - (threshold - feather)) / (2 * feather), 0, 1)
    shape_alpha = t * t * (3.0 - 2.0 * t)
    alpha = shape_alpha * float(rng.uniform(0.76, 0.96))

    # Spatial variations stop clouds from being perfectly uniform blocks.
    texture = 0.88 + 0.18 * smooth_noise(rng, size, 30)
    cloud_dn = np.array([760, 820, 850], dtype=np.float32)
    cloud_dn = cloud_dn[None, None, :] * texture[:, :, None]

    clean = clean_dn.astype(np.float32)
    cloudy = clean * (1.0 - alpha[..., None]) + cloud_dn * alpha[..., None]
    cloudy = np.clip(cloudy, 0, 1023).astype(np.float32)

    # Avoid changing missing pixels; these do not count toward cloud coverage.
    valid = valid_mask.astype(bool)
    alpha[~valid] = 0
    cloudy[~valid] = clean[~valid]
    binary = (alpha >= 0.35) & valid

    return cloudy, alpha.astype(np.float32), binary.astype(np.uint8)


def false_color(dn, valid):
    view = np.clip(
        (dn.astype(np.float32) - DISPLAY_LOW) / (DISPLAY_HIGH - DISPLAY_LOW),
        0, 1
    )
    view[~valid.astype(bool)] = 0
    return (view * 255).astype(np.uint8)


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest missing: {MANIFEST}")
    with MANIFEST.open(newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))

    # Use approved training targets only; preserve validation/test for evaluation.
    if len(records) != 649:
        raise ValueError(f"Expected 649 QC-approved targets, found {len(records)}")
    if len({r["patch_id"] for r in records}) != len(records):
        raise ValueError("Duplicate patch IDs in approved manifest")
    if any(r.get("patch_id") == "C0137" for r in records):
        raise ValueError("Uncertain patch C0137 must not be used")
    candidates = [
        r for r in records
        if r.get("split") == "train"
        and r.get("manual_status") == "approved"
        and float(r.get("valid_fraction", "0")) >= 0.995
    ]
    print(f"Approved pilot source: {len(candidates)} training targets", flush=True)
    by_type = {
        key: [r for r in candidates if r.get("surface_proxy") == key]
        for key in ("land_or_mixed", "water_like")
    }
    rng = np.random.default_rng(SEED)
    chosen = []
    for name, count in (("land_or_mixed", 4), ("water_like", 2)):
        pool = by_type[name]
        if len(pool) < count:
            raise ValueError(f"Not enough {name} training examples: {len(pool)}")
        indices = rng.choice(len(pool), count, replace=False)
        chosen.extend(pool[int(i)] for i in indices)

    OUT.mkdir(parents=True, exist_ok=True)
    tile = 240
    label_h = 34
    columns = 3
    margin = 15
    header = 52
    row_h = tile + label_h
    canvas = Image.new(
        "RGB", (columns * tile + (columns + 1) * margin,
                header + len(chosen) * (row_h + margin)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), "Mumbai LISS-IV | Synthetic Cloud Pilot (illustrative)", fill="black")
    draw.text((margin, 29), "Original    |    Simulated cloudy input    |    Synthetic cloud mask", fill="black")

    stats = []
    for index, record in enumerate(chosen, 1):
        image_path = ROOT / record["image_path"]
        valid_path = ROOT / record["valid_mask_path"]
        clean = np.load(image_path, allow_pickle=False)
        valid = np.load(valid_path, allow_pickle=False).astype(bool)
        if clean.shape != (256, 256, 3) or clean.dtype != np.uint16:
            raise ValueError(f"Unexpected saved data: {record['patch_id']}")
        cloudy, alpha, binary = synthetic_clouds(clean, valid, rng)
        cloud_share = float(binary[valid].mean())

        mask_rgb = np.repeat((binary[..., None] * 255).astype(np.uint8), 3, axis=2)
        panels = (
            false_color(clean, valid),
            false_color(cloudy, valid),
            mask_rgb
        )
        y = header + (index - 1) * (row_h + margin)
        for col, panel in enumerate(panels):
            x = margin + col * (tile + margin)
            preview = Image.fromarray(panel, mode="RGB")
            canvas.paste(preview.resize((tile, tile), Image.Resampling.BILINEAR), (x, y))
            caption = (
                f"{record['patch_id']} | {record['surface_proxy']}"
                if col == 0
                else (f"Cloud coverage: {cloud_share:.1%}" if col == 1 else "White = synthetic cloud")
            )
            draw.text((x, y + tile + 5), caption, fill="black")

        if index == 1:
            # A single machine-readable example to verify downstream conventions.
            np.savez_compressed(
                OUT / "pilot_example.npz",
                clean_dn=clean,
                cloudy_dn=cloudy,
                cloud_alpha=alpha,
                cloud_binary=binary,
                valid_mask=valid.astype(np.uint8)
            )
        stats.append((record["patch_id"], cloud_share, record["surface_proxy"]))
        print(f"Generated {index}/{len(chosen)}: {record['patch_id']} | cloud={cloud_share:.1%}", flush=True)

    preview_path = OUT / "synthetic_cloud_comparison.png"
    canvas.save(preview_path)
    print("\nSYNTHETIC CLOUD PILOT COMPLETE", flush=True)
    print(f"Preview: {preview_path}", flush=True)
    print(f"Example arrays: {OUT / 'pilot_example.npz'}", flush=True)
    print("Note: Approximate DN-space simulation, not physical cloud radiative transfer.")
    print("Note: Source targets are visually approved for the prototype, not scientifically certified cloud-free.")


if __name__ == "__main__":
    main()
