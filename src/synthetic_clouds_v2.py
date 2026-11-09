"""Six-patch synthetic cloud pilot v2 for approved Mumbai LISS-IV targets.

Reads saved [NIR, Red, Green] 10-bit DN patches from approved_target_manifest.csv.
Creates reproducible multi-opacity cloud simulations, corresponding alpha/binary
masks and simple adjacent cloud shadows. This is an illustrative, uncalibrated
DN-space augmentation and does not reproduce atmospheric radiative transfer.

Runs only on SIX APPROVED TRAINING patches, preserving originals and existing v1
pilot files. Requires NumPy and Pillow.
"""

from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "cloud_reconstruction_v1"
MANIFEST = DATASET / "metadata" / "approved_target_manifest.csv"
OUT = ROOT / "outputs" / "synthetic_cloud_pilot_v2"

PILOT_IDS = ("C0016", "C0590", "C0443", "C0125", "C0414", "C0263")
MAX_DN = 1023.0
LOW = np.array([45.0, 50.0, 60.0], dtype=np.float32)
HIGH = np.array([280.0, 210.0, 200.0], dtype=np.float32)


def smooth_noise(rng, height, width, grids, blur=0):
    """Bicubic upsampling of a small random field, with optional smoothing."""
    gh, gw = grids
    cells = (rng.random((gh, gw)) * 255).astype(np.uint8)
    picture = Image.fromarray(cells, mode="L").resize((width, height), Image.Resampling.BICUBIC)
    if blur:
        picture = picture.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(picture, dtype=np.float32) / 255.0


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def shift_without_wrap(array, dy, dx):
    """Shift a 2D array and fill newly exposed pixels with zeros."""
    h, w = array.shape
    out = np.zeros_like(array)
    y0 = max(0, -dy)
    y1 = min(h, h - dy)
    x0 = max(0, -dx)
    x1 = min(w, w - dx)
    if y1 > y0 and x1 > x0:
        out[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = array[y0:y1, x0:x1]
    return out


def smooth_float(array, radius):
    return np.asarray(
        Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8), mode="L")
        .filter(ImageFilter.GaussianBlur(radius)),
        dtype=np.float32
    ) / 255.0


def synthesize(clear_dn, valid, rng):
    h, w, c = clear_dn.shape
    if (h, w, c) != (256, 256, 3):
        raise ValueError(f"Expected 256x256x3, got {clear_dn.shape}")
    if valid.shape != (h, w):
        raise ValueError("Validity-mask shape mismatch")
    if not np.any(valid):
        raise ValueError("No valid pixels in patch")
    if np.min(clear_dn) < 0 or np.max(clear_dn) > MAX_DN:
        raise ValueError("Patch contains DN outside 10-bit range")

    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    field = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(3, 7))):
        cx = rng.uniform(-0.2 * w, 1.2 * w)
        cy = rng.uniform(-0.2 * h, 1.2 * h)
        rx, ry = rng.uniform(30, 98, 2)
        angle = rng.uniform(0, np.pi)
        vx = np.cos(angle) * (xx - cx) + np.sin(angle) * (yy - cy)
        vy = -np.sin(angle) * (xx - cx) + np.cos(angle) * (yy - cy)
        blob = np.exp(-0.5 * ((vx / rx) ** 2 + (vy / ry) ** 2))
        field = np.maximum(field, blob.astype(np.float32))

    field = 0.72 * field + 0.22 * smooth_noise(rng, h, w, (10, 10), blur=4) + 0.06 * smooth_noise(rng, h, w, (36, 36), blur=2)
    field = (field - np.percentile(field[valid], 2)) / max(
        np.percentile(field[valid], 98) - np.percentile(field[valid], 2), 1e-6
    )
    field = np.clip(field, 0, 1)

    target_cover = float(rng.uniform(0.15, 0.38))
    threshold = float(np.quantile(field[valid], 1.0 - target_cover))
    # Broad semi-transparent edges and denser cores, not a hard-edged white stencil.
    feather = float(rng.uniform(0.17, 0.27))
    core = smoothstep((field - threshold + feather / 2) / feather)
    texture = 0.73 + 0.27 * smooth_noise(rng, h, w, (25, 25), blur=2)
    peak_alpha = float(rng.uniform(0.64, 0.90))
    alpha = np.clip(peak_alpha * core * texture, 0, 0.92)
    alpha[~valid] = 0.0

    # Simple displaced, smoothed cloud shadow adjacent to cloud bodies.
    dy = int(rng.integers(5, 14)) * (1 if rng.random() > .5 else -1)
    dx = int(rng.integers(5, 14)) * (1 if rng.random() > .5 else -1)
    shifted = smooth_float(shift_without_wrap(core, dy, dx), radius=5)
    shadow = np.clip(shifted - 0.7 * core, 0, 1)
    shadow = np.clip(shadow * float(rng.uniform(0.10, 0.23)), 0, 0.23)
    shadow[~valid] = 0.0

    # Approximate 10-bit cloud DN values, not radiometrically calibrated reflectance.
    cloud_rgb = np.array([
        rng.uniform(530, 700),  # NIR
        rng.uniform(630, 820),  # Red
        rng.uniform(650, 840),  # Green
    ], dtype=np.float32)
    cloud_texture = 0.87 + 0.20 * smooth_noise(rng, h, w, (32, 32), blur=1)
    cloud_brightness = cloud_rgb[None, None, :] * cloud_texture[..., None]

    clean = clear_dn.astype(np.float32)
    shaded = clean * (1.0 - shadow[..., None])
    cloudy = shaded * (1.0 - alpha[..., None]) + cloud_brightness * alpha[..., None]
    cloudy = np.clip(cloudy, 0, MAX_DN).astype(np.float32)
    cloudy[~valid] = clean[~valid]

    binary = (alpha >= 0.30) & valid
    coverage = float(binary.sum() / valid.sum())
    return cloudy, alpha.astype(np.float32), binary.astype(np.uint8), shadow.astype(np.float32), coverage


def false_color(dn, valid):
    """Same highlight-preserving tone curve for clean and cloudy previews.

    Pure linear DN stretching clipped the synthetic clouds to white; this
    compression reveals simulated cloud brightness differences. Display only.
    """
    x = np.maximum((dn.astype(np.float32) - LOW) / (HIGH - LOW), 0.0)
    view = np.power(x / (x + 1.1), 0.8)
    view = np.clip(view, 0, 1)
    view[~valid] = 0
    return (view * 255).astype(np.uint8)


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    with MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 649:
        raise ValueError(f"Expected 649 approved targets, got {len(rows)}")
    by_id = {row["patch_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate patch IDs")
    if "C0137" in by_id:
        raise ValueError("Uncertain C0137 present in approved manifest")

    OUT.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, patch_id in enumerate(PILOT_IDS, 1):
        if patch_id not in by_id:
            raise ValueError(f"Pilot patch missing from approved manifest: {patch_id}")
        record = by_id[patch_id]
        if record["split"] != "train" or record["manual_status"] != "approved":
            raise ValueError(f"Pilot patch not approved for training: {patch_id}")
        # Paths in extraction manifest are relative to project root.
        clear_path = ROOT / record["image_path"]
        mask_path = ROOT / record["valid_mask_path"]
        clear = np.load(clear_path, allow_pickle=False)
        valid = np.load(mask_path, allow_pickle=False).astype(bool)
        rng = np.random.default_rng(33007 + int(patch_id[1:]))
        cloudy, alpha, binary, shadow, coverage = synthesize(clear, valid, rng)
        samples.append((record, clear, cloudy, alpha, binary, shadow, valid, coverage))
        if i == 1:
            np.savez_compressed(
                OUT / "pilot_example_v2.npz", clear_dn=clear, cloudy_dn=cloudy,
                cloud_alpha=alpha, cloud_binary=binary,
                cloud_shadow=shadow, valid_mask=valid.astype(np.uint8),
            )
        print(f"Generated {i}/{len(PILOT_IDS)}: {patch_id} | cloud={coverage:.1%}", flush=True)

    tile = 245
    heading = 55
    lineheight = 275
    gap = 12
    columns = 4
    canvas = Image.new("RGB", (columns * tile + (columns + 1) * gap, heading + len(samples) * lineheight), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), "Mumbai LISS-IV | Improved synthetic cloud pilot v2", fill="black")
    draw.text((gap, 29), "Clear target | Simulated cloudy | Cloud alpha | Binary cloud mask", fill="black")
    for row_index, (record, clear, cloudy, alpha, binary, shadow, valid, coverage) in enumerate(samples):
        panels = [
            false_color(clear, valid),
            false_color(cloudy, valid),
            np.repeat((alpha[..., None] * 255).astype(np.uint8), 3, axis=2),
            np.repeat((binary[..., None] * 255).astype(np.uint8), 3, axis=2),
        ]
        y = heading + row_index * lineheight
        for col, frame in enumerate(panels):
            x = gap + col * (tile + gap)
            picture = Image.fromarray(frame).resize((tile, tile), Image.Resampling.BILINEAR)
            canvas.paste(picture, (x, y))
            caption = (
                f"{record['patch_id']} | {record['surface_proxy']}" if col == 0
                else f"Cloud fraction: {coverage:.1%}" if col == 1
                else "Grey = partial opacity" if col == 2 else "White = cloud (alpha>=0.30)"
            )
            draw.text((x, y + tile + 4), caption, fill="black")

    preview = OUT / "synthetic_cloud_comparison_v2.png"
    canvas.save(preview)
    print("\nSYNTHETIC CLOUD PILOT V2 COMPLETE")
    print(f"Preview: {preview}")
    print(f"Example arrays: {OUT / 'pilot_example_v2.npz'}")
    print("DN-space augmentation only; cloud physics and real-scene performance are unvalidated.")


if __name__ == "__main__":
    main()
