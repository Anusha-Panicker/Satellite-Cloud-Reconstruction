from pathlib import Path
from collections import Counter
import csv

import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
processed = root / 'data' / 'processed' / 'cloud_reconstruction_v1'
patch_root = processed / 'patches'
mask_root = processed / 'valid_masks'
metadata_dir = processed / 'metadata'
preview_dir = processed / 'previews'
qc_csv = root / 'outputs' / 'large_patch_dataset' / 'quality_review' / 'patch_quality_audit.csv'
raw_dir = root / 'data' / 'raw' / 'mumbai_scenes' / '2025-04-16'

metadata_dir.mkdir(parents=True, exist_ok=True)
preview_dir.mkdir(parents=True, exist_ok=True)

with open(qc_csv, newline='', encoding='utf-8') as f:
    records = list(csv.DictReader(f))

expected = {'train': 500, 'val': 75, 'test': 75}
if len(records) != 650 or Counter(r['split'] for r in records) != expected:
    raise ValueError('QC catalog must contain 500 train, 75 val, and 75 test records.')
if len({r['patch_id'] for r in records}) != len(records):
    raise ValueError('Duplicate patch IDs in QC catalog.')

existing_images = list(patch_root.glob('*/*.npy'))
existing_masks = list(mask_root.glob('*/*.npy'))
print(f'Found {len(existing_images)} image arrays and {len(existing_masks)} masks.', flush=True)
if len(existing_images) != 650 or len(existing_masks) != 650:
    raise ValueError('Expected 650 images and 650 masks. Do not finalize until counts match.')

matches = list(raw_dir.rglob('BAND4.tif'))
if len(matches) != 1:
    raise ValueError('Expected exactly one April 2025 BAND4.tif for spatial metadata.')

manifest = []
with rasterio.open(matches[0]) as ds:
    for idx, record in enumerate(records, start=1):
        patch_id = record['patch_id']
        split = record['split']
        col = int(record['col'])
        row = int(record['row'])
        size = 256

        image_path = patch_root / split / f'{patch_id}.npy'
        mask_path = mask_root / split / f'{patch_id}_valid.npy'
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f'Missing image/mask for {patch_id}')

        image = np.load(image_path, mmap_mode='r', allow_pickle=False)
        mask = np.load(mask_path, mmap_mode='r', allow_pickle=False)
        if image.shape != (size, size, 3) or image.dtype != np.uint16:
            raise ValueError(f'Unexpected image shape or type: {patch_id}')
        if mask.shape != (size, size) or mask.dtype not in (np.uint8, np.bool_):
            raise ValueError(f'Unexpected mask shape or type: {patch_id}')
        if image.max() > 1023:
            raise ValueError(f'Invalid 10-bit DN range in {patch_id}')
        valid_fraction = float(np.mean(mask > 0))
        if valid_fraction < 0.995:
            raise ValueError(f'Too many invalid pixels in {patch_id}')
        if not (0 <= col <= ds.width - size and 0 <= row <= ds.height - size):
            raise ValueError(f'Patch is outside the April scene: {patch_id}')

        left, bottom, right, top = window_bounds(
            Window(col, row, size, size), ds.transform
        )
        manifest.append({
            'patch_id': patch_id,
            'split': split,
            'col': col,
            'row': row,
            'size': size,
            'bands': 'BAND4_NIR,BAND3_Red,BAND2_Green',
            'source_scene': matches[0].parent.name,
            'crs': str(ds.crs),
            'left': left,
            'bottom': bottom,
            'right': right,
            'top': top,
            'valid_fraction': round(valid_fraction, 6),
            'surface_proxy': record.get('surface_proxy', ''),
            'qc_priority': record.get('qc_priority', ''),
            'quality_status': record.get('quality_status', 'pending_manual_review'),
            'image_path': image_path.relative_to(root).as_posix(),
            'valid_mask_path': mask_path.relative_to(root).as_posix(),
            'dn_normalization_divisor': 1023,
        })
        if idx == 1 or idx % 100 == 0:
            print(f'Validated {idx}/650 saved patches.', flush=True)

manifest_path = metadata_dir / 'patch_manifest.csv'
with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
    writer.writeheader()
    writer.writerows(manifest)
print(f'Saved manifest: {manifest_path}', flush=True)

rng = np.random.default_rng(2026)
review = []
for split, count in (('train', 15), ('val', 5), ('test', 5)):
    group = [r for r in manifest if r['split'] == split]
    choice = rng.choice(len(group), size=count, replace=False)
    review.extend(group[int(i)] for i in choice)

cols, rows, tile = 5, 5, 180
cell_w, cell_h, margin, header = 196, 210, 16, 36
canvas = Image.new('RGB', (2 * margin + cols * cell_w,
                           header + margin + rows * cell_h), 'white')
draw = ImageDraw.Draw(canvas)
draw.text((margin, 12), 'April 2025 - Saved patch verification', fill='black')
low = np.array([45, 50, 60], dtype=np.float32)
high = np.array([280, 210, 200], dtype=np.float32)
resample = Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR
for i, record in enumerate(review):
    image = np.load(root / record['image_path'], allow_pickle=False)
    valid = np.load(root / record['valid_mask_path'], allow_pickle=False).astype(bool)
    rgb = np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)
    rgb[~valid] = 0
    rgb = (rgb * 255).astype(np.uint8)
    thumb = Image.fromarray(rgb).resize((tile, tile), resample)
    r, c = divmod(i, cols)
    x, y = margin + c * cell_w, header + r * cell_h
    canvas.paste(thumb, (x, y))
    draw.text((x, y + tile + 5), f"{record['split']} | {record['patch_id']}", fill='black')

preview_path = preview_dir / 'extracted_patch_overview.png'
canvas.save(preview_path)
print(f'Saved preview: {preview_path}', flush=True)
print('\nDATASET FINALIZATION COMPLETE')
print('Split counts:', dict(Counter(r['split'] for r in manifest)))
print('Quality status counts:', dict(Counter(r['quality_status'] for r in manifest)))
print('Original DN arrays have not been modified.')
