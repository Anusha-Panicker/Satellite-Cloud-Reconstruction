
from pathlib import Path
from contextlib import ExitStack
from collections import Counter
import csv

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
raw_dir = root / 'data' / 'raw' / 'mumbai_scenes'
catalog_dir = root / 'outputs' / 'large_patch_dataset'
out_dir = catalog_dir / 'quality_review'
out_dir.mkdir(parents=True, exist_ok=True)

catalog_path = catalog_dir / 'balanced_candidate_catalog.csv'
audit_path = out_dir / 'patch_quality_audit.csv'

with open(catalog_path, newline='', encoding='utf-8') as file:
    records = list(csv.DictReader(file))

if not records:
    raise ValueError('Balanced candidate catalog is empty')

matches = list((raw_dir / '2025-04-16').rglob('BAND2.tif'))

if len(matches) != 1:
    raise ValueError('Expected exactly one April satellite scene')

scene_dir = matches[0].parent

size = 256
rng = np.random.default_rng(42)

display_low = np.array([45, 50, 60], dtype=np.float32)
display_high = np.array([280, 210, 200], dtype=np.float32)

needed_columns = {
    'measured_valid_fraction',
    'bright_fraction',
    'very_bright_fraction',
    'visible_p99_dn',
    'saturated_fraction',
    'qc_priority',
    'quality_status'
}

reuse_audit = False

if audit_path.exists():
    with open(audit_path, newline='', encoding='utf-8') as file:
        saved = list(csv.DictReader(file))

    keys = ('patch_id', 'col', 'row', 'split', 'surface_proxy')

    reuse_audit = (
        len(saved) == len(records)
        and bool(saved)
        and needed_columns.issubset(saved[0].keys())
        and all(
            tuple(a.get(k) for k in keys)
            == tuple(b.get(k) for k in keys)
            for a, b in zip(records, saved)
        )
    )

    if reuse_audit:
        records = saved
        print(
            f'Reusing completed quality audit: {len(records)} patches',
            flush=True
        )

with rasterio.Env(GDAL_CACHEMAX=64):
    with ExitStack() as stack:

        datasets = [
            stack.enter_context(
                rasterio.open(scene_dir / name)
            )
            for name in (
                'BAND4.tif',
                'BAND3.tif',
                'BAND2.tif'
            )
        ]

        ref = datasets[0]

        for ds in datasets[1:]:
            if (
                ds.crs != ref.crs
                or ds.transform != ref.transform
                or ds.width != ref.width
                or ds.height != ref.height
            ):
                raise ValueError('Satellite band grid mismatch')

        def load_patch(record):
            col = int(record['col'])
            row = int(record['row'])

            if (
                col < 0
                or row < 0
                or col + size > ref.width
                or row + size > ref.height
            ):
                raise ValueError(
                    f"Patch outside scene: {record['patch_id']}"
                )

            window = Window(col, row, size, size)

            image = np.stack([
                ds.read(1, window=window)
                for ds in datasets
            ], axis=-1).astype(np.float32)

            mask_ok = np.logical_and.reduce([
                ds.read_masks(1, window=window) > 0
                for ds in datasets
            ])

            valid = mask_ok & np.all(image > 0, axis=-1)

            return image, valid

        if not reuse_audit:

            print(
                f'Starting quality audit: {len(records)} patches',
                flush=True
            )

            for idx, record in enumerate(records, start=1):

                image, valid = load_patch(record)

                if not valid.any():
                    raise ValueError(
                        f"No valid pixels: {record['patch_id']}"
                    )

                nir = image[..., 0]
                red = image[..., 1]
                green = image[..., 2]

                visible = (red + green) / 2.0

                bright = (
                    valid
                    & (nir >= 220)
                    & (red >= 220)
                    & (green >= 220)
                )

                very_bright = (
                    valid
                    & (nir >= 350)
                    & (red >= 300)
                    & (green >= 300)
                )

                saturated = (
                    valid
                    & np.any(image >= 1020, axis=-1)
                )

                record['measured_valid_fraction'] = round(
                    float(valid.mean()), 5
                )

                record['bright_fraction'] = round(
                    float(bright.mean()), 5
                )

                record['very_bright_fraction'] = round(
                    float(very_bright.mean()), 5
                )

                record['visible_p99_dn'] = round(
                    float(np.percentile(visible[valid], 99)), 2
                )

                record['saturated_fraction'] = round(
                    float(saturated.mean()), 5
                )

                needs_review = (
                    float(record['bright_fraction']) >= 0.01
                    or float(record['very_bright_fraction']) >= 0.001
                    or float(record['visible_p99_dn']) >= 230
                )

                record['qc_priority'] = (
                    'high_brightness_review'
                    if needs_review
                    else 'routine_visual_review'
                )

                record['quality_status'] = 'pending_manual_review'

                if idx == 1 or idx % 25 == 0:
                    print(
                        f'Processed {idx}/{len(records)} patches',
                        flush=True
                    )

            with open(
                audit_path, 'w', newline='', encoding='utf-8'
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(records[0].keys())
                )
                writer.writeheader()
                writer.writerows(records)

            print(f'Saved CSV: {audit_path}', flush=True)

        def save_sheet(group, filename, heading):

            print(
                f'Creating image: {filename}',
                flush=True
            )

            columns = 5
            rows = 5
            tile = 190
            cell_width = 210
            cell_height = 225
            margin = 16
            header = 48

            canvas = Image.new(
                'RGB',
                (
                    2 * margin + columns * cell_width,
                    header + margin + rows * cell_height
                ),
                'white'
            )

            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()

            resample = (
                Image.Resampling.BILINEAR
                if hasattr(Image, 'Resampling')
                else Image.BILINEAR
            )

            for idx, record in enumerate(group[:25]):

                image, valid = load_patch(record)

                preview = np.clip(
                    (image - display_low)
                    / (display_high - display_low),
                    0,
                    1
                )

                preview[~valid] = 0

                rgb = np.round(
                    preview * 255
                ).astype(np.uint8)

                thumb = Image.fromarray(
                    rgb, mode='RGB'
                ).resize(
                    (tile, tile),
                    resample
                )

                r, c = divmod(idx, columns)

                x = margin + c * cell_width
                y = header + r * cell_height

                canvas.paste(thumb, (x, y))

                draw.text(
                    (x, y + tile + 5),
                    f"{record['patch_id']} | "
                    f"bright {float(record['bright_fraction']):.1%}",
                    fill='black',
                    font=font
                )

                if idx == 0 or (idx + 1) % 5 == 0:
                    print(
                        f'Rendered {idx + 1}/'
                        f'{min(len(group), 25)} patches',
                        flush=True
                    )

            path = out_dir / filename

            canvas.save(path, format='PNG')

            print(f'Saved: {path}', flush=True)

        ranked = sorted(
            records,
            key=lambda r: (
                float(r['bright_fraction']),
                float(r['very_bright_fraction']),
                float(r['visible_p99_dn'])
            ),
            reverse=True
        )

        save_sheet(
            ranked[:25],
            'high_brightness_review.png',
            'Highest brightness candidates (not confirmed clouds)'
        )

        for split in ('train', 'val', 'test'):

            subset = [
                r for r in records
                if r['split'] == split
            ]

            water = [
                r for r in subset
                if r['surface_proxy'] == 'water_like'
            ]

            land = [
                r for r in subset
                if r['surface_proxy'] != 'water_like'
            ]

            rng.shuffle(water)
            rng.shuffle(land)

            sample = water[:6] + land[:19]
            rng.shuffle(sample)

            save_sheet(
                sample,
                f'{split}_sample_review.png',
                f'{split.title()} set: representative candidates'
            )

print('\nQUALITY AUDIT FINISHED', flush=True)
print(f'Total audited: {len(records)}')

print(
    'Priority counts:',
    dict(Counter(r['qc_priority'] for r in records))
)

print('All quality statuses are pending_manual_review')
print(f'CSV: {audit_path}')
