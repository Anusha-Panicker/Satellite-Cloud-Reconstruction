"""Mumbai April/June UNREGISTERED temporal-reference pilot (read-only).

A visualization to decide if April contains useful surface information where June
appears cloudy. This is NOT validated cloud removal or registration.

Uses existing project scripts to read 256x256 patches on the declared June grid.
NEVER applies unverified band/inter-date shifts and never writes GeoTIFFs.
Cloud core and April quality are DN-based PROXIES, not ground truth.

From project root:
  .\\.venv\\Scripts\\python.exe -u .\\src\\preview_real_temporal_transfer_unregistered.py --self-test
  .\\.venv\\Scripts\\python.exe -u .\\src\\preview_real_temporal_transfer_unregistered.py

Dependencies: numpy, rasterio, pillow and the previous project scripts; no pandas.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.transform import from_origin

from audit_april_june_temporal_reference import ROOT, preview_rgb
from review_april_june_cloud_and_alignment import candidate_masks, display_auto
from verify_april_june_landmark_registration import load_tile

QC_CSV = ROOT / 'outputs/april_june_targeted_qc/sampled_cloud_reference_qc.csv'
DEFAULT_OUT = ROOT / 'outputs/april_june_unregistered_temporal_pilot'
NOTICE = (
    'UNREGISTERED / UNCERTAIN: images only share declared geographic coordinates. '
    'No within-scene or between-date pixel shifts are approved. Masks are unvalidated '
    'DN screening proxies, not real cloud truth. April pixels are 48 days earlier and '
    'are not June ground truth. A visual transfer is not a scientifically validated '
    'cloud-free reconstruction. DO NOT use this as a finished GeoTIFF or assessment of model accuracy.'
)


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--qc-csv', type=Path, default=QC_CSV)
    p.add_argument('--april-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-04-16')
    p.add_argument('--june-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-06-03')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--tiles', type=str, default='T105,T106,T118,T012,T041,T058',
                   help='Comma separated IDs from the existing targeted QC report')
    p.add_argument('--tile-size', type=int, default=256)
    return p.parse_args()


def qc_records(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'QC manifest missing: {path}\nRun review_april_june_cloud_and_alignment.py first.')
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        records = list(csv.DictReader(fh))
    out = {}
    for rec in records:
        tid = rec.get('tile_id', '').upper().strip()
        if not tid or tid in out:
            raise ValueError(f'Empty/duplicate tile ID in QC CSV: {tid}')
        for col in ('june_col', 'june_row'):
            if col not in rec or int(rec[col]) < 0:
                raise ValueError(f'Invalid {col} for {tid}')
        out[tid] = rec
    return out


def make_candidate(a: np.ndarray, av: np.ndarray, j: np.ndarray, jv: np.ndarray) -> dict:
    if a.shape != j.shape or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError('Expected matching HWC (NIR, Red, Green) arrays')
    if av.shape != a.shape[:2] or jv.shape != av.shape:
        raise ValueError('Invalid source validity-mask dimensions')
    masks = candidate_masks(a, j, av, jv)
    candidate = masks['core'] & masks['april_ref']
    # The intentionally unregistered, uncalibrated demonstration is restricted
    # to pixels satisfying BOTH heuristics. Do not use a warped/band-shifted array.
    demo = j.copy()
    demo[candidate] = a[candidate]
    safe = masks['valid']
    return {'masks': masks, 'selected': candidate, 'demo': demo,
            'selected_fraction_valid': float(candidate.sum() / max(int(safe.sum()), 1)),
            'strict_core_fraction_valid': float(masks['core'].sum() / max(int(safe.sum()), 1)),
            'april_ref_fraction_valid': float(masks['april_ref'].sum() / max(int(safe.sum()), 1))}


def mask_rgb(mask, valid):
    res = np.full((*mask.shape, 3), 255, np.uint8)
    res[valid] = [12, 12, 12]
    res[mask] = [255, 255, 255]
    return Image.fromarray(res, 'RGB')


def overlay_rgb(j, valid, core, selected):
    arr = np.asarray(display_auto(j, valid)).astype(np.float32).copy()
    # Orange: strict cloud core proxy; green: April reference ALSO appears usable.
    arr[core] = .45 * arr[core] + .55 * np.array([255, 128, 10])
    arr[selected] = .4 * arr[selected] + .6 * np.array([40, 240, 90])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGB')


def draw_preview(rows, dest):
    if not rows:
        raise RuntimeError('No eligible tiles to preview')
    gap, size = 10, 220
    cols = ['April fixed', 'June fixed', 'April auto*', 'June auto*',
            'Proxy overlay', 'Potential April transfer**', 'Selected proxy mask']
    top, rh = 72, size + 46
    w = gap + len(cols) * (size + gap)
    h = top + rh * len(rows) + 25
    out = Image.new('RGB', (w, h), 'white')
    dr = ImageDraw.Draw(out)
    dr.text((10, 4), 'APRIL / JUNE | UNREGISTERED TEMPORAL-REFERENCE FEASIBILITY PREVIEW', fill='black')
    dr.text((10, 21), '*Auto stretch for display only. **Masked April replacement is a candidate illustration, NOT cloud removal.', fill='black')
    dr.text((10, 38), 'Overlay: orange = strict bright proxy, green = selected + April reference proxy. No shifts applied.', fill='black')
    for k, c in enumerate(cols):
        dr.text((gap + k * (size + gap), 58), c, fill='black')
    for i, row in enumerate(rows):
        a, av, j, jv, result = row['a'], row['av'], row['j'], row['jv'], row['result']
        masks = result['masks']
        # Fixed scaling on both dates and auto scaling shown side by side.
        images = [preview_rgb(a, av), preview_rgb(j, jv), display_auto(a, av),
                  display_auto(j, jv), overlay_rgb(j, jv, masks['core'], result['selected']),
                  preview_rgb(result['demo'], masks['valid']), mask_rgb(result['selected'], masks['valid'])]
        y = top + i * rh
        for k, image in enumerate(images):
            if image.size != (size, size):
                image = image.resize((size, size), Image.Resampling.NEAREST)
            out.paste(image, (gap + k * (size + gap), y))
        dr.text((gap, y + size + 5),
                f"{row['tile_id']} | core {result['strict_core_fraction_valid']:.1%} | "
                f"selected {result['selected_fraction_valid']:.1%} | "
                f"April-ref proxy {result['april_ref_fraction_valid']:.1%} | UNREGISTERED",
                fill='black')
    dr.text((10, h - 18), 'NO TRAINING, NO ACCURACY METRICS, NO GEOTIFF WRITES, NO ACCEPTED REGISTRATION.', fill='black')
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest)


def run(opts):
    if opts.tile_size < 32 or opts.tile_size > 1024:
        raise ValueError('tile-size must be in [32,1024]')
    lookup = qc_records(opts.qc_csv)
    tile_ids = [t.strip().upper() for t in opts.tiles.split(',') if t.strip()]
    if not tile_ids or len(tile_ids) != len(set(tile_ids)):
        raise ValueError('Provide distinct tile IDs in --tiles')
    missing = [tid for tid in tile_ids if tid not in lookup]
    if missing:
        raise ValueError(f'Unknown tiles: {missing}; available: {len(lookup)}')

    rows = []
    for tid in tile_ids:
        rec = lookup[tid]
        a, av, j, jv, px = load_tile(opts, rec)
        result = make_candidate(a, av, j, jv)
        r = {'tile_id': tid, 'qc_group': rec.get('qc_group', ''),
             'june_col': int(rec['june_col']), 'june_row': int(rec['june_row']),
             'pixel_spacing_m': px, 'geographic_grid': 'June declared grid',
             'valid_overlap_fraction': float(result['masks']['valid'].mean()),
             'strict_core_proxy_fraction_valid': result['strict_core_fraction_valid'],
             'april_reference_proxy_fraction_valid': result['april_ref_fraction_valid'],
             'potential_transfer_fraction_valid': result['selected_fraction_valid'],
             'april_june_registration_approved': False,
             'validated_cloud_mask': False,
             'claim': 'unregistered_illustrative_transfer_only'}
        rows.append({'tile_id': tid, 'a': a, 'av': av, 'j': j, 'jv': jv,
                     'result': result, 'report': r})
        print(f"{tid}: valid {r['valid_overlap_fraction']:.1%} | strict core {r['strict_core_proxy_fraction_valid']:.1%} | "
              f"selected for possible transfer {r['potential_transfer_fraction_valid']:.1%} | registration UNVERIFIED", flush=True)
    opts.out.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]['report'])
    csvpath = opts.out / 'unregistered_temporal_candidate_tiles.csv'
    with csvpath.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row['report'])
    preview = opts.out / 'unregistered_temporal_candidate_preview.png'
    draw_preview(rows, preview)
    metadata = {
        'purpose': 'Real-data source suitability visualization, NOT a registered reconstruction',
        'notice': NOTICE,
        'source_april_date': '2025-04-16', 'source_june_date': '2025-06-03',
        'source_grid': 'Geographic June grid using temporary WarpedVRT for April',
        'no_band_shift': True, 'no_inter_date_pixel_shift': True,
        'no_geotiff_output': True, 'no_model_or_test_access': True,
        'selection_method': 'strict_cloud_core_proxy AND april_reference_proxy',
        'source_qc_manifest': str(opts.qc_csv),
        'tiles': [r['report'] for r in rows]
    }
    js = opts.out / 'unregistered_temporal_candidate_provenance.json'
    js.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (opts.out / 'READ_ME_UNREGISTERED.txt').write_text(NOTICE + '\n', encoding='utf-8')
    print('\nUNREGISTERED TEMPORAL SOURCE PILOT COMPLETE — NO RECONSTRUCTION APPROVED')
    print('QC CSV:', csvpath)
    print('Visual:', preview)
    print('Provenance:', js)
    print(NOTICE)
    return rows


def self_test():
    rng = np.random.default_rng(181)
    h = w = 96
    yy, xx = np.indices((h, w))
    ref = (110 + 15 * np.sin(xx / 9) + 9 * np.cos(yy / 13) + rng.normal(0, 2, (h, w)))
    a = np.stack([ref + 45, ref - 18, ref - 5], axis=-1).astype(np.float32)
    j = a.copy()
    j[20:78, 20:82] = [410, 340, 330]
    valid = np.ones((h, w), bool)
    result = make_candidate(a, valid, j, valid)
    assert result['selected'][30:60, 30:60].all()
    assert not result['selected'][0:12, 0:12].any()
    assert np.allclose(result['demo'][result['selected']], a[result['selected']])
    assert np.allclose(result['demo'][~result['selected']], j[~result['selected']])
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name, dat in [('apr', a), ('jun', j)]:
            folder = root / name
            folder.mkdir()
            for idx, band in enumerate((4, 3, 2)):
                with rasterio.open(folder / f'BAND{band}.tif', 'w',
                                   driver='GTiff', width=w, height=h, count=1, dtype='uint16',
                                   crs='EPSG:32643', transform=from_origin(500000, 2100000, 5, 5),
                                   nodata=0) as ds:
                    ds.write(np.uint16(np.clip(dat[..., idx], 1, 1023)), 1)
        qc = root / 'qc.csv'
        with qc.open('w', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=['tile_id', 'june_col', 'june_row', 'qc_group'])
            writer.writeheader()
            writer.writerow({'tile_id': 'T001', 'june_col': 0, 'june_row': 0, 'qc_group': 'cloud_candidate'})
        opts = SimpleNamespace(qc_csv=qc, april_dir=root / 'apr', june_dir=root / 'jun',
                               out=root / 'out', tiles='T001', tile_size=96)
        out_rows = run(opts)
        assert len(out_rows) == 1
        assert out_rows[0]['result']['selected_fraction_valid'] > .2
        for file in ('unregistered_temporal_candidate_tiles.csv',
                     'unregistered_temporal_candidate_preview.png',
                     'unregistered_temporal_candidate_provenance.json', 'READ_ME_UNREGISTERED.txt'):
            assert (opts.out / file).is_file(), file
    print('\nUNREGISTERED TEMPORAL PILOT SELF-TEST PASSED')
    print('Only temporary synthetic rasters were used; no project data touched.')


if __name__ == '__main__':
    args = get_args()
    if args.self_test:
        self_test()
    else:
        run(args)
