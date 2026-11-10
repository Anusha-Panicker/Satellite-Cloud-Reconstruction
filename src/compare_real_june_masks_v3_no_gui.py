r"""Headless (NO WINDOW) cloud-mask sensitivity check for the frozen Mumbai V3 model.

Place this file in Cloud_Reconstruction/src/. Run from the project root:
  .\.venv\Scripts\python.exe -u .\src\compare_real_june_masks_v3_no_gui.py --self-test
  .\.venv\Scripts\python.exe -u .\src\compare_real_june_masks_v3_no_gui.py

Creates PNGs, a CSV, and JSON for June-only mask variants on T041 / T058.
Not a manual mask review, cloud-truth evaluation, or validated cloud removal.
Original GeoTIFFs and trained model files are never modified. No GUI, matplotlib,
OpenCV, pandas, training, April transfer, scene shifts, or GeoTIFF output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from rasterio.transform import from_origin

from audit_april_june_temporal_reference import ROOT, preview_rgb
from preview_real_temporal_transfer_unregistered import qc_records, mask_rgb
from verify_april_june_landmark_registration import load_tile
from review_april_june_cloud_and_alignment import display_auto
from preview_real_june_unet_v3 import june_only_cloud_proxy, predict
from train_unet_residual_v3 import BEST, ResidualCloudUNet, load_checkpoint

OUT = ROOT / 'outputs' / 'real_june_mask_sensitivity_v3'
QC_CSV = ROOT / 'outputs' / 'april_june_targeted_qc' / 'sampled_cloud_reference_qc.csv'


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--tiles', default='T041,T058')
    p.add_argument('--qc-csv', type=Path, default=QC_CSV)
    p.add_argument('--april-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-04-16')
    p.add_argument('--june-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-06-03')
    p.add_argument('--checkpoint', type=Path, default=BEST)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--tile-size', type=int, default=256)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return p.parse_args()


def dilate(mask: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    if mask.shape != valid.shape or mask.ndim != 2 or radius < 0:
        raise ValueError('Invalid binary mask / valid mask / radius')
    if radius == 0:
        return mask.astype(bool) & valid
    src = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))[None, None]
    expanded = F.max_pool2d(src, kernel_size=radius * 2 + 1, stride=1, padding=radius)
    return (expanded[0, 0].numpy() > .5) & valid


def mask_variants(june: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Only June DN values and June validity affect all variants."""
    base = june_only_cloud_proxy(june, valid)
    nir, red, green = np.moveaxis(june.astype(np.float32), -1, 0)
    ndvi = (nir - red) / np.maximum(nir + red, 1.0)
    # Sensitive proxy: deliberately NOT an endorsed cloud detector.
    loose = (nir >= 180) & (red >= 180) & (green >= 180) & (ndvi < .36) & valid
    # New loose positives must be within 12 px of a base cloud pixel.
    nearby = loose & dilate(base, valid, 12)
    variants = {
        'original_proxy': base,
        'expanded_3px': dilate(base, valid, 3),
        'expanded_8px': dilate(base, valid, 8),
        'nearby_looser_proxy': (base | nearby) & valid,
    }
    for m in variants.values():
        if m.dtype != np.bool_ or np.any(m & ~valid):
            raise AssertionError('Mask includes invalid pixels')
    return variants


def png_sheet(tile, a, av, j, jv, variants, predictions, reports, path):
    names = list(variants)
    labels = ['April fixed (reference ONLY)', 'June fixed (input)', 'June auto (display only)']
    labels += [n.replace('_', ' ') for n in names]
    labels += ['V3 ' + n.replace('_', ' ') for n in names]
    panel_w, gap, top, panel_h = 195, 11, 94, 195
    width = gap + len(labels) * (panel_w + gap)
    height = top + panel_h + 70
    sheet = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 6), f'{tile} | V3 MASK SENSITIVITY | AUTOMATIC ONLY | NO EDITOR WINDOW', fill='black')
    draw.text((gap, 25), 'June-only DN screening; all masks unvalidated. April is DISPLAY ONLY.', fill='black')
    draw.text((gap, 43), 'Fixed-scale outputs allow side-by-side inspection; NO real-cloud accuracy claims.', fill='black')
    draw.text((gap, 61), 'Increased coverage can simply replace more valid pixels with inaccurate estimates.', fill='black')
    images = [preview_rgb(a, av), preview_rgb(j, jv), display_auto(j, jv)]
    images += [mask_rgb(variants[n], jv) for n in names]
    images += [preview_rgb(predictions[n], jv) for n in names]
    for col, (label, im) in enumerate(zip(labels, images)):
        x = gap + col * (panel_w + gap)
        draw.text((x, top - 18), label[:29], fill='black')
        resampling = Image.Resampling.NEAREST if col in range(3, 3 + len(names)) else Image.Resampling.BILINEAR
        sheet.paste(im.resize((panel_w, panel_h), resampling), (x, top))
    for k, name in enumerate(names):
        x = gap + (3 + k) * (panel_w + gap)
        r = reports[name]
        draw.text((x, top + panel_h + 8), f"mask {r['mask_fraction']:.1%}", fill='black')
        draw.text((x, top + panel_h + 25), r['status'].replace('_', ' ')[:28], fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def evaluate(opts):
    if opts.tile_size != 256:
        raise ValueError('V3 expects --tile-size 256')
    tiles = [s.strip().upper() for s in opts.tiles.split(',') if s.strip()]
    if not tiles or len(tiles) != len(set(tiles)):
        raise ValueError('Specify distinct tile IDs')
    records = qc_records(opts.qc_csv)
    for tile in tiles:
        if tile not in records:
            raise ValueError(f'{tile} is missing from the targeted QC CSV: {opts.qc_csv}')
    if not opts.checkpoint.is_file():
        raise FileNotFoundError(f'V3 checkpoint not found: {opts.checkpoint}')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if opts.device == 'auto' else opts.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; use --device cpu')
    model, ck = load_checkpoint(opts.checkpoint, device, residual_scale=.25)
    model.eval()
    sha = hashlib.sha256(opts.checkpoint.read_bytes()).hexdigest()
    outputs = []
    opts.out.mkdir(parents=True, exist_ok=True)
    print('REAL-JUNE V3 | NON-INTERACTIVE MASK SENSITIVITY', flush=True)
    print(f'Device: {device} | checkpoint epoch: {ck.get("epoch", "unknown")} | tiles: {tiles}', flush=True)
    print('No GUI, no training, no April transfer, no scene registration, no ground truth.', flush=True)
    for tile in tiles:
        a, av, j, jv, _ = load_tile(opts, records[tile])
        variants = mask_variants(j, jv)
        predictions = {}
        rpts = {}
        for name, mask in variants.items():
            f = float(mask.sum() / max(1, jv.sum()))
            if f < .02 or f > .90 or (jv & ~mask).sum() < 100:
                out = j.astype(np.float32).copy()
                status = 'skipped_unsafe_proxy_coverage'
                changed = None
            else:
                out = predict(model, j, jv, mask, device)
                status = 'qualitative_inference_only'
                changed = float(np.abs(out.astype(np.float32) - j.astype(np.float32))[mask].mean())
            if not np.array_equal(out[~mask], j[~mask]):
                raise AssertionError('Outside-mask June pixels changed')
            predictions[name] = out
            item = dict(tile_id=tile, variant=name, valid_fraction=round(float(jv.mean()), 4),
                        mask_fraction=round(f, 4), original_proxy_fraction=round(float(variants['original_proxy'][jv].mean()), 4),
                        added_vs_original=int((mask & ~variants['original_proxy']).sum()),
                        mean_masked_change_dn=None if changed is None else round(changed, 2),
                        status=status, mask_is_ground_truth=False, reconstruction_accuracy_available=False,
                        trained_on_real_clouds=False)
            rpts[name] = item
            outputs.append(item)
            print(f'  {tile} {name:21s} | selected={f:5.1%} | changed={item["mean_masked_change_dn"]} DN | {status}', flush=True)
        png_sheet(tile, a, av, j, jv, variants, predictions, rpts, opts.out / f'{tile}_mask_sensitivity.png')
        # Save only analysis masks, not full-resolution/georeferenced reconstructions.
        np.savez_compressed(opts.out / f'{tile}_mask_variants.npz', **{k: v.astype(np.uint8) for k, v in variants.items()})
    table = opts.out / 'mask_sensitivity_metrics.csv'
    with table.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(outputs[0]))
        w.writeheader()
        w.writerows(outputs)
    prov = opts.out / 'mask_sensitivity_provenance.json'
    prov.write_text(json.dumps(dict(
        experiment='Non-interactive, automatic June-only mask sensitivity for frozen Residual U-Net V3',
        warning='NOT validated cloud removal. No known clear June target. DN-based masks are unvalidated; larger mask can erase real surface pixels.',
        checkpoint=str(opts.checkpoint), checkpoint_epoch=ck.get('epoch'), checkpoint_sha256=sha,
        april_used_for_model=False, april_only_reference_visual=True,
        any_shifts_applied=False, cloud_mask_ground_truth=False,
        real_cloud_accuracy_available=False, geotiff_output=False,
        thresholds=dict(original='June-only NIR/Red/Green>=220 and NDVI<0.28',
                        loose_nearby='NIR/Red/Green>=180 and NDVI<0.36, ONLY within 12px of original proxy'),
        cases=outputs), indent=2), encoding='utf-8')
    print('NON-INTERACTIVE PREVIEW COMPLETE', flush=True)
    print('Images:', ', '.join(str(opts.out / f'{tile}_mask_sensitivity.png') for tile in tiles), flush=True)
    print('CSV:', table, flush=True)
    print('Provenance:', prov, flush=True)
    return outputs


def self_test():
    n = 256
    yy, xx = np.indices((n, n))
    rng = np.random.default_rng(13)
    base = 125 + 10 * np.sin(xx / 20) + 10 * np.cos(yy / 17) + rng.normal(0, 2, (n, n))
    a = np.stack([base + 50, base, base + 10], axis=-1).clip(1, 1023).astype(np.uint16)
    j = a.copy()
    j[44:140, 60:168] = [460, 420, 420]
    j[38:146, 54:174] = np.maximum(j[38:146, 54:174], [192, 191, 191])
    valid = np.ones((n, n), bool)
    valid[:8] = False
    a[~valid] = 0
    j[~valid] = 0
    variants = mask_variants(j, valid)
    base_mask = variants['original_proxy']
    assert base_mask.any() and not base_mask[0].any()
    assert variants['expanded_8px'].sum() >= variants['expanded_3px'].sum() >= base_mask.sum()
    assert variants['nearby_looser_proxy'].sum() >= base_mask.sum()
    model = ResidualCloudUNet(residual_scale=.25).eval()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for folder, arr in [('apr', a), ('jun', j)]:
            folder = root / folder
            folder.mkdir()
            for b, i in [(4, 0), (3, 1), (2, 2)]:
                with rasterio.open(folder / f'BAND{b}.tif', 'w', driver='GTiff', height=n, width=n,
                                   count=1, dtype='uint16', crs='EPSG:32643',
                                   transform=from_origin(500000, 2100000, 5, 5), nodata=0) as ds:
                    ds.write(arr[..., i], 1)
        qc = root / 'qc.csv'
        qc.write_text('tile_id,june_col,june_row\nT001,0,0\n', encoding='utf-8')
        ckpt = root / 'ckpt.pt'
        torch.save(dict(model_state_dict=model.state_dict(), experiment='residual_unet_v3',
                        normalization_divisor=1023.0, residual_scale=.25, epoch=0), ckpt)
        opts = SimpleNamespace(tiles='T001', qc_csv=qc, april_dir=root / 'apr', june_dir=root / 'jun',
                               checkpoint=ckpt, out=root / 'out', tile_size=n, device='cpu')
        results = evaluate(opts)
        assert len(results) == 4 and all(x['status'] == 'qualitative_inference_only' for x in results)
        assert (root/'out'/'T001_mask_sensitivity.png').is_file()
        assert (root/'out'/'T001_mask_variants.npz').is_file()
        assert (root/'out'/'mask_sensitivity_metrics.csv').is_file()
        assert (root/'out'/'mask_sensitivity_provenance.json').is_file()
    print('NON-INTERACTIVE V3 MASK SENSITIVITY SELF-TEST PASSED', flush=True)


if __name__ == '__main__':
    options = arguments()
    if options.self_test:
        self_test()
    else:
        evaluate(options)
