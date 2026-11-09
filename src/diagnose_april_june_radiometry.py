r"""Read-only Mumbai April/June LISS-IV radiometry + preview diagnosis.

Place next to audit_april_june_temporal_reference.py in project src/.
From project root:
  .\.venv\Scripts\python.exe -u .\src\diagnose_april_june_radiometry.py --self-test
  .\.venv\Scripts\python.exe -u .\src\diagnose_april_june_radiometry.py

This is NOT a cloud detector or a geometric correction. Uses no pandas.
"""
import argparse
import csv
import json
import math
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from audit_april_june_temporal_reference import (
    ROOT, BANDS, locate_bands, open_scene, overlap_window, patch, classify,
    preview_rgb,
)

LABELS = ("NIR_BAND4", "RED_BAND3", "GREEN_BAND2")
OUT_DEFAULT = ROOT / "outputs" / "april_june_radiometry_diagnostic"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--april-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-04-16')
    p.add_argument('--june-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-06-03')
    p.add_argument('--out', type=Path, default=OUT_DEFAULT)
    p.add_argument('--grid', type=int, default=12)
    p.add_argument('--tile', type=int, default=256)
    p.add_argument('--preview', type=int, default=8)
    return p


def channel_stats(arr):
    z = np.asarray(arr, dtype=np.float32)
    if len(z) == 0:
        return None
    return {
        'count': int(len(z)),
        'min': float(np.min(z)),
        'p01': float(np.percentile(z, 1)),
        'p10': float(np.percentile(z, 10)),
        'p50': float(np.percentile(z, 50)),
        'p90': float(np.percentile(z, 90)),
        'p99': float(np.percentile(z, 99)),
        'max': float(np.max(z)),
        'fraction_ge_1023': float(np.mean(z >= 1023)),
        'fraction_gt_1023': float(np.mean(z > 1023)),
        'fraction_ge_800': float(np.mean(z >= 800)),
    }


def sample_report(apr_samples, jun_samples):
    if not len(apr_samples) or not len(jun_samples):
        raise ValueError('No valid sampled radiometry')
    result = {}
    for key, arr in (('april', apr_samples), ('june', jun_samples)):
        result[key] = {label: channel_stats(arr[:, i]) for i, label in enumerate(LABELS)}
    return result


def auto_rgb(a, valid):
    """Independent per-tile visualization only, never used for detection or stats."""
    if np.count_nonzero(valid) < 50:
        z = np.zeros_like(a, dtype=np.uint8)
    else:
        pixels = a[valid]
        lo = np.percentile(pixels, 2, axis=0)
        hi = np.percentile(pixels, 98, axis=0)
        span = np.maximum(hi - lo, 1)
        z = (np.clip((a - lo) / span, 0, 1) * 255).astype(np.uint8)
        z[~valid] = 0
    return Image.fromarray(z, 'RGB')


def make_preview(records, saved, out, count):
    n = min(count, len(records))
    records = sorted(records, key=lambda r: r['candidate_fraction'], reverse=True)
    nh = n // 2
    selected = records[:nh] + list(reversed(records[-(n - nh):]))
    w, h, gap = 200, 200, 10
    columns = ['April fixed', 'June fixed', 'April auto*', 'June auto*', 'Candidate mask']
    width = 2 * gap + len(columns) * (w + gap)
    height = 52 + n * (h + 32)
    outimg = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(outimg)
    draw.text((gap, 8), 'April vs June: raw DN check. *Auto-stretch = display ONLY, not radiometric correction', fill='black')
    for j, title in enumerate(columns):
        draw.text((gap + j * (w + gap), 30), title, fill='black')
    for i, r in enumerate(selected):
        a, b, av, bv, mask = saved[r['tile_id']]
        imgs = [preview_rgb(a, av), preview_rgb(b, bv),
                auto_rgb(a, av), auto_rgb(b, bv),
                Image.fromarray((mask.astype(np.uint8) * 255), 'L').convert('RGB')]
        y = 52 + i * (h + 32)
        for j, im in enumerate(imgs):
            outimg.paste(im.resize((w,h)),(gap + j*(w+gap), y))
        draw.text((gap, y+h+3),
                  f"{r['tile_id']} | candidate={r['candidate_fraction']:.1%} | June/April Red median={r['red_median_ratio']:.2f}", fill='black')
    out.parent.mkdir(parents=True, exist_ok=True)
    outimg.save(out)


def analyze(args):
    if args.grid < 2 or args.tile < 32 or args.preview < 2:
        raise ValueError('Use --grid >=2, --tile >=32, --preview >=2')
    apr_samples = []
    jun_samples = []
    records = []
    saved = {}
    with rasterio.Env(GDAL_CACHEMAX=64):
        with ExitStack() as stack:
            apr, ap_paths = open_scene(stack, args.april_dir, 'April 16')
            jun, ju_paths = open_scene(stack, args.june_dir, 'June 03')
            if apr[0].crs != jun[0].crs:
                raise ValueError('CRS mismatch; stop and check inputs')
            box, _ = overlap_window(apr[0], jun[0])
            col0, row0, col1, row1 = box
            warps = [stack.enter_context(WarpedVRT(d, crs=jun[0].crs,
                    transform=jun[0].transform, width=jun[0].width,
                    height=jun[0].height, src_nodata=0, nodata=0,
                    resampling=Resampling.bilinear)) for d in apr]
            xs = np.unique(np.rint(np.linspace(col0, col1 - args.tile, args.grid)).astype(int))
            ys = np.unique(np.rint(np.linspace(row0, row1 - args.tile, args.grid)).astype(int))
            for k, (y, x) in enumerate(((int(y), int(x)) for y in ys for x in xs), 1):
                win = Window(x, y, args.tile, args.tile)
                a, av = patch(warps, win)
                b, bv = patch(jun, win)
                valid = av & bv
                if np.mean(valid) < 0.70:
                    continue
                aa, bb, keep = a[::4, ::4], b[::4, ::4], valid[::4, ::4]
                apr_samples.append(aa[keep])
                jun_samples.append(bb[keep])
                maps = classify(a, b, av, bv, 145, 140, 30)
                c = maps['temporal_candidates']
                fraction = float(np.mean(c[valid]))
                ra = float(np.median(a[:, :, 1][valid]))
                rb = float(np.median(b[:, :, 1][valid]))
                ga = float(np.median(a[:, :, 2][valid]))
                gb = float(np.median(b[:, :, 2][valid]))
                row = {
                    'tile_id': f'T{k:03d}', 'june_col':x, 'june_row':y,
                    'valid_fraction':round(float(np.mean(valid)),4),
                    'candidate_fraction':round(fraction,5),
                    'april_red_median_dn':round(ra,2), 'june_red_median_dn':round(rb,2),
                    'april_green_median_dn':round(ga,2), 'june_green_median_dn':round(gb,2),
                    'red_median_ratio':round(rb / max(ra,1),3),
                    'green_median_ratio':round(gb / max(ga,1),3),
                    'june_red_ge145':round(float(np.mean(b[:, :, 1][valid] >= 145)),4),
                    'june_green_ge140':round(float(np.mean(b[:, :, 2][valid] >= 140)),4),
                }
                records.append(row)
                saved[row['tile_id']] = (a,b,av,bv,c)
                if k % 24 == 0:
                    print(f'Sampled {k}/{len(xs)*len(ys)} grid locations | retained {len(records)}',flush=True)
            if not records:
                raise RuntimeError('No overlapping valid samples. Inspect source directories and band masks.')
            sampled = sample_report(np.concatenate(apr_samples), np.concatenate(jun_samples))
            candidate_median = float(np.median([r['candidate_fraction'] for r in records]))
            over95 = sum(r['candidate_fraction'] >= 0.95 for r in records)
            over75 = sum(r['candidate_fraction'] >= 0.75 for r in records)
            sources = {'april':ap_paths,'june':ju_paths}
            meta = {}
            for key, ds in (('april',apr),('june',jun)):
                meta[key] = []
                for label, d in zip(LABELS,ds):
                    meta[key].append({'band':label,'dtype':d.dtypes[0],
                        'scale':float(d.scales[0]), 'offset':float(d.offsets[0]),
                        'nodata':d.nodata, 'tags':{str(k):str(v) for k,v in d.tags().items()}})
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    with (out/'scene_band_statistics.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['scene','band']+list(channel_stats(np.array([1,2])).keys()))
        writer.writeheader()
        for scene,bands in sampled.items():
            for band,st in bands.items():writer.writerow({'scene':scene,'band':band,**st})
    with (out/'tile_radiometry_screening.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    preview = out/'radiometry_debug_preview.png'
    make_preview(records,saved,preview,args.preview)
    summary={'sources':sources,'geotiff_metadata':meta,'sampled_tiles':len(records),
        'high_candidate_tiles_ge75pct':over75,'near_total_candidate_tiles_ge95pct':over95,
        'median_tile_candidate_fraction':candidate_median,
        'sampled_band_statistics':sampled,
        'warning':'Proxy candidate masks are NOT validated real cloud maps; independent display stretch is NOT a radiometric correction.'}
    with (out/'radiometry_diagnostic_summary.json').open('w',encoding='utf-8') as f:
        json.dump(summary,f,indent=2,allow_nan=False)
    print('\nRADIOMETRY DIAGNOSTIC COMPLETE')
    print(f'Retained tiles: {len(records)} | candidate fraction median: {candidate_median:.1%}')
    print(f'Tiles >=75% candidates: {over75}; tiles >=95% candidates: {over95}')
    for band in LABELS:
        a=sampled['april'][band]; b=sampled['june'][band]
        print(f'{band:12s} April p10/p50/p90={a["p10"]:.1f}/{a["p50"]:.1f}/{a["p90"]:.1f} '
              f'| June={b["p10"]:.1f}/{b["p50"]:.1f}/{b["p90"]:.1f} '
              f'| June >=1023: {b["fraction_ge_1023"]:.2%}')
    print('Statistics:',out/'scene_band_statistics.csv')
    print('Summary:',out/'radiometry_diagnostic_summary.json')
    print('Preview:',preview)
    print('No source GeoTIFFs, V3 outputs, or held-out benchmark changed.')


def self_test():
    rng=np.random.default_rng(9)
    a=rng.uniform(50,170,size=(64,64,3)).astype(np.float32)
    b=a.copy();b[4:28,4:28,1:3]+=120
    v=np.ones((64,64),dtype=bool)
    m=classify(a,b,v,v,145,140,30)['temporal_candidates']
    assert m.any()
    a_stats=sample_report(a.reshape(-1,3),b.reshape(-1,3))
    assert a_stats['june']['RED_BAND3']['p90'] > a_stats['april']['RED_BAND3']['p90']
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'example.png'
        rows=[{'tile_id':'T001','candidate_fraction':.1,'red_median_ratio':1.2},
              {'tile_id':'T002','candidate_fraction':.4,'red_median_ratio':1.3}]
        saved={r['tile_id']:(a,b,v,v,m) for r in rows}
        make_preview(rows,saved,p,2)
        assert p.exists() and p.stat().st_size > 1000
    print('APRIL–JUNE RADIOMETRY SELF-TEST PASSED')
    print('No satellite files, models, or existing project outputs accessed.')


if __name__=='__main__':
    args=build_parser().parse_args()
    if args.self_test: self_test()
    else: analyze(args)
