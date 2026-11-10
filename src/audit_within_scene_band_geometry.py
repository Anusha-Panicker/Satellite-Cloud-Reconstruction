"""Audit INTERNAL spectral-band alignment in April and June Mumbai scenes (read-only).

Reason: Red/NIR-based inter-date shifts improve Red/NIR correlation, but worsen Green.
This audit uses WITHIN-DATE Red as a spatial reference and searches how far NIR
and Green would need to shift to match Red, separately on April and June tiles.

These are feature-correlation DIAGNOSTICS, not ground-truth registration. The
spatial reflectance of different spectral bands may naturally differ. Do not
correct or warp full-scene data based on these results alone.

From Cloud_Reconstruction project root:
  .\\.venv\\Scripts\\python.exe -u .\\src\\audit_within_scene_band_geometry.py --self-test
  .\\.venv\\Scripts\\python.exe -u .\\src\\audit_within_scene_band_geometry.py

Requires numpy, Pillow, rasterio, and existing project scripts; NO pandas.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from audit_april_june_temporal_reference import ROOT, shift_no_wrap
from auto_register_april_june_v1 import corr, edge_feature
from verify_april_june_landmark_registration import get_record, load_tile

DEFAULT_QC = ROOT / 'outputs/april_june_targeted_qc/sampled_cloud_reference_qc.csv'
DEFAULT_OUT = ROOT / 'outputs/april_june_within_scene_band_qc'


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--tiles', default='T140,T103,T080,T104,T032')
    p.add_argument('--max-shift', type=int, default=6)
    p.add_argument('--tile-size', type=int, default=256)
    p.add_argument('--qc-csv', type=Path, default=DEFAULT_QC)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--april-dir', type=Path, default=ROOT/'data/raw/mumbai_scenes/2025-04-16')
    p.add_argument('--june-dir', type=Path, default=ROOT/'data/raw/mumbai_scenes/2025-06-03')
    return p.parse_args()


def fixed_safe_support(valid, radius):
    """Select pixels where every test shift has valid data (no shifting denominator)."""
    good = np.asarray(valid, dtype=bool).copy()
    if radius < 0 or radius > 16:
        raise ValueError('max-shift must be in [0,16]')
    h, w = good.shape
    if min(h, w) <= 2*radius + 20:
        raise ValueError('Tile too small for chosen max-shift')
    support = good.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            support &= shift_no_wrap(good, dy, dx)
    support[:radius+2] = False
    support[-radius-2:] = False
    support[:, :radius+2] = False
    support[:, -radius-2:] = False
    return support


def search_internal_shift(target_red, source_band, valid, radius=6, min_pixels=1500):
    """Shift to apply to SOURCE band to align it with RED from the SAME date."""
    target_red = np.asarray(target_red)
    source_band = np.asarray(source_band)
    valid = np.asarray(valid, bool)
    if target_red.shape != source_band.shape or target_red.shape != valid.shape:
        raise ValueError('Band and mask shapes differ')
    support = fixed_safe_support(valid, radius)
    n = int(support.sum())
    result = {'support_pixels':n,'zero_corr':None,'best_corr':None,
              'best_dx':None,'best_dy':None,'gain':None,'offset_magnitude_px':None,
              'heatmap':None,'status':'insufficient_support'}
    if n < min_pixels:
        return result
    target = edge_feature(target_red, valid)
    source = edge_feature(source_band, valid)
    y=[]
    best=(-float('inf'),0,0)
    zero=None
    for dy in range(-radius,radius+1):
        row=[]
        for dx in range(-radius,radius+1):
            score = corr(target, shift_no_wrap(source,dy,dx), support, minimum=min_pixels)
            row.append(float(score))
            if dx == 0 and dy == 0:
                zero = float(score)
            if np.isfinite(score) and score > best[0]:
                best = (float(score),dx,dy)
        y.append(row)
    if zero is None or not np.isfinite(zero) or not np.isfinite(best[0]):
        result['status']='flat_or_uncorrelated'
        return result
    gain=best[0]-zero
    magnitude=float(np.hypot(best[1],best[2]))
    result.update({'zero_corr':round(zero,5), 'best_corr':round(best[0],5),
                   'best_dx':best[1], 'best_dy':best[2], 'gain':round(gain,5),
                   'offset_magnitude_px':round(magnitude,3), 'heatmap':y,
                   'status': ('zero_shift_supported' if magnitude < 0.5 else
                              'nonzero_peak_inconclusive' if gain < .025 else
                              'possible_internal_offset_needs_verification')})
    return result


def analyze_tile(a,av,j,jv,radius,tid):
    out=[]
    for date, image, valid in [('April',a,av),('June',j,jv)]:
        # HWC: NIR (0), Red (1), Green (2)
        for source, k in [('NIR',0),('Green',2)]:
            res=search_internal_shift(image[...,1], image[...,k], valid, radius=radius)
            out.append({'tile_id':tid,'date':date,'source_band':source,
                        'target_band':'Red','shift_source_to_red':(res['best_dx'],res['best_dy']),
                        **res})
    return out


def save_outputs(results, outdir, radius):
    outdir.mkdir(parents=True, exist_ok=True)
    fields=['tile_id','date','target_band','source_band','support_pixels','zero_corr','best_corr',
            'best_dx','best_dy','gain','offset_magnitude_px','status']
    with (outdir/'within_scene_band_alignment.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k:r.get(k) for k in fields})
    summary={
        'question':'Are Green and NIR internally spatially aligned with Red within each date?',
        'method':'Edge-magnitude correlation on exactly the same fixed pixels for every shift',
        'limit':'Different bands see different scene structures; nonzero peaks are NOT proof of true spatial offsets',
        'warning':'DO NOT resample any GeoTIFF or accept April/June registration from these scores alone.',
        'results':[{k:v for k,v in r.items() if k not in ('heatmap',)} for r in results]
    }
    (outdir/'within_scene_band_alignment.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    make_figure(results,outdir/'within_scene_band_alignment.png',radius)


def make_figure(results,path,radius):
    # Heatmap of candidate correlations. Each row has April NIR, April Green,
    # June NIR, June Green. Values share one fixed support per panel.
    ids=list(dict.fromkeys(r['tile_id'] for r in results))
    s=2*radius+1
    cell=12
    panel_w=232
    row_h=210
    sheet=Image.new('RGB',(5*panel_w+15,45+len(ids)*row_h),'white')
    draw=ImageDraw.Draw(sheet)
    draw.text((10,6),'WITHIN-SCENE BAND GEOMETRY | SOURCE shifted to same-date Red | DIAGNOSTIC ONLY',fill=(10,10,10))
    draw.text((10,23),'NIR/Green shifts are NOT georegistration approval. Center square = zero shift.',fill=(70,70,70))
    for i,tid in enumerate(ids):
        y0=47+i*row_h
        draw.text((9,y0+16),tid,fill=(0,0,0))
        current=[r for r in results if r['tile_id']==tid]
        for j,r in enumerate(current):
            x0=panel_w*(j+1)+5
            draw.text((x0,y0+4),f'{r["date"]} {r["source_band"]}->Red',fill=(25,25,25))
            hm=r['heatmap']
            if hm is not None:
                arr=np.array(hm,dtype=float)
                for yi in range(s):
                    for xi in range(s):
                        val=arr[yi,xi]
                        t=max(0.,min(1.,(val+0.2)/1.0)) if np.isfinite(val) else 0.
                        color=(int(230*t+25*(1-t)), int(170*t+40*(1-t)),int(30*t+115*(1-t)))
                        x=x0+2+xi*cell; y=y0+26+yi*cell
                        draw.rectangle((x,y,x+cell-1,y+cell-1),fill=color)
                cx=x0+2+radius*cell;cy=y0+26+radius*cell
                draw.rectangle((cx,cy,cx+cell-1,cy+cell-1),outline=(255,255,255),width=2)
                bx=x0+2+(r['best_dx']+radius)*cell;by=y0+26+(r['best_dy']+radius)*cell
                draw.rectangle((bx,by,bx+cell-1,by+cell-1),outline=(0,0,0),width=2)
            scores=f"zero={r['zero_corr']} best={r['best_corr']}"
            offset=f"shift=({r['best_dx']},{r['best_dy']}) gain={r['gain']}"
            draw.text((x0,y0+28+s*cell),scores,fill=(50,50,50))
            draw.text((x0,y0+42+s*cell),offset,fill=(50,50,50))
    sheet.save(path)


def self_test():
    rng=np.random.default_rng(94)
    base=rng.normal(130,28,(160,160)).astype(np.float32)
    dr=Image.fromarray(np.uint8(np.clip(base,0,255)),'L')
    painter=ImageDraw.Draw(dr)
    for x in range(12,150,28):
        painter.line((x,7,x+14,149),fill=240,width=4)
    for y in range(18,155,35):
        painter.line((8,y,152,y+4),fill=20,width=3)
    red=np.array(dr,dtype=np.float32)
    green=shift_no_wrap(red,3,-2)  # To align with RED needs dx=+2, dy=-3.
    nir=red*1.1+15
    valid=np.ones(red.shape,bool)
    g=search_internal_shift(red,green,valid,radius=5)
    n=search_internal_shift(red,nir,valid,radius=5)
    assert (g['best_dx'],g['best_dy'])==(2,-3),g
    assert (n['best_dx'],n['best_dy'])==(0,0),n
    arr=np.stack([nir,red,green],axis=-1)
    result=analyze_tile(arr,valid,arr,valid,5,'T999')
    with tempfile.TemporaryDirectory() as td:
        outdir=Path(td)
        save_outputs(result,outdir,5)
        assert all((outdir/name).exists() for name in (
            'within_scene_band_alignment.csv','within_scene_band_alignment.json',
            'within_scene_band_alignment.png'))
        assert len(list(csv.DictReader((outdir/'within_scene_band_alignment.csv').open())))==4
    print('WITHIN-SCENE BAND QC SELF-TEST PASSED')
    print('Synthetic Green-to-Red (+2,-3) detected; aligned NIR-to-Red (0,0) detected.')
    print('CSV/JSON/PNG generation verified; no real satellite data accessed.')


def main():
    args=get_args()
    if args.self_test:
        self_test();return
    tiles=[t.strip().upper() for t in args.tiles.split(',') if t.strip()]
    if not tiles:raise ValueError('Specify one or more tiles')
    results=[]
    print('WITHIN-SCENE CROSS-BAND ALIGNMENT QC — APRIL AND JUNE')
    print('Bands already have matching declared geospatial grids; testing feature-level internal alignment.')
    print('All shifts scored on identical support. No data will be changed.')
    for tid in tiles:
        record=get_record(args.qc_csv,tid)
        a,av,j,jv,_=load_tile(args,record)
        rs=analyze_tile(a,av,j,jv,args.max_shift,tid)
        results.extend(rs)
        for r in rs:
            print(f"  {tid} {r['date']:5s} {r['source_band']:5s}->Red: "
                  f"shift {r['shift_source_to_red']} | corr {r['zero_corr']} -> {r['best_corr']} "
                  f"| gain {r['gain']} | {r['status']}",flush=True)
    save_outputs(results,args.out,args.max_shift)
    print('\nWITHIN-SCENE BAND QC COMPLETE')
    print('Results:',args.out/'within_scene_band_alignment.csv')
    print('Preview:',args.out/'within_scene_band_alignment.png')
    print('No inter-date alignment approved, no imagery warped, no V3 test data modified.')

if __name__=='__main__':
    main()
