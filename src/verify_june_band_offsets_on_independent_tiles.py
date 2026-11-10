"""Independent CHECK of June 2025 within-scene spectral-band feature offsets.

Previous in-sample diagnostics suggested June GREEN->RED=(-2,+3) and
NIR->RED=(+1,-2) pixels. This script freezes those hypotheses and tests them
on OTHER, clear-looking land tiles that were not used to suggest the shifts.
April GREEN->RED (0,0) is a comparison control. Neither imagery nor existing
checkpoints are modified. These are image-feature offsets, not ground truth.

Requirements: numpy, Pillow, rasterio and earlier Cloud_Reconstruction/src scripts.
No pandas, scipy or OpenCV required.

From the project root (one command at a time):
    .\\.venv\\Scripts\\python.exe -u .\\src\\verify_june_band_offsets_on_independent_tiles.py --self-test
    .\\.venv\\Scripts\\python.exe -u .\\src\\verify_june_band_offsets_on_independent_tiles.py
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFilter
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from audit_april_june_temporal_reference import ROOT, open_scene, patch, shift_no_wrap
from audit_within_scene_band_geometry import fixed_safe_support
from auto_register_april_june_v1 import edge_feature, corr
from review_april_june_cloud_and_alignment import candidate_masks, display_auto

QC_DEFAULT = ROOT / 'outputs/april_june_targeted_qc/sampled_cloud_reference_qc.csv'
OUT_DEFAULT = ROOT / 'outputs/april_june_independent_band_qc'
USED_TILES = {'T140', 'T103', 'T080', 'T104', 'T032'}
SHIFTS = {'june_green': (-2, +3), 'june_nir': (+1, -2), 'april_green_control': (-2, +3)}
BANDS = {'june_green': 2, 'june_nir': 0, 'april_green_control': 2}


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--max-tiles', type=int, default=12)
    p.add_argument('--qc-csv', type=Path, default=QC_DEFAULT)
    p.add_argument('--out', type=Path, default=OUT_DEFAULT)
    p.add_argument('--tile-size', type=int, default=256)
    p.add_argument('--april-dir', type=Path, default=ROOT/'data/raw/mumbai_scenes/2025-04-16')
    p.add_argument('--june-dir', type=Path, default=ROOT/'data/raw/mumbai_scenes/2025-06-03')
    return p.parse_args()


def select_new_tiles(csv_path, max_tiles):
    if max_tiles < 1:
        raise ValueError('max-tiles must be >=1')
    if not csv_path.is_file():
        raise FileNotFoundError(f'Missing QC tile manifest: {csv_path}\nRun review_april_june_cloud_and_alignment.py first.')
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    if not records:
        raise ValueError('QC tile manifest is empty')
    names = ['tile_id','valid_fraction','old_broad_fraction','strict_core_fraction',
             'stable_land_fraction','june_col','june_row']
    if any(n not in records[0] for n in names):
        raise ValueError(f'QC CSV is missing required columns: {[n for n in names if n not in records[0]]}')
    candidates = []
    for r in records:
        try:
            if r['tile_id'] in USED_TILES:
                continue
            valid = float(r['valid_fraction'])
            broad = float(r['old_broad_fraction'])
            core = float(r['strict_core_fraction'])
            land = float(r['stable_land_fraction'])
            if valid < 0.75 or land < 0.12 or core > 0.10 or broad > 0.32:
                continue
            candidates.append((land - 0.35*broad, r))
        except (ValueError, KeyError):
            continue
    candidates.sort(key=lambda x: (-x[0], x[1]['tile_id']))
    picked=[r for _,r in candidates[:max_tiles]]
    return picked, len(candidates)


def roi_slices(n):
    h = w = n
    return {'NW':(slice(0,h//2),slice(0,w//2)),
            'NE':(slice(0,h//2),slice(w//2,w)),
            'SW':(slice(h//2,h),slice(0,w//2)),
            'SE':(slice(h//2,h),slice(w//2,w))}


def measure_one(image, valid, conservative_safe, source_band, frozen_shift, min_pixels=1000):
    """Measure on same fixed pixels for zero shift and preselected shift.

    All masks are diagnostics. We never refit the shift to the evaluated tiles.
    """
    assert image.ndim == 3 and image.shape[-1] == 3
    dx,dy = frozen_shift
    radius = max(4, abs(dx), abs(dy))
    support = fixed_safe_support(np.asarray(valid, bool) & np.asarray(conservative_safe, bool),radius)
    red = edge_feature(image[...,1], valid)
    band = edge_feature(image[...,source_band], valid)
    moved = shift_no_wrap(band,dy,dx)
    zero = corr(red,band,support,minimum=min_pixels)
    shifted = corr(red,moved,support,minimum=min_pixels)
    quad=[]
    for label,(ys,xs) in roi_slices(image.shape[0]).items():
        submask = np.zeros(support.shape,dtype=bool)
        submask[ys,xs] = support[ys,xs]
        a = corr(red,band,submask,minimum=max(350,min_pixels//2))
        b = corr(red,moved,submask,minimum=max(350,min_pixels//2))
        quad.append({'quadrant':label,'support':int(submask.sum()),
                     'zero':float(a) if np.isfinite(a) else None,
                     'shifted':float(b) if np.isfinite(b) else None,
                     'gain':float(b-a) if np.isfinite(a) and np.isfinite(b) else None})
    gain = float(shifted-zero) if np.isfinite(zero) and np.isfinite(shifted) else None
    eligible=[q for q in quad if q['gain'] is not None]
    strong = sum(q['gain']>=0.01 for q in eligible)
    # Conservative per-tile criterion: overall AND multiple spatial regions agree.
    confirmed = bool(gain is not None and gain >= 0.025 and shifted >= 0.18
                     and len(eligible)>=3 and strong>=3)
    return {'zero_corr':float(zero) if np.isfinite(zero) else None,
            'frozen_corr':float(shifted) if np.isfinite(shifted) else None,
            'gain':gain,'support_pixels':int(support.sum()),
            'quadrants_checked':len(eligible),'quadrants_positive':strong,
            'tile_confirms_frozen_offset':confirmed,
            'support_mask':support,'quad_results':quad}


def same_date_tests(a,av,j,jv):
    masks = candidate_masks(a,j,av,jv)
    # Remove proxy-bright/cloud-changed pixels AND restrict to land-like controls.
    # This mask is heuristic, *not* a guaranteed clear-sky map.
    unsafe = masks['old'] | masks['core']
    unsafe = np.asarray(Image.fromarray(np.uint8(unsafe)*255,mode='L').filter(
        ImageFilter.MaxFilter(9)),dtype=np.uint8)>0
    safe = masks['stable'] & ~unsafe & av & jv
    results={}
    for name,(img,valid) in {
        'june_green':(j,jv),'june_nir':(j,jv),'april_green_control':(a,av)
    }.items():
        results[name] = measure_one(img,valid,safe,BANDS[name],SHIFTS[name])
    return results, safe


def safe_num(v, digits=4):
    return 'NA' if v is None else f'{v:+.{digits}f}'


def save_report(rows, previews, outdir, available):
    outdir.mkdir(parents=True, exist_ok=True)
    columns=['tile_id','trial','frozen_dx','frozen_dy','support_pixels','corr_zero',
             'corr_frozen','gain','quadrants_checked','quadrants_positive',
             'tile_confirms_frozen_offset']
    csvfile=outdir/'independent_band_offset_results.csv'
    with csvfile.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader()
        for r in rows: w.writerow({c:r.get(c) for c in columns})
    summary={'method':'Frozen candidate shifts evaluated on previously unused clear-looking QC tiles',
             'candidate_offsets':SHIFTS,
             'selected_tiles':sorted(set(r['tile_id'] for r in rows)),
             'eligible_holdout_tiles_available':available,
             'sampling_warning':'Selection and cloud screening are heuristic; clear-looking tiles may contain haze or scene changes',
             'interpretation_warning':'Feature correlation offsets are NOT certified physical sensor-band displacements, georegistration, or approval to resample',
             'results':{}}
    for name in SHIFTS:
        subset=[r for r in rows if r['trial']==name]
        good=[r for r in subset if r['gain'] is not None]
        confirmed=sum(bool(r['tile_confirms_frozen_offset']) for r in good)
        gains=[float(r['gain']) for r in good]
        summary['results'][name]={'evaluated_tiles':len(good),'confirming_tiles':confirmed,
             'median_corr_gain':float(np.median(gains)) if gains else None,
             'frozen_shift_supported_on_new_tiles':bool(len(good)>=4 and confirmed>=max(4,int(np.ceil(.75*len(good)))))}
    (outdir/'independent_band_offset_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    save_figure(previews, outdir/'independent_band_offset_preview.png')
    return summary


def red_green_overlay(image, valid, offset=(0,0)):
    red=edge_feature(image[...,1],valid)
    green=edge_feature(image[...,2],valid)
    green=shift_no_wrap(green, offset[1],offset[0])
    # Green edges = green, Red edges = magenta; overlap becomes pale
    r=np.uint8(np.clip(red*140,0,255))
    g=np.uint8(np.clip(green*140,0,255))
    b=np.uint8(np.clip(red*110,0,255))
    out=np.stack([r,g,b],axis=-1)
    out[~valid]=0
    return Image.fromarray(out,'RGB')


def save_figure(previews, path):
    if not previews:
        return
    previews=previews[:6]
    thumb=192
    columns=['June NIR/Red/Green','June Green vs Red: original',
             'June Green vs Red: frozen offset','April Green vs Red: zero']
    margin=12; panel_w=thumb+12; row_h=thumb+42; top=58
    canvas=Image.new('RGB',(margin+4*panel_w,top+len(previews)*row_h),'white')
    d=ImageDraw.Draw(canvas)
    d.text((margin,5),'Independent June within-scene band-offset check | diagnostics only',fill='black')
    d.text((margin,22),'Frozen shifts tested on NEW QC tiles. Red edges magenta, Green edges green.',fill=(75,75,75))
    for k,h in enumerate(columns):
        d.text((margin+k*panel_w,42),h[:26],fill='black')
    for i,(tid,j,jv,a,av,results) in enumerate(previews):
        imgs=[display_auto(j,jv),red_green_overlay(j,jv),
              red_green_overlay(j,jv,SHIFTS['june_green']),red_green_overlay(a,av)]
        y=top+i*row_h
        for k,im in enumerate(imgs):
            canvas.paste(im.resize((thumb,thumb)),(margin+k*panel_w,y))
        f=results['june_green']
        d.text((margin,y+thumb+5),f'{tid}: June Green->Red gain {safe_num(f["gain"])}; ' +
               f'{f["quadrants_positive"]}/{f["quadrants_checked"]} quadrants improved',fill='black')
    canvas.save(path)


def synthetic_self_test():
    rng=np.random.default_rng(123)
    h=w=256
    base=rng.normal(100,20,(h,w)).astype(np.float32)
    pil=Image.fromarray(np.uint8(np.clip(base,0,255)),'L')
    draw=ImageDraw.Draw(pil)
    for y in range(12,250,24): draw.line((6,y,250,y),fill=220,width=3)
    for x in range(12,250,35): draw.line((x,6,x,250),fill=20,width=4)
    red=np.asarray(pil,dtype=np.float32)
    # June needs shift(-2,+3) on Green and shift(+1,-2) on NIR.
    green_j=shift_no_wrap(red,-3,+2)
    nir_j=shift_no_wrap(red,+2,-1)
    june=np.stack([nir_j,red,green_j],axis=-1)
    april=np.stack([red*1.2,red,red],axis=-1)
    valid=np.ones((h,w),bool)
    # Entire synthetic image used, no cloud mask; verification of the fixed-support calculator.
    rj=measure_one(june,valid,valid,2,(-2,+3))
    rn=measure_one(june,valid,valid,0,(+1,-2))
    ra=measure_one(april,valid,valid,2,(-2,+3))
    assert rj['gain'] is not None and rj['gain'] > 0.10, rj
    assert rn['gain'] is not None and rn['gain'] > 0.10, rn
    assert ra['gain'] is not None and ra['gain'] < -0.05, ra
    rows=[]
    tests={'june_green':rj,'june_nir':rn,'april_green_control':ra}
    for name,r in tests.items():
        dx,dy=SHIFTS[name]
        rows.append({'tile_id':'T999','trial':name,'frozen_dx':dx,'frozen_dy':dy,
                     'support_pixels':r['support_pixels'],'corr_zero':r['zero_corr'],
                     'corr_frozen':r['frozen_corr'],'gain':r['gain'],
                     'quadrants_checked':r['quadrants_checked'],
                     'quadrants_positive':r['quadrants_positive'],
                     'tile_confirms_frozen_offset':r['tile_confirms_frozen_offset']})
    with tempfile.TemporaryDirectory() as td:
        sm=save_report(rows,[('T999',june,valid,april,valid,tests)],Path(td),1)
        assert len(list(csv.DictReader((Path(td)/'independent_band_offset_results.csv').open()))) == 3
        assert (Path(td)/'independent_band_offset_preview.png').is_file()
    print('INDEPENDENT BAND QC SELF-TEST PASSED')
    print('Known synthetic June shifts recognized; aligned April Green control not falsely approved.')
    print('CSV, JSON and PNG generation passed; no real satellite files opened.')


def main():
    args=args_parser()
    if args.self_test:
        synthetic_self_test(); return
    selected,available=select_new_tiles(args.qc_csv,args.max_tiles)
    if not selected:
        raise RuntimeError('No independent clear-looking land tiles passed screening. '
                           'Do NOT fall back to the original five tiles. '
                           'Inspect the QC CSV or consider other acquisition dates.')
    print('INDEPENDENT JUNE BAND-OFFSET QUALITY CHECK — READ ONLY')
    print('Original five discovery tiles EXCLUDED:',', '.join(sorted(USED_TILES)))
    print('Eligible new tiles:',available,'| Testing:',len(selected))
    print('Selected:', ', '.join(r['tile_id'] for r in selected))
    print('Frozen offsets: June Green->Red (-2,+3), June NIR->Red (+1,-2)')
    print('April Green control: test the SAME nonzero offset, expected to worsen if aligned')
    rows=[]; previews=[]
    with rasterio.Env(GDAL_CACHEMAX=96):
        with ExitStack() as stack:
            apr,_=open_scene(stack,args.april_dir,'April 16')
            jun,_=open_scene(stack,args.june_dir,'June 03')
            if apr[0].crs != jun[0].crs:
                raise ValueError('CRS mismatch')
            aw=[stack.enter_context(WarpedVRT(d,crs=jun[0].crs,transform=jun[0].transform,
                 width=jun[0].width,height=jun[0].height,src_nodata=0,nodata=0,
                 resampling=Resampling.bilinear)) for d in apr]
            for ix,record in enumerate(selected,1):
                tid=record['tile_id']
                win=Window(int(record['june_col']),int(record['june_row']),args.tile_size,args.tile_size)
                a,av=patch(aw,win);j,jv=patch(jun,win)
                res,safe=same_date_tests(a,av,j,jv)
                for name,r in res.items():
                    dx,dy=SHIFTS[name]
                    rows.append({'tile_id':tid,'trial':name,'frozen_dx':dx,'frozen_dy':dy,
                        'support_pixels':r['support_pixels'],'corr_zero':r['zero_corr'],
                        'corr_frozen':r['frozen_corr'],'gain':r['gain'],
                        'quadrants_checked':r['quadrants_checked'],
                        'quadrants_positive':r['quadrants_positive'],
                        'tile_confirms_frozen_offset':r['tile_confirms_frozen_offset']})
                print(f'{tid} {ix}/{len(selected)} | usable {safe.mean():.1%} | '
                      f'June Green Δ {safe_num(res["june_green"]["gain"])} '
                      f'({res["june_green"]["quadrants_positive"]}/{res["june_green"]["quadrants_checked"]} quads) | '
                      f'June NIR Δ {safe_num(res["june_nir"]["gain"])} | '
                      f'April Green control Δ {safe_num(res["april_green_control"]["gain"])}', flush=True)
                if len(previews)<6:
                    previews.append((tid,j,jv,a,av,res))
    summary=save_report(rows,previews,args.out,available)
    print('\nINDEPENDENT BAND QC COMPLETE')
    for name,stat in summary['results'].items():
        print(f'  {name}: confirming {stat["confirming_tiles"]}/{stat["evaluated_tiles"]}; '
              f'median gain {safe_num(stat["median_corr_gain"])}; '
              f'new-tile support={stat["frozen_shift_supported_on_new_tiles"]}')
    print('CSV:',args.out/'independent_band_offset_results.csv')
    print('Summary:',args.out/'independent_band_offset_summary.json')
    print('Preview:',args.out/'independent_band_offset_preview.png')
    print('Candidate pixel shifts are NOT approved for resampling or inter-date registration.')


if __name__ == '__main__':
    main()
