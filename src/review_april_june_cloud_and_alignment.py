r"""Mumbai April/June LISS-IV — conservative cloud/reference screening + registration QC.

Put this file inside Cloud_Reconstruction/src/ next to
`audit_april_june_temporal_reference.py`.

Run from project root:
    .\.venv\Scripts\python.exe -u .\src\review_april_june_cloud_and_alignment.py --self-test
    .\.venv\Scripts\python.exe -u .\src\review_april_june_cloud_and_alignment.py

This is an exploratory QUALITY-CONTROL REPORT, not a verified cloud mask,
a registered composite, or a cloud-reconstruction method.
Original imagery and existing V3/checkpoint/test outputs are never modified.
Uses NumPy, Pillow, Rasterio and Python standard library; NO pandas/SciPy.
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
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from audit_april_june_temporal_reference import (
    ROOT, open_scene, overlap_window, patch, preview_rgb, ndvi,
    highpass, estimate_shift, shift_no_wrap,
)

DEFAULT_OUT = ROOT / 'outputs' / 'april_june_targeted_qc'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--april-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-04-16')
    p.add_argument('--june-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-06-03')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--grid', type=int, default=12)
    p.add_argument('--tile', type=int, default=256)
    p.add_argument('--rows-per-group', type=int, default=4)
    p.add_argument('--min-valid', type=float, default=0.70)
    return p.parse_args()


def candidate_masks(a, j, av, jv):
    """Screening proxies, never true cloud/clear labels. NIR, RED, GREEN order."""
    both = av & jv
    dr = j[..., 1] - a[..., 1]
    dg = j[..., 2] - a[..., 2]
    # Reproduce original broad heuristic for a controlled comparison.
    old = (both & (j[..., 1] >= 145) & (j[..., 2] >= 140)
           & (dr >= 30) & (dg >= 30)
           & ~((a[..., 1] >= 145) & (a[..., 2] >= 140)))
    # Much more restrictive BRIGHT-CLOUD CORE proxy. Thresholds are
    # deliberately explicit DNs, not tuned against truth labels.
    jnd = ndvi(j)
    core = (both & (j[..., 0] >= 220) & (j[..., 1] >= 220)
            & (j[..., 2] >= 220) & (dr >= 75) & (dg >= 75)
            & (jnd < 0.28))
    # April quality SCREEN ONLY. This is not a scientifically verified clear mask.
    apr_ref = (both & (a[..., 1] < 190) & (a[..., 2] < 190)
               & (a[..., 0] < 340))
    # Land-like areas with spectral vegetation response on BOTH dates, to
    # avoid matching ocean wave texture or thick June clouds for registration.
    land = (both & (ndvi(a) > 0.14) & (jnd > 0.10)
            & (a[..., 0] > 105) & (j[..., 0] > 105))
    stable = land & (~old) & (~core) & apr_ref
    return {'valid':both, 'old':old, 'core':core, 'april_ref':apr_ref,
            'land':land, 'stable':stable}


def fraction(arr, valid):
    return float(np.count_nonzero(arr & valid) / max(1, np.count_nonzero(valid)))


def stats_for_tile(a, j, masks, tile_id, col, row):
    valid = masks['valid']
    hp_a, hp_j = highpass(a), highpass(j)
    dx, dy, psr, before, after, status = estimate_shift(hp_a, hp_j, masks['stable'], maxshift=8)
    return {
        'tile_id':tile_id, 'june_col':int(col),'june_row':int(row),
        'valid_fraction':round(float(valid.mean()),4),
        'old_broad_fraction':round(fraction(masks['old'],valid),4),
        'strict_core_fraction':round(fraction(masks['core'],valid),4),
        'april_reference_proxy_fraction':round(fraction(masks['april_ref'],valid),4),
        'land_like_fraction':round(fraction(masks['land'],valid),4),
        'stable_land_fraction':round(fraction(masks['stable'],valid),4),
        'june_april_red_median_ratio':round(float(np.median(j[...,1][valid])/max(1,np.median(a[...,1][valid]))),3),
        'local_shift_june_to_april_dx':int(dx),
        'local_shift_june_to_april_dy':int(dy),
        'shift_peak_snr':round(float(psr),3) if np.isfinite(psr) else None,
        'edge_corr_before':round(float(before),4) if np.isfinite(before) else None,
        'edge_corr_after':round(float(after),4) if np.isfinite(after) else None,
        'shift_status':status,
        'qc_group':'unselected',
    }


def select_groups(records, n):
    """Distinct representative samples. No claims of real cloud truth."""
    ordered = list(records)
    seen = set()
    def pick(name, pool, key, count):
        chosen = []
        for r in sorted(pool, key=key, reverse=True):
            if r['tile_id'] not in seen:
                seen.add(r['tile_id'])
                r['qc_group'] = name
                chosen.append(r)
            if len(chosen) == count:
                break
        return chosen

    # Most obvious opaque-looking candidates (strict pixels + potential April reference).
    cloudy = pick('cloud_candidate', [r for r in ordered if r['strict_core_fraction'] > 0.10 and r['april_reference_proxy_fraction'] > 0.45],
                  lambda r: (r['strict_core_fraction'],r['april_reference_proxy_fraction']),n)
    # Mixed boundaries: substantial but not whole-tile cloud core.
    mixed = pick('mixed_transition', [r for r in ordered if 0.02 < r['strict_core_fraction'] < 0.65],
                 lambda r: (-abs(r['strict_core_fraction']-0.25),r['april_reference_proxy_fraction']),n)
    # Land texture controls must be sufficiently clear-looking on BOTH dates.
    controls = pick('clear_land_control', [r for r in ordered if r['strict_core_fraction'] < 0.02 and r['old_broad_fraction'] < 0.20
                    and r['land_like_fraction'] > 0.30 and r['stable_land_fraction'] > 0.20],
                    lambda r: (r['stable_land_fraction'],r['edge_corr_after'] if r['edge_corr_after'] is not None else -1),n)
    return {'cloud_candidate':cloudy,'mixed_transition':mixed,'clear_land_control':controls}


def display_auto(a, valid):
    """Independent 2–98 percentile display stretch; strictly visualization."""
    if valid.sum() < 40:
        return Image.fromarray(np.zeros((*valid.shape,3),np.uint8),'RGB')
    v=a[valid]
    lo,hi=np.percentile(v,[2,98],axis=0)
    rgb=np.uint8(255*np.clip((a-lo)/np.maximum(hi-lo,1),0,1))
    rgb[~valid]=0
    return Image.fromarray(rgb,'RGB')


def binary_image(mask):
    return Image.fromarray(np.where(mask[...,None],255,0).astype(np.uint8).repeat(3,axis=2),'RGB')


def overlay_image(j, jvalid, core, broad):
    rgb=np.array(display_auto(j,jvalid)).astype(np.float32)
    # Core in red, other broad candidates in yellow. Avoid mixing the two.
    halo=broad & ~core
    rgb[halo]=.55*rgb[halo]+.45*np.array([255,200,0])
    rgb[core]=.45*rgb[core]+.55*np.array([255,40,40])
    return Image.fromarray(np.uint8(np.clip(rgb,0,255)),'RGB')


def checker(a, j, av, jv, n=16):
    ai=np.asarray(display_auto(a,av)); ji=np.asarray(display_auto(j,jv))
    yy,xx=np.indices(av.shape)
    selector=((yy//n+xx//n)%2).astype(bool)
    output=np.where(selector[...,None],ai,ji)
    return Image.fromarray(np.uint8(output),'RGB')


def quality_sheet(groups, tiles, dest, size=190):
    cols=['April fixed','June fixed','June auto*','Broad proxy','Strict core','Overlay auto*']
    ordered=[(name,r) for name,rr in groups.items() for r in rr]
    if not ordered:
        raise ValueError('No QC records selected')
    gap=10; cell=size; top=55; row_height=size+35
    width=gap+len(cols)*(cell+gap)
    out=Image.new('RGB',(width,top+len(ordered)*row_height),'white')
    pen=ImageDraw.Draw(out)
    pen.text((gap,5),'APRIL–JUNE | Cloud candidate QC | Screening ONLY, not validated masks',fill='black')
    pen.text((gap,22),'*Auto display stretched independently; no radiometric correction applied',fill='black')
    for ix,col in enumerate(cols):pen.text((gap+ix*(cell+gap),40),col,fill='black')
    last_group=None
    for i,(group,r) in enumerate(ordered):
        a,j,av,jv,m=tiles[r['tile_id']]
        ims=[preview_rgb(a,av),preview_rgb(j,jv),display_auto(j,jv),binary_image(m['old']),
             binary_image(m['core']),overlay_image(j,jv,m['core'],m['old'])]
        y=top+i*row_height
        for k,im in enumerate(ims):out.paste(im.resize((size,size)),(gap+k*(size+gap),y))
        text=(f"{r['tile_id']} | {group} | broad {r['old_broad_fraction']:.0%}, "
              f"strict core {r['strict_core_fraction']:.0%}, April ref-proxy {r['april_reference_proxy_fraction']:.0%}")
        pen.text((gap,y+size+5),text,fill='black')
    dest.parent.mkdir(parents=True,exist_ok=True)
    out.save(dest)


def alignment_sheet(groups,tiles,dest,size=208):
    chosen=groups['clear_land_control']
    if not chosen:return False
    columns=['April auto*','June auto*','Checkerboard current','Checkerboard candidate-shift']
    gap=12;top=53; row_h=size+44
    out=Image.new('RGB',(gap+len(columns)*(size+gap),top+len(chosen)*row_h),'white')
    pen=ImageDraw.Draw(out)
    pen.text((gap,5),'CLEAR-LAND REGISTRATION CONTROLS — shifts are unvalidated candidates',fill='black')
    pen.text((gap,22),'Independent display stretches, not radiometric correction. Look for discontinuous roads/edges.',fill='black')
    for i,c in enumerate(columns):pen.text((gap+i*(size+gap),40),c,fill='black')
    for i,r in enumerate(chosen):
        a,j,av,jv,m=tiles[r['tile_id']]
        dx,dy=r['local_shift_june_to_april_dx'],r['local_shift_june_to_april_dy']
        # shift only in temporary preview arrays, NEVER source files.
        js=shift_no_wrap(j,dy,dx); jvs=shift_no_wrap(jv,dy,dx)
        ims=[display_auto(a,av),display_auto(j,jv),checker(a,j,av,jv),checker(a,js,av,jvs)]
        y=top+i*row_h
        for k,im in enumerate(ims):out.paste(im.resize((size,size)),(gap+k*(size+gap),y))
        pen.text((gap,y+size+3),f"{r['tile_id']} | shift candidate dx={dx} dy={dy} | {r['shift_status']} "
                 f"| corr {r['edge_corr_before']} -> {r['edge_corr_after']}",fill='black')
    dest.parent.mkdir(parents=True,exist_ok=True)
    out.save(dest)
    return True


def run(args):
    if not 0 < args.min_valid <= 1 or args.grid < 2 or args.tile < 32 or args.rows_per_group < 1:
        raise ValueError('Invalid grid/tile/rows-per-group/min-valid settings')
    with rasterio.Env(GDAL_CACHEMAX=64):
        with ExitStack() as stack:
            apr,ap_paths=open_scene(stack,args.april_dir,'April 16')
            jun,ju_paths=open_scene(stack,args.june_dir,'June 03')
            if apr[0].crs!=jun[0].crs:
                raise ValueError('CRS mismatch; no arbitrary transform is applied')
            col0,row0,col1,row1=overlap_window(apr[0],jun[0])[0]
            if col1-col0<args.tile or row1-row0<args.tile:
                raise ValueError('Overlap too small for selected tile size')
            warps=[stack.enter_context(WarpedVRT(d,crs=jun[0].crs,transform=jun[0].transform,
                     width=jun[0].width,height=jun[0].height,src_nodata=0,nodata=0,
                     resampling=Resampling.bilinear)) for d in apr]
            xs=np.unique(np.rint(np.linspace(col0,col1-args.tile,args.grid)).astype(int))
            ys=np.unique(np.rint(np.linspace(row0,row1-args.tile,args.grid)).astype(int))
            rows=[];tiles={};count=0
            for y in ys:
                for x in xs:
                    count+=1
                    win=Window(int(x),int(y),args.tile,args.tile)
                    a,av=patch(warps,win)
                    j,jv=patch(jun,win)
                    mask=candidate_masks(a,j,av,jv)
                    if float(np.mean(mask['valid']))<args.min_valid:continue
                    tid=f'T{count:03d}'
                    r=stats_for_tile(a,j,mask,tid,int(x),int(y))
                    rows.append(r)
                    tiles[tid]=(a,j,av,jv,mask)
                    if count % max(1,len(xs)*2)==0:
                        print(f'Analyzed {count}/{len(xs)*len(ys)} windows | retained {len(rows)}',flush=True)
    if not rows:
        raise RuntimeError('No valid sample windows were retained')
    groups=select_groups(rows,args.rows_per_group)
    args.out.mkdir(parents=True,exist_ok=True)
    csvpath=args.out/'sampled_cloud_reference_qc.csv'
    with csvpath.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    qcpath=args.out/'cloud_reference_qc_contact_sheet.png'
    quality_sheet(groups,tiles,qcpath)
    regpath=args.out/'clear_land_registration_controls.png'
    had_controls=alignment_sheet(groups,tiles,regpath)
    summary={
        'note':'All classifications are UNCALIBRATED heuristics. No cloud truth, alignment transform or reconstruction generated.',
        'tiles_sampled':count,'tiles_retained':len(rows),
        'group_selection':{name:[r['tile_id'] for r in rr] for name,rr in groups.items()},
        'old_broad_coverage_median':float(np.median([r['old_broad_fraction'] for r in rows])),
        'strict_core_coverage_median':float(np.median([r['strict_core_fraction'] for r in rows])),
        'strict_core_ge75pct_tiles':sum(r['strict_core_fraction']>=.75 for r in rows),
        'april_source_files':ap_paths,'june_source_files':ju_paths,
        'q_caveat':'Independent auto display only for inspection; clouds must be confirmed manually. Registration control shifts are tentative.'
    }
    with (args.out/'targeted_qc_summary.json').open('w',encoding='utf-8') as f:
        json.dump(summary,f,indent=2,allow_nan=False)
    print('\nTARGETED APRIL–JUNE QC COMPLETE')
    print(f"Sampled {count} locations | retained {len(rows)}")
    print(f"Broad mask median: {summary['old_broad_coverage_median']:.1%} | strict opaque-proxy median: {summary['strict_core_coverage_median']:.1%}")
    print(f"Strict core >=75%: {summary['strict_core_ge75pct_tiles']} tiles")
    for group,rr in groups.items():
        print(group+': '+(', '.join(r['tile_id'] for r in rr) or '(none found)'))
    print('CSV:',csvpath)
    print('Cloud/reference QC:',qcpath)
    print('Land-registration controls:',regpath if had_controls else 'NOT GENERATED (no sufficiently clear land controls in samples)')
    print('No source imagery, test data or model checkpoints modified.')
    return summary


def self_test():
    rng=np.random.default_rng(77)
    h=w=128
    yy,xx=np.indices((h,w))
    a0=155+26*np.sin(xx/5)+17*np.cos(yy/7)+rng.normal(0,4,(h,w))
    a=np.stack([a0+30, a0-40, a0-30],axis=-1).astype(np.float32)
    j=a.copy();j[:64,:64,:]=[360,300,325]
    v=np.ones((h,w),bool)
    cm=candidate_masks(a,j,v,v)
    assert cm['core'][:64,:64].mean()>.95
    assert cm['core'][64:,64:].mean()<.05
    assert cm['old'][:64,:64].mean()>.85
    assert cm['stable'].shape==(h,w)
    r=stats_for_tile(a,j,cm,'T001',0,0)
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)
        # Exercise the full read/write route with correctly georeferenced sample rasters.
        for name,img in [('april',a),('june',j)]:
            d=root/name;d.mkdir()
            for bi,b in enumerate((4,3,2)):
                with rasterio.open(d/f'BAND{b}.tif','w',driver='GTiff',height=h,width=w,
                     count=1,dtype='uint16',transform=from_origin(500000,2100000,5,5),crs='EPSG:32643',nodata=0) as ds:
                    ds.write(np.uint16(np.clip(img[...,bi],1,1023)),1)
        opt=argparse.Namespace(april_dir=root/'april',june_dir=root/'june',out=root/'out',
                               grid=3,tile=32,rows_per_group=2,min_valid=.7)
        summary=run(opt)
        assert summary['tiles_retained']==9
        assert (opt.out/'sampled_cloud_reference_qc.csv').is_file()
        assert (opt.out/'cloud_reference_qc_contact_sheet.png').is_file()
        assert (opt.out/'targeted_qc_summary.json').is_file()
    print('\nTARGETED APRIL–JUNE QC SELF-TEST PASSED')
    print('No real satellite data or project output directories accessed.')


if __name__=='__main__':
    args=parse_args()
    if args.self_test:self_test()
    else:run(args)
