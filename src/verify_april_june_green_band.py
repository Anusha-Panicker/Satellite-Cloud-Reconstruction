r"""Independent-band verification of frozen April/June candidate translations.

The earlier auto_register_april_june_v1.py chose shifts from Red and NIR edges.
This script TESTS those FROZEN shifts with Green-band edges, which were not
used in that shift search. Thus it is independent *spectrally*, not a fully
independent geographic tie-point validation. It NEVER searches to update the
frozen candidates or changes the underlying source imagery.

Cloud screening is still a heuristic, not a verified real cloud mask.
Outputs are exploratory local QC only; NOT registration approval.

From project root:
 .\\.venv\\Scripts\\python.exe -u .\\src\\verify_april_june_green_band.py --self-test
 .\\.venv\\Scripts\\python.exe -u .\\src\\verify_april_june_green_band.py

Dependencies: NumPy, Pillow, Rasterio, stdlib. No pandas/OpenCV/SciPy.
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
from auto_register_april_june_v1 import (
    corr, edge_feature, screen_mask, candidate_windows,
    search_region, DEFAULT_QC,
)
from verify_april_june_landmark_registration import get_record, load_tile
from review_april_june_cloud_and_alignment import checker, display_auto

DEFAULT_AUTO = ROOT / 'outputs/april_june_automatic_registration_qc/automatic_registration_summary.csv'
DEFAULT_OUT = ROOT / 'outputs/april_june_green_band_qc'


def args_parser():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--auto-csv',type=Path,default=DEFAULT_AUTO)
    p.add_argument('--qc-csv',type=Path,default=DEFAULT_QC)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--april-dir',type=Path,default=ROOT/'data/raw/mumbai_scenes/2025-04-16')
    p.add_argument('--june-dir',type=Path,default=ROOT/'data/raw/mumbai_scenes/2025-06-03')
    p.add_argument('--tile-size',type=int,default=256)
    p.add_argument('--max-shift',type=int,default=8)
    p.add_argument('--tiles',type=str,default='T032,T140,T103,T080,T104')
    return p.parse_args()


def frozen_rows(csv_path: Path) -> dict:
    if not csv_path.is_file():
        raise FileNotFoundError(f'Automatic registration summary missing: {csv_path}\nRun auto_register_april_june_v1.py first.')
    with csv_path.open('r',newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    found={}
    for r in rows:
        t=r.get('tile_id','').strip().upper()
        if not t or t in found:
            raise ValueError(f'Missing or duplicate tile_id in auto CSV: {t}')
        if not r.get('status','') or not r.get('dx','') or not r.get('dy',''):
            raise ValueError(f'Missing frozen shift/status for {t}')
        found[t]=r
    return found


def score_green(ae,je,safe,dx,dy,roi,min_pixels=350):
    h,w=safe.shape
    y0,y1,x0,x1=roi
    region=np.zeros((h,w),bool)
    region[y0:y1,x0:x1]=True
    # The SAME intersection supports both the baseline and candidate scores.
    # Without this, a changing denominator/region could manufacture a 'gain'.
    common=safe & region & (shift_no_wrap(safe.astype(np.uint8),dy,dx)>0)
    n=int(common.sum())
    if n<min_pixels:
        return {'before':None,'after':None,'gain':None,'pixels':n}
    base=corr(ae,je,common,min_pixels)
    shifted=corr(ae,shift_no_wrap(je,dy,dx),common,min_pixels)
    if not np.isfinite(base) or not np.isfinite(shifted):
        return {'before':None,'after':None,'gain':None,'pixels':n}
    return {'before':float(base),'after':float(shifted),
            'gain':float(shifted-base),'pixels':n}


def analyze_green(a,av,j,jv,dx,dy,maxshift=8):
    if a.shape!=j.shape or a.ndim!=3 or a.shape[2]!=3:
        raise ValueError('Expected aligned HWC three-channel satellite patches')
    safe,_,_=screen_mask(a,av,j,jv,maxshift)
    ae=edge_feature(a[...,2],av)
    je=edge_feature(j[...,2],jv)
    h,w=safe.shape
    q=candidate_windows(h,w)
    quads={k:score_green(ae,je,safe,dx,dy,roi)
           for k,roi in q.items() if k!='center'}
    full=score_green(ae,je,safe,dx,dy,(0,h,0,w),min_pixels=1200)
    positive=sum(1 for x in quads.values() if x['gain'] is not None and x['gain']>=.02)
    negative=sum(1 for x in quads.values() if x['gain'] is not None and x['gain']<=-.02)
    available=sum(x['gain'] is not None for x in quads.values())
    # Independent GREEN-only search. This is diagnostic; never supersedes the
    # frozen red/NIR-selected shift. Compare its best offset with the original.
    fg={'red':ae,'nir':ae}
    gg={'red':je,'nir':je}
    green_best=search_region(fg,gg,safe,(0,h,0,w),maxshift=min(maxshift,6),min_pixels=1200)
    if green_best is None:
        gx=gy=None
        dist=None
    else:
        gx=int(green_best['dx']);gy=int(green_best['dy'])
        dist=float(np.hypot(dx-gx,dy-gy))
    supported=(full['gain'] is not None and full['gain']>=.035 and full['after']>=.25
               and available>=2 and positive>=2 and negative==0
               and dist is not None and dist<=1.5)
    if dx==0 and dy==0:
        supported=False
    # Explicitly label as spectral evidence, NOT registration acceptance.
    status='green_supports_candidate_provisionally' if supported else 'green_does_not_confirm'
    return {'status':status,'dx':dx,'dy':dy,'safe_fraction':float(safe.mean()),
            'green_corr_at_frozen_shift':full,
            'green_best_dx':gx,'green_best_dy':gy,'green_best_distance_px':dist,
            'quadrants_available':available,'quadrants_improve':positive,
            'quadrants_worsen':negative,'quadrants':quads,
            'green_edges_april':ae,'green_edges_june':je,
            'safe_mask':safe}


def edge_overlay(a_edge,j_edge,valid,dx,dy):
    sj=shift_no_wrap(j_edge,dy,dx)
    good=valid & (shift_no_wrap(valid.astype('uint8'),dy,dx)>0)
    ar=np.clip(a_edge/1.6,0,1)
    jr=np.clip(sj/1.6,0,1)
    rgb=np.stack([ar,jr,np.zeros_like(ar)],axis=2)
    rgb[~good]=0
    return Image.fromarray(np.uint8(rgb*255),'RGB')


def make_panel(a,av,j,jv,res,tile):
    dx,dy=res['dx'],res['dy']
    js=shift_no_wrap(j,dy,dx)
    jvs=shift_no_wrap(jv.astype('uint8'),dy,dx)>0
    ae,je,safe=res['green_edges_april'],res['green_edges_june'],res['safe_mask']
    views=[('April (independent stretch)',display_auto(a,av)),
           ('June (independent stretch)',display_auto(j,jv)),
           ('Green edges: NO SHIFT',edge_overlay(ae,je,safe,0,0)),
           ('Green edges: frozen shift',edge_overlay(ae,je,safe,dx,dy)),
           ('Checkerboard: frozen shift',checker(a,js,av,jvs,n=16))]
    width=5*218+6*8
    im=Image.new('RGB',(width,275),'white')
    dr=ImageDraw.Draw(im)
    for i,(title,image) in enumerate(views):
        x=8+i*226
        dr.text((x,8),title,fill='black')
        im.paste(image.resize((210,210)),(x,32))
    main=res['green_corr_at_frozen_shift']
    score=('unavailable' if main['gain'] is None else
           f"{main['before']:.3f}->{main['after']:.3f} (gain {main['gain']:+.3f})")
    dr.text((8,250),f'{tile} frozen ({dx:+d},{dy:+d}) | Green corr {score} | {res["status"]}',fill='black')
    return im


def write_outputs(rows,panels,out):
    out.mkdir(parents=True,exist_ok=True)
    fields=['tile_id','prior_status','green_status','frozen_dx','frozen_dy','safe_fraction',
            'green_corr_before','green_corr_after','green_corr_gain','green_pixels',
            'green_best_dx','green_best_dy','green_best_distance_px',
            'quadrants_available','quadrants_improve','quadrants_worsen']
    with (out/'green_band_verification.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r in rows:w.writerow({k:r.get(k) for k in fields})
    meta={'scope':'Spectral holdout using Green band, not independent geographic registration',
          'warning':'No shifts applied. Positive evidence is provisional; cloud masks are heuristic.',
          'results':rows}
    (out/'green_band_verification.json').write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8')
    if panels:
        sheet=Image.new('RGB',(max(x.width for x in panels),len(panels)*275+30),'white')
        pen=ImageDraw.Draw(sheet)
        pen.text((8,10),'APRIL/JUNE | GREEN-BAND CHECK OF FROZEN RED/NIR SHIFTS | NO WARP APPLIED',fill='black')
        for i,p in enumerate(panels):sheet.paste(p,(0,30+i*275))
        sheet.save(out/'green_band_verification.png')


def self_test():
    rng=np.random.default_rng(11)
    a0=rng.uniform(70,180,(256,256)).astype('float32')
    ai=Image.fromarray(np.uint8(a0),'L')
    d=ImageDraw.Draw(ai)
    for x in (22,65,104,164,211):d.line((x,10,x+8,242),fill=230,width=3)
    for y in (40,91,159,208):d.line((10,y,245,y+4),fill=55,width=4)
    x=np.asarray(ai,np.float32)
    a=np.stack([80+.58*x,40+.35*x,52+.30*x],axis=-1).astype('float32')
    desired_dx,desired_dy=2,-3
    j=shift_no_wrap(a,-desired_dy,-desired_dx)
    # Mild date-dependent radiometry shifts with all bands still preserving edges.
    j=j*1.04+3
    v=np.ones((256,256),bool)
    # This particular test uses all pixels to test the held-out scoring math.
    ae=edge_feature(a[...,2],v)
    je=edge_feature(j[...,2],v)
    roi=(0,256,0,256)
    good=score_green(ae,je,v,desired_dx,desired_dy,roi)
    bad=score_green(ae,je,v,-desired_dx,-desired_dy,roi)
    assert good['after'] is not None and good['gain']>.10,good
    assert good['after']>bad['after']+.1,(good,bad)
    r=analyze_green(a,v,j,v,desired_dx,desired_dy)
    assert r['green_best_dx'] is not None and abs(r['green_best_dx']-desired_dx)<=1,r
    assert abs(r['green_best_dy']-desired_dy)<=1,r
    entry={'tile_id':'T999','prior_status':'provisional_local_candidate',
           'green_status':r['status'],'frozen_dx':desired_dx,'frozen_dy':desired_dy,
           'safe_fraction':r['safe_fraction'],'green_corr_before':good['before'],
           'green_corr_after':good['after'],'green_corr_gain':good['gain'],
           'green_pixels':good['pixels'],'green_best_dx':r['green_best_dx'],
           'green_best_dy':r['green_best_dy'],'green_best_distance_px':r['green_best_distance_px'],
           'quadrants_available':r['quadrants_available'],
           'quadrants_improve':r['quadrants_improve'],'quadrants_worsen':r['quadrants_worsen']}
    with tempfile.TemporaryDirectory() as td:
        t=Path(td)
        panel=make_panel(a,v,j,v,r,'T999')
        write_outputs([entry],[panel],t)
        assert (t/'green_band_verification.csv').exists()
        assert (t/'green_band_verification.json').exists()
        assert (t/'green_band_verification.png').exists()
        dat=json.loads((t/'green_band_verification.json').read_text())
        assert len(dat['results'])==1
    print('GREEN-BAND REGISTRATION SELF-TEST PASSED')
    print(f'Synthetic known shift ({desired_dx:+d},{desired_dy:+d}) verified, wrong shift rejected.')
    print('CSV, JSON and preview generation passed. No real satellite data accessed.')


def main():
    args=args_parser()
    if args.self_test:
        self_test();return
    rows=[]; panels=[]
    auto=frozen_rows(args.auto_csv)
    tile_ids=[x.strip().upper() for x in args.tiles.split(',') if x.strip()]
    if not tile_ids:raise ValueError('No tile IDs selected')
    print('APRIL/JUNE | GREEN-BAND VERIFICATION OF PREVIOUSLY FROZEN SHIFTS')
    print('Red/NIR candidates are evaluated with Green edges, WITHOUT retuning.')
    for tid in tile_ids:
        if tid not in auto:raise ValueError(f'{tid} not in automatic registration CSV')
        prior=auto[tid]
        dx=int(prior['dx']);dy=int(prior['dy'])
        record=get_record(args.qc_csv,tid)
        a,av,j,jv,_=load_tile(args,record)
        v=analyze_green(a,av,j,jv,dx,dy,args.max_shift)
        if prior['status']!='provisional_local_candidate':
            # Never promote a previously rejected candidate based on another band.
            v['status']='prior_shift_rejected_not_approved'
        score=v['green_corr_at_frozen_shift']
        result={'tile_id':tid,'prior_status':prior['status'],'green_status':v['status'],
                'frozen_dx':dx,'frozen_dy':dy,'safe_fraction':v['safe_fraction'],
                'green_corr_before':score['before'],'green_corr_after':score['after'],
                'green_corr_gain':score['gain'],'green_pixels':score['pixels'],
                'green_best_dx':v['green_best_dx'],'green_best_dy':v['green_best_dy'],
                'green_best_distance_px':v['green_best_distance_px'],
                'quadrants_available':v['quadrants_available'],
                'quadrants_improve':v['quadrants_improve'],
                'quadrants_worsen':v['quadrants_worsen']}
        rows.append(result)
        panels.append(make_panel(a,av,j,jv,v,tid))
        b='n/a' if score['before'] is None else f'{score["before"]:.3f}->{score["after"]:.3f}'
        print(f'{tid}: prior {prior["status"]} | green corr {b} | '
              f'green best ({v["green_best_dx"]},{v["green_best_dy"]}) vs frozen ({dx:+d},{dy:+d}) | '
              f'quadrants {v["quadrants_improve"]}/{v["quadrants_available"]} improved | '
              f'{v["status"]}',flush=True)
    write_outputs(rows,panels,args.out)
    print('\nGREEN-BAND QC COMPLETE')
    print('CSV:',args.out/'green_band_verification.csv')
    print('JSON:',args.out/'green_band_verification.json')
    print('Preview:',args.out/'green_band_verification.png')
    print('No scene shifts applied. No model/test benchmark modified.')

if __name__=='__main__':
    main()
