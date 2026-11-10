r"""April/June Mumbai: fair-pixel registration shift audit (READ ONLY).

Purpose: The Red/NIR search and later Green-band test disagreed. The original
Red/NIR printed 'before' and 'after' correlations using *different* valid pixel
supports. This audit re-scores ZERO, frozen Red/NIR, reverse Red/NIR, and Green
suggested shifts on the SAME safe pixels, for ALL THREE spectral bands.

It DOES NOT register/reproject any final image or validate a cloud mask. It
uses prior candidate CSVs as frozen inputs and does not tune or approve shifts.

From project root:
  .\\.venv\\Scripts\\python.exe -u .\\src\\audit_april_june_registration_bias.py --self-test
  .\\.venv\\Scripts\\python.exe -u .\\src\\audit_april_june_registration_bias.py

Requires numpy, rasterio, Pillow, and existing project QC scripts; no pandas.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from audit_april_june_temporal_reference import ROOT, shift_no_wrap
from auto_register_april_june_v1 import edge_feature, corr, screen_mask
from verify_april_june_landmark_registration import get_record, load_tile

AUTO_CSV = ROOT / 'outputs/april_june_automatic_registration_qc/automatic_registration_summary.csv'
GREEN_CSV = ROOT / 'outputs/april_june_green_band_qc/green_band_verification.csv'
QC_CSV = ROOT / 'outputs/april_june_targeted_qc/sampled_cloud_reference_qc.csv'
OUTPUT = ROOT / 'outputs/april_june_registration_bias_audit'
BANDS = ('NIR_B4', 'RED_B3', 'GREEN_B2')


def arguments():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--auto-csv', type=Path, default=AUTO_CSV)
    ap.add_argument('--green-csv', type=Path, default=GREEN_CSV)
    ap.add_argument('--qc-csv', type=Path, default=QC_CSV)
    ap.add_argument('--out', type=Path, default=OUTPUT)
    ap.add_argument('--april-dir', type=Path, default=ROOT/'data/raw/mumbai_scenes/2025-04-16')
    ap.add_argument('--june-dir', type=Path, default=ROOT/'data/raw/mumbai_scenes/2025-06-03')
    ap.add_argument('--tile-size', type=int, default=256)
    ap.add_argument('--tiles', default='T032,T140,T103,T080,T104')
    ap.add_argument('--max-shift', type=int, default=8)
    return ap.parse_args()


def rows_by_tile(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input CSV: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    out={}
    for row in rows:
        tid=row.get('tile_id','').upper().strip()
        if not tid or tid in out:
            raise ValueError(f'Missing or duplicated tile_id: {tid}')
        out[tid]=row
    return out


def audit_patch(a, av, j, jv, rednir, green, maxshift=8, min_pixels=1200):
    if a.shape != j.shape or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError('Expect paired HWC=3 arrays')
    if av.shape != a.shape[:2] or jv.shape != av.shape:
        raise ValueError('Invalid mask dimensions')
    safe,_,_ = screen_mask(a,av,j,jv,maxshift)
    shifts={
        'no_extra_shift':(0,0),
        'frozen_rednir':tuple(int(v) for v in rednir),
        'reverse_rednir':tuple(-int(v) for v in rednir),
        'independent_green':tuple(int(v) for v in green),
    }
    # A FIXED spatial support for ALL tested shifts AND baseline. This avoids
    # reporting spurious gain from changing which pixels are included.
    common=safe.copy()
    for dx,dy in shifts.values():
        common &= shift_no_wrap(safe.astype(np.uint8),dy,dx)>0
    n=int(common.sum())
    result={'support_pixels':n,'safe_fraction':float(safe.mean()),
            'fixed_support_fraction':float(common.mean()),'shifts':shifts,'scores':{},
            'note':'Same fixed spatial support across all candidate shifts and bands'}
    if n < min_pixels:
        result['status']='insufficient_fixed_support'
        return result
    fa=[edge_feature(a[:,:,k],av) for k in range(3)]
    fj=[edge_feature(j[:,:,k],jv) for k in range(3)]
    for name,(dx,dy) in shifts.items():
        values={}
        for k,label in enumerate(BANDS):
            v=corr(fa[k],shift_no_wrap(fj[k],dy,dx),common,minimum=min_pixels)
            values[label]=round(float(v),6) if np.isfinite(v) else None
        # Explicit fixed 60% Red + 40% NIR score matching earlier search.
        a_mix=.6*fa[1]+.4*fa[0]
        j_mix=.6*fj[1]+.4*fj[0]
        s=corr(a_mix,shift_no_wrap(j_mix,dy,dx),common,minimum=min_pixels)
        values['RED_NIR_COMBINED']=round(float(s),6) if np.isfinite(s) else None
        result['scores'][name]=values
    baseline=result['scores']['no_extra_shift']
    frozen=result['scores']['frozen_rednir']
    diffs={k:(round(frozen[k]-baseline[k],6) if frozen[k] is not None and baseline[k] is not None else None)
           for k in list(BANDS)+['RED_NIR_COMBINED']}
    result['frozen_gain_same_pixels']=diffs
    # A diagnostic ONLY, not approval. A candidate with negative Green evidence
    # or no Red+NIR improvement must not be accepted for reconstruction.
    if any(x is None for x in diffs.values()):
        status='insufficient_or_flat_features'
    elif diffs['GREEN_B2'] < -0.005:
        status='rejected_green_band_worsened'
    elif diffs['RED_NIR_COMBINED'] <= 0.01:
        status='rejected_no_robust_rednir_gain'
    elif min(diffs['NIR_B4'],diffs['RED_B3']) <= 0:
        status='rejected_band_disagreement'
    else:
        status='provisional_only_needs_geographic_tiepoints'
    result['status']=status
    return result


def score_csv(rows, out):
    out.mkdir(parents=True, exist_ok=True)
    fields=['tile_id','status','prior_status','rednir_shift','green_shift',
            'support_pixels','safe_fraction','fixed_support_fraction','band','zero_corr',
            'frozen_corr','reverse_corr','green_best_corr','frozen_gain_fixed',
            'prior_reported_corr_before','prior_reported_corr_after']
    with (out/'registration_shift_bias_per_band.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        for r in rows:
            for band in list(BANDS)+['RED_NIR_COMBINED']:
                d={'tile_id':r['tile_id'],'status':r['result']['status'],
                   'prior_status':r['prior_status'],
                   'rednir_shift':str(tuple(r['result']['shifts']['frozen_rednir'])),
                   'green_shift':str(tuple(r['result']['shifts']['independent_green'])),
                   'support_pixels':r['result']['support_pixels'],
                   'safe_fraction':round(r['result']['safe_fraction'],4),
                   'fixed_support_fraction':round(r['result']['fixed_support_fraction'],4),
                   'band':band,
                   'prior_reported_corr_before':r.get('prior_corr_before'),
                   'prior_reported_corr_after':r.get('prior_corr_after')}
                for variant,field in [('no_extra_shift','zero_corr'),('frozen_rednir','frozen_corr'),
                                      ('reverse_rednir','reverse_corr'),('independent_green','green_best_corr')]:
                    d[field]=r['result']['scores'].get(variant,{}).get(band)
                d['frozen_gain_fixed']=r['result'].get('frozen_gain_same_pixels',{}).get(band)
                w.writerow(d)
    payload={'warning':'QC only. No shifts applied; do not treat provisional candidates as registered.',
             'comparison':'All shifts and spectral bands scored on the SAME fixed valid pixels',
             'tiles':rows}
    (out/'registration_shift_bias_summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')


def make_plot(rows, out):
    out.mkdir(parents=True,exist_ok=True)
    width=1100; row_h=200
    im=Image.new('RGB',(width,48+len(rows)*row_h),'white')
    draw=ImageDraw.Draw(im)
    draw.text((15,10),'APRIL–JUNE SHIFT BIAS AUDIT | All variants scored on SAME pixels',fill='#141414')
    draw.text((15,27),'Bars show edge correlation; higher is better, but not proof of geospatial registration.',fill='#555555')
    names=['no_extra_shift','frozen_rednir','reverse_rednir','independent_green']
    colors=['#596274','#c16b5a','#8b73ae','#349c8c']
    for i,r in enumerate(rows):
        y=48+i*row_h
        meta=r['result']
        draw.rectangle((5,y,width-8,y+row_h-4),outline='#dddddd')
        title=f"{r['tile_id']} | {meta['status']} | support={meta['support_pixels']} px | frozen {meta['shifts']['frozen_rednir']} | Green-best {meta['shifts']['independent_green']}"
        draw.text((14,y+8),title,fill='#111111')
        for band_i,band in enumerate(['RED_NIR_COMBINED','GREEN_B2']):
            by=y+33+band_i*78
            draw.text((15,by),band,fill='#333333')
            for k,name in enumerate(names):
                v=meta['scores'].get(name,{}).get(band)
                lab=name.replace('no_extra_shift','zero').replace('frozen_rednir','frozen').replace('reverse_rednir','inverse').replace('independent_green','green-best')
                start_x=220; end_x=830
                yy=by+k*15
                draw.text((97,yy),lab,fill=colors[k])
                draw.line((start_x,yy+6,end_x,yy+6),fill='#eeeeee',width=2)
                if v is not None:
                    stop=start_x+max(0,min(1,float(v)))*(end_x-start_x)
                    draw.line((start_x,yy+6,stop,yy+6),fill=colors[k],width=6)
                    draw.text((850,yy),f'{v:.3f}',fill='#333333')
    im.save(out/'registration_shift_bias_comparison.png')


def self_test():
    from PIL import ImageFilter
    rng=np.random.default_rng(20)
    raw=rng.uniform(0,255,(256,256)).astype(np.uint8)
    pic=Image.fromarray(raw,'L')
    draw=ImageDraw.Draw(pic)
    for xx in (32,76,124,194):draw.line((xx,7,xx+7,245),fill=230,width=5)
    for yy in (45,99,161,217):draw.line((8,yy,242,yy+5),fill=35,width=5)
    sm=np.asarray(pic.filter(ImageFilter.GaussianBlur(1.0)),dtype=np.float32)
    a=np.stack([145+.17*sm,65+.25*sm,89+.20*sm],axis=-1).astype(np.float32)
    true=(3,-2)
    j=shift_no_wrap(a,-true[1],-true[0])
    j=j*1.025+1
    av=np.ones((256,256),bool);jv=av.copy()
    good=audit_patch(a,av,j,jv,true,true)
    assert good['support_pixels']>15000,good
    for band in BANDS:
        assert good['frozen_gain_same_pixels'][band]>.08,(band,good)
    # A mismatched Green channel must lead to a warning, not blind promotion.
    bad_j=j.copy()
    bad_j[:,:,2]=shift_no_wrap(a[:,:,2],-3,1)
    bad=audit_patch(a,av,bad_j,jv,true,(-1,3))
    assert bad['frozen_gain_same_pixels']['GREEN_B2']<-.005,bad
    assert 'rejected' in bad['status'],bad
    with tempfile.TemporaryDirectory() as td:
        record={'tile_id':'T999','prior_status':'provisional_local_candidate','result':good,
                'prior_corr_before':'0.20','prior_corr_after':'0.60'}
        score_csv([record],Path(td));make_plot([record],Path(td))
        assert (Path(td)/'registration_shift_bias_per_band.csv').is_file()
        assert (Path(td)/'registration_shift_bias_comparison.png').is_file()
        assert json.loads((Path(td)/'registration_shift_bias_summary.json').read_text())['tiles']
    print('FAIR-SUPPORT REGISTRATION AUDIT SELF-TEST PASSED')
    print('Known shift recovered in all three bands; contradictory Green evidence rejected.')
    print('Outputs verified on sample data; real satellite files not accessed.')


def main():
    args=arguments()
    if args.self_test:
        self_test();return
    auto=rows_by_tile(args.auto_csv)
    green=rows_by_tile(args.green_csv)
    tiles=[s.strip().upper() for s in args.tiles.split(',') if s.strip()]
    print('APRIL–JUNE | FIXED-SUPPORT REGISTRATION BIAS AUDIT | NO SHIFTS APPLIED')
    print('Reason: old Red/NIR before/after reports may use different pixel supports.')
    output=[]
    for tid in tiles:
        if tid not in auto or tid not in green:
            raise ValueError(f'Tile {tid} absent from automatic or Green CSV')
        ar=auto[tid];gr=green[tid]
        if gr['green_best_dx'] in ('',None) or gr['green_best_dy'] in ('',None):
            print(f'{tid}: no independent Green offset available: skipped')
            continue
        frozen=(int(ar['dx']),int(ar['dy']))
        independent=(int(gr['green_best_dx']),int(gr['green_best_dy']))
        rec=get_record(args.qc_csv,tid)
        a,av,j,jv,_=load_tile(args,rec)
        s=audit_patch(a,av,j,jv,frozen,independent,args.max_shift)
        row={'tile_id':tid,'prior_status':ar['status'],'prior_corr_before':ar.get('corr_before'),
             'prior_corr_after':ar.get('corr_after'),'result':s}
        output.append(row)
        gains=s.get('frozen_gain_same_pixels',{})
        def fmt(v):return 'N/A' if v is None else f'{v:+.4f}'
        print(f"{tid}: frozen {frozen}; Green-best {independent}; support {s['support_pixels']}; "
              f"FAIR gains Red/NIR={fmt(gains.get('RED_NIR_COMBINED'))}, Green={fmt(gains.get('GREEN_B2'))}; "
              f"{s['status']}")
    if not output:raise RuntimeError('No tiles evaluated')
    score_csv(output,args.out)
    make_plot(output,args.out)
    print('\nAUDIT COMPLETE — diagnostic only, no accepted shifts')
    print('Table:',args.out/'registration_shift_bias_per_band.csv')
    print('Summary:',args.out/'registration_shift_bias_summary.json')
    print('Graphic:',args.out/'registration_shift_bias_comparison.png')
    print('Do NOT shift any full-resolution scene based on this audit.')


if __name__=='__main__':
    main()
