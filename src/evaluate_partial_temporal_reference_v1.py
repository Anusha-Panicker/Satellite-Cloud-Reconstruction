r"""Mumbai April/June restricted temporal reference feasibility gate.

IMPORTANT: output is diagnostic ONLY. No georegistration or real cloud
reconstruction is approved or written. April is 48 days earlier and NOT June
truth. Uses same-date Red band only for local alignment diagnostic to avoid
confounding disagreement among June spectral bands.

From Cloud_Reconstruction root (PowerShell):
    .\.venv\Scripts\python.exe -u .\src\evaluate_partial_temporal_reference_v1.py --self-test
    .\.venv\Scripts\python.exe -u .\src\evaluate_partial_temporal_reference_v1.py

Depends on existing project modules and NumPy / Pillow / Rasterio, no pandas.
"""
from __future__ import annotations
import argparse
import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import rasterio
from rasterio.transform import from_origin

from audit_april_june_temporal_reference import ROOT, shift_no_wrap, preview_rgb
from preview_real_temporal_transfer_unregistered import qc_records, make_candidate, mask_rgb
from verify_april_june_landmark_registration import load_tile
from review_april_june_cloud_and_alignment import display_auto

DEFAULT_CSV = ROOT/'outputs/april_june_targeted_qc/sampled_cloud_reference_qc.csv'
DEFAULT_OUT = ROOT/'outputs/april_june_partial_reference_gate'
NOTICE = ('READ-ONLY FEASIBILITY TEST. Red-only local alignment candidates are NOT approved '
          'georegistration. Cloud masks and April clear-reference thresholds are unvalidated. '
          'April imagery is not June ground truth; 48-day seasonal changes matter. '
          'No reconstruction or GeoTIFF is created.')


def args_parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--tiles', default='T041,T058')
    p.add_argument('--qc-csv', type=Path, default=DEFAULT_CSV)
    p.add_argument('--april-dir', type=Path, default=ROOT/'data/raw/mumbai_scenes/2025-04-16')
    p.add_argument('--june-dir', type=Path, default=ROOT/'data/raw/mumbai_scenes/2025-06-03')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--tile-size', type=int, default=256)
    p.add_argument('--max-shift', type=int, default=4)
    return p.parse_args()


def red_gradients(arr, valid):
    red = np.asarray(arr[..., 1], dtype=np.float32)
    arr0 = red[valid]
    if arr0.size < 500:
        return np.zeros((*red.shape, 2), np.float32)
    p2, p98 = np.percentile(arr0, (2, 98))
    scl = max(float(p98-p2), 1.0)
    im = np.uint8(np.clip((red-p2)*255.0/scl, 0, 255))
    blurred = np.asarray(Image.fromarray(im).filter(ImageFilter.GaussianBlur(radius=1.0)), np.float32)
    gy, gx = np.gradient(blurred)
    return np.stack([gx, gy], axis=-1)


def pearson(x, y, mask):
    a = np.asarray(x[mask], np.float64)
    b = np.asarray(y[mask], np.float64)
    if a.size < 400:
        return float('nan')
    a -= a.mean(); b -= b.mean()
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/norm) if norm > 1e-7 else float('nan')


def score(g1, g2, mask):
    z = [pearson(g1[..., k], g2[..., k], mask) for k in (0,1)]
    return float(np.mean(z)) if all(np.isfinite(z)) else float('nan')


def shift_mixture(a,j,valid,maxshift=4):
    """Frozen-support local diagnostic. Select shift on spatial blocks, report on other blocks."""
    from review_april_june_cloud_and_alignment import candidate_masks
    m = candidate_masks(a,j,valid,valid)
    safe = m['valid'] & m['april_ref'] & (~m['old']) & (~m['core'])
    # Require an 11x11 (for radius 5) neighborhood free of flagged pixels;
    # ensures the SAME geographical pixel support for all candidate shifts.
    radius = maxshift + 2
    filtered = Image.fromarray(np.uint8(safe)*255).filter(ImageFilter.MinFilter(2*radius+1))
    stable = (np.asarray(filtered) == 255)
    stable[:radius,:] = False; stable[-radius:,:] = False
    stable[:,:radius] = False; stable[:,-radius:] = False
    h,w = stable.shape
    y,x = np.indices((h,w))
    grid = ((y>=h//2) ^ (x>=w//2))
    train = stable & grid
    held = stable & (~grid)
    data = {'stable_frac':float(stable.mean()), 'train_pixels':int(train.sum()),
            'holdout_pixels':int(held.sum()), 'candidate_dx':0,'candidate_dy':0,
            'train_corr_before':None,'train_corr_after':None,
            'holdout_corr_before':None,'holdout_corr_after':None,
            'holdout_gain':None,'status':'insufficient_clear_reference_controls'}
    if train.sum()<800 or held.sum()<800:
        return data, stable
    ga = red_gradients(a, valid)
    gj = red_gradients(j, valid)
    tr_before = score(ga,gj,train)
    ho_before = score(ga,gj,held)
    if not np.isfinite(tr_before) or not np.isfinite(ho_before):
        data['status']='insufficient_stable_texture'
        return data,stable
    choices=[]
    for dy in range(-maxshift,maxshift+1):
        for dx in range(-maxshift,maxshift+1):
            shifted=shift_no_wrap(gj,dy,dx)
            s=score(ga,shifted,train)
            if np.isfinite(s): choices.append((s,dx,dy))
    if not choices:
        data['status']='insufficient_stable_texture'; return data,stable
    train_best,dx,dy = max(choices)
    held_best=score(ga,shift_no_wrap(gj,dy,dx),held)
    gain=float(held_best-ho_before) if np.isfinite(held_best) else float('nan')
    data.update(candidate_dx=int(dx),candidate_dy=int(dy),
                train_corr_before=round(tr_before,4),train_corr_after=round(train_best,4),
                holdout_corr_before=round(ho_before,4),
                holdout_corr_after=round(held_best,4) if np.isfinite(held_best) else None,
                holdout_gain=round(gain,4) if np.isfinite(gain) else None)
    # This is NOT a georegistration approval even if a local shift is supported.
    if dx==dy==0:
        data['status']='zero_shift_best_locally_not_georegistration'
    elif (np.isfinite(gain) and train_best-tr_before>0.08 and
          gain>0.08 and held_best>=0.35):
        data['status']='provisional_red_local_candidate_not_georegistration'
    else:
        data['status']='local_shift_not_independently_supported'
    return data,stable


def create_preview(rows,path):
    size=226; gap=9; titles=['April fixed','June fixed','April auto (reference)','June auto (clouds)',
                             'Strict proxy (white)','Nominal-grid transfer*','Red control support (white)']
    top=76; rh=size+54
    w=gap+len(titles)*(size+gap)
    h=top+len(rows)*rh+29
    canvas=Image.new('RGB',(w,h),'white')
    d=ImageDraw.Draw(canvas)
    d.text((10,5),'APRIL-JUNE | PARTIAL-TILE TEMPORAL REFERENCE FEASIBILITY GATE',fill='black')
    d.text((10,23),'*Illustrative nominal-grid copy only. NO approved registration or validated June reconstruction.',fill='black')
    d.text((10,41),'Red-band local shifts tested on separate training and held-out spatial blocks; NOT APPLIED.',fill='black')
    for k,t in enumerate(titles): d.text((gap+k*(size+gap),top-17),t,fill='black')
    for i,r in enumerate(rows):
        a,av,j,jv,res,stable = (r['a'],r['av'],r['j'],r['jv'],r['result'],r['stable'])
        imgs=[preview_rgb(a,av),preview_rgb(j,jv),display_auto(a,av),display_auto(j,jv),
              mask_rgb(res['selected'],res['masks']['valid']),preview_rgb(res['demo'],res['masks']['valid']),
              mask_rgb(stable,res['masks']['valid'])]
        y=top+i*rh
        for k,img in enumerate(imgs):
            canvas.paste(img.resize((size,size),Image.Resampling.NEAREST),(gap+k*(size+gap),y))
        q=r['report']
        d.text((gap,y+size+7),f"{q['tile_id']}: selected {q['selected_fraction']:.1%}, stable Red controls {q['stable_frac']:.1%}, "
                              f"holdout gain={q['holdout_gain']}, status={q['status']}",fill='black')
    d.text((9,h-21),'NO MODEL TRAINING. NO GEOTIFF. NO APPROVED CLOUD MASK OR REGISTRATION.',fill='black')
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path)


def run(opts):
    if opts.max_shift<1 or opts.max_shift>10: raise ValueError('max-shift must be 1..10')
    if opts.tile_size<64 or opts.tile_size>1024:raise ValueError('tile-size must be 64..1024')
    ids=[s.strip().upper() for s in opts.tiles.split(',') if s.strip()]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Unique tile IDs required')
    qc=qc_records(opts.qc_csv)
    if any(t not in qc for t in ids):raise ValueError(f'Tile missing from QC CSV: {[t for t in ids if t not in qc]}')
    rows=[]
    print('PARTIAL-TILE TEMPORAL SOURCE GATE | NO RECONSTRUCTION APPROVED')
    for tid in ids:
        a,av,j,jv,_=load_tile(opts,qc[tid])
        res=make_candidate(a,av,j,jv)
        if np.any(res['selected'] & ~(av & jv)):
            raise AssertionError('Selected mask outside common source-valid pixels')
        metric,stable=shift_mixture(a,j,av&jv,opts.max_shift)
        report={'tile_id':tid,'selected_fraction':round(res['selected_fraction_valid'],4),
                'strict_core_fraction':round(res['strict_core_fraction_valid'],4),
                'april_ref_proxy_fraction':round(res['april_ref_fraction_valid'],4),
                'valid_fraction':round(float((av & jv).mean()),4),**metric,
                'registration_approved':False,'real_cloud_mask_validated':False,
                'data_written':'none'}
        rows.append(dict(tile_id=tid,a=a,av=av,j=j,jv=jv,result=res,stable=stable,report=report))
        print(f"  {tid}: selected={report['selected_fraction']:.1%} | Red stable={report['stable_frac']:.1%} "
              f"| candidate ({report['candidate_dx']:+d},{report['candidate_dy']:+d}) "
              f"| holdout gain {report['holdout_gain']} | {report['status']}",flush=True)
    opts.out.mkdir(parents=True,exist_ok=True)
    path=opts.out/'partial_temporal_feasibility.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]['report']))
        writer.writeheader();writer.writerows([r['report'] for r in rows])
    png=opts.out/'partial_temporal_reference_preview.png'
    create_preview(rows,png)
    prov=opts.out/'partial_temporal_feasibility.json'
    prov.write_text(json.dumps({'notice':NOTICE,'tile_metrics':[r['report'] for r in rows],
                      'shifts_are_not_applied':True,'uses_red_band_only_for_shift_diagnostic':True,
                      'april_acquisition':'2025-04-16','june_acquisition':'2025-06-03'},indent=2),encoding='utf8')
    print('FEASIBILITY GATE COMPLETE\nCSV:',path,'\nPNG:',png,'\nProvenance:',prov)
    print(NOTICE)
    return rows


def self_test():
    from rasterio.transform import from_origin
    rng=np.random.default_rng(21)
    h=w=128; yy,xx=np.indices((h,w)); pix=np.zeros((h,w,3),np.float32)
    noise=rng.normal(0,12,(h,w)); base=110+35*np.sin(xx/7)+27*np.cos(yy/11)+noise
    pix[...,0]=np.clip(base+65,1,300);pix[...,1]=np.clip(base,1,180);pix[...,2]=np.clip(base+10,1,180)
    jun=pix.copy()
    jun[23:69,21:72]=[420,380,370]
    val=np.ones((h,w),bool)
    res=make_candidate(pix,val,jun,val)
    assert float(res['selected'].mean())>0.03
    assert np.all(res['demo'][~res['selected']]==jun[~res['selected']])
    stat,stable=shift_mixture(pix,jun,val,maxshift=4)
    assert 'status' in stat and stable.shape == (h,w)
    with tempfile.TemporaryDirectory() as td:
        r=Path(td)
        for title,data in [('apr',pix),('jun',jun)]:
            folder=r/title;folder.mkdir()
            for idx,band in enumerate((4,3,2)):
                with rasterio.open(folder/f'BAND{band}.tif','w',driver='GTiff',count=1,
                    width=w,height=h,dtype='uint16',crs='EPSG:32643',
                    transform=from_origin(500000,2100000,5,5),nodata=0) as ds:
                    ds.write(np.uint16(data[:,:,idx]),1)
        q=r/'qc.csv';q.write_text('tile_id,june_col,june_row\nT001,0,0\n',encoding='utf8')
        opts=SimpleNamespace(april_dir=r/'apr',june_dir=r/'jun',qc_csv=q,out=r/'out',
                             tiles='T001',tile_size=128,max_shift=4)
        rows=run(opts)
        assert len(rows)==1
        for fname in ('partial_temporal_feasibility.csv','partial_temporal_feasibility.json',
                      'partial_temporal_reference_preview.png'):
            assert (r/'out'/fname).is_file(),fname
    print('PARTIAL TEMPORAL PILOT SELF-TEST PASSED | temporary synthetic GeoTIFFs only')

if __name__=='__main__':
    opts=args_parse()
    if opts.self_test:self_test()
    else:run(opts)
