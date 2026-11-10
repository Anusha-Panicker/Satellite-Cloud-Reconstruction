r"""Read-only automatic LOCAL registration QC for Mumbai April 16 / June 03, 2025.

Uses existing April/June QC tile coordinates and original GeoTIFFs reprojected
onto the June pixel grid by the existing load_tile() helper.

Algorithm: cloud-screened, normalized edge-map translation search over multiple
independent image subwindows; strict agreement/quality checks reject uncertain
results rather than applying a speculative shift. NIR/red independent sanity
checks are included. Requires only numpy, pillow, rasterio and existing scripts.

THIS IS A QC TOOL. It neither applies a correction to full scenes nor validates
cloud masks or proves temporal surface correspondence. The registration remains
provisional until verified using unmistakable stable structures on multiple
independent locations. No models, benchmarks, or source GeoTIFFs modified.

From Cloud_Reconstruction project root:
 .\.venv\Scripts\python.exe -u .\src\auto_register_april_june_v1.py --self-test
 .\.venv\Scripts\python.exe -u .\src\auto_register_april_june_v1.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from audit_april_june_temporal_reference import ROOT, shift_no_wrap
from review_april_june_cloud_and_alignment import candidate_masks, checker, display_auto
from verify_april_june_landmark_registration import get_record, load_tile

DEFAULT_QC = ROOT / 'outputs/april_june_targeted_qc/sampled_cloud_reference_qc.csv'
DEFAULT_OUT = ROOT / 'outputs/april_june_automatic_registration_qc'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--tiles', default='T032,T140,T103,T080,T104', help='Comma-separated tile IDs from QC CSV')
    p.add_argument('--max-shift', type=int, default=8, help='Integer pixel offsets +/- this magnitude')
    p.add_argument('--tile-size', type=int, default=256)
    p.add_argument('--april-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-04-16')
    p.add_argument('--june-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-06-03')
    p.add_argument('--qc-csv', type=Path, default=DEFAULT_QC)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def dilate(mask, size=11):
    if size % 2 != 1:
        raise ValueError('Dilate filter size must be odd')
    z = Image.fromarray(np.uint8(mask) * 255, 'L').filter(ImageFilter.MaxFilter(size))
    return np.asarray(z) > 0


def edge_feature(channel, valid):
    """Radiometry-tolerant edge magnitude: independent per-image, per-band stretch."""
    good = channel[np.asarray(valid, bool)]
    if good.size < 200:
        return np.zeros_like(channel, dtype=np.float32)
    p2, p98 = np.percentile(good, [2, 98])
    if p98 <= p2 + 1:
        return np.zeros_like(channel, dtype=np.float32)
    z = np.clip((channel - p2) / max(p98 - p2, 1.), 0., 1.)
    blurred = np.asarray(Image.fromarray(np.uint8(z * 255), 'L').filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32) / 255
    gy, gx = np.gradient(blurred)
    edge = np.hypot(gx, gy)
    scale = np.percentile(edge[valid], 93) if valid.any() else 1.
    if scale < .0005:
        return np.zeros_like(channel, dtype=np.float32)
    return np.clip(edge / scale, 0., 3.).astype(np.float32)


def features(img, valid):
    return {'red': edge_feature(img[..., 1], valid),
            'nir': edge_feature(img[..., 0], valid)}


def corr(a, b, mask, minimum=500):
    n = int(mask.sum())
    if n < minimum:
        return float('nan')
    aa = a[mask].astype(np.float64)
    bb = b[mask].astype(np.float64)
    aa -= aa.mean(); bb -= bb.mean()
    denom = np.sqrt(np.dot(aa, aa)*np.dot(bb, bb))
    if denom < 1e-8:
        return float('nan')
    return float(np.dot(aa, bb)/denom)


def screen_mask(a, av, j, jv, maxshift):
    m = candidate_masks(a, j, av, jv)
    # The broad proxy is deliberately conservative; it is NOT a real cloud map.
    cloud_like = m['old'] | m['core']
    # Include buffer around cloud edges to avoid matching evolving cloud contours.
    unsafe = dilate(cloud_like, size=11)
    valid = av & jv & ~unsafe
    # Suppress source borders / any shift-dependent edge pads.
    border = maxshift+3
    valid[:border] = False; valid[-border:] = False
    valid[:, :border] = False; valid[:, -border:] = False
    return valid, cloud_like, m


def search_region(a_feat, j_feat, safe, roi, maxshift, min_pixels=650):
    """Find shift to APPLY TO June so edge features match April."""
    h, w = safe.shape
    y0, y1, x0, x1 = roi
    if y1 > h or x1 > w:
        raise ValueError('ROI exceeds tile')
    region = np.zeros_like(safe, dtype=bool)
    region[y0:y1, x0:x1] = True
    if np.count_nonzero(safe & region) < min_pixels:
        return None
    j_safe = safe.astype(np.uint8)
    ag = 0.6 * a_feat['red'] + 0.4 * a_feat['nir']
    jg = 0.6 * j_feat['red'] + 0.4 * j_feat['nir']
    positions = []
    scores = []
    for dy in range(-maxshift, maxshift+1):
        for dx in range(-maxshift, maxshift+1):
            overlap = region & safe & (shift_no_wrap(j_safe, dy, dx) > 0)
            if overlap.sum() < min_pixels:
                continue
            shifted = shift_no_wrap(jg, dy, dx)
            score = corr(ag, shifted, overlap, min_pixels)
            if np.isfinite(score):
                positions.append((dx,dy))
                scores.append(score)
    if not positions:
        return None
    best_id = int(np.argmax(scores))
    dx, dy = positions[best_id]
    best = float(scores[best_id])
    # Best unrelated alternative excludes the immediate 2px neighborhood.
    others = [s for (x,y),s in zip(positions,scores) if abs(x-dx)>2 or abs(y-dy)>2]
    second = max(others) if others else float('nan')
    # Check independent NIR and RED correlations at selected offset.
    overlap = region & safe & (shift_no_wrap(j_safe, dy, dx) > 0)
    red = corr(a_feat['red'],shift_no_wrap(j_feat['red'],dy,dx),overlap,min_pixels)
    nir = corr(a_feat['nir'],shift_no_wrap(j_feat['nir'],dy,dx),overlap,min_pixels)
    before = corr(ag,jg,region & safe,min_pixels)
    return {'dx':dx,'dy':dy,'correlation':best,
            'gain':float(best-before) if np.isfinite(before) else float('nan'),
            'alternative_gap':float(best-second) if np.isfinite(second) else float('nan'),
            'red_corr':red,'nir_corr':nir,'sample_pixels':int(overlap.sum()),
            'roi':list(roi)}


def candidate_windows(h, w):
    """Four mostly independent quadrants and one wider central control."""
    if h < 128 or w < 128:
        raise ValueError('Expected >= 128x128 tiles')
    cy, cx = h//2, w//2
    # In 256 images, quadrants each 128x128; plus a larger central section.
    return {
        'upper_left': (0,cy,0,cx),
        'upper_right': (0,cy,cx,w),
        'lower_left': (cy,h,0,cx),
        'lower_right': (cy,h,cx,w),
        'center':(h//8,h-h//8,w//8,w-w//8),
    }


def analyze(a,av,j,jv,maxshift=8):
    if a.shape!=j.shape or a.ndim!=3 or a.shape[2]!=3 or av.shape!=a.shape[:2] or jv.shape!=av.shape:
        raise ValueError('Image dimensions mismatch')
    h,w=av.shape
    safe, cloud, masks = screen_mask(a,av,j,jv,maxshift)
    f_a,f_j = features(a,av),features(j,jv)
    safe_fraction = float(safe.mean())
    sub=[]
    for label,roi in candidate_windows(h,w).items():
        res=search_region(f_a,f_j,safe,roi,maxshift, min_pixels=max(500,(roi[1]-roi[0])*(roi[3]-roi[2])//10))
        if res is not None:
            res['window']=label
            sub.append(res)
    # Derive consensus from only independent spatial windows, not one global best.
    candidates=[s for s in sub if s['correlation']>=.18 and s['gain']>=.025
                and s['red_corr']>.08 and s['nir_corr']>.08]
    candidates_sorted=sorted(candidates,key=lambda z:z['correlation'],reverse=True)
    best_group=[]
    for anchor in candidates_sorted:
        group=[s for s in candidates if np.hypot(s['dx']-anchor['dx'],s['dy']-anchor['dy'])<=1.5]
        # Number of window names ensures candidate isn't double counted.
        if len(group)>len(best_group):
            best_group=group
    if best_group:
        dx=int(np.rint(np.median([r['dx'] for r in best_group])))
        dy=int(np.rint(np.median([r['dy'] for r in best_group])))
    else:
        dx=dy=0
    # Require at least two independent quadrant matches, not just center and a quadrant.
    n_quads=sum(s['window']!='center' for s in best_group)
    # Robust leave-in-place correlation on the broad safe area at the consensus offset.
    all_roi=(0,h,0,w)
    a_comb=.6*f_a['red']+.4*f_a['nir']; j_comb=.6*f_j['red']+.4*f_j['nir']
    j_safe=safe.astype(np.uint8)
    shifted_valid=safe & (shift_no_wrap(j_safe,dy,dx)>0)
    after=corr(a_comb,shift_no_wrap(j_comb,dy,dx),shifted_valid,800)
    before=corr(a_comb,j_comb,safe,800)
    gain=after-before if np.isfinite(after) and np.isfinite(before) else float('nan')
    agreement_fraction=(len(best_group)/max(1,len(sub)))
    accept=bool(len(best_group)>=3 and n_quads>=2 and agreement_fraction>=.55
                and np.isfinite(after) and after>=.28 and np.isfinite(gain) and gain>=.035
                and safe_fraction>=.18 and abs(dx)<maxshift and abs(dy)<maxshift)
    reason=('provisional_local_candidate' if accept else
            'uncertain_no_consistent_spatial_features')
    return {
        'status':reason,'dx':dx,'dy':dy,'safe_fraction':safe_fraction,
        'broad_cloud_proxy_fraction':float((cloud & av & jv).sum()/max(1,(av&jv).sum())),
        'n_windows_tested':len(sub),'n_qualified_windows':len(candidates),
        'n_consensus_windows':len(best_group),'n_consensus_quadrants':n_quads,
        'corr_before':float(before),'corr_after':float(after),'corr_gain':float(gain),
        'windows':sub,'consensus_window_names':[x['window'] for x in best_group],
        'note':'Translation estimate LOCAL and provisional, never automatically applied to satellite files.'
    }


def figure_for_tile(a,av,j,jv,r):
    # Show shift only as QC visualization, even if rejected; status overlaid.
    dx,dy=r['dx'],r['dy']
    js=shift_no_wrap(j,dy,dx)
    jvs=shift_no_wrap(jv,dy,dx)
    width=6*225+7*8
    canvas=Image.new('RGB',(width,279),'white')
    d=ImageDraw.Draw(canvas)
    titles=['April (auto stretch)','June (auto stretch)','Checkerboard at 0','Checkerboard candidate',
            'June shift candidate','Cloud-screened region']
    safe,_,_=screen_mask(a,av,j,jv,8)
    safe_im=Image.fromarray(np.uint8(np.repeat(safe[...,None],3,axis=2))*255,'RGB')
    ims=[display_auto(a,av),display_auto(j,jv),checker(a,j,av,jv,n=16),
         checker(a,js,av,jvs,n=16),display_auto(js,jvs),safe_im]
    for k,(title,im) in enumerate(zip(titles,ims)):
        x=8+k*233
        canvas.paste(im.resize((225,225)),(x,33))
        d.text((x,10),title,fill='black')
    d.text((8,262),f'{r["tile_id"]}: dx={dx:+d},dy={dy:+d} | {r["status"]} | {r["n_consensus_windows"]}/{r["n_windows_tested"]} windows agree',fill='black')
    return canvas


def create_outputs(rows,views,out):
    out.mkdir(parents=True,exist_ok=True)
    fields=['tile_id','qc_group','status','dx','dy','safe_fraction','broad_cloud_proxy_fraction',
            'n_windows_tested','n_qualified_windows','n_consensus_windows','n_consensus_quadrants',
            'corr_before','corr_after','corr_gain']
    with (out/'automatic_registration_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fields)
        writer.writeheader()
        writer.writerows([{k:r.get(k) for k in fields} for r in rows])
    winfields=['tile_id','window','dx','dy','correlation','gain','alternative_gap','red_corr','nir_corr','sample_pixels']
    with (out/'automatic_registration_windows.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,winfields);w.writeheader()
        for r in rows:
            for sub in r['windows']:
                w.writerow({k:(r['tile_id'] if k=='tile_id' else sub.get(k)) for k in winfields})
    def safe_json(value):
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return None
        if isinstance(value, dict):
            return {k:safe_json(v) for k,v in value.items()}
        if isinstance(value, list):
            return [safe_json(v) for v in value]
        return value
    digest={'warning':'No registration has been applied. Any candidates must be independently verified; mixed date radiometry and clouds limit confidence.',
            'n_tiles':len(rows),'n_provisional':sum(r['status']=='provisional_local_candidate' for r in rows),
            'results':[{k:v for k,v in r.items() if k!='windows'} for r in rows]}
    (out/'automatic_registration_summary.json').write_text(json.dumps(safe_json(digest),indent=2,allow_nan=False),encoding='utf-8')
    if views:
        w=max(im.width for im in views)
        sheet=Image.new('RGB',(w,len(views)*279+30),'white')
        pen=ImageDraw.Draw(sheet)
        pen.text((8,10),'APRIL–JUNE | AUTOMATIC LOCAL REGISTRATION QC | Candidates are NOT applied',fill='black')
        for k,im in enumerate(views):sheet.paste(im,(0,30+k*279))
        sheet.save(out/'automatic_registration_comparison.png')


def self_test():
    from PIL import ImageDraw
    rng=np.random.default_rng(1007)
    h=w=256
    base=np.asarray(Image.fromarray(np.uint8(rng.uniform(0,255,(h,w))), 'L').filter(ImageFilter.GaussianBlur(1.1)),np.float32)
    # Distinct persistent roads and shapes spanning multiple quadrants.
    base=np.uint8(base)
    im=Image.fromarray(base,'L');p=ImageDraw.Draw(im)
    for x in (21,62,113,169,218):p.line((x,2,x+11,255),fill=235,width=3)
    for y in (29,91,146,207):p.line((0,y,255,y+3),fill=45,width=3)
    for i in range(16):
        x=int(rng.integers(10,235));y=int(rng.integers(10,235))
        p.rectangle((x,y,x+int(rng.integers(5,19)),y+int(rng.integers(4,19))),fill=int(rng.integers(30,225)))
    base=np.asarray(im,np.float32)
    a=np.stack([100+.62*base,55+.40*base,70+.40*base],axis=2).astype('float32')
    dx,dy=3,-4
    j=shift_no_wrap(a,-dy,-dx)
    # Mild multi-date contrast changes, preserve geometry.
    j[...,0]=j[...,0]*1.08+5
    j[...,1]=j[...,1]*1.12+7
    j[...,2]=j[...,2]*1.10+6
    av=np.ones((h,w),bool);jv=np.ones((h,w),bool)
    # June cloudy corner; higher visible and lower NIR spectral contrast.
    j[18:75,15:85,:]=[370,355,350]
    res=analyze(a,av,j,jv,maxshift=8)
    if abs(res['dx']-dx)>1 or abs(res['dy']-dy)>1:
        raise AssertionError(f'Known synthetic shift {(dx,dy)} not recovered: {res}')
    # Unrelated random images must NOT produce an accepted correction.
    random_j=rng.uniform(80,200,(h,w,3)).astype('float32')
    noise=analyze(a,av,random_j,jv,maxshift=8)
    assert noise['status']!='provisional_local_candidate',noise
    res.update(tile_id='T999',qc_group='synthetic_test')
    with tempfile.TemporaryDirectory() as tmp:
        t=Path(tmp)
        view=figure_for_tile(a,av,j,jv,res)
        create_outputs([res],[view],t)
        assert all((t/n).exists() for n in ('automatic_registration_summary.csv',
            'automatic_registration_summary.json','automatic_registration_windows.csv',
            'automatic_registration_comparison.png'))
        assert json.loads((t/'automatic_registration_summary.json').read_text())['n_tiles']==1
    print(f'AUTOMATIC REGISTRATION SELF-TEST PASSED | expected ({dx:+d},{dy:+d}), found ({res["dx"]:+d},{res["dy"]:+d})')
    print('Synthetic mismatch correctly rejected; CSV/JSON/PNG outputs verified. No real imagery accessed.')


def main():
    args=parse_args()
    if args.self_test:
        self_test()
        return
    if args.max_shift<2 or args.max_shift>16:
        raise ValueError('Use --max-shift between 2 and 16 pixels')
    tiles=[x.strip().upper() for x in args.tiles.split(',') if x.strip()]
    if not tiles:
        raise ValueError('No tile IDs selected')
    print('MUMBAI APRIL–JUNE | ROBUST AUTOMATIC REGISTRATION QC')
    print('Method: red/NIR structural edges + multiple cloud-screened subwindows')
    print('NO scene warps saved; NO new ground-truth or reconstruction asserted.\n')
    rows=[]; views=[]
    for tile in tiles:
        record=get_record(args.qc_csv,tile)
        a,av,j,jv,pixel_metres=load_tile(args,record)
        r=analyze(a,av,j,jv,maxshift=args.max_shift)
        r.update(tile_id=tile,qc_group=record.get('qc_group',''),pixel_spacing_m=pixel_metres)
        rows.append(r)
        views.append(figure_for_tile(a,av,j,jv,r))
        print(f'{tile}: {r["status"]} | shift June->April ({r["dx"]:+d},{r["dy"]:+d}) px | '
              f'{r["n_consensus_windows"]}/{r["n_windows_tested"]} agreement | '
              f'corr {r["corr_before"]:.3f}->{r["corr_after"]:.3f} | '
              f'usable {r["safe_fraction"]:.1%}',flush=True)
    create_outputs(rows,views,args.out)
    print('\nAUTOMATIC REGISTRATION QC COMPLETE')
    print(f'Provisional candidates: {sum(r["status"]=="provisional_local_candidate" for r in rows)}/{len(rows)}')
    print(f'Summary: {args.out / "automatic_registration_summary.csv"}')
    print(f'Window diagnostics: {args.out / "automatic_registration_windows.csv"}')
    print(f'Visual: {args.out / "automatic_registration_comparison.png"}')
    print('IMPORTANT: Do NOT apply any estimated shifts to the full-resolution scenes yet.')

if __name__ == '__main__':
    main()
