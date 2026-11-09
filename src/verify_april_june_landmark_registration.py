r"""Manually verify LOCAL April/June registration using corresponding stable landmarks.

Run from Cloud_Reconstruction project root:
    .\.venv\Scripts\python.exe -u .\src\verify_april_june_landmark_registration.py --self-test
    .\.venv\Scripts\python.exe -u .\src\verify_april_june_landmark_registration.py --tile T140 --points 8

Dependencies: numpy, matplotlib (interactive Tk/Qt backend), Pillow, rasterio.
Uses the previous QC CSV and April/June BAND4/3/2 GeoTIFF files.
Does not modify any source image, model, training data, or locked test results.

Click APRIL landmark, then corresponding JUNE landmark; repeat 6-10 times.
Select persistent landmarks (road junctions, bridge ends, stable building corners),
NOT clouds, shadows, tree crowns, waves, waterlines, or seasonal boundaries.
Press Enter when finished (>= 4 pairs); Backspace removes last pending/complete click.

Measured shifts are LOCAL, not a global correction. A low residual is necessary
but insufficient evidence of correct matches/registration. April is 48 days older.
"""
from __future__ import annotations

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

from audit_april_june_temporal_reference import ROOT, open_scene, patch, shift_no_wrap
from review_april_june_cloud_and_alignment import display_auto, checker

DEFAULT_QC = ROOT / "outputs" / "april_june_targeted_qc" / "sampled_cloud_reference_qc.csv"
DEFAULT_OUT = ROOT / "outputs" / "april_june_landmark_registration"


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--tile", default="T140")
    p.add_argument("--points", type=int, default=8)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--april-dir", type=Path, default=ROOT/"data/raw/mumbai_scenes/2025-04-16")
    p.add_argument("--june-dir", type=Path, default=ROOT/"data/raw/mumbai_scenes/2025-06-03")
    p.add_argument("--qc-csv", type=Path, default=DEFAULT_QC)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def robust_shift(pairs):
    """Pairs are (april_x, april_y, june_x, june_y); return provisional shift and residuals."""
    p=np.asarray(pairs,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=4 or p.shape[0]<4 or not np.isfinite(p).all():
        raise ValueError("Need >=4 finite pairs of corresponding landmark coordinates")
    offsets=p[:,0:2]-p[:,2:4]
    med=np.median(offsets,axis=0)
    radial=np.linalg.norm(offsets-med,axis=1)
    # Conservative agreement check for individually clicked landmark pixels.
    good=radial<=2.0
    if good.sum()>=4:
        med=np.median(offsets[good],axis=0)
        radial=np.linalg.norm(offsets-med,axis=1)
        good=radial<=2.0
    rms=float(np.sqrt(np.mean(radial[good]**2))) if good.any() else float('nan')
    return {'dx':float(med[0]),'dy':float(med[1]),
            'n_pairs':int(len(p)),'n_agree_within_2px':int(good.sum()),
            'agreement_fraction':float(good.mean()),
            'median_residual_px':float(np.median(radial)),
            'rms_agreeing_residual_px':rms,
            'per_pair_residual_px':radial.tolist(),
            'per_pair_agrees':good.tolist(),
            'preliminary_status':('consistent_local_translation_candidate' if good.sum()>=6 and good.mean()>=0.75 and rms<=1.5 else 'uncertain_or_inconsistent_matches')}


def get_record(path, tid):
    if not path.is_file():
        raise FileNotFoundError(f"QC CSV not found: {path}\nFirst run review_april_june_cloud_and_alignment.py")
    with path.open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    matching=[r for r in rows if r['tile_id'].strip().upper()==tid.upper()]
    if len(matching)!=1:
        raise ValueError(f"Tile {tid} not found uniquely in QC CSV (found {len(matching)})")
    return matching[0]


def load_tile(args,record):
    row,col=int(record['june_row']),int(record['june_col'])
    with rasterio.Env(GDAL_CACHEMAX=64):
        with ExitStack() as stack:
            apr,_=open_scene(stack,args.april_dir,'April 16')
            jun,_=open_scene(stack,args.june_dir,'June 03')
            if apr[0].crs != jun[0].crs:
                raise ValueError("CRS mismatch")
            warps=[stack.enter_context(WarpedVRT(d,crs=jun[0].crs,transform=jun[0].transform,
                       width=jun[0].width,height=jun[0].height,src_nodata=0,nodata=0,
                       resampling=Resampling.bilinear)) for d in apr]
            w=Window(col,row,args.tile_size,args.tile_size)
            a,av=patch(warps,w)
            j,jv=patch(jun,w)
            pixel_metres=float(abs(jun[0].transform.a))
    if a.shape!=(args.tile_size,args.tile_size,3) or j.shape!=a.shape:
        raise ValueError('Tile dimensions wrong')
    return a,av,j,jv,pixel_metres


def interact(a,av,j,jv,tile,npoints):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("Install matplotlib in your venv: python -m pip install matplotlib") from e
    backend=plt.get_backend().lower()
    if 'agg' in backend and not ('tkagg' in backend or 'qtagg' in backend):
        raise RuntimeError(f"Matplotlib backend {backend} does not support mouse clicks. Run locally using interactive TkAgg/QtAgg.")
    aa=np.asarray(display_auto(a,av));jj=np.asarray(display_auto(j,jv))
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,7),layout='constrained')
    ax1.imshow(aa);ax2.imshow(jj)
    ax1.set_title('April 16 — click landmark first');ax2.set_title('June 03 — click SAME landmark second')
    for ax in (ax1,ax2):
        ax.set_xlim(0,a.shape[1]);ax.set_ylim(a.shape[0],0);ax.set_xlabel('Pixel x');ax.set_ylabel('Pixel y')
    pairs=[];pending=None;dots=[];done=[False]
    def redraw():
        nonlocal dots
        for artist in dots:artist.remove()
        dots=[]
        for i,(ax,ay,jx,jy) in enumerate(pairs,1):
            dots += [ax1.plot(ax,ay,'yo',ms=6)[0],ax2.plot(jx,jy,'yo',ms=6)[0]]
            dots += [ax1.text(ax+2,ay+2,str(i),color='white',fontsize=10),
                     ax2.text(jx+2,jy+2,str(i),color='white',fontsize=10)]
        if pending is not None:
            dots.append(ax1.plot(*pending,'co',ms=8)[0])
        order='Click JUNE matching point' if pending is not None else 'Click APRIL landmark'
        fig.suptitle(f'{tile}: {len(pairs)}/{npoints} matching pairs | {order}\n'
                     'Press Enter to finish (4+ pairs); Backspace to undo. Never click clouds/waves.',fontsize=12)
        fig.canvas.draw_idle()
    def on_click(event):
        nonlocal pending
        if event.button!=1 or event.xdata is None or event.ydata is None:return
        if len(pairs)>=npoints:return
        pt=(float(event.xdata),float(event.ydata))
        if pending is None:
            if event.inaxes!=ax1:return
            pending=pt
        else:
            if event.inaxes!=ax2:return
            pairs.append((*pending,*pt));pending=None
        redraw()
    def on_key(event):
        nonlocal pending
        if event.key in ('backspace','delete'):
            if pending is not None:pending=None
            elif pairs:pairs.pop()
            redraw()
        elif event.key in ('enter','return'):
            if len(pairs)<4:
                print('Need at least 4 corresponding landmark pairs before saving.',flush=True)
                return
            done[0]=True
            plt.close(fig)
    fig.canvas.mpl_connect('button_press_event',on_click)
    fig.canvas.mpl_connect('key_press_event',on_key)
    redraw()
    print('\nINTERACTIVE LANDMARK CHECK:',flush=True)
    print('Select distinct, stable surface points spread across the tile.',flush=True)
    print('Click APRIL -> matching JUNE -> repeat; Enter saves; Backspace undoes.',flush=True)
    plt.show(block=True)
    if not done[0]:
        raise RuntimeError('Window closed without pressing Enter; no output saved')
    return pairs


def report(a,av,j,jv,pairs,stats,tile,record,out,pixel_metres):
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    rounded_dx=int(round(stats['dx']));rounded_dy=int(round(stats['dy']))
    js=shift_no_wrap(j,rounded_dy,rounded_dx)
    jvs=shift_no_wrap(jv,rounded_dy,rounded_dx)
    base=out/f'manual_registration_{tile}'
    with (base.with_suffix('.csv')).open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f);wr.writerow(['pair','april_x','april_y','june_x','june_y','dx_june_to_april','dy_june_to_april','residual_px','agrees_2px'])
        for k,(ax,ay,jx,jy) in enumerate(pairs):
            wr.writerow([k+1,ax,ay,jx,jy,ax-jx,ay-jy,stats['per_pair_residual_px'][k],stats['per_pair_agrees'][k]])
    summary={k:v for k,v in stats.items() if k not in ('per_pair_residual_px','per_pair_agrees')}
    summary.update({'tile_id':tile,'june_row':int(record['june_row']),'june_col':int(record['june_col']),
                    'pixel_spacing_m':pixel_metres,'candidate_dx_from_prior_qc':int(record['local_shift_june_to_april_dx']),
                    'candidate_dy_from_prior_qc':int(record['local_shift_june_to_april_dy']),
                    'shift_display_rounding_px':[rounded_dx,rounded_dy],
                    'warning':'LOCAL manual candidate only; do NOT automatically align full scenes or treat April as June ground truth.'})
    with (base.with_suffix('.json')).open('w',encoding='utf-8') as f:
        json.dump(summary,f,indent=2)
    figs,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained')
    panels=[('April 16',np.asarray(display_auto(a,av))),
            ('June 03',np.asarray(display_auto(j,jv))),
            ('Shifted June (rounded)',np.asarray(display_auto(js,jvs))),
            ('Checkerboard original',np.asarray(checker(a,j,av,jv,n=16))),
            ('Checkerboard candidate',np.asarray(checker(a,js,av,jvs,n=16))),
            ('A: red / shifted J: green',None)]
    # Edge overlay from independently normalized highpasses (display diagnostic).
    from audit_april_june_temporal_reference import highpass
    ah=highpass(a);jh=highpass(js)
    ga=np.hypot(*np.gradient(ah));gj=np.hypot(*np.gradient(jh))
    def norm(x,ok):
        hi=float(np.percentile(x[ok],97)) if ok.any() else 1.
        return np.clip(x/max(hi,1e-6),0,1)
    both=av&jvs
    ea=norm(ga,both);ej=norm(gj,both)
    overlay=np.zeros((*both.shape,3),np.float32)
    overlay[...,0]=ea;overlay[...,1]=ej;overlay[...,2]=ea
    overlay[~both]=0
    panels[-1]=(panels[-1][0],np.uint8(overlay*255))
    for ax,(name,img) in zip(axes.flat,panels):
        ax.imshow(img);ax.set_title(name);ax.axis('off')
    for i,(ax,ay,jx,jy) in enumerate(pairs,1):
        axes[0,0].plot(ax,ay,'yo',markersize=3);axes[0,0].text(ax+2,ay+2,str(i),color='white',fontsize=7)
        axes[0,1].plot(jx,jy,'yo',markersize=3);axes[0,1].text(jx+2,jy+2,str(i),color='white',fontsize=7)
    figs.suptitle(f'{tile} | manual June-to-April shift dx={stats["dx"]:+.2f}, dy={stats["dy"]:+.2f} px '
                  f'| {stats["n_agree_within_2px"]}/{stats["n_pairs"]} agreeing | {stats["preliminary_status"]}',fontsize=12)
    figs.savefig(base.with_suffix('.png'),dpi=135,bbox_inches='tight');plt.close(figs)
    return base


def self_test():
    rng=np.random.default_rng(37)
    june=rng.uniform(20,220,(256,256,3)).astype('float32')
    april=shift_no_wrap(june,-3,4)
    points=[]
    for jx,jy in [(33,54),(70,160),(110,90),(170,180),(194,43),(212,213),(140,137),(55,210)]:
        points.append((jx+4+rng.normal(0,.12),jy-3+rng.normal(0,.12),jx,jy))
    points.append((10,10,240,240)) # intentionally bad matching pair
    result=robust_shift(points)
    assert abs(result['dx']-4)<.2 and abs(result['dy']+3)<.2 and result['n_agree_within_2px']==8, result
    av=np.ones((256,256),bool);jv=av.copy()
    record={'june_col':'500','june_row':'700','local_shift_june_to_april_dx':'4','local_shift_june_to_april_dy':'-3'}
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)
        stem=report(april,av,june,jv,points,result,'T999',record,out,5.)
        assert all(stem.with_suffix(s).is_file() for s in ('.csv','.json','.png'))
        data=json.loads(stem.with_suffix('.json').read_text())
        assert data['n_agree_within_2px']==8
    print('MANUAL REGISTRATION SELF-TEST PASSED',flush=True)
    print('Known synthetic translation (+4, -3) recovered; CSV/JSON/PNG outputs verified.',flush=True)


def main():
    args=parse_args()
    if args.self_test:
        self_test();return
    if args.points<4 or args.points>20 or args.tile_size<64:
        raise ValueError('Use 4..20 point pairs and tile size >=64')
    record=get_record(args.qc_csv,args.tile)
    print(f'Tile {args.tile} | QC group: {record["qc_group"]} | prior shift candidate:'
          f' ({record["local_shift_june_to_april_dx"]},{record["local_shift_june_to_april_dy"]})',flush=True)
    a,av,j,jv,metres=load_tile(args,record)
    print(f'Common valid fraction: {float((av&jv).mean()):.1%} | pixel size {metres:.1f} m',flush=True)
    pairs=interact(a,av,j,jv,args.tile,args.points)
    stats=robust_shift(pairs)
    output=report(a,av,j,jv,pairs,stats,args.tile,record,args.out,metres)
    print('\nMANUAL LANDMARK REGISTRATION QC COMPLETE',flush=True)
    print(f'Matched pairs: {stats["n_pairs"]} | within 2px of median: {stats["n_agree_within_2px"]}',flush=True)
    print(f'Candidate June-to-April translation dx={stats["dx"]:+.2f}, dy={stats["dy"]:+.2f} px',flush=True)
    print(f'RMS agreeing residual: {stats["rms_agreeing_residual_px"]:.2f} px | status: {stats["preliminary_status"]}',flush=True)
    print(f'Points: {output.with_suffix(".csv")}',flush=True)
    print(f'Summary: {output.with_suffix(".json")}',flush=True)
    print(f'Visual: {output.with_suffix(".png")}',flush=True)
    print('Manual alignment is local/provisional; no source files or checkpoints modified.',flush=True)

if __name__=='__main__':
    main()
