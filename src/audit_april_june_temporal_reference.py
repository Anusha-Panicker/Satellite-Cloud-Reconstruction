r"""Mumbai LISS-IV: read-only April/June temporal alignment + reference-screening audit.

Place inside Cloud_Reconstruction/src/ and run from the project root:
  .\.venv\Scripts\python.exe -u .\src\audit_april_june_temporal_reference.py --self-test
  .\.venv\Scripts\python.exe -u .\src\audit_april_june_temporal_reference.py

Reads the ORIGINAL April 16 and June 03 2025 BAND4/3/2 GeoTIFF scenes.
Samples a regular set of small windows in the geospatial intersection, in a
common June grid using rasterio WarpedVRT. Never creates or changes an aligned
scene or model checkpoint. No pandas, SciPy, OpenCV, or ML libraries required.

Cloud candidate and April reference are ONLY heuristic screening indicators;
not validated cloud masks or ground truth. Phase correlation estimates local
translation candidates, not a precise co-registration solution. Inspection of
full-resolution source imagery and manual masking are required before fusion.
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
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "april_june_temporal_audit"
BANDS = ("BAND4.tif", "BAND3.tif", "BAND2.tif")  # NIR, Red, Green
VIS_LOW = np.array([45.0, 50.0, 60.0], dtype=np.float32)
VIS_HIGH = np.array([280.0, 210.0, 200.0], dtype=np.float32)
FIELDS = (
    "tile_id", "jun_col", "jun_row", "center_easting_m", "center_northing_m",
    "overlap_valid_fraction", "april_bright_fraction_valid", "june_bright_fraction_valid",
    "june_cloud_candidate_fraction_valid", "april_reference_candidate_fraction_valid",
    "temporal_candidate_pixels", "temporal_candidate_fraction_valid", "stable_clear_fraction_valid", "stable_edge_std",
    "shift_june_to_april_dx_px", "shift_june_to_april_dy_px", "phase_peak_snr",
    "edge_corr_before", "edge_corr_after", "shift_status",
    "median_red_delta_june_minus_april_stable_dn", "median_ndvi_april_stable",
    "median_ndvi_june_stable", "preview_rank",
)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--self-test", action="store_true", help="No source satellite images needed")
    p.add_argument("--april-dir", type=Path, default=ROOT / "data/raw/mumbai_scenes/2025-04-16")
    p.add_argument("--june-dir", type=Path, default=ROOT / "data/raw/mumbai_scenes/2025-06-03")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--grid", type=int, default=12, help="Regular grid of N by N sampled windows")
    p.add_argument("--tile", type=int, default=256, help="Square window size, pixels")
    p.add_argument("--preview", type=int, default=12, help="Number of diagnostic rows in image")
    p.add_argument("--min-valid", type=float, default=0.70)
    p.add_argument("--max-shift", type=int, default=8, help="Maximum local translation searched in pixels")
    p.add_argument("--red-min", type=int, default=145, help="Conservative bright-red DN screening threshold")
    p.add_argument("--green-min", type=int, default=140, help="Conservative bright-green DN screening threshold")
    p.add_argument("--min-brightening", type=int, default=30, help="June versus April rise in both visible bands")
    return p


def locate_bands(date_dir):
    if not date_dir.is_dir():
        raise FileNotFoundError(f"Satellite date directory missing: {date_dir}")
    files = {}
    for name in BANDS:
        hits = sorted(date_dir.rglob(name))
        if len(hits) != 1:
            raise RuntimeError(f"Expected exactly ONE {name} under {date_dir}; found {len(hits)}. "
                               "Pass --april-dir/--june-dir if data live elsewhere.")
        files[name] = hits[0]
    return files


def open_scene(stack, dirpath, label):
    paths = locate_bands(dirpath)
    datasets = [stack.enter_context(rasterio.open(paths[name])) for name in BANDS]
    ref = datasets[0]
    if ref.crs is None or not ref.crs.is_projected:
        raise ValueError(f"{label} requires projected CRS with metre units for this audit")
    if any(ds.count != 1 or ds.dtypes[0] != 'uint16' for ds in datasets):
        raise ValueError(f'{label}: expected one uint16 band per input GeoTIFF')
    for ds in datasets[1:]:
        if (ds.crs != ref.crs or ds.transform != ref.transform
                or ds.width != ref.width or ds.height != ref.height):
            raise ValueError(f"{label} satellite bands are not on one identical grid")
    print(f"{label}: {ref.width} x {ref.height} | CRS {ref.crs} | pixel {ref.res[0]:.3f} x {ref.res[1]:.3f} m")
    return datasets, {k: str(v) for k, v in paths.items()}


def overlap_window(april, june):
    a, j = april.bounds, june.bounds
    left, right = max(a.left, j.left), min(a.right, j.right)
    bottom, top = max(a.bottom, j.bottom), min(a.top, j.top)
    if right <= left or top <= bottom:
        raise ValueError("April and June scene extents do not overlap")
    win = from_bounds(left, bottom, right, top, june.transform)
    col0 = max(0, math.ceil(win.col_off - 1e-8))
    row0 = max(0, math.ceil(win.row_off - 1e-8))
    col1 = min(june.width, math.floor(win.col_off + win.width + 1e-8))
    row1 = min(june.height, math.floor(win.row_off + win.height + 1e-8))
    if col1 <= col0 or row1 <= row0:
        raise ValueError("Empty pixel-aligned geographic overlap")
    return (col0, row0, col1, row1), (left, bottom, right, top)


def patch(ds_list, win):
    data = np.stack([ds.read(1, window=win) for ds in ds_list], axis=-1).astype(np.float32)
    valid = np.all(data > 0, axis=-1)
    for ds in ds_list:
        valid &= ds.read_masks(1, window=win) > 0
    valid &= np.all(np.isfinite(data), axis=2)
    return data, valid


def preview_rgb(arr, valid):
    z = np.clip((arr.astype(np.float32) - VIS_LOW) / (VIS_HIGH - VIS_LOW), 0, 1)
    rgb = (z * 255).astype(np.uint8)
    rgb[~valid] = [5, 5, 5]
    return Image.fromarray(rgb, "RGB")


def ndvi(arr):
    return (arr[:, :, 0] - arr[:, :, 1]) / np.maximum(arr[:, :, 0] + arr[:, :, 1], 1)


def classify(apr, jun, av, jv, red_min, green_min, brightening):
    both = av & jv
    # Conservative but uncalibrated brightness proxies: NOT confirmed clouds.
    apr_bright = av & (apr[:, :, 1] >= red_min) & (apr[:, :, 2] >= green_min)
    jun_bright = jv & (jun[:, :, 1] >= red_min) & (jun[:, :, 2] >= green_min)
    strong_change = ((jun[:, :, 1] - apr[:, :, 1]) >= brightening
                     ) & ((jun[:, :, 2] - apr[:, :, 2]) >= brightening)
    june_cloud_candidate = both & jun_bright & strong_change
    april_reference_candidate = both & ~apr_bright
    candidates = june_cloud_candidate & april_reference_candidate
    stable = both & ~apr_bright & ~jun_bright
    return dict(both=both, apr_bright=apr_bright, jun_bright=jun_bright,
                june_cloud_candidate=june_cloud_candidate,
                april_reference_candidate=april_reference_candidate,
                temporal_candidates=candidates, stable=stable)


def highpass(arr):
    # Independent contrast scaling reduces impact of seasonal radiometric shifts.
    gray = 0.30 * arr[:, :, 0] + 0.45 * arr[:, :, 1] + 0.25 * arr[:, :, 2]
    qlo, qhi = np.percentile(gray, (2, 98))
    if qhi <= qlo + 2:
        return np.zeros_like(gray, np.float32)
    gray8 = np.clip(255 * (gray - qlo) / (qhi - qlo), 0, 255).astype(np.uint8)
    smooth = np.asarray(Image.fromarray(gray8).filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32)
    return gray8.astype(np.float32) - smooth


def shift_no_wrap(arr, dy, dx):
    out = np.zeros_like(arr)
    h, w = arr.shape[:2]
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[max(0,dy):max(0,dy)+(src_y1-src_y0),
            max(0,dx):max(0,dx)+(src_x1-src_x0)] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def masked_corr(a, b, valid):
    if np.count_nonzero(valid) < 500:
        return float("nan")
    x, y = a[valid].astype(np.float64), b[valid].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    d = math.sqrt(float(x @ x) * float(y @ y))
    return float((x @ y) / d) if d > 1e-8 else float("nan")


def estimate_shift(april_hp, june_hp, stable, maxshift=8):
    h, w = april_hp.shape
    if (h, w) != june_hp.shape or stable.shape != (h,w):
        raise ValueError("Alignment inputs have different dimensions")
    if np.count_nonzero(stable) < 1000:
        return (0, 0, float("nan"), float("nan"), float("nan"), "insufficient_stable")
    # Smooth the valid selection to suppress strong synthetic mask edges.
    stable_img = Image.fromarray((stable * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))
    weight = np.asarray(stable_img, np.float32) / 255
    weight *= np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    if weight.sum() < 1:
        return (0, 0, float("nan"), float("nan"), float("nan"), "insufficient_stable")
    a = (april_hp - np.sum(april_hp*weight) / weight.sum()) * weight
    b = (june_hp - np.sum(june_hp*weight) / weight.sum()) * weight
    edge_std = min(float(np.std(a[stable])),float(np.std(b[stable])))
    if edge_std < 0.75:
        return (0, 0, float("nan"), float("nan"), float("nan"), "low_texture")
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-8)
    correlation = np.fft.fftshift(np.abs(np.fft.ifft2(cross)))
    cy, cx = h//2, w//2
    radius = min(maxshift, cy-3, cx-3)
    local = correlation[cy-radius:cy+radius+1, cx-radius:cx+radius+1]
    oy, ox = np.unravel_index(np.argmax(local), local.shape)
    dy, dx = int(oy-radius), int(ox-radius)
    peak = float(local[oy, ox])
    bg = local.copy()
    bg[max(0,oy-2):oy+3,max(0,ox-2):ox+3] = np.nan
    noise_mean = float(np.nanmean(bg))
    noise_std = float(np.nanstd(bg))
    snr = (peak-noise_mean) / max(noise_std, 1e-8)
    edge_before = masked_corr(april_hp, june_hp, stable)
    shifted = shift_no_wrap(june_hp, dy, dx)
    valid_after = stable & shift_no_wrap(stable, dy, dx)
    edge_after = masked_corr(april_hp, shifted, valid_after)
    if snr < 5 or not np.isfinite(edge_after):
        status = "uncertain_peak"
    elif edge_after < 0.15:
        status = "weak_structural_agreement"
    elif dx == 0 and dy == 0:
        status = "nominal_grid_shift_0"
    elif (not np.isfinite(edge_before)) or edge_after <= edge_before + 0.02:
        status = "uncertain_improvement"
    else:
        status = "possible_local_shift"
    return (dx, dy, float(snr), edge_before, edge_after, status)


def metric_median(x, mask):
    return round(float(np.median(x[mask])), 4) if np.count_nonzero(mask) else None


def analyze_one(apr, jun, av, jv, row, col, args, ds_jun):
    maps = classify(apr,jun,av,jv,args.red_min,args.green_min,args.min_brightening)
    both, stable = maps['both'], maps['stable']
    nvalid = int(np.count_nonzero(both))
    if not nvalid:
        return None, maps
    apr_hp, jun_hp = highpass(apr), highpass(jun)
    if stable.sum() >= 1000:
        dx, dy, psr, before, after, status = estimate_shift(apr_hp,jun_hp,stable,args.max_shift)
    else:
        dx=dy=0;psr=before=after=float('nan');status='insufficient_stable'
    px_center, py_center = ds_jun.transform * (col + apr.shape[1]/2, row+apr.shape[0]/2)
    data = {
        'jun_col':col, 'jun_row':row, 'center_easting_m':round(px_center,1),
        'center_northing_m':round(py_center,1),
        'overlap_valid_fraction':round(nvalid/both.size,4),
        'april_bright_fraction_valid':round(np.count_nonzero(maps['apr_bright'] & both)/nvalid,4),
        'june_bright_fraction_valid':round(np.count_nonzero(maps['jun_bright'] & both)/nvalid,4),
        'june_cloud_candidate_fraction_valid':round(np.count_nonzero(maps['june_cloud_candidate'])/nvalid,4),
        'april_reference_candidate_fraction_valid':round(np.count_nonzero(maps['april_reference_candidate'])/nvalid,4),
        'temporal_candidate_pixels':int(np.count_nonzero(maps['temporal_candidates'])),
        'temporal_candidate_fraction_valid':round(np.count_nonzero(maps['temporal_candidates'])/nvalid,4),
        'stable_clear_fraction_valid':round(np.count_nonzero(stable)/nvalid,4),
        'stable_edge_std':round(min(float(np.std(apr_hp[stable])),float(np.std(jun_hp[stable]))),4) if stable.any() else None,
        'shift_june_to_april_dx_px':dx, 'shift_june_to_april_dy_px':dy,
        'phase_peak_snr':round(psr,3) if np.isfinite(psr) else None,
        'edge_corr_before':round(before,4) if np.isfinite(before) else None,
        'edge_corr_after':round(after,4) if np.isfinite(after) else None,
        'shift_status':status,
        'median_red_delta_june_minus_april_stable_dn':metric_median(jun[:,:,1]-apr[:,:,1],stable),
        'median_ndvi_april_stable':metric_median(ndvi(apr),stable),
        'median_ndvi_june_stable':metric_median(ndvi(jun),stable),
        'preview_rank': '',
    }
    return data, maps


def thumbnail_panel(apr, jun, av, jv, masks, sz):
    a = preview_rgb(apr,av).resize((sz,sz))
    j = preview_rgb(jun,jv).resize((sz,sz))
    c = np.asarray(preview_rgb(jun,jv)).copy()
    candidate = masks['temporal_candidates']
    # Mark true candidate regions in yellow. Only a heuristic.
    c[candidate] = (0.40*c[candidate] + 0.60*np.array([255,215,0])).astype(np.uint8)
    cimg = Image.fromarray(c,"RGB").resize((sz,sz))
    return a,j,cimg


def make_preview(records, tiles, outpath, maxrows):
    valid_recs = [r for r in records if r['tile_id'] in tiles]
    valid_recs.sort(key=lambda r: int(r['temporal_candidate_pixels']), reverse=True)
    top = valid_recs[:maxrows//2]
    remainder = [r for r in valid_recs if r not in top]
    # Also show lower-candidate / more representative controls.
    tail = remainder[::max(1,len(remainder)//max(1,maxrows-len(top)))][:maxrows-len(top)]
    selected = (top + tail)[:maxrows]
    for ix, r in enumerate(selected,1):
        r['preview_rank'] = ix
    width=3*220 + 16*4
    row_h=239
    im=Image.new('RGB',(width,43+row_h*len(selected)), 'white')
    pen=ImageDraw.Draw(im)
    pen.text((12,8),'April 2025 -> June 2025 | TEMPORAL SCREENING (sampled tiles only)',fill='black')
    for i, label in enumerate(('April NIR/Red/Green','June NIR/Red/Green','June candidate overlay (yellow)')):
        pen.text((12+i*236,27),label,fill='black')
    for i,r in enumerate(selected):
        apr,jun,av,jv,masks=tiles[r['tile_id']]
        ims=thumbnail_panel(apr,jun,av,jv,masks,220)
        y=43+i*row_h
        for k,t in enumerate(ims):im.paste(t,(12+k*236,y))
        subtitle=(f"{r['tile_id']} valid={float(r['overlap_valid_fraction']):.0%} "
                  f"candidate={int(r['temporal_candidate_pixels'])} "
                  f"local shift jun->apr=({r['shift_june_to_april_dx_px']},"
                  f"{r['shift_june_to_april_dy_px']}) [{r['shift_status']}]")
        pen.text((12,y+222),subtitle,fill='black')
    outpath.parent.mkdir(parents=True,exist_ok=True)
    im.save(outpath)


def clean_json(obj):
    if isinstance(obj, dict): return {str(k):clean_json(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)): return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj): return None
    return obj


def perform(args):
    if args.grid < 2 or args.tile < 32 or args.preview < 1 or args.max_shift < 1:
        raise ValueError('Require --grid >= 2, --tile >= 32, --preview >=1, --max-shift >=1')
    with rasterio.Env(GDAL_CACHEMAX=64):
        with ExitStack() as stack:
            april, ap_paths=open_scene(stack,args.april_dir,'April 16')
            june, jun_paths=open_scene(stack,args.june_dir,'June 03')
            if april[0].crs != june[0].crs:
                raise ValueError('CRS differ. Reprojection of different CRSs is not yet vetted by this audit.')
            a_ref,j_ref=april[0],june[0]
            box,bounds=overlap_window(a_ref,j_ref)
            col0,row0,col1,row1=box
            print(f'Geographic overlap on June grid: {col1-col0} x {row1-row0} px '
                  f'({(col1-col0)*abs(j_ref.res[0])/1000:.2f} x {(row1-row0)*abs(j_ref.res[1])/1000:.2f} km)')
            if col1-col0 < args.tile or row1-row0 < args.tile:
                raise ValueError(f'Overlap smaller than {args.tile}x{args.tile} sample tile')
            warps=[stack.enter_context(WarpedVRT(src, crs=j_ref.crs, transform=j_ref.transform,
                          width=j_ref.width,height=j_ref.height, src_nodata=0,nodata=0,
                          resampling=Resampling.bilinear)) for src in april]
            xs=np.unique(np.rint(np.linspace(col0,col1-args.tile,args.grid)).astype(int))
            ys=np.unique(np.rint(np.linspace(row0,row1-args.tile,args.grid)).astype(int))
            rows=[]; tiles={}; checked=0; skipped=0
            for yy in ys:
                for xx in xs:
                    checked+=1
                    win=Window(int(xx),int(yy),args.tile,args.tile)
                    a,av=patch(warps,win)
                    j,jv=patch(june,win)
                    valid_fraction=float(np.mean(av & jv))
                    if valid_fraction < args.min_valid:
                        skipped+=1;continue
                    row,masks=analyze_one(a,j,av,jv,int(yy),int(xx),args,j_ref)
                    if row is None:skipped+=1;continue
                    row['tile_id']=f'T{checked:03d}'
                    rows.append(row)
                    tiles[row['tile_id']]=(a,j,av,jv,masks) if args.preview else None
                    if checked % max(1,args.grid) == 0:
                        print(f'Analyzed {checked}/{len(xs)*len(ys)} locations; eligible tiles {len(rows)}',flush=True)
            if not rows:
                raise RuntimeError('No tiles passed valid-data threshold. Try --min-valid 0.4 or inspect satellite source paths.')
            good_shifts=[r for r in rows if r['shift_status'] in ('possible_local_shift','nominal_grid_shift_0')]
            dx=[r['shift_june_to_april_dx_px'] for r in good_shifts]
            dy=[r['shift_june_to_april_dy_px'] for r in good_shifts]
            refs=[r for r in rows if r['temporal_candidate_pixels'] >= 100]
            summary={
              'scene_dates':{'reference':'2025-04-16','target':'2025-06-03','days_apart':48},
              'bands_hwc_order':['BAND4_NIR','BAND3_Red','BAND2_Green'],
              'april_source_files':ap_paths,'june_source_files':jun_paths,
              'april_grid':{'crs':str(a_ref.crs),'width':a_ref.width,'height':a_ref.height,'transform':list(a_ref.transform)[:6]},
              'june_grid':{'crs':str(j_ref.crs),'width':j_ref.width,'height':j_ref.height,'transform':list(j_ref.transform)[:6]},
              'overlap_bounds_m':{'left':bounds[0],'bottom':bounds[1],'right':bounds[2],'top':bounds[3]},
              'overlap_window_on_june':{'col0':col0,'row0':row0,'width':col1-col0,'height':row1-row0},
              'sample_settings':{'grid':args.grid,'tile_size_px':args.tile,'min_valid_fraction':args.min_valid,
                     'max_local_shift_px':args.max_shift,'red_min_dn':args.red_min,
                     'green_min_dn':args.green_min,'min_visible_band_brightening_dn':args.min_brightening},
              'sampled_tiles_total':checked,'sampled_tiles_kept':len(rows),'sampled_tiles_low_valid_skipped':skipped,
              'sampled_tiles_with_at_least_100_temporal_candidate_pixels':len(refs),
              'sum_candidate_pixels_sample_only':sum(r['temporal_candidate_pixels'] for r in rows),
              'mean_temporal_candidate_fraction_valid_sample':float(np.mean([r['temporal_candidate_fraction_valid'] for r in rows])),
              'reliable_or_zero_shift_tiles_count':len(good_shifts),
              'median_candidate_shift_june_to_april_px':[float(np.median(dx)),float(np.median(dy))] if dx else None,
              'shift_note':'Local phase-correlation estimates are NOT a guaranteed co-registration transform. Examine geometry on stable terrain before fusion.',
              'cloud_note':'Brightness/change thresholds only flag possible June clouds over possible April reference. Not scientifically validated cloud masks.',
              'representativeness_note':'Sparse regular samples: counts and fractions do NOT measure full-scene cloud or usable-reference coverage.',
            }
            args.out.mkdir(parents=True,exist_ok=True)
            imgpath=args.out/'temporal_alignment_reference_preview.png'
            make_preview(rows,tiles,imgpath,min(args.preview,len(rows)))
            csvpath=args.out/'sampled_temporal_reference_tiles.csv'
            with csvpath.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=FIELDS)
                writer.writeheader();writer.writerows(rows)
            with (args.out/'temporal_audit_summary.json').open('w',encoding='utf-8') as f:
                json.dump(clean_json(summary),f,indent=2)
            print('\nAPRIL–JUNE AUDIT COMPLETE')
            print(f'Sampled tiles: {checked}; valid retained: {len(rows)}; skipped: {skipped}')
            print(f'Tiles with >=100 candidate pixels: {len(refs)} (proxy, NOT validated cloud mask)')
            print(f'Candidate-shift estimates with adequate confidence: {len(good_shifts)}/{len(rows)}')
            print('Candidate median June->April shift (dx,dy):',summary['median_candidate_shift_june_to_april_px'])
            print('CSV:',csvpath)
            print('Summary:',args.out/'temporal_audit_summary.json')
            print('Preview:',imgpath)
            print('No images were resampled to disk or modified. No actual cloud reconstruction yet.')
            return summary


def self_test():
    rng=np.random.default_rng(5)
    n=128
    gray=rng.normal(130,24,(n,n)).astype(np.float32)
    yy,xx=np.indices((n,n))
    gray+=12*np.sin(xx*.28)+10*np.cos(yy*.23)
    april=np.stack([gray+25,gray,gray+10],axis=-1).astype(np.float32)
    june=shift_no_wrap(april,3,-4)
    valid=np.ones((n,n),bool)
    valid[:12,:]=False;valid[-12:,:]=False;valid[:,:12]=False;valid[:,-12:]=False
    aprhp,junhp=highpass(april),highpass(june)
    est=estimate_shift(aprhp,junhp,valid,maxshift=8)
    assert (est[0],est[1]) == (4,-3), f'Expected (+4,-3) shift, got {est}'
    assert est[4] > est[3], f'Shift did not improve correlation: {est}'
    mask=classify(april,june,valid,valid,145,140,30)
    assert mask['temporal_candidates'].shape==(n,n)
    assert shift_no_wrap(np.ones((4,4)),1,1).sum()==9
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'sample.png'
        sample=[{'tile_id':'T001','preview_rank':'', 'overlap_valid_fraction':1,
                 'temporal_candidate_pixels':100,'shift_june_to_april_dx_px':0,
                 'shift_june_to_april_dy_px':0,'shift_status':'nominal_grid_shift_0'}]
        make_preview(sample,{'T001':(april,june,valid,valid,mask)},p,1)
        assert p.is_file()
    print(f'TEMPORAL AUDIT SELF-TEST PASSED | example known shift June->April = ({est[0]}, {est[1]})')
    print('No real satellite data accessed; no project outputs written.')


if __name__=='__main__':
    args=parser().parse_args()
    if args.self_test:
        self_test()
    else:
        perform(args)
