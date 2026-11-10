r"""Real-June qualitative preview for the frozen residual U-Net V3 model.

Run from Cloud_Reconstruction project root:
  .\.venv\Scripts\python.exe -u .\src\preview_real_june_unet_v3.py --self-test
  .\.venv\Scripts\python.exe -u .\src\preview_real_june_unet_v3.py

Uses: src/train_unet_residual_v3.py, src/verify_april_june_landmark_registration.py,
      src/preview_real_temporal_transfer_unregistered.py,
      src/audit_april_june_temporal_reference.py,
      src/review_april_june_cloud_and_alignment.py.

This is NOT validated real-cloud removal:
- June-only fixed DN thresholds produce an unvalidated cloud proxy.
- April only appears as an older visual reference (not used for masking/model input).
- No April-June shifts, band offsets or reference pixel transfers are applied.
- No accuracy scores or cloud-free GeoTIFF are produced.
- Frozen V3 weights are loaded only for inference; no training or test-set access.
- Some clouds, haze and cloud shadows can remain outside the selected proxy mask.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw
from rasterio.transform import from_origin
import rasterio

from audit_april_june_temporal_reference import ROOT, preview_rgb
from preview_real_temporal_transfer_unregistered import qc_records, mask_rgb
from verify_april_june_landmark_registration import load_tile
from review_april_june_cloud_and_alignment import display_auto
from train_unet_residual_v3 import BEST, ResidualCloudUNet, load_checkpoint

OUT = ROOT / 'outputs' / 'real_june_v3_qualitative_preview'
QC_CSV = ROOT / 'outputs' / 'april_june_targeted_qc' / 'sampled_cloud_reference_qc.csv'
WARNING = (
    'QUALITATIVE EXPERIMENT ONLY: June cloud mask is a fixed unvalidated DN proxy; '
    'V3 was trained on synthetic clouds; April is an older visual reference, NOT June truth. '
    'No accepted April-June registration, quantitative real-cloud accuracy, or GeoTIFF output.'
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--tiles', default='T041,T058')
    p.add_argument('--qc-csv', type=Path, default=QC_CSV)
    p.add_argument('--april-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-04-16')
    p.add_argument('--june-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-06-03')
    p.add_argument('--checkpoint', type=Path, default=BEST)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--tile-size', type=int, default=256)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return p.parse_args()


def june_only_cloud_proxy(june, june_valid):
    """Transparent fixed heuristic, no April pixels or synthetic cloud labels."""
    if june.ndim != 3 or june.shape[-1] != 3 or june_valid.shape != june.shape[:2]:
        raise ValueError('Expected June HWC (NIR, Red, Green) and matching validity')
    a = np.asarray(june, np.float32)
    if not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 1023)):
        raise ValueError('June contains invalid DN; expected values 0..1023')
    nir, red, green = a[..., 0], a[..., 1], a[..., 2]
    ndvi = (nir - red) / np.maximum(nir + red, 1.0)
    mask = (june_valid & (nir >= 220) & (red >= 220) &
            (green >= 220) & (ndvi < 0.28))
    return mask


def make_input(june, june_valid, mask, device):
    h, w, c = june.shape
    if (h % 16) or (w % 16) or c != 3 or june_valid.shape != (h,w) or mask.shape != (h,w):
        raise ValueError('Expected 3-channel 16-divisible input with matching masks')
    if np.any(mask & ~june_valid):
        raise ValueError('Cloud proxy selected invalid June pixels')
    if (june_valid & ~mask).sum() < 100:
        raise ValueError('Insufficient observed pixels for V3 local baseline')
    z = np.clip(june.astype(np.float32) / 1023.0, 0.0, 1.0)
    four = np.concatenate([z, mask[..., None].astype(np.float32)], axis=-1)
    x = torch.from_numpy(np.ascontiguousarray(four.transpose(2, 0, 1)[None])).to(device)
    valid = torch.from_numpy(np.ascontiguousarray(june_valid.astype(np.float32)[None,None])).to(device)
    return x, valid


@torch.inference_mode()
def predict(model, june, june_valid, mask, device):
    x, v = make_input(june, june_valid, mask, device)
    composite, _ = model(x, v)
    out = np.asarray(composite[0].permute(1, 2, 0).cpu(), np.float32) * 1023.0
    # Preserve observed and invalid June pixels EXACTLY, in original DN units.
    out[~mask] = june[~mask]
    if not np.array_equal(out[~mask], june[~mask]):
        raise AssertionError('Model modified pixels outside the selected June-only mask')
    if out.shape != june.shape or not np.isfinite(out).all():
        raise AssertionError('Invalid model output')
    return out


def change_visual(june, result, mask, valid):
    difference = np.mean(np.abs(result.astype(np.float32) - june.astype(np.float32)), axis=2)
    # Same scale for every tile, blue (0 DN) to orange (>=100 DN). VISUALIZATION ONLY.
    t = np.clip(difference / 100.0, 0, 1)
    rgb = np.zeros((*t.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255*t).astype(np.uint8)
    rgb[..., 1] = (170*t).astype(np.uint8)
    rgb[..., 2] = (80 + 90*(1-t)).astype(np.uint8)
    rgb[~mask & valid] = [18,18,18]
    rgb[~valid] = [255,255,255]
    return Image.fromarray(rgb, 'RGB')


def preview(rows, path):
    headers = ['April fixed (reference)', 'June fixed (input)', 'June auto*',
               'June-only proxy mask', 'V3 composite fixed', 'V3 composite auto*',
               'Changed DN (masked)']
    tile, gap, y0, rh = 195, 10, 86, 243
    w = gap + len(headers)*(tile+gap)
    h = y0 + len(rows)*rh + 38
    fig = Image.new('RGB', (w,h), 'white')
    dr = ImageDraw.Draw(fig)
    dr.text((10, 6), 'MUMBAI | REAL JUNE V3 INFERENCE | UNVALIDATED CLOUD MASK', fill='black')
    dr.text((10, 23), 'April is display-only: NO APRIL pixels used by cloud mask or V3. No shifts applied.', fill='black')
    dr.text((10, 40), '*Auto-stretch is display-only and independently computed; do not compare brightness across auto panels.', fill='black')
    dr.text((10, 57), 'DN difference colors are illustrative, NOT reconstruction accuracy or cloud truth.', fill='black')
    for k,head in enumerate(headers):
        dr.text((gap + k*(tile+gap), y0-17), head, fill='black')
    for idx,record in enumerate(rows):
        a,av,j,jv,m,rest = (record[k] for k in ('a','av','j','jv','mask','output'))
        imgs = [preview_rgb(a,av), preview_rgb(j,jv), display_auto(j,jv),
                mask_rgb(m,jv),preview_rgb(rest,jv),display_auto(rest,jv),
                change_visual(j,rest,m,jv)]
        y = y0 + idx*rh
        for k,im in enumerate(imgs):
            fig.paste(im.resize((tile,tile),Image.Resampling.BILINEAR if k not in (3,6) else Image.Resampling.NEAREST),
                      (gap + k*(tile+gap), y))
        r=record['report']
        dr.text((gap, y + tile + 8),
                f"{r['tile_id']} | proxy fraction {r['june_only_proxy_fraction']:.1%} | "
                f"inside-mask mean abs change {r['mean_masked_change_dn']:.1f} DN | NOT validation",
                fill='black')
    dr.text((10,h-23),'NO GEOTIFF, NO GROUND TRUTH, NO REAL-CLOUD ACCURACY, NO TRAINING.',fill='black')
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.save(path)


def run(opts):
    if opts.tile_size != 256:
        raise ValueError('V3 real-image preview uses 256x256 tiles; supply --tile-size 256')
    ids=[s.strip().upper() for s in opts.tiles.split(',') if s.strip()]
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('Provide distinct tile IDs')
    source=qc_records(opts.qc_csv)
    for tid in ids:
        if tid not in source: raise ValueError(f'Unknown tile {tid} in {opts.qc_csv}')
    if not opts.checkpoint.is_file():
        raise FileNotFoundError(f'V3 checkpoint missing: {opts.checkpoint}')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if opts.device=='auto' else opts.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA not available; use --device cpu')
    model,state = load_checkpoint(opts.checkpoint, device, residual_scale=0.25)
    model.eval()
    digest=hashlib.sha256(opts.checkpoint.read_bytes()).hexdigest()
    records=[]
    print('REAL JUNE RESIDUAL U-NET V3 — QUALITATIVE PREVIEW ONLY')
    print('Device:',device,'| checkpoint epoch:',state.get('epoch','unknown'))
    print('June-only DN proxy used as input cloud mask; April is DISPLAY-ONLY.')
    print('No training, no shifts, no April transfer, no cloud-free GeoTIFF.')
    for tid in ids:
        a,av,j,jv,_=load_tile(opts,source[tid])
        mask=june_only_cloud_proxy(j,jv)
        fraction=float(mask.sum()/max(1,jv.sum()))
        report = dict(tile_id=tid,valid_june_fraction=round(float(jv.mean()),4),
                      june_only_proxy_fraction=round(fraction,4),
                      mean_masked_change_dn=0.0,mean_outside_change_dn=0.0,
                      status='not_evaluated',model='residual_unet_v3_best_epoch3',
                      real_cloud_mask_validated=False,april_used_as_model_input=False,
                      april_to_june_registration_approved=False,real_cloud_accuracy_available=False)
        if not (0.02 <= fraction <= 0.90):
            report['status']='skipped_proxy_fraction_outside_2_to_90_percent'
            output=j.astype(np.float32).copy()
        elif (jv & ~mask).sum() < 100:
            report['status']='skipped_insufficient_visible_pixels'
            output=j.astype(np.float32).copy()
        else:
            output=predict(model,j,jv,mask,device)
            report['status']='qualitative_inference_only_unvalidated'
            report['mean_masked_change_dn']=round(float(np.abs(output-j)[mask].mean()),2)
            if (~mask & jv).any():
                report['mean_outside_change_dn']=float(np.max(np.abs(output-j)[~mask & jv]))
            if report['mean_outside_change_dn'] != 0:
                raise AssertionError('Visible June pixels were modified')
        records.append(dict(a=a,av=av,j=j,jv=jv,mask=mask,output=output,report=report))
        print(f"  {tid}: June-only proxy={fraction:.1%} | mean changed {report['mean_masked_change_dn']} DN | {report['status']}")
    opts.out.mkdir(parents=True,exist_ok=True)
    csvfile=opts.out/'real_june_v3_qualitative_metrics.csv'
    with csvfile.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0]['report']))
        wr.writeheader();wr.writerows([r['report'] for r in records])
    img=opts.out/'real_june_v3_qualitative_comparison.png'
    preview(records,img)
    prov=opts.out/'real_june_v3_qualitative_provenance.json'
    prov.write_text(json.dumps(dict(warning=WARNING,checkpoint=str(opts.checkpoint),
                   checkpoint_sha256=digest,checkpoint_epoch=state.get('epoch'),
                   input_date='2025-06-03',comparison_date='2025-04-16',
                   cloud_proxy='June ONLY: NIR/Red/Green >=220 DN and NDVI <0.28',
                   pixel_registration_approved=False,uses_april_as_input=False,
                   real_cloud_ground_truth=False,quantitative_accuracy_claims=False,
                   test_split_loaded=False,geotiff_output=False,tiles=[r['report'] for r in records]),
                   indent=2),encoding='utf-8')
    print('PREVIEW COMPLETE\nCSV:',csvfile,'\nPNG:',img,'\nProvenance:',prov)
    print(WARNING)
    return records


def self_test():
    # Exercise the model and file I/O with temporary, fully synthetic rasters.
    rng=np.random.default_rng(122)
    n=256
    yy,xx=np.indices((n,n))
    base = 130 + 22*np.sin(xx/12)+15*np.cos(yy/19)+rng.normal(0,3,(n,n))
    a=np.stack([base+50,base,base+12],axis=2).clip(1,1023).astype(np.uint16)
    j=a.copy()
    j[45:143,80:171,:]=np.array([440,415,410],np.uint16)
    valid=np.ones((n,n),bool)
    mask=june_only_cloud_proxy(j,valid)
    assert mask.mean() > 0.05 and mask.mean() < 0.40
    model=ResidualCloudUNet(residual_scale=0.25).eval()
    pred=predict(model,j,valid,mask,torch.device('cpu'))
    assert np.array_equal(pred[~mask],j[~mask]) and np.all(np.isfinite(pred))
    with tempfile.TemporaryDirectory() as d:
        r=Path(d)
        for dir_name,arr in [('apr',a),('jun',j)]:
            folder=r/dir_name;folder.mkdir()
            for i,b in enumerate((4,3,2)):
                with rasterio.open(folder/f'BAND{b}.tif','w',driver='GTiff',
                     height=n,width=n,count=1,dtype='uint16',crs='EPSG:32643',
                     transform=from_origin(500000,2100000,5,5),nodata=0) as ds:
                    ds.write(arr[...,i],1)
        q=r/'qc.csv';q.write_text('tile_id,june_col,june_row\nT001,0,0\n',encoding='utf-8')
        ckpt=r/'test.pt'
        torch.save({'experiment':'residual_unet_v3','normalization_divisor':1023.0,
                    'residual_scale':0.25,'model_state_dict':model.state_dict(),'epoch':0},ckpt)
        opts=SimpleNamespace(tiles='T001',qc_csv=q,april_dir=r/'apr',june_dir=r/'jun',
                             checkpoint=ckpt,out=r/'outputs',tile_size=256,device='cpu')
        rows=run(opts)
        assert len(rows)==1 and rows[0]['report']['status']=='qualitative_inference_only_unvalidated'
        for file in ('real_june_v3_qualitative_metrics.csv','real_june_v3_qualitative_comparison.png',
                     'real_june_v3_qualitative_provenance.json'):
            assert (r/'outputs'/file).is_file(),file
        assert json.loads((r/'outputs'/'real_june_v3_qualitative_provenance.json').read_text())['test_split_loaded'] is False
    print('REAL-JUNE V3 QUALITATIVE SELF-TEST PASSED')
    print('Only temporary synthetic GeoTIFFs/checkpoint were used; actual April/June data not opened.')


if __name__=='__main__':
    a=parse_args()
    if a.self_test:self_test()
    else:run(a)
