r'''FINAL HELD-OUT SYNTHETIC TEST for the locked single-image Mumbai V3 model.

Save as: Cloud_Reconstruction/src/evaluate_v3_locked_test.py

Dependencies already in your src/ folder:
  evaluate_v3_structure.py, evaluate_v3_cloud_boundaries.py,
  train_unet_residual_v3.py, test_pytorch_dataloader.py,
  synthetic_clouds_v2.py, unet_model.py, train_unet_detail_v2.py

Commands from repository root:
  .\.venv\Scripts\python.exe -u .\src\evaluate_v3_locked_test.py --self-test
  .\.venv\Scripts\python.exe -u .\src\evaluate_v3_locked_test.py

The V3 model checkpoint, baseline methods and 3px dilation + 3px feather
variant were selected using validation data BEFORE the held-out test run.
This script never trains, adjusts weights, chooses parameters, or reads any
non-test image array. Do not make model choices based on these results.

NOTE: This benchmark tests synthetic clouds applied to visually screened
April satellite patches. It does not verify real-cloud reconstruction.
No pandas, scikit-image or scipy dependency.
'''
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from synthetic_clouds_v2 import false_color
from test_pytorch_dataloader import ROOT, CloudPairsDataset
from train_unet_residual_v3 import BEST, DN_MAX, load_checkpoint, read_manifest
from evaluate_v3_cloud_boundaries import load_synthetic_diagnostics
from evaluate_v3_structure import (
    REPORT_FIELDS, VARIANTS, LABEL_FIELD,
    gaussian_kernel, predictions, structural_metrics, summarize, save_csv,
)

OUTPUT = ROOT / 'outputs' / 'unet_final_test_v3'
DETAIL_FILE = OUTPUT / 'test_structural_metrics_by_patch.csv'
SUMMARY_FILE = OUTPUT / 'test_structural_summary.csv'
PREVIEW_FILE = OUTPUT / 'test_reconstruction_comparison.png'
PROVENANCE_FILE = OUTPUT / 'locked_test_provenance.json'


def make_preview(samples, output, epoch):
    heads = ['Clear target', 'Synthetic cloudy', 'Visible mean', 'Local fill',
             'V3 original', 'V3 dilate3+feather3', 'Cloud mask']
    tile, gap, caption, top = 172, 10, 35, 76
    w = gap + len(heads)*(tile+gap)
    height = top + len(samples)*(tile+caption+gap)+gap
    canvas = Image.new('RGB', (w,height), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), f'Mumbai | HELD-OUT SYNTHETIC TEST | V3 best epoch {epoch}', fill='black')
    draw.text((gap, 29), 'Fixed false-colour mapping | No model fitting or test-based parameter selection', fill='black')
    for i,h in enumerate(heads):
        draw.text((gap+i*(tile+gap),55),h,fill='black')
    for j,sample in enumerate(samples):
        y = top+j*(tile+caption+gap)
        for i,panel in enumerate(sample['images']):
            canvas.paste(Image.fromarray(panel).resize((tile,tile),Image.Resampling.BILINEAR),
                         (gap+i*(tile+gap),y))
        draw.text((gap,y+tile+5),
                  f"{sample['id']} | {sample['surface']} | cloud {sample['cover']:.1f}%",
                  fill='black')
        draw.text((gap+4*(tile+gap),y+tile+5),
                  f"Core MAE {sample['mae_v3']:.4f} -> {sample['mae_ref']:.4f}",fill='black')
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def self_test():
    # Reuse exactly the proven structural smoke test: math, gradients, SSIM,
    # baselines, 7-channel V3 inference, 3px dilation and feathering.
    from evaluate_v3_structure import self_test as structure_self_test
    structure_self_test()
    import tempfile
    sample = np.zeros((64,64,3),dtype=np.uint8)
    mask = np.full((64,64,3),255,dtype=np.uint8)
    with tempfile.TemporaryDirectory() as d:
        output = Path(d)/'preview.png'
        make_preview([{'id':'TEST000','surface':'water_like','cover':25.0,
                       'mae_v3':.01,'mae_ref':.02,
                       'images':[sample]*6+[mask]}],output,3)
        assert output.is_file() and Image.open(output).size[0] > 900
    print('LOCKED TEST EVALUATOR SELF-TEST PASSED')
    print('This self-test does NOT read held-out test data or model checkpoints.')


@torch.inference_mode()
def evaluate():
    if OUTPUT.exists() and any(p.exists() for p in
        (DETAIL_FILE, SUMMARY_FILE, PREVIEW_FILE, PROVENANCE_FILE)):
        raise FileExistsError(
            'A final test report already exists. Not overwriting it: '+str(OUTPUT)+
            '\nUse the existing report; do not repeatedly rerun test-based experiments.'
        )
    if not BEST.is_file():
        raise FileNotFoundError(f'V3 checkpoint missing: {BEST}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = load_checkpoint(BEST, device, residual_scale=0.25)
    model.eval()
    records = read_manifest()  # metadata only; no train/validation image arrays read
    ds = CloudPairsDataset(records,'test')
    if len(ds)!=75 or len({r['patch_id'] for r in ds.rows})!=75:
        raise AssertionError('Expected exactly 75 unique held-out test pairs')

    digest = hashlib.sha256(BEST.read_bytes()).hexdigest()
    kernel = gaussian_kernel(device=device)
    chosen = set(np.linspace(0,74,6).round().astype(int).tolist())
    detail, samples = [], []

    print('LOCKED RESIDUAL U-NET V3 — HELD-OUT SYNTHETIC TEST',flush=True)
    print(f"Checkpoint epoch {checkpoint['epoch']} | Device {device} | Test patches {len(ds)}",flush=True)
    print('Frozen variants: cloudy input, mean-fill, local-fill, V3 original, V3 3px+feather3',flush=True)
    print('No training or validation-based changes during this run.',flush=True)
    for idx in range(len(ds)):
        item = ds[idx]
        record = ds.rows[idx]
        x = item['input'].unsqueeze(0).to(device)
        t = item['target'].unsqueeze(0).to(device)
        m = item['cloud_mask'].unsqueeze(0).to(device)
        valid = item['valid_mask'].unsqueeze(0).to(device)
        alpha,_shadow = load_synthetic_diagnostics(record)
        alpha = alpha.to(device)
        preds = predictions(model,x,valid)
        group = record.get(LABEL_FIELD,'').strip() or 'unlabeled'
        calculated = {}
        for variant in VARIANTS:
            mt = structural_metrics(preds[variant],t,m,valid,alpha,kernel)
            calculated[variant] = mt
            detail.append({'patch_id':record['patch_id'],
                           'surface_proxy':group,
                           'cloud_fraction':record['cloud_fraction'],
                           'variant':variant,**mt})
        if idx in chosen:
            val_np = item['valid_mask'][0].numpy()>.5
            clear = item['target'].numpy().transpose(1,2,0)*DN_MAX
            cloudy = item['input'][:3].numpy().transpose(1,2,0)*DN_MAX
            images = [false_color(clear,val_np),false_color(cloudy,val_np)]
            for variant in VARIANTS[1:]:
                val = preds[variant][0].cpu().numpy().transpose(1,2,0)*DN_MAX
                images.append(false_color(val,val_np))
            mask = (item['cloud_mask'][0].numpy()>.5).astype(np.uint8)*255
            images.append(np.repeat(mask[...,None],3,axis=2))
            if len(images)!=7:
                raise AssertionError('Unexpected preview channel count')
            samples.append({'id':record['patch_id'], 'surface':group,
                            'cover':float(record['cloud_fraction'])*100,
                            'mae_v3':calculated['v3_original']['core_mae'],
                            'mae_ref':calculated['v3_dilate3_feather3']['core_mae'],
                            'images':images})
        if (idx+1)%15==0:
            print(f'Evaluated {idx+1}/75 held-out test patches',flush=True)

    if len(detail)!=75*len(VARIANTS):
        raise AssertionError('Incomplete test evaluation')
    summary=summarize(detail)
    print('\nHELD-OUT TEST — PER-PATCH UNWEIGHTED MEANS',flush=True)
    print(f"{'Method':<24} {'Core MAE':>9} {'SSIM7':>9} {'Grad corr':>10} {'Grad MAE':>9} {'Halo MAE':>9}")
    for row in summary:
        if row['surface_proxy']=='ALL':
            print(f"{row['variant']:<24} {row['core_mae']:9.5f} {row['core_ssim7_interior']:9.4f} "
                  f"{row['gradient_corr']:10.4f} {row['gradient_mae']:9.5f} {row['halo_mae']:9.5f}",flush=True)

    print('\nPER-SURFACE PROXY RESULTS (coarse groups, not land-cover truth)',flush=True)
    for group in sorted({r['surface_proxy'] for r in summary if r['surface_proxy']!='ALL'}):
        rows_group=[r for r in summary if r['surface_proxy']==group]
        print(f"  {group}: {rows_group[0]['patches']} patches",flush=True)
        for row in rows_group:
            if row['variant'] in ('visible_mean','v3_original','v3_dilate3_feather3'):
                print(f"    {row['variant']:<23} core MAE {row['core_mae']:.5f} | SSIM7 {row['core_ssim7_interior']:.4f}",flush=True)

    # Write only after completing all metrics successfully, preserving baseline
    # metadata for reproducing the single frozen final test run.
    OUTPUT.mkdir(parents=True,exist_ok=True)
    save_csv(DETAIL_FILE,detail,('patch_id','surface_proxy','cloud_fraction','variant',*REPORT_FIELDS))
    save_csv(SUMMARY_FILE,summary,('surface_proxy','variant','patches',*REPORT_FIELDS,'ssim_available_patches'))
    make_preview(samples,PREVIEW_FILE,checkpoint['epoch'])
    provenance={
        'benchmark':'Mumbai April 2025 synthetic clouds; held-out spatial test',
        'checkpoint_path_relative':str(BEST.relative_to(ROOT)),
        'checkpoint_sha256':digest,
        'checkpoint_epoch':int(checkpoint['epoch']),
        'test_patch_count':75,
        'patch_ids':[r['patch_id'] for r in ds.rows],
        'methods':list(VARIANTS),
        'variant_choice':'Original V3 for core; 3px dilation + 3px feather as optional boundary tradeoff (validation preselected)',
        'no_retraining':True,
        'synthetic_alpha_used_for_metrics_only':True,
        'metric_aggregation':'Unweighted mean of valid per-patch metric values',
        'caveat':'Synthetic LISS-IV DN-space cloud augmentation; not evidence of real-cloud ground-truth recovery',
    }
    PROVENANCE_FILE.write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('\nHELD-OUT TEST COMPLETE',flush=True)
    print('Summary:',SUMMARY_FILE,flush=True)
    print('Per-patch metrics:',DETAIL_FILE,flush=True)
    print('Comparison preview:',PREVIEW_FILE,flush=True)
    print('Locked benchmark provenance:',PROVENANCE_FILE,flush=True)
    print('Do not adjust V3 or postprocessing based on test results.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        evaluate()

if __name__=='__main__':
    main()
