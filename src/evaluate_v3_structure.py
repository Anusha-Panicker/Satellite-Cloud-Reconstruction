r'''Mumbai LISS-IV | validation-only structural reconstruction audit for Residual U-Net V3.

Put this script inside Cloud_Reconstruction/src/, alongside:
  train_unet_residual_v3.py, evaluate_v3_cloud_boundaries.py,
  test_pytorch_dataloader.py, synthetic_clouds_v2.py, unet_model.py.

Run (no dataset/checkpoint required for first command):
  .\.venv\Scripts\python.exe -u .\src\evaluate_v3_structure.py --self-test
  .\.venv\Scripts\python.exe -u .\src\evaluate_v3_structure.py

READS: 75 fixed validation pairs, saved V3 best checkpoint, synthetic alpha labels.
Never reads training or TEST image arrays; never trains or changes model checkpoints.
Alpha is used for diagnostic region labels only, NOT by any reconstruction method.
Surface proxy labels are coarse (water_like vs land_or_mixed): they are NOT
verified per-pixel water/urban/vegetation classifications.

This implementation measures (per patch, normalized NIR/Red/Green DN/1023):
  - cloud-core MAE and PSNR, halo MAE and all-valid MAE
  - interior-cloud SSIM (7x7 Gaussian windows wholly inside core+valid)
  - within-core gradient MAE and signed-gradient Pearson correlation
  - within-core texture standard-deviation ratio and gradient ratio
These do not establish true recovery under real thick cloud cover.

No pandas, scipy, sklearn, skimage or torchvision required.
'''
import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from synthetic_clouds_v2 import false_color
from test_pytorch_dataloader import CloudPairsDataset, ROOT
from train_unet_residual_v3 import read_manifest, load_checkpoint, BEST, DN_MAX
from evaluate_v3_cloud_boundaries import predict, load_synthetic_diagnostics

OUT = ROOT / 'outputs' / 'unet_structural_evaluation_v3'
VARIANTS = ('cloudy_input', 'visible_mean', 'local_fill', 'v3_original', 'v3_dilate3_feather3')
REPORT_FIELDS = (
    'core_mae', 'core_psnr_db', 'core_ssim7_interior', 'core_ssim_centers',
    'halo_mae', 'all_valid_mae', 'texture_std_ratio',
    'gradient_mae', 'gradient_ratio', 'gradient_corr',
)
LABEL_FIELD = 'surface_proxy'
EPS = 1e-12


def gaussian_kernel(size=7, sigma=1.5, device='cpu'):
    axis = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    w = torch.exp(-(axis ** 2) / (2 * sigma * sigma))
    w = w / w.sum()
    k = torch.outer(w, w)
    return k[None, None].repeat(3, 1, 1, 1)


def ssim_map(x, y, kernel):
    '''Standard SSIM local map, 3 bands independently, range normalized to 0..1.'''
    if x.shape != y.shape or x.ndim != 4 or x.shape[1] != 3:
        raise ValueError('SSIM expects matching [B,3,H,W] tensors')
    pad = kernel.shape[-1] // 2
    # Reflect padding avoids injecting artificial black image borders.
    def filt(a):
        return F.conv2d(F.pad(a, (pad, pad, pad, pad), mode='reflect'),
                        kernel, groups=3)
    ux, uy = filt(x), filt(y)
    vx = (filt(x*x) - ux*ux).clamp_min(0.)
    vy = (filt(y*y) - uy*uy).clamp_min(0.)
    cov = filt(x*y) - ux*uy
    c1, c2 = 0.01**2, 0.03**2
    return ((2*ux*uy+c1)*(2*cov+c2)) / ((ux*ux+uy*uy+c1)*(vx+vy+c2))


def region_mae(a, b, region):
    mask = region.expand_as(a) > 0.5
    if not mask.any().item():
        return float('nan')
    return (a[mask] - b[mask]).abs().mean().item()


def structural_metrics(pred, target, cloud_mask, valid, alpha, kernel):
    '''All arguments batch size 1, device matched; oracle alpha labels EVAL only.'''
    if pred.shape != target.shape or pred.shape[0] != 1:
        raise ValueError('Metrics require one patch with three equal-sized bands')
    if not all(z.shape == cloud_mask.shape for z in (valid, alpha)):
        raise ValueError('Mask/valid/alpha dimensions differ')
    if not torch.isfinite(pred).all().item():
        raise ValueError('Non-finite prediction')
    core = (cloud_mask > 0.5) & (valid > 0.5)
    halo = (alpha > 0.02) & (alpha < 0.30) & (cloud_mask <= 0.5) & (valid > 0.5)
    validpix = valid > 0.5
    if core.sum().item() < 100:
        raise ValueError('Too few cloud-core pixels to evaluate')

    # Masked SSIM is sampled ONLY from valid, fully interior cloud windows.
    # This prevents clean surroundings leaking into the SSIM score, while making
    # the metric explicitly an *interior* structural score rather than full SSIM.
    window = kernel.shape[-1]
    centers = F.avg_pool2d(core.float(), window, stride=1, padding=window//2) > (1 - 1e-6)
    ncenters = int(centers.sum().item())
    structural = ssim_map(pred, target, kernel)
    interior_ssim = structural[centers.expand_as(structural)].mean().item() if ncenters else float('nan')

    m = core[0, 0]
    p, t = pred[0], target[0]
    pc, tc = p[:, m], t[:, m]
    std_target = tc.std(dim=1, unbiased=False).mean().item()
    std_pred = pc.std(dim=1, unbiased=False).mean().item()

    gx = m[:, 1:] & m[:, :-1]
    gy = m[1:, :] & m[:-1, :]
    dxp = (p[:, :, 1:] - p[:, :, :-1])[:, gx].reshape(-1)
    dxt = (t[:, :, 1:] - t[:, :, :-1])[:, gx].reshape(-1)
    dyp = (p[:, 1:, :] - p[:, :-1, :])[:, gy].reshape(-1)
    dyt = (t[:, 1:, :] - t[:, :-1, :])[:, gy].reshape(-1)
    diff_pred = torch.cat((dxp, dyp))
    diff_target = torch.cat((dxt, dyt))
    if diff_pred.numel() < 20:
        raise ValueError('Insufficient interior cloud neighbours for gradient statistics')
    g_t = diff_target.abs().mean().item()
    g_p = diff_pred.abs().mean().item()
    grad_err = (diff_pred-diff_target).abs().mean().item()
    centered_p = diff_pred - diff_pred.mean()
    centered_t = diff_target - diff_target.mean()
    denominator = torch.linalg.vector_norm(centered_p) * torch.linalg.vector_norm(centered_t)
    corr = ((centered_p*centered_t).sum()/denominator).item() if denominator.item() > EPS else float('nan')

    core_mse = (pc - tc).square().mean().item()
    return {
        'core_mae': region_mae(pred, target, core),
        'core_psnr_db': -10.0 * math.log10(max(core_mse, EPS)),
        'core_ssim7_interior': float(interior_ssim),
        'core_ssim_centers': ncenters,
        'halo_mae': region_mae(pred, target, halo),
        'all_valid_mae': region_mae(pred, target, validpix),
        'texture_std_ratio': std_pred/max(std_target, 1e-8),
        'gradient_mae': grad_err,
        'gradient_ratio': g_p/max(g_t, 1e-8),
        'gradient_corr': corr,
    }


def visible_mean_composite(x, valid):
    cloudy, m = x[:, :3], x[:, 3:4]
    seen = (valid > .5) & (m <= .5)
    count = seen.sum(dim=(2,3), keepdim=True)
    if (count < 1).any().item():
        raise ValueError('Patch has no visible valid input pixels')
    mu = (cloudy * seen).sum(dim=(2,3), keepdim=True)/count
    return torch.where(m > .5, mu.expand_as(cloudy), cloudy)


def predictions(model, x, valid):
    '''All inference uses ONLY observed cloudy input, binary mask, valid mask.'''
    with torch.inference_mode():
        v3, local = model(x, valid)
        v3_exact = predict(model, x, valid, 0, 0)
        if not torch.allclose(v3, v3_exact, atol=1e-7, rtol=0):
            raise AssertionError('Original V3 prediction mismatch')
        refined = predict(model, x, valid, 3, 3)
        mean = visible_mean_composite(x, valid)
        return {
            'cloudy_input': x[:, :3],
            'visible_mean': mean,
            'local_fill': local,
            'v3_original': v3,
            'v3_dilate3_feather3': refined,
        }


def save_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(detail):
    summary = []
    labels = sorted({d['surface_proxy'] for d in detail})
    for group in ('ALL', *labels):
        for variant in VARIANTS:
            subset = [r for r in detail if r['variant'] == variant and (group == 'ALL' or r['surface_proxy'] == group)]
            if not subset:
                continue
            row = {'surface_proxy': group, 'variant': variant, 'patches': len(subset)}
            for k in REPORT_FIELDS:
                a = np.asarray([float(r[k]) for r in subset], dtype=np.float64)
                good = np.isfinite(a)
                row[k] = float(a[good].mean()) if good.any() else float('nan')
            row['ssim_available_patches'] = sum(np.isfinite(float(r['core_ssim7_interior'])) for r in subset)
            summary.append(row)
    return summary


def draw_preview(samples, path, epoch):
    heads = ['Clear target', 'Synthetic cloudy', 'Visible mean', 'Local fill',
             'V3 original', 'V3 dilate3+feather3', 'Cloud mask']
    tile, gap, caption, top = 172, 10, 35, 76
    w = gap + len(heads)*(tile+gap)
    height = top + len(samples)*(tile+caption+gap)+gap
    canvas = Image.new('RGB', (w, height), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), f'Mumbai | V3 epoch {epoch} | FIXED VALIDATION structural evaluation', fill='black')
    draw.text((gap, 29), 'Fixed false-colour display | Predictions do NOT use clear targets or synthetic alpha', fill='black')
    for i, h in enumerate(heads):
        draw.text((gap+i*(tile+gap), 55), h, fill='black')
    for j, sample in enumerate(samples):
        row_y = top+j*(tile+caption+gap)
        for i, panel in enumerate(sample['images']):
            canvas.paste(Image.fromarray(panel).resize((tile,tile),Image.Resampling.BILINEAR),
                         (gap+i*(tile+gap),row_y))
        draw.text((gap, row_y+tile+5),
                  f"{sample['id']} | {sample['surface']} | cloud {sample['cover']:.1f}%", fill='black')
        draw.text((gap+4*(tile+gap), row_y+tile+5),
                  f"SSIM {sample['ssim_v3']:.3f} -> {sample['ssim_ref']:.3f}", fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def self_test():
    # Synthetic shapes, known-ground-truth metrics; doesn't need project files.
    torch.manual_seed(123)
    t = torch.rand(1,3,64,64)*.4 + .1
    m = torch.zeros(1,1,64,64)
    m[..., 10:53, 12:56] = 1
    valid = torch.ones_like(m)
    alpha = .7*m
    k = gaussian_kernel(device=t.device)
    perfect = structural_metrics(t,t,m,valid,alpha,k)
    assert abs(perfect['core_ssim7_interior'] - 1.0) < 1e-5
    assert perfect['core_mae'] == 0.0
    assert perfect['gradient_mae'] == 0.0
    assert perfect['core_ssim_centers'] > 500
    pred = t.clone()
    pred[..., 10:53, 12:56] = .3
    flat = structural_metrics(pred,t,m,valid,alpha,k)
    assert flat['core_ssim7_interior'] < perfect['core_ssim7_interior']
    assert flat['core_mae'] > 0 and flat['gradient_mae'] > 0
    cloudy = t*(1-m)+.9*m
    x = torch.cat((cloudy,m),dim=1)
    mean = visible_mean_composite(x,valid)
    assert torch.equal(mean*(1-m),cloudy*(1-m))
    from train_unet_residual_v3 import ResidualCloudUNet
    model = ResidualCloudUNet().eval()
    outputs = predictions(model,x,valid)
    assert len(outputs) == len(VARIANTS)
    for key, output in outputs.items():
        assert output.shape==t.shape and torch.isfinite(output).all(), key
    print('STRUCTURAL EVALUATION SELF-TEST PASSED')
    print('Interior SSIM perfect target = 1.0; smoothed fill < 1.0; V3 inference and baselines passed.')


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true', help='No dataset/checkpoints required')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not BEST.exists():
        raise FileNotFoundError(f'V3 best checkpoint missing: {BEST}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, state = load_checkpoint(BEST, device, residual_scale=0.25)
    model.eval()
    rows = read_manifest()
    ds = CloudPairsDataset(rows, 'val')
    if len(ds) != 75:
        raise AssertionError('Only the 75 validation patches are allowed')
    chosen = set(np.linspace(0, 74, 6).round().astype(int).tolist())
    kernel = gaussian_kernel(device=device)
    detail = []
    examples = []
    print('V3 STRUCTURAL RECONSTRUCTION EVALUATION — FIXED VALIDATION ONLY', flush=True)
    print(f"Checkpoint epoch {state['epoch']} | device {device} | {len(ds)} validation patches", flush=True)
    print('Metrics: core MAE/PSNR, 7x7 INTERIOR masked SSIM, gradient/texture fidelity, halo MAE.',flush=True)
    print('Surface group: water_like vs land_or_mixed coarse labels (NOT urban/vegetation ground truth).',flush=True)
    print('TEST data and model training are not accessed. Alpha is for diagnostic labels ONLY.',flush=True)
    for idx in range(len(ds)):
        item = ds[idx]
        record = ds.rows[idx]
        x = item['input'][None].to(device)
        t = item['target'][None].to(device)
        m = item['cloud_mask'][None].to(device)
        valid = item['valid_mask'][None].to(device)
        alpha, _ = load_synthetic_diagnostics(record)
        alpha = alpha.to(device)
        pred_dict = predictions(model,x,valid)
        group = record.get(LABEL_FIELD,'').strip() or 'unlabeled'
        stored = {}
        for variant in VARIANTS:
            metr = structural_metrics(pred_dict[variant],t,m,valid,alpha,kernel)
            detail.append({'patch_id': record['patch_id'], 'surface_proxy': group,
                           'cloud_fraction': record['cloud_fraction'], 'variant': variant, **metr})
            stored[variant] = metr
        if idx in chosen:
            vnp = item['valid_mask'][0].numpy() > .5
            clear = item['target'].numpy().transpose(1,2,0) * DN_MAX
            cloudy = item['input'][:3].numpy().transpose(1,2,0) * DN_MAX
            cols = [false_color(clear,vnp),false_color(cloudy,vnp)]
            for label in VARIANTS[1:]:
                cols.append(false_color(pred_dict[label][0].cpu().numpy().transpose(1,2,0)*DN_MAX,vnp))
            mask_rgb = np.repeat((item['cloud_mask'][0].numpy()>0.5).astype(np.uint8)[...,None]*255,3,axis=2)
            cols.append(mask_rgb)
            assert len(cols)==7
            examples.append({'id': record['patch_id'], 'surface': group,
                             'cover': float(record['cloud_fraction'])*100,
                             'ssim_v3':stored['v3_original']['core_ssim7_interior'],
                             'ssim_ref':stored['v3_dilate3_feather3']['core_ssim7_interior'],
                             'images':cols})
        if (idx + 1) % 15 == 0:
            print(f'Evaluated {idx+1}/75 validation patches', flush=True)
    if len(detail) != len(VARIANTS)*75:
        raise AssertionError('Incomplete validation evaluation')
    summary = summarize(detail)
    print('\nALL VALIDATION PATCHES — PER-PATCH UNWEIGHTED MEANS', flush=True)
    print(f"{'Variant':<23} {'Core MAE':>9} {'SSIM7':>9} {'Grad corr':>10} {'Grad MAE':>9} {'Std ratio':>10} {'Halo MAE':>10}")
    for r in summary:
        if r['surface_proxy'] == 'ALL':
            print(f"{r['variant']:<23} {r['core_mae']:9.5f} {r['core_ssim7_interior']:9.4f} "
                  f"{r['gradient_corr']:10.4f} {r['gradient_mae']:9.5f} "
                  f"{r['texture_std_ratio']:10.3f} {r['halo_mae']:10.5f}",flush=True)
    print('\nPER-SURFACE PROXY GROUPS',flush=True)
    for group in sorted(set(r['surface_proxy'] for r in summary if r['surface_proxy']!='ALL')):
        group_rows = [r for r in summary if r['surface_proxy']==group]
        print(f"  {group} ({group_rows[0]['patches']} patches)",flush=True)
        for r in group_rows:
            print(f"    {r['variant']:<23} MAE {r['core_mae']:.5f} | SSIM7 {r['core_ssim7_interior']:.4f}",flush=True)
    detail_file = OUT / 'validation_structural_metrics_by_patch.csv'
    summary_file = OUT / 'validation_structural_summary.csv'
    preview_file = OUT / 'validation_structural_comparison.png'
    save_csv(detail_file,detail,('patch_id','surface_proxy','cloud_fraction','variant',*REPORT_FIELDS))
    save_csv(summary_file,summary,('surface_proxy','variant','patches',*REPORT_FIELDS,'ssim_available_patches'))
    draw_preview(examples,preview_file,state['epoch'])
    print('\nSTRUCTURAL EVALUATION COMPLETE',flush=True)
    print(f'Summary: {summary_file}',flush=True)
    print(f'Per-patch metrics: {detail_file}',flush=True)
    print(f'Visual comparison: {preview_file}',flush=True)
    print('REMINDER: interior SSIM excludes boundaries; group labels are coarse; REAL-cloud truth unknown.')


if __name__ == '__main__':
    main()
