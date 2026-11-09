r"""Validation-only cloud-boundary postprocessing experiment for Mumbai Residual U-Net V3.

Save in Cloud_Reconstruction/src/, alongside train_unet_residual_v3.py.
Run:
    .\.venv\Scripts\python.exe -u .\src\evaluate_v3_cloud_boundaries.py --self-test
    .\.venv\Scripts\python.exe -u .\src\evaluate_v3_cloud_boundaries.py

Read-only inputs: saved V3 best checkpoint and 75 fixed validation patches.
Outputs (new directory): outputs/unet_boundary_refinement_v3/

IMPORTANT: synthetic alpha and shadow arrays are used ONLY to measure error,
never to construct the model input, expanded masks, or the compositing weights.
At real-scene inference, an external estimated cloud mask is required.
No pandas, SciPy, sklearn, or torchvision required.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from synthetic_clouds_v2 import false_color
from test_pytorch_dataloader import CloudPairsDataset, ROOT
from train_unet_residual_v3 import read_manifest, load_checkpoint, BEST, DN_MAX, local_visible_fill

OUT = ROOT / 'outputs' / 'unet_boundary_refinement_v3'
CONFIGS = [
    ('original', 0, 0),
    ('dilate3', 3, 0),
    ('dilate3_feather3', 3, 3),
    ('dilate6', 6, 0),
    ('dilate6_feather3', 6, 3),
    ('dilate10_feather3', 10, 3),
]
METRICS = ('core_mae', 'halo_mae', 'shadow_mae', 'clean_mae', 'all_valid_mae')


def dilate_mask(mask, radius, valid):
    if radius == 0:
        return mask * valid
    k = radius * 2 + 1
    return F.max_pool2d(mask, k, stride=1, padding=radius) * valid


def predict(model, x, valid, radius, feather):
    """Use only observed cloudy pixels, validity and the estimated BINARY cloud mask."""
    cloudy, original = x[:, :3], x[:, 3:4]
    expanded = dilate_mask(original, radius, valid)
    # The original config is exactly the saved V3 inference behavior.
    if radius == 0 and feather == 0:
        out, _ = model(x, valid)
        return out
    baseline = local_visible_fill(cloudy, expanded, valid)
    features = torch.cat((cloudy, expanded, baseline), dim=1)
    raw = model.refiner(features)
    inferred = (baseline + (raw - 0.5) * (2.0 * model.residual_scale)).clamp(0, 1)
    if feather == 0:
        weights = expanded
    else:
        k = feather * 2 + 1
        soft = F.avg_pool2d(expanded, k, stride=1, padding=feather)
        # Keep the full core correction; blend only near the outer boundary.
        weights = torch.maximum(original, soft) * valid
    return cloudy * (1.0 - weights) + inferred * weights


def load_synthetic_diagnostics(record):
    def load(field):
        p = Path(record[field])
        if not p.is_absolute():
            p = ROOT / p
        a = np.load(p, allow_pickle=False)
        if a.shape != (256, 256) or not np.isfinite(a).all():
            raise ValueError(f"Unexpected {field} for {record['patch_id']}: {a.shape}")
        return torch.from_numpy(np.ascontiguousarray(a.astype(np.float32)))[None, None]
    return load('alpha_path'), load('shadow_path')


def masked_mae(pred, target, region):
    region = region.expand_as(pred)
    if not region.any().item():
        return float('nan')
    return float((pred[region] - target[region]).abs().mean().item())


def get_metrics(pred, target, orig_binary, valid, alpha, shadow):
    m = orig_binary > .5
    v = valid > .5
    # Alpha/shadow are synthetic oracle masks: evaluation only, never input.
    core = m & v
    halo = (alpha > 0.02) & (alpha < 0.30) & (~m) & v
    shadow_only = (alpha <= 0.02) & (shadow > 0.025) & v
    clean = (alpha <= 0.02) & (shadow <= 0.025) & v
    regions = (core, halo, shadow_only, clean, v)
    return dict(zip(METRICS, [masked_mae(pred, target, r) for r in regions]))


def make_sheet(examples, path, epoch):
    heads = ['Clear target', 'Synthetic cloudy', 'V3 original', 'Dilate 3 + feather',
             'Dilate 6 + feather', 'Binary mask']
    tile, gap, line = 190, 12, 230
    top = 80
    sheet = Image.new('RGB', (gap + len(heads) * (tile + gap), top + len(examples) * line), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 9), f'Mumbai | V3 best epoch {epoch} | Boundary refinement | FIXED VALIDATION only', fill='black')
    draw.text((gap, 29), 'Same false-colour mapping | Alpha/shadow used ONLY for evaluation metrics', fill='black')
    for i, title in enumerate(heads):
        draw.text((gap + i * (tile + gap), 55), title, fill='black')
    for r, (patch_id, data) in enumerate(examples):
        y = top + r * line
        panels = data['panels']
        for j, panel in enumerate(panels):
            image = Image.fromarray(panel.astype(np.uint8)).resize((tile, tile), Image.Resampling.BILINEAR)
            sheet.paste(image, (gap + j * (tile + gap), y))
        draw.text((gap, y + tile + 8), f'{patch_id} | cloud {data["coverage"]:.1f}%', fill='black')
        draw.text((gap + 2 * (tile + gap), y + tile + 8), f'Core MAE {data["core"]:.4f}', fill='black')
        draw.text((gap + 4 * (tile + gap), y + tile + 8), f'Core MAE {data["refined"]:.4f}', fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def summary_rows(rows):
    result = []
    for name, _, _ in CONFIGS:
        subset = [r for r in rows if r['variant'] == name]
        item = {'variant': name, 'patches': len(subset)}
        for metric in METRICS:
            nums = np.asarray([r[metric] for r in subset], dtype=float)
            item[metric] = float(np.mean(nums[np.isfinite(nums)])) if np.isfinite(nums).any() else float('nan')
        result.append(item)
    return result


def save_csv(path, rows, keys):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    from train_unet_residual_v3 import ResidualCloudUNet
    torch.manual_seed(3)
    mdl = ResidualCloudUNet().eval()
    valid = torch.ones(1, 1, 64, 64)
    m = torch.zeros_like(valid)
    m[..., 15:45, 18:48] = 1.0
    target = .1 + .3 * torch.rand(1, 3, 64, 64)
    cloudy = torch.where(m.expand_as(target) > .5, torch.ones_like(target) * .8, target)
    x = torch.cat([cloudy, m], 1)
    with torch.inference_mode():
        original = predict(mdl, x, valid, 0, 0)
        direct, _ = mdl(x, valid)
        assert torch.allclose(original, direct, atol=0, rtol=0)
        for _, radius, feather in CONFIGS:
            out = predict(mdl, x, valid, radius, feather)
            assert out.shape == target.shape and torch.isfinite(out).all()
            assert torch.equal(out[..., :3, :3], cloudy[..., :3, :3])
        alpha = m * .7
        shadow = torch.zeros_like(m)
        metrics = get_metrics(original, target, m, valid, alpha, shadow)
        assert np.isfinite(metrics['core_mae'])
    print('BOUNDARY DIAGNOSTIC SELF-TEST PASSED')
    print('Only binary masks affect inference; synthetic alpha/shadow used for evaluation.')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not BEST.is_file():
        raise FileNotFoundError(f'Missing trained V3 best checkpoint: {BEST}')
    model, checkpoint = load_checkpoint(BEST, device, residual_scale=0.25)
    model.eval()
    records = read_manifest()
    ds = CloudPairsDataset(records, 'val')
    example_indices = sorted(set(np.linspace(0, 74, 6).round().astype(int)))
    results, examples = [], []
    print('V3 CLOUD-BOUNDARY REFINEMENT — FIXED VALIDATION ONLY', flush=True)
    print(f'Checkpoint: best epoch {checkpoint["epoch"]} | Device: {device}', flush=True)
    print('Variants: ' + ', '.join(name for name, _, _ in CONFIGS), flush=True)
    print('Alpha/shadows used ONLY to evaluate true synthetic corruption; never as model features.', flush=True)
    for n in range(len(ds)):
        item = ds[n]
        record = ds.rows[n]
        x = item['input'].unsqueeze(0).to(device)
        v = item['valid_mask'].unsqueeze(0).to(device)
        t = item['target'].unsqueeze(0).to(device)
        m = item['cloud_mask'].unsqueeze(0).to(device)
        alpha, shadow = load_synthetic_diagnostics(record)
        alpha, shadow = alpha.to(device), shadow.to(device)
        predictions = {}
        patch_id = record['patch_id']
        for name, radius, feather in CONFIGS:
            out = predict(model, x, v, radius, feather)
            metrics = get_metrics(out, t, m, v, alpha, shadow)
            results.append({'patch_id': patch_id, 'variant': name, **metrics})
            if n in example_indices:
                predictions[name] = out[0].cpu().numpy().transpose(1, 2, 0) * DN_MAX
        if n in example_indices:
            valid_np = item['valid_mask'][0].numpy() > .5
            cloudy_dn = item['input'][:3].numpy().transpose(1, 2, 0) * DN_MAX
            clear_dn = item['target'].numpy().transpose(1, 2, 0) * DN_MAX
            cloudmask = item['cloud_mask'][0].numpy() > .5
            panels = [false_color(clear_dn, valid_np), false_color(cloudy_dn, valid_np),
                      false_color(predictions['original'], valid_np),
                      false_color(predictions['dilate3_feather3'], valid_np),
                      false_color(predictions['dilate6_feather3'], valid_np),
                      np.repeat((cloudmask.astype(np.uint8) * 255)[..., None], 3, axis=2)]
            core0 = get_metrics(torch.from_numpy(predictions['original'].transpose(2,0,1))[None].to(device) / DN_MAX, t, m, v, alpha, shadow)['core_mae']
            core1 = get_metrics(torch.from_numpy(predictions['dilate6_feather3'].transpose(2,0,1))[None].to(device) / DN_MAX, t, m, v, alpha, shadow)['core_mae']
            examples.append((patch_id, {'panels': panels, 'coverage': float(record['cloud_fraction']) * 100,
                                        'core': core0, 'refined': core1}))
        if (n + 1) % 15 == 0 or n + 1 == len(ds):
            print(f'Validated {n+1}/{len(ds)} patches', flush=True)
    if len(results) != 75 * len(CONFIGS):
        raise AssertionError('Expected 75 patches x 6 variants')
    summaries = summary_rows(results)
    print('\nAVERAGE VALIDATION MAE (per-patch, lower is better)', flush=True)
    print(f'{"Variant":<22} {"Core":>9} {"Halo":>9} {"Shadow":>9} {"Clean":>9} {"All":>9}')
    for s in summaries:
        print(f'{s["variant"]:<22} {s["core_mae"]:9.5f} {s["halo_mae"]:9.5f} {s["shadow_mae"]:9.5f} {s["clean_mae"]:9.5f} {s["all_valid_mae"]:9.5f}', flush=True)
    if not (0.006 <= summaries[0]['core_mae'] <= 0.025):
        raise AssertionError('Original V3 core MAE differs unusually from expected ~0.013; investigate input')
    save_csv(OUT / 'validation_boundary_metrics.csv', results, ['patch_id','variant',*METRICS])
    save_csv(OUT / 'validation_boundary_summary.csv', summaries, ['variant','patches',*METRICS])
    make_sheet(examples, OUT / 'validation_boundary_comparison.png', checkpoint['epoch'])
    print('\nBOUNDARY EXPERIMENT COMPLETE', flush=True)
    print(f'Summary: {OUT / "validation_boundary_summary.csv"}', flush=True)
    print(f'Preview: {OUT / "validation_boundary_comparison.png"}', flush=True)
    print('NOTE: Alpha/shadow masks are synthetic evaluation labels, not available for real clouds.')
    print('An improvement in halo errors does not prove hidden texture recovery.')


if __name__ == '__main__':
    main()
