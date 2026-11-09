r"""Diagnose U-Net V1 flat/oversmoothed cloud fills on held-out VALIDATION data.

Does NOT train, change checkpoints or touch the TEST split. Compares the U-Net
against a simple per-image *visible-pixel channel mean* fill, and measures
within-cloud spatial detail. No pandas or torchvision required.

Run at project root:
  .\.venv\Scripts\python.exe -u .\src\diagnose_unet_validation_v1.py

Output:
  outputs/unet_training_v1/validation_diagnostics/validation_diagnostics.csv
"""

import csv
from pathlib import Path

import numpy as np
import torch

from test_pytorch_dataloader import CloudPairsDataset, MANIFEST, ROOT
from preview_unet_validation_v1 import load_checkpoint

OUT = ROOT / "outputs" / "unet_training_v1" / "validation_diagnostics"
FIELDS = [
    "patch_id", "cloud_percent", "model_cloud_mae", "visible_mean_cloud_mae",
    "cloudy_input_cloud_mae", "model_nir_mae", "model_red_mae", "model_green_mae",
    "target_cloud_std", "model_cloud_std", "std_ratio",
    "target_incloud_gradient", "model_incloud_gradient", "gradient_ratio",
    "incloud_gradient_error",
]


def mean_abs(a, b, region):
    return float(np.abs(a - b)[region].mean())


def cloud_std(arr, region):
    """Mean of per-band standard deviations within the cloud mask."""
    pixels = arr[region]
    return float(pixels.std(axis=0).mean())


def internal_gradients(pred, target, region):
    """Compare adjacent-pixel changes only where BOTH pixels are in cloud."""
    horizontal = region[:, :-1] & region[:, 1:]
    vertical = region[:-1, :] & region[1:, :]
    pieces = []
    for common, axis in ((horizontal, 1), (vertical, 0)):
        if not np.any(common):
            continue
        if axis == 1:
            a = pred[:, 1:, :] - pred[:, :-1, :]
            b = target[:, 1:, :] - target[:, :-1, :]
        else:
            a = pred[1:, :, :] - pred[:-1, :, :]
            b = target[1:, :, :] - target[:-1, :, :]
        pieces.append((np.abs(a[common]), np.abs(b[common])))
    if not pieces:
        raise ValueError("No adjacent cloudy pixels found")
    gp = np.concatenate([p[0].reshape(-1) for p in pieces])
    gt = np.concatenate([p[1].reshape(-1) for p in pieces])
    # |(pred_i-pred_j) - (target_i-target_j)| is a detail error.
    # Store explicitly signed gradients for correct evaluation.
    err_num = 0.0
    total = 0
    for common, axis in ((horizontal, 1), (vertical, 0)):
        if not np.any(common):
            continue
        if axis == 1:
            dp = pred[:, 1:, :] - pred[:, :-1, :]
            dt = target[:, 1:, :] - target[:, :-1, :]
        else:
            dp = pred[1:, :, :] - pred[:-1, :, :]
            dt = target[1:, :, :] - target[:-1, :, :]
        dif = np.abs(dp[common] - dt[common])
        err_num += float(dif.sum())
        total += dif.size
    return float(gp.mean()), float(gt.mean()), err_num / total


def compute_metrics(pred, x, target, cloud_mask, valid_mask, patch_id):
    region = cloud_mask & valid_mask
    visible = (~cloud_mask) & valid_mask
    if region.sum() < 2 or visible.sum() < 2:
        raise ValueError(f"{patch_id}: insufficient valid cloudy or visible pixels")
    if pred.shape != target.shape or pred.shape != x.shape:
        raise ValueError(f"{patch_id}: invalid prediction/input/target shapes")
    if not np.isfinite(pred).all():
        raise ValueError(f"{patch_id}: prediction contains NaN/Inf")

    # Baseline uses ONLY observed input outside the binary cloud mask.
    mean_visible_spectrum = x[visible].mean(axis=0)
    mean_fill = np.broadcast_to(mean_visible_spectrum, target.shape)
    pred_g, target_g, detail_error = internal_gradients(pred, target, region)
    pstd, tstd = cloud_std(pred, region), cloud_std(target, region)

    row = {
        "patch_id": patch_id,
        "cloud_percent": 100.0 * float(region.sum()) / float(valid_mask.sum()),
        "model_cloud_mae": mean_abs(pred, target, region),
        "visible_mean_cloud_mae": mean_abs(mean_fill, target, region),
        "cloudy_input_cloud_mae": mean_abs(x, target, region),
        "target_cloud_std": tstd,
        "model_cloud_std": pstd,
        "std_ratio": pstd / max(tstd, 1e-8),
        "target_incloud_gradient": target_g,
        "model_incloud_gradient": pred_g,
        "gradient_ratio": pred_g / max(target_g, 1e-8),
        "incloud_gradient_error": detail_error,
    }
    for c, label in enumerate(("nir", "red", "green")):
        row[f"model_{label}_mae"] = float(np.abs(pred[..., c] - target[..., c])[region].mean())
    return row


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Cannot find dataset manifest: {MANIFEST}")
    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    val = CloudPairsDataset(rows, "val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_checkpoint(device)
    print("U-NET V1 — VALIDATION RECONSTRUCTION DIAGNOSTICS", flush=True)
    print(f"Best checkpoint epoch: {epoch} | Device: {device}", flush=True)
    print(f"Validation patches: {len(val)} | TEST split NOT loaded", flush=True)

    results = []
    with torch.inference_mode():
        for k in range(len(val)):
            item = val[k]
            pred = model(item["input"].unsqueeze(0).to(device))[0].cpu().numpy()
            pred = np.clip(pred.transpose(1, 2, 0), 0.0, 1.0)
            x = item["input"][:3].numpy().transpose(1, 2, 0)
            target = item["target"].numpy().transpose(1, 2, 0)
            cloud = item["cloud_mask"][0].numpy() > 0.5
            valid = item["valid_mask"][0].numpy() > 0.5
            results.append(compute_metrics(pred, x, target, cloud, valid, item["patch_id"]))
            if k == 0 or (k + 1) % 15 == 0 or k + 1 == len(val):
                print(f"Checked {k + 1}/{len(val)} patches", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "validation_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow({key: (f"{row[key]:.7f}" if isinstance(row[key], float) else row[key]) for key in FIELDS})

    def avg(name):
        return sum(r[name] for r in results) / len(results)

    better = sum(r["model_cloud_mae"] < r["visible_mean_cloud_mae"] for r in results)
    print("\nVALIDATION DIAGNOSTIC SUMMARY (per-patch unweighted means)")
    print(f"U-Net cloud MAE:           {avg('model_cloud_mae'):.5f}")
    print(f"Visible-mean fill MAE:     {avg('visible_mean_cloud_mae'):.5f}")
    print(f"Unchanged cloudy MAE:      {avg('cloudy_input_cloud_mae'):.5f}")
    print(f"U-Net beats mean fill on:  {better}/{len(results)} patches")
    print(f"Target cloud-region std:   {avg('target_cloud_std'):.5f}")
    print(f"U-Net cloud-region std:    {avg('model_cloud_std'):.5f}")
    print(f"U-Net / target std ratio:  {avg('std_ratio'):.3f}")
    print(f"Target within-cloud grad:  {avg('target_incloud_gradient'):.5f}")
    print(f"U-Net within-cloud grad:   {avg('model_incloud_gradient'):.5f}")
    print(f"U-Net / target grad ratio: {avg('gradient_ratio'):.3f}")
    print(f"Within-cloud grad error:   {avg('incloud_gradient_error'):.5f}")
    print(f"CSV: {csv_path}")
    print("Interpretation: mean-fill comparisons and texture ratios reveal whether")
    print("low pixel error mainly reflects average-colour estimation rather than detail recovery.")
    print("No model was trained; test split was not used.")


if __name__ == "__main__":
    main()
