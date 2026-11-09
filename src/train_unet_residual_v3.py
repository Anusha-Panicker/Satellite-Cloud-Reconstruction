r"""Residual U-Net V3 for Mumbai LISS-IV synthetic cloud reconstruction.

Save as: Cloud_Reconstruction/src/train_unet_residual_v3.py
Uses existing src/ files: unet_model.py, test_pytorch_dataloader.py,
train_unet_detail_v2.py, synthetic_clouds_v2.py.

3-epoch controlled experiment (test split is never opened):
  .\.venv\Scripts\python.exe -u .\src\train_unet_residual_v3.py --epochs 3 --batch-size 2

Resume later to 6 TOTAL epochs, if warranted:
  .\.venv\Scripts\python.exe -u .\src\train_unet_residual_v3.py --epochs 6 --batch-size 2 --resume

Sanity test without the satellite dataset:
  .\.venv\Scripts\python.exe -u .\src\train_unet_residual_v3.py --self-test

NO pandas, torchvision, scikit-learn, or SciPy required.
This is experimental single-image inpainting, NOT recovery of real hidden truth.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from synthetic_clouds_v2 import false_color
from test_pytorch_dataloader import CloudPairsDataset, MANIFEST, EXPECTED, ROOT
from train_unet_detail_v2 import DynamicTrainDataset
from unet_model import CloudUNet

OUT = ROOT / "outputs" / "unet_training_residual_v3"
CKPT_DIR = OUT / "checkpoints"
BEST = CKPT_DIR / "best.pt"
LATEST = CKPT_DIR / "latest.pt"
HISTORY = OUT / "history.csv"
CONFIG = OUT / "config.json"
VAL_CSV = OUT / "best_validation_metrics.csv"
PREVIEW = OUT / "validation_comparison_v2_vs_v3.png"
V2_BEST = ROOT / "outputs" / "unet_training_detail_v2" / "checkpoints" / "best.pt"
DN_MAX = 1023.0
FIELDNAMES = (
    "epoch", "train_cloud_mae", "val_v3_cloud_mae", "val_visible_mean_mae",
    "val_local_fill_mae", "val_cloudy_mae", "val_v3_beats_mean_patches",
    "val_v3_beats_local_patches", "val_texture_std_ratio",
    "val_gradient_ratio", "val_cloud_psnr_db", "epoch_seconds",
    "peak_cuda_mib", "best_val_cloud_mae",
)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=3, help="Total desired epochs")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--grad-weight", type=float, default=0.10)
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--self-test", action="store_true", help="No dataset or output files needed")
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.log_every < 1:
        p.error("epochs, batch-size, log-every must be positive")
    if a.lr <= 0 or a.grad_weight < 0 or not (0 < a.residual_scale <= 1):
        p.error("lr > 0, grad-weight >= 0, and 0 < residual-scale <= 1 required")
    return a


def read_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing synthetic manifest: {MANIFEST}")
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    counts = dict(Counter(r["split"].strip() for r in rows))
    ids = [r["patch_id"] for r in rows]
    if counts != EXPECTED or len(rows) != 649:
        raise ValueError(f"Unexpected dataset counts: {counts}; expected {EXPECTED}")
    if len(ids) != len(set(ids)) or "C0137" in ids:
        raise ValueError("Duplicate IDs or excluded patch C0137 in manifest")
    return rows


@torch.no_grad()
def local_visible_fill(cloudy, mask, valid):
    """Multi-scale, ONLY-visible-pixel interpolation; no target-image leakage.

    cloudy: [B,3,H,W] normalized, mask & valid: [B,1,H,W].
    Uses pooled sums/counts of OBSERVED pixels at scales 256 -> ... -> 1;
    bilinear coarse-to-fine interpolation fills gaps while preserving visible pixels.
    """
    observed = (valid > 0.5).to(cloudy.dtype) * (mask <= 0.5).to(cloudy.dtype)
    if torch.any(observed.flatten(1).sum(1) == 0):
        raise ValueError("A patch has no visible valid pixels for local fill")
    numerators = [cloudy * observed]
    denominators = [observed]
    while min(numerators[-1].shape[-2:]) > 1:
        numerators.append(F.avg_pool2d(numerators[-1], 2, stride=2))
        denominators.append(F.avg_pool2d(denominators[-1], 2, stride=2))
    coarse_count = denominators[-1]
    if torch.any(coarse_count == 0):
        raise ValueError("Could not construct a coarse visible-pixel baseline")
    fill = numerators[-1] / coarse_count.clamp_min(1e-7)
    for n, d in zip(reversed(numerators[:-1]), reversed(denominators[:-1])):
        fill = F.interpolate(fill, size=n.shape[-2:], mode="bilinear", align_corners=False)
        local_mean = n / d.clamp_min(1e-7)
        fill = d * local_mean + (1 - d) * fill
    # Exactly retain the visible pixels from the original input.
    fill = torch.where(observed.expand_as(cloudy) > 0.5, cloudy, fill)
    return fill.clamp(0, 1)


class SevenChannelCloudUNet(CloudUNet):
    """Use the exact proven U-Net layers with a 7-channel input check.

    The original CloudUNet.forward intentionally hardcodes four channels;
    this subclass changes only that validation while keeping src/unet_model.py
    unchanged for V1 and V2.
    """

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 7:
            raise ValueError(f"Expected [B,7,H,W], received {tuple(x.shape)}")
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("Height and width must be divisible by 16")
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        z = self.bottleneck(self.pool(s4))
        z = self.dec4(z, s4)
        z = self.dec3(z, s3)
        z = self.dec2(z, s2)
        z = self.dec1(z, s1)
        return torch.sigmoid(self.output(z))


class ResidualCloudUNet(nn.Module):
    def __init__(self, residual_scale=0.25):
        super().__init__()
        self.residual_scale = float(residual_scale)
        # The U-Net now receives the RGB-like NIR/Red/Green input, cloud mask,
        # AND a 3-band locally interpolated baseline: 3 + 1 + 3 = 7 channels.
        self.refiner = SevenChannelCloudUNet(in_channels=7, out_channels=3, base_channels=16)
        # Start exactly at the local-fill baseline. Later learn corrections.
        nn.init.zeros_(self.refiner.output.weight)
        nn.init.zeros_(self.refiner.output.bias)

    def forward(self, x, valid):
        if x.ndim != 4 or x.shape[1] != 4 or valid.shape != x[:, :1].shape:
            raise ValueError("Expected x=[B,4,H,W], valid=[B,1,H,W]")
        cloudy = x[:, :3]
        mask = x[:, 3:4]
        baseline = local_visible_fill(cloudy.detach(), mask.detach(), valid.detach())
        features = torch.cat([cloudy, mask, baseline], dim=1)
        raw = self.refiner(features)  # sigmoid in [0,1]
        residual = (raw - 0.5) * (2 * self.residual_scale)
        pred = (baseline + residual).clamp(0, 1)
        # Never alter originally visible pixels in the final composite.
        composite = torch.where(mask > 0.5, pred, cloudy)
        return composite, baseline


def region_mae(pred, target, region):
    # Counts pixels * spectral channels, not merely pixels.
    return ((pred - target).abs() * region).sum() / (
        region.sum() * pred.shape[1]
    ).clamp_min(1.0)


def interior_gradient_loss(pred, target, mask, valid):
    """Only compare neighbouring valid pixels BOTH inside the cloud mask."""
    region = (mask > 0.5).to(pred.dtype) * (valid > 0.5).to(pred.dtype)
    rx = region[..., 1:] * region[..., :-1]
    ry = region[..., 1:, :] * region[..., :-1, :]
    px = pred[..., 1:] - pred[..., :-1]
    tx = target[..., 1:] - target[..., :-1]
    py = pred[..., 1:, :] - pred[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    num = ((px - tx).abs() * rx).sum() + ((py - ty).abs() * ry).sum()
    denom = (rx.sum() + ry.sum()) * pred.shape[1]
    return num / denom.clamp_min(1.0)


def step_loss(composite, batch, grad_weight):
    region = batch["cloud_mask"] * batch["valid_mask"]
    cloud_mae = region_mae(composite, batch["target"], region)
    grad = interior_gradient_loss(composite, batch["target"],
                                  batch["cloud_mask"], batch["valid_mask"])
    return cloud_mae + grad_weight * grad, cloud_mae


def move(batch, device):
    return {k: batch[k].to(device, non_blocking=device.type == "cuda")
            for k in ("input", "target", "cloud_mask", "valid_mask")}


def one_patch_metrics(composite, baseline, batch, i):
    target = batch["target"][i]
    cloudy = batch["input"][i, :3]
    pred = composite[i]
    local = baseline[i]
    mask = (batch["cloud_mask"][i, 0] > 0.5)
    valid = (batch["valid_mask"][i, 0] > 0.5)
    covered = mask & valid
    seen = (~mask) & valid
    if covered.sum() < 2 or seen.sum() < 2:
        raise ValueError("Validation patch lacks enough valid masked/unmasked pixels")
    visible_avg = cloudy[:, seen].mean(dim=1)
    u = pred[:, covered]
    t = target[:, covered]
    ll = local[:, covered]
    mean_err = (visible_avg[:, None] - t).abs().mean().item()
    local_err = (ll - t).abs().mean().item()
    model_err = (u - t).abs().mean().item()
    cloudy_err = (cloudy[:, covered] - t).abs().mean().item()
    mse = (u - t).square().mean().item()
    st = t.std(dim=1, unbiased=False).mean().item()
    sp = u.std(dim=1, unbiased=False).mean().item()
    ex = covered[:, 1:] & covered[:, :-1]
    ey = covered[1:, :] & covered[:-1, :]

    def grad(v):
        dx = (v[:, :, 1:] - v[:, :, :-1]).abs()[:, ex]
        dy = (v[:, 1:, :] - v[:, :-1, :]).abs()[:, ey]
        return ((dx.sum() + dy.sum()) / max(1, dx.numel() + dy.numel())).item()

    return {
        "model_mae": model_err,
        "mean_mae": mean_err,
        "local_mae": local_err,
        "cloudy_mae": cloudy_err,
        "std_ratio": sp / max(st, 1e-8),
        "gradient_ratio": grad(pred) / max(grad(target), 1e-8),
        "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
        "mae_nir": (u[0] - t[0]).abs().mean().item(),
        "mae_red": (u[1] - t[1]).abs().mean().item(),
        "mae_green": (u[2] - t[2]).abs().mean().item(),
    }


def run_train_epoch(model, loader, optimizer, device, grad_weight, log_every):
    model.train()
    total, count = 0.0, 0
    for j, batch in enumerate(loader, start=1):
        b = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        composite, baseline = model(b["input"], b["valid_mask"])
        loss, cloud_mae = step_loss(composite, b, grad_weight)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite loss at batch {j}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n = len(batch["patch_id"])
        total += cloud_mae.detach().item() * n
        count += n
        if j == 1 or j % log_every == 0 or j == len(loader):
            print(f"  Train batch {j}/{len(loader)}", flush=True)
    return total / count


@torch.inference_mode()
def run_validation(model, loader, device, return_rows=False):
    model.eval()
    detail = []
    for batch in loader:
        b = move(batch, device)
        composite, baseline = model(b["input"], b["valid_mask"])
        for i, patch_id in enumerate(batch["patch_id"]):
            item = one_patch_metrics(composite, baseline, b, i)
            item["patch_id"] = patch_id
            detail.append(item)
    if len(detail) != EXPECTED["val"]:
        raise ValueError(f"Validation should contain 75 patches, got {len(detail)}")
    def avg(key):
        return float(np.mean([r[key] for r in detail]))
    summary = {
        "model_mae": avg("model_mae"),
        "mean_mae": avg("mean_mae"),
        "local_mae": avg("local_mae"),
        "cloudy_mae": avg("cloudy_mae"),
        "wins_mean": sum(r["model_mae"] < r["mean_mae"] for r in detail),
        "wins_local": sum(r["model_mae"] < r["local_mae"] for r in detail),
        "std_ratio": avg("std_ratio"),
        "gradient_ratio": avg("gradient_ratio"),
        "psnr_db": avg("psnr_db"),
    }
    if return_rows:
        return summary, detail
    return summary


def save_checkpoint(path, epoch, model, optimizer, best, args):
    state = {
        "experiment": "residual_unet_v3",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_cloud_mae": float(best),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "grad_weight": float(args.grad_weight),
        "residual_scale": float(args.residual_scale),
        "normalization_divisor": DN_MAX,
        "input_channels": 7,
        "target_channels": 3,
        "dataset": "cloud_reconstruction_v2_synthetic",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def load_checkpoint(path, device, residual_scale):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("experiment") != "residual_unet_v3":
        raise ValueError("This is not a V3 residual checkpoint")
    if checkpoint.get("normalization_divisor") != 1023.0:
        raise ValueError("Checkpoint normalization is not DN/1023")
    if abs(float(checkpoint["residual_scale"]) - float(residual_scale)) > 1e-10:
        raise ValueError("Residual scale mismatch")
    model = ResidualCloudUNet(residual_scale=residual_scale).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def append_history(row):
    existing = HISTORY.is_file()
    with HISTORY.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not existing:
            writer.writeheader()
        writer.writerow(row)


def get_v2_model(device):
    if not V2_BEST.is_file():
        return None, None
    state = torch.load(V2_BEST, map_location="cpu", weights_only=True)
    model = CloudUNet(in_channels=4, out_channels=3, base_channels=16).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model, int(state["epoch"])


@torch.inference_mode()
def make_preview(rows, device, residual_scale):
    model, state = load_checkpoint(BEST, device, residual_scale)
    model.eval()
    v2_model, v2_epoch = get_v2_model(device)
    ds = CloudPairsDataset(rows, "val")
    sorted_rows = sorted(ds.rows, key=lambda r: (float(r["cloud_fraction"]), r["patch_id"]))
    selected = [sorted_rows[i] for i in np.linspace(0, len(sorted_rows)-1, 6).round().astype(int)]
    id_to_index = {r["patch_id"]: i for i, r in enumerate(ds.rows)}
    headers = ["Clear target", "Synthetic cloudy", "Visible mean", "Local baseline",
               "U-Net V2" if v2_model is not None else "V2 unavailable", "Residual U-Net V3", "Cloud mask"]
    tile, gap, row_step, top = 170, 12, 204, 80
    canvas = Image.new("RGB", (gap + len(headers) * (tile + gap), top + row_step*len(selected)+gap), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), "Mumbai satellite reconstruction | FIXED VAL only | consistent false-colour scaling", fill="black")
    draw.text((gap, 26), f"V3 best epoch: {state['epoch']}; V2 best epoch: {v2_epoch if v2_epoch is not None else 'not found'}", fill="black")
    for col, name in enumerate(headers):
        draw.text((gap + col*(tile+gap), 53), name, fill="black")
    for rownum, record in enumerate(selected):
        item = ds[id_to_index[record["patch_id"]]]
        x = item["input"].unsqueeze(0).to(device)
        valid = item["valid_mask"].unsqueeze(0).to(device)
        comp, base = model(x, valid)
        mask_np = (item["cloud_mask"][0].numpy() > .5)
        valid_np = (item["valid_mask"][0].numpy() > .5)
        clear_dn = item["target"].numpy().transpose(1, 2, 0) * DN_MAX
        cloudy_dn = item["input"][:3].numpy().transpose(1, 2, 0) * DN_MAX
        visible = valid_np & (~mask_np)
        mean_dn = cloudy_dn[visible].mean(axis=0)
        mean_comp = np.where(mask_np[..., None], mean_dn, cloudy_dn)
        baseline_comp = np.where(mask_np[..., None],
                                 base[0].cpu().numpy().transpose(1, 2, 0)*DN_MAX, cloudy_dn)
        pred3 = comp[0].cpu().numpy().transpose(1, 2, 0)*DN_MAX
        if v2_model is not None:
            pred2 = v2_model(x)[0].cpu().numpy().transpose(1, 2, 0)*DN_MAX
            comp2 = np.where(mask_np[..., None], pred2, cloudy_dn)
        else:
            comp2 = cloudy_dn
        panels = [
            false_color(clear_dn, valid_np), false_color(cloudy_dn, valid_np),
            false_color(mean_comp, valid_np), false_color(baseline_comp, valid_np),
            false_color(comp2, valid_np), false_color(pred3, valid_np),
            np.repeat((mask_np.astype(np.uint8)*255)[..., None], 3, axis=2),
        ]
        y = top + rownum*row_step
        for col, panel in enumerate(panels):
            frame = Image.fromarray(panel).resize((tile, tile), Image.Resampling.BILINEAR)
            canvas.paste(frame, (gap + col*(tile+gap), y))
        draw.text((gap, y+tile+5),
                  f"{record['patch_id']} | cloud={float(record['cloud_fraction']):.1%}", fill="black")
        cloud_valid = mask_np & valid_np
        mae3 = np.abs(pred3-clear_dn)[cloud_valid].mean()/DN_MAX
        draw.text((gap+5*(tile+gap), y+tile+5), f"V3 MAE={mae3:.4f}", fill="black")
    canvas.save(PREVIEW)
    return PREVIEW


def run_self_test():
    torch.manual_seed(12)
    target = torch.rand(1, 3, 64, 64)*0.3
    m = torch.zeros(1, 1, 64, 64)
    m[..., 14:52, 12:56] = 1
    v = torch.ones_like(m)
    cloudy = target*(1-m) + 0.9*m
    x = torch.cat((cloudy, m), dim=1)
    model = ResidualCloudUNet(residual_scale=0.25)
    comp, baseline = model(x, v)
    assert comp.shape == target.shape == (1, 3, 64, 64)
    assert torch.equal(comp*(1-m), cloudy*(1-m))
    assert torch.allclose(comp*m, baseline*m, atol=1e-6), "Initial output must equal local fill"
    altered_cloud = cloudy*(1-m) + 0.1*m
    baseline2 = local_visible_fill(altered_cloud, m, v)
    assert torch.allclose(baseline, baseline2, atol=1e-6), "Baseline leaked cloudy pixels"
    batch = {"target": target, "cloud_mask": m, "valid_mask": v}
    loss, mae = step_loss(comp, batch, 0.10)
    loss.backward()
    assert torch.isfinite(loss).item() and torch.isfinite(mae).item()
    assert any(p.grad is not None and p.grad.abs().sum().item()>0 for p in model.parameters())
    print(f"V3 SELF-TEST PASSED | shape={tuple(comp.shape)} | masked MAE={mae.item():.6f}", flush=True)
    print("Baseline uses only visible input; visible composite pixels remain unchanged.")


def main():
    a = args_parser()
    if a.self_test:
        run_self_test()
        return
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_manifest()
    train_ds = DynamicTrainDataset(rows, a.seed)
    val_ds = CloudPairsDataset(rows, "val")
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    model = ResidualCloudUNet(residual_scale=a.residual_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    start, best_val = 1, float("inf")
    if a.resume:
        if not LATEST.exists() or not BEST.exists() or not HISTORY.exists():
            raise FileNotFoundError("V3 latest/best checkpoint or history is missing")
        state = torch.load(LATEST, map_location="cpu", weights_only=True)
        if state.get("experiment") != "residual_unet_v3":
            raise ValueError("Cannot resume from a different experiment")
        for key, wanted in (("seed",a.seed),("batch_size",a.batch_size),("lr",a.lr),
                            ("grad_weight",a.grad_weight),("residual_scale",a.residual_scale)):
            if state.get(key) != wanted:
                raise ValueError(f"Resume parameter {key} changed; keep prior settings")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start = int(state["epoch"]) + 1
        best_val = float(state["best_val_cloud_mae"])
        with HISTORY.open("r", encoding="utf-8", newline="") as f:
            history = list(csv.DictReader(f))
        if not history or int(history[-1]["epoch"]) != start - 1:
            raise ValueError("Checkpoint epoch disagrees with history; fix before resuming")
    elif any(p.exists() for p in (LATEST, BEST, HISTORY, CONFIG)):
        raise FileExistsError("V3 outputs already exist. Use --resume instead of overwriting.")
    if start > a.epochs:
        raise ValueError("--epochs must exceed the number of completed epochs")
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG.exists():
        with CONFIG.open("w", encoding="utf-8") as f:
            json.dump({"experiment":"residual_unet_v3", "model":"CloudUNet input 7, output 3, base 16",
                       "baseline":"local multiscale visible-only interpolation", "training":"dynamic cloud resynthesis",
                       "validation":"fixed 75 patches", "test":"reserved - never loaded",
                       "loss":"cloud L1 + grad_weight * within-cloud gradient L1",
                       "epochs_requested": a.epochs, "batch_size":a.batch_size, "lr":a.lr,
                       "seed":a.seed, "grad_weight":a.grad_weight, "residual_scale":a.residual_scale,
                       "normalization": "DN / 1023"}, f, indent=2)
    print("MUMBAI SATELLITE RECONSTRUCTION — RESIDUAL U-NET V3", flush=True)
    print(f"Device: {device} | {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}", flush=True)
    print(f"TRAIN: {len(train_ds)} dynamic clouds/epoch | VAL: {len(val_ds)} fixed", flush=True)
    print("TEST: reserved; not loaded. V1 and V2 outputs remain untouched.", flush=True)
    print("Baseline: visible-only local multiscale interpolation; correction ±"+str(a.residual_scale), flush=True)
    print(f"Loss: masked L1 + {a.grad_weight} * within-cloud gradient L1", flush=True)
    print(f"Epochs {start}..{a.epochs} | batch size {a.batch_size} | learning rate {a.lr}", flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    for epoch in range(start, a.epochs + 1):
        train_ds.epoch = epoch
        # Seed shuffling independently per epoch to make resumed runs reproducible.
        generator = torch.Generator().manual_seed(a.seed + 1009*epoch)
        train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                                  generator=generator, num_workers=0,
                                  pin_memory=device.type == "cuda")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        print(f"\nEpoch {epoch}/{a.epochs}", flush=True)
        train_mae = run_train_epoch(model, train_loader, optimizer, device,
                                    a.grad_weight, a.log_every)
        val = run_validation(model, val_loader, device)
        improved = val["model_mae"] < best_val
        if improved:
            best_val = val["model_mae"]
            save_checkpoint(BEST, epoch, model, optimizer, best_val, a)
        save_checkpoint(LATEST, epoch, model, optimizer, best_val, a)
        seconds = time.perf_counter()-started
        mem = torch.cuda.max_memory_allocated()/1048576 if device.type=="cuda" else 0
        append_history({
            "epoch":epoch, "train_cloud_mae":f"{train_mae:.8f}",
            "val_v3_cloud_mae":f"{val['model_mae']:.8f}",
            "val_visible_mean_mae":f"{val['mean_mae']:.8f}",
            "val_local_fill_mae":f"{val['local_mae']:.8f}",
            "val_cloudy_mae":f"{val['cloudy_mae']:.8f}",
            "val_v3_beats_mean_patches":val["wins_mean"],
            "val_v3_beats_local_patches":val["wins_local"],
            "val_texture_std_ratio":f"{val['std_ratio']:.5f}",
            "val_gradient_ratio":f"{val['gradient_ratio']:.5f}",
            "val_cloud_psnr_db":f"{val['psnr_db']:.3f}",
            "epoch_seconds":f"{seconds:.2f}",
            "peak_cuda_mib":f"{mem:.1f}",
            "best_val_cloud_mae":f"{best_val:.8f}",
        })
        print(f"  Train cloud MAE: {train_mae:.5f} | Val V3 cloud MAE: {val['model_mae']:.5f}", flush=True)
        print(f"  Visible-mean: {val['mean_mae']:.5f} | Local baseline: {val['local_mae']:.5f}", flush=True)
        print(f"  V3 beats mean: {val['wins_mean']}/75 | beats local: {val['wins_local']}/75", flush=True)
        print(f"  Texture std ratio: {val['std_ratio']:.3f} | gradient ratio: {val['gradient_ratio']:.3f}", flush=True)
        print(f"  Val PSNR: {val['psnr_db']:.2f} dB | time: {seconds:.1f}s | CUDA peak: {mem:.1f} MiB", flush=True)
        print(f"  Best model saved: {'YES' if improved else 'no'}", flush=True)
    best_model, state = load_checkpoint(BEST, device, a.residual_scale)
    best_model.eval()
    _, detailed = run_validation(best_model, val_loader, device, return_rows=True)
    with VAL_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patch_id"] +
                 [k for k in detailed[0] if k != "patch_id"])
        writer.writeheader()
        writer.writerows(detailed)
    picture = make_preview(rows, device, a.residual_scale)
    print("\nRESIDUAL U-NET V3 COMPLETE", flush=True)
    print(f"Best validation cloud MAE: {best_val:.6f} (epoch {state['epoch']})", flush=True)
    print(f"Best checkpoint: {BEST}", flush=True)
    print(f"Latest checkpoint: {LATEST}", flush=True)
    print(f"History: {HISTORY}", flush=True)
    print(f"Best-checkpoint validation CSV: {VAL_CSV}", flush=True)
    print(f"Comparison preview: {picture}", flush=True)
    print("Synthetic validation only; low MAE or apparent texture is not real-cloud recovery.")


if __name__ == "__main__":
    main()
