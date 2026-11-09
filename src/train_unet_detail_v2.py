r"""Train an experimental detail-aware U-Net on the Mumbai synthetic cloud dataset.

V2 is a SEPARATE experiment from U-Net V1. Its goal is to test whether
on-the-fly synthetic clouds and a masked spatial-gradient loss improve
reconstruction beyond a simple visible-mean fill. It may not.

Requirements: numpy, Pillow, torch, and these existing files in src/:
  unet_model.py, test_pytorch_dataloader.py, synthetic_clouds_v2.py

Run from project root:
  .\.venv\Scripts\python.exe -u .\src\train_unet_detail_v2.py --epochs 3 --batch-size 2

For further training of V2 ONLY:
  .\.venv\Scripts\python.exe -u .\src\train_unet_detail_v2.py --epochs 6 --batch-size 2 --resume

No pandas, no torchvision. This NEVER loads or changes the TEST split.
All validation cloud simulations remain fixed. Training inputs are newly
synthesized from approved TRAIN targets, deterministically by epoch and ID.
Uncalibrated DN simulations are not a substitute for real-cloud evaluation.
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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from synthetic_clouds_v2 import synthesize, false_color
from test_pytorch_dataloader import CloudPairsDataset, EXPECTED, MANIFEST, ROOT
from unet_model import CloudUNet

OUT = ROOT / "outputs" / "unet_training_detail_v2"
CHECKPOINTS = OUT / "checkpoints"
LATEST = CHECKPOINTS / "latest.pt"
BEST = CHECKPOINTS / "best.pt"
HISTORY = OUT / "history.csv"
CONFIG = OUT / "config.json"
PREVIEW = OUT / "validation_comparison_v1_vs_v2.png"
V1_BEST = ROOT / "outputs" / "unet_training_v1" / "checkpoints" / "best.pt"
DIVISOR = 1023.0
CLEAR_WEIGHT = 0.10
GRADIENT_WEIGHT = 1.0
FIELDS = (
    "epoch", "train_cloud_mae", "val_cloud_mae", "val_mean_fill_mae",
    "val_cloudy_mae", "val_model_beats_mean_patches", "val_texture_std_ratio",
    "val_gradient_ratio", "val_cloud_psnr_db", "epoch_seconds",
    "peak_cuda_mib", "best_val_cloud_mae",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3, help="TOTAL target epochs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.log_every) < 1 or args.lr <= 0:
        parser.error("epochs, batch size, learning rate, and log interval must be positive")
    return args


def load_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    counts = dict(Counter(row["split"].strip() for row in rows))
    ids = [r["patch_id"] for r in rows]
    if counts != EXPECTED or len(rows) != 649:
        raise ValueError(f"Incorrect split counts: {counts}")
    if len(ids) != len(set(ids)) or "C0137" in ids:
        raise ValueError("Duplicate patch IDs or excluded C0137 found")
    return rows


def source_array(row, field):
    path = Path(row[field])
    if not path.is_absolute():
        path = ROOT / path
    return np.load(path, allow_pickle=False)


class DynamicTrainDataset(Dataset):
    """New cloud realization per TRAIN patch per epoch; fixed val is untouched."""

    def __init__(self, rows, seed):
        self.rows = [r for r in rows if r["split"].strip() == "train"]
        if len(self.rows) != EXPECTED["train"]:
            raise ValueError("Incorrect training split size")
        self.seed = seed
        self.epoch = 1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        patch_id = row["patch_id"]
        clear = source_array(row, "target_path")
        valid = source_array(row, "valid_mask_path").astype(bool)
        if clear.shape != (256, 256, 3) or valid.shape != (256, 256):
            raise ValueError(f"Invalid data shapes for {patch_id}")
        if clear.min() < 0 or clear.max() > 1023 or not valid.any():
            raise ValueError(f"Invalid DN range or validity for {patch_id}")
        # Distinct, reproducible cloud fields for every patch and training epoch.
        seed = self.seed + 1000003 * self.epoch + 1009 * int(patch_id[1:])
        rng = np.random.default_rng(seed)
        cloudy, alpha, cloud, shadow, frac = synthesize(clear, valid, rng)
        del alpha, shadow, frac
        # Paired flip and 90-degree rotation only; target stays pixel-aligned.
        rotation = int(rng.integers(4))
        flip = rng.random() < 0.5
        if rotation:
            cloudy = np.rot90(cloudy, rotation)
            clear = np.rot90(clear, rotation)
            cloud = np.rot90(cloud, rotation)
            valid = np.rot90(valid, rotation)
        if flip:
            cloudy = np.flip(cloudy, axis=1)
            clear = np.flip(clear, axis=1)
            cloud = np.flip(cloud, axis=1)
            valid = np.flip(valid, axis=1)
        model_input = np.concatenate(
            [cloudy.transpose(2, 0, 1) / DIVISOR, cloud.astype(np.float32)[None]],
            axis=0,
        ).astype(np.float32)
        target = (clear.transpose(2, 0, 1).astype(np.float32) / DIVISOR)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(model_input)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "cloud_mask": torch.from_numpy(np.ascontiguousarray(cloud.astype(np.float32)[None])),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid.astype(np.float32)[None])),
            "patch_id": patch_id,
        }


def move(batch, device):
    return {
        k: batch[k].to(device, non_blocking=device.type == "cuda")
        for k in ("input", "target", "cloud_mask", "valid_mask")
    }


def region_mean(err, region):
    # err: [B,3,H,W], region: [B,1,H,W]
    return (err * region).sum() / (region.sum() * err.shape[1]).clamp_min(1)


def spatial_gradient_error(pred, target, cloud, valid):
    """Signed x/y gradient L1 on valid edges intersecting synthetic clouds.

    This encourages details but cannot recover truly occluded physical truth.
    """
    dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) - (
        target[:, :, :, 1:] - target[:, :, :, :-1]
    )
    dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]) - (
        target[:, :, 1:, :] - target[:, :, :-1, :]
    )
    ex = ((cloud[:, :, :, 1:] + cloud[:, :, :, :-1]) > 0).to(pred.dtype)
    ey = ((cloud[:, :, 1:, :] + cloud[:, :, :-1, :]) > 0).to(pred.dtype)
    ex = ex * valid[:, :, :, 1:] * valid[:, :, :, :-1]
    ey = ey * valid[:, :, 1:, :] * valid[:, :, :-1, :]
    numerator = (dx.abs() * ex).sum() + (dy.abs() * ey).sum()
    denominator = (ex.sum() + ey.sum()) * pred.shape[1]
    return numerator / denominator.clamp_min(1)


def objective(pred, batch):
    cloud = batch["cloud_mask"] * batch["valid_mask"]
    clear = (1 - batch["cloud_mask"]) * batch["valid_mask"]
    target = batch["target"]
    error = (pred - target).abs()
    cloud_l1 = region_mean(error, cloud)
    clear_l1 = region_mean(error, clear)
    grad = spatial_gradient_error(pred, target, batch["cloud_mask"], batch["valid_mask"])
    total = cloud_l1 + CLEAR_WEIGHT * clear_l1 + GRADIENT_WEIGHT * grad
    return total, cloud_l1


def validation_per_patch(pred, batch):
    """Return per-example normalized metrics, using VAL only."""
    rows = []
    for k in range(pred.shape[0]):
        p = pred[k].detach()
        t = batch["target"][k]
        x = batch["input"][k, :3]
        cloud = (batch["cloud_mask"][k, 0] > .5) & (batch["valid_mask"][k, 0] > .5)
        visible = (batch["cloud_mask"][k, 0] < .5) & (batch["valid_mask"][k, 0] > .5)
        if cloud.sum() < 2 or visible.sum() < 2:
            raise ValueError("No valid cloudy or visible pixels in validation patch")
        mean_visible = x[:, visible].mean(dim=1)
        mean_fill_error = (mean_visible[:, None] - t[:, cloud]).abs().mean().item()
        model_error = (p[:, cloud] - t[:, cloud]).abs().mean().item()
        cloudy_error = (x[:, cloud] - t[:, cloud]).abs().mean().item()
        model_mse = (p[:, cloud] - t[:, cloud]).square().mean().item()
        # Per-patch average channel standard deviations.
        ps = p[:, cloud].std(dim=1, unbiased=False).mean().item()
        ts = t[:, cloud].std(dim=1, unbiased=False).mean().item()
        ex = cloud[:, 1:] & cloud[:, :-1]
        ey = cloud[1:, :] & cloud[:-1, :]
        def internal_grad(a):
            dx = (a[:, :, 1:] - a[:, :, :-1]).abs()[:, ex]
            dy = (a[:, 1:, :] - a[:, :-1, :]).abs()[:, ey]
            count = dx.numel() + dy.numel()
            return ((dx.sum() + dy.sum()) / max(count, 1)).item()
        gp = internal_grad(p)
        gt = internal_grad(t)
        rows.append({
            "model_mae": model_error,
            "mean_mae": mean_fill_error,
            "cloudy_mae": cloudy_error,
            "mse": model_mse,
            "std_ratio": ps / max(ts, 1e-8),
            "gradient_ratio": gp / max(gt, 1e-8),
        })
    return rows


def train_epoch(model, loader, optimizer, device, log_every):
    model.train()
    numerator, denominator = 0.0, 0
    for step, batch in enumerate(loader, 1):
        b = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(b["input"])
        loss, cloud_mae = objective(pred, b)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite loss at training batch {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n = len(batch["patch_id"])
        numerator += float(cloud_mae.detach().item()) * n
        denominator += n
        if step == 1 or step % log_every == 0 or step == len(loader):
            print(f"  Train batch {step}/{len(loader)}", flush=True)
    return numerator / denominator


def validate(model, loader, device):
    model.eval()
    metrics = []
    with torch.inference_mode():
        for batch in loader:
            b = move(batch, device)
            pred = model(b["input"])
            metrics.extend(validation_per_patch(pred, b))
    return {
        "model_mae": np.mean([r["model_mae"] for r in metrics]),
        "mean_mae": np.mean([r["mean_mae"] for r in metrics]),
        "cloudy_mae": np.mean([r["cloudy_mae"] for r in metrics]),
        "wins": sum(r["model_mae"] < r["mean_mae"] for r in metrics),
        "std_ratio": np.mean([r["std_ratio"] for r in metrics]),
        "gradient_ratio": np.mean([r["gradient_ratio"] for r in metrics]),
        "psnr": np.mean([-10. * math.log10(max(r["mse"], 1e-12)) for r in metrics]),
        "n": len(metrics),
    }


def save_checkpoint(path, epoch, model, optimizer, best_val, args):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_cloud_mae": float(best_val),
        "in_channels": 4, "out_channels": 3, "base_channels": 16,
        "normalization_divisor": DIVISOR,
        "loss": "cloud L1 + 0.1 clear L1 + 1.0 masked gradient L1",
        "dynamic_train_clouds": True,
        "fixed_validation_clouds": True,
        "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pt.tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def write_history(row):
    exists = HISTORY.is_file()
    with HISTORY.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_best_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("in_channels") != 4 or saved.get("out_channels") != 3:
        raise ValueError(f"Invalid checkpoint architecture in {path}")
    if float(saved.get("normalization_divisor", 0)) != 1023.:
        raise ValueError(f"Incorrect DN normalization in {path}")
    model = CloudUNet(base_channels=int(saved["base_channels"]))
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return model.to(device).eval(), int(saved["epoch"])


def render_preview(rows, device):
    """Simple fixed-validation comparison of original/cloudy/mean/V1/V2/mask."""
    val_rows = sorted((r for r in rows if r["split"].strip() == "val"),
                      key=lambda r: (float(r["cloud_fraction"]), r["patch_id"]))
    selected_rows = [val_rows[int(round(i))] for i in np.linspace(0, 74, 6)]
    dataset = CloudPairsDataset(rows, "val")
    index = {r["patch_id"]: i for i, r in enumerate(dataset.rows)}
    v2, v2_epoch = read_best_model(BEST, device)
    v1 = None
    v1_epoch = None
    if V1_BEST.exists():
        v1, v1_epoch = read_best_model(V1_BEST, device)
    headers = ["Clear target", "Synthetic cloudy", "Visible mean", "U-Net V1 composite", "U-Net V2 composite", "Cloud mask"]
    tile, gap, row_h, top = 180, 10, 215, 75
    canvas = Image.new("RGB", (gap + 6 * (tile + gap), top + row_h * len(selected_rows) + gap), "white")
    drawer = ImageDraw.Draw(canvas)
    drawer.text((gap, 6), "Mumbai cloud reconstruction | FIXED VALIDATION | same display mapping", fill="black")
    drawer.text((gap, 24), f"V1 best: epoch {v1_epoch if v1 is not None else 'missing'}; V2 best: epoch {v2_epoch}", fill="black")
    for j, head in enumerate(headers):
        drawer.text((gap + j * (tile + gap), 50), head, fill="black")
    with torch.inference_mode():
        for n, row in enumerate(selected_rows):
            item = dataset[index[row["patch_id"]]]
            x = item["input"].unsqueeze(0).to(device)
            clear = item["target"].numpy().transpose(1, 2, 0) * DIVISOR
            cloudy = item["input"][:3].numpy().transpose(1, 2, 0) * DIVISOR
            mask = item["cloud_mask"][0].numpy() > 0.5
            valid = item["valid_mask"][0].numpy() > 0.5
            seen = valid & ~mask
            mean_dn = cloudy[seen].mean(axis=0)
            mean_fill = np.where(mask[..., None], mean_dn, cloudy)
            pred2 = v2(x)[0].cpu().numpy().transpose(1, 2, 0) * DIVISOR
            comp2 = np.where(mask[..., None], pred2, cloudy)
            if v1 is not None:
                pred1 = v1(x)[0].cpu().numpy().transpose(1, 2, 0) * DIVISOR
                comp1 = np.where(mask[..., None], pred1, cloudy)
                v1_mae = np.abs(pred1 - clear)[mask & valid].mean() / DIVISOR
            else:
                comp1 = cloudy
                v1_mae = float("nan")
            v2_mae = np.abs(pred2 - clear)[mask & valid].mean() / DIVISOR
            panels = [
                false_color(clear, valid), false_color(cloudy, valid),
                false_color(mean_fill, valid), false_color(comp1, valid),
                false_color(comp2, valid),
                np.repeat(mask[..., None].astype(np.uint8) * 255, 3, axis=2),
            ]
            yy = top + n * row_h
            for col, panel in enumerate(panels):
                frame = Image.fromarray(panel).resize((tile, tile), Image.Resampling.BILINEAR)
                canvas.paste(frame, (gap + col * (tile + gap), yy))
            drawer.text((gap, yy + tile + 5),
                        f"{row['patch_id']} cloud={float(row['cloud_fraction']):.0%}", fill="black")
            drawer.text((gap + 3 * (tile + gap), yy + tile + 5), f"V1 MAE={v1_mae:.4f}", fill="black")
            drawer.text((gap + 4 * (tile + gap), yy + tile + 5), f"V2 MAE={v2_mae:.4f}", fill="black")
    canvas.save(PREVIEW)
    return PREVIEW


def main():
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_manifest()
    train_ds = DynamicTrainDataset(rows, args.seed)
    val_ds = CloudPairsDataset(rows, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    model = CloudUNet(base_channels=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start, best = 1, float("inf")
    if args.resume:
        if not LATEST.exists() or not HISTORY.exists():
            raise FileNotFoundError("V2 latest checkpoint or training history missing")
        state = torch.load(LATEST, map_location="cpu", weights_only=True)
        if (state["seed"] != args.seed or state["batch_size"] != args.batch_size
                or state["lr"] != args.lr):
            raise ValueError("Resume requires same seed, batch size and learning rate")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start, best = int(state["epoch"]) + 1, float(state["best_val_cloud_mae"])
        with HISTORY.open(newline="", encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        if not history or int(history[-1]["epoch"]) != start - 1:
            raise ValueError("V2 history and latest checkpoint do not agree")
    elif any(p.exists() for p in (LATEST, BEST, HISTORY)):
        raise FileExistsError("V2 outputs already exist; use --resume or a different experiment")
    if start > args.epochs:
        raise ValueError("--epochs must exceed number of epochs already completed")
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if not CONFIG.exists():
        with CONFIG.open("w", encoding="utf-8") as f:
            json.dump({
                "model": "CloudUNet base_channels=16", "train": 499, "val": 75,
                "test_reserved": 75, "dynamic_train_synthesis": True,
                "validation_fixed_synthesis": True,
                "cloud_L1_weight": 1.0, "clear_L1_weight": CLEAR_WEIGHT,
                "masked_gradient_L1_weight": GRADIENT_WEIGHT,
                "dn_divisor": DIVISOR, "learning_rate": args.lr,
                "batch_size": args.batch_size, "seed": args.seed,
            }, f, indent=2)
    print("MUMBAI CLOUD RECONSTRUCTION — DETAIL-AWARE U-NET V2", flush=True)
    print(f"Device: {device}; GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}", flush=True)
    print(f"TRAIN: {len(train_ds)} dynamically resynthesized / epoch | VAL: {len(val_ds)} fixed", flush=True)
    print("TEST: reserved, never accessed. V1 checkpoint untouched.", flush=True)
    print("Loss: cloud L1 + 0.10 clear L1 + 1.0 masked spatial-gradient L1", flush=True)
    print(f"Epochs: {start}..{args.epochs}, batch size: {args.batch_size}", flush=True)

    for epoch in range(start, args.epochs + 1):
        train_ds.epoch = epoch
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_mae = train_epoch(model, train_loader, optimizer, device, args.log_every)
        v = validate(model, val_loader, device)
        if v["n"] != EXPECTED["val"]:
            raise ValueError("Validation count mismatch")
        is_best = float(v["model_mae"]) < best
        if is_best:
            best = float(v["model_mae"])
            save_checkpoint(BEST, epoch, model, optimizer, best, args)
        save_checkpoint(LATEST, epoch, model, optimizer, best, args)
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1048576 if device.type == "cuda" else 0.0
        write_history({
            "epoch": epoch, "train_cloud_mae": f"{train_mae:.8f}",
            "val_cloud_mae": f"{v['model_mae']:.8f}",
            "val_mean_fill_mae": f"{v['mean_mae']:.8f}",
            "val_cloudy_mae": f"{v['cloudy_mae']:.8f}",
            "val_model_beats_mean_patches": v["wins"],
            "val_texture_std_ratio": f"{v['std_ratio']:.6f}",
            "val_gradient_ratio": f"{v['gradient_ratio']:.6f}",
            "val_cloud_psnr_db": f"{v['psnr']:.4f}",
            "epoch_seconds": f"{elapsed:.2f}", "peak_cuda_mib": f"{peak:.1f}",
            "best_val_cloud_mae": f"{best:.8f}",
        })
        print(f"  Train cloud MAE: {train_mae:.5f} | Validation cloud MAE: {v['model_mae']:.5f}", flush=True)
        print(f"  Mean-fill baseline MAE: {v['mean_mae']:.5f} | V2 beats mean on {v['wins']}/75", flush=True)
        print(f"  Texture std ratio: {v['std_ratio']:.3f} | gradient ratio: {v['gradient_ratio']:.3f}", flush=True)
        print(f"  Cloud PSNR: {v['psnr']:.2f} dB | time: {elapsed:.1f}s | peak CUDA: {peak:.1f} MiB", flush=True)
        print(f"  Best checkpoint saved: {'YES' if is_best else 'no'}", flush=True)
    img = render_preview(rows, device)
    print("\nU-NET DETAIL V2 TRAINING COMPLETE", flush=True)
    print(f"Best V2 validation cloud MAE: {best:.5f}", flush=True)
    print(f"Best V2 checkpoint: {BEST}", flush=True)
    print(f"History: {HISTORY}", flush=True)
    print(f"V1-vs-V2 preview: {img}", flush=True)
    print("Validation only; no test or real-cloud conclusions. More texture does not guarantee accuracy.", flush=True)


if __name__ == "__main__":
    main()
