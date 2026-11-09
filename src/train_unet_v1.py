r"""First supervised training run for the Mumbai LISS-IV synthetic cloud dataset.

Requires the already verified scripts alongside this file:
  src/unet_model.py
  src/test_pytorch_dataloader.py

Reads the 499 approved TRAIN pairs and monitors 75 VAL pairs. It never loads
TEST examples during fitting. No pandas, torchvision, or matplotlib required.

Suggested first run (3 total epochs):
  .\.venv\Scripts\python.exe -u .\src\train_unet_v1.py --epochs 3

Later continue to 12 total epochs, without discarding optimizer state:
  .\.venv\Scripts\python.exe -u .\src\train_unet_v1.py --epochs 12 --resume

Outputs:
  outputs/unet_training_v1/checkpoints/latest.pt
  outputs/unet_training_v1/checkpoints/best.pt
  outputs/unet_training_v1/training_history.csv
  outputs/unet_training_v1/training_config.json
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
from torch.utils.data import DataLoader

from test_pytorch_dataloader import CloudPairsDataset, MANIFEST, EXPECTED, ROOT
from unet_model import CloudUNet

OUTPUT_DIR = ROOT / "outputs" / "unet_training_v1"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
HISTORY_CSV = OUTPUT_DIR / "training_history.csv"
CONFIG_JSON = OUTPUT_DIR / "training_config.json"
LATEST_PT = CHECKPOINT_DIR / "latest.pt"
BEST_PT = CHECKPOINT_DIR / "best.pt"
CLEAR_WEIGHT = 0.10

HISTORY_FIELDS = [
    "epoch", "train_cloud_mae", "train_clear_mae",
    "val_cloud_mae", "val_clear_mae", "val_cloudy_baseline_mae",
    "val_cloud_mae_dn", "val_cloud_psnr_db", "val_cloudy_baseline_psnr_db",
    "epoch_seconds", "peak_cuda_allocated_mib", "best_val_cloud_mae",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3,
                        help="Target TOTAL epochs, including any resumed epochs (default: 3).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest.pt and continue to --epochs total.")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.log_every < 1:
        parser.error("epochs, batch-size, lr and log-every must be positive")
    return args


def read_rows():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Synthetic-pair manifest missing: {MANIFEST}")
    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    counts = dict(Counter(r["split"].strip() for r in rows))
    ids = [r["patch_id"] for r in rows]
    if len(rows) != sum(EXPECTED.values()) or counts != EXPECTED:
        raise ValueError(f"Unexpected dataset split counts: {counts} (expected {EXPECTED})")
    if len(ids) != len(set(ids)) or "C0137" in ids:
        raise ValueError("Manifest contains duplicate IDs or excluded C0137")
    return rows


def make_loader(rows, split, batch_size, use_cuda):
    ds = CloudPairsDataset(rows, split)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,   # Stable on Windows and avoids pandas dependencies
        pin_memory=use_cuda,
        drop_last=False,
    )
    return loader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    transfer = {key: batch[key].to(device, non_blocking=(device.type == "cuda"))
                for key in ("input", "target", "cloud_mask", "valid_mask")}
    return transfer


def get_region_errors(pred, target, input_img, cloud_mask, valid_mask):
    # Masks are [B,1,H,W] and broadcast across exactly three spectral bands.
    cloud_region = cloud_mask * valid_mask
    clear_region = (1.0 - cloud_mask) * valid_mask
    count_cloud = cloud_region.sum() * 3.0
    count_clear = clear_region.sum() * 3.0
    if count_cloud.item() <= 0 or count_clear.item() <= 0:
        raise ValueError("A batch has zero valid cloudy or clear pixels")

    errors = (pred - target).abs()
    sq_errors = (pred - target).square()
    baseline_errors = (input_img[:, :3] - target).abs()
    baseline_sq_errors = (input_img[:, :3] - target).square()
    return {
        "cloud_abs": (errors * cloud_region).sum(),
        "cloud_sq": (sq_errors * cloud_region).sum(),
        "clear_abs": (errors * clear_region).sum(),
        "baseline_cloud_abs": (baseline_errors * cloud_region).sum(),
        "baseline_cloud_sq": (baseline_sq_errors * cloud_region).sum(),
        "cloud_count": count_cloud,
        "clear_count": count_clear,
    }


def blank_totals():
    return {name: 0.0 for name in (
        "cloud_abs", "cloud_sq", "clear_abs", "baseline_cloud_abs",
        "baseline_cloud_sq", "cloud_count", "clear_count"
    )}


def accumulate(totals, errors):
    for key in totals:
        totals[key] += float(errors[key].detach().item())


def finish_metrics(totals):
    n_cloud = totals["cloud_count"]
    n_clear = totals["clear_count"]
    if n_cloud <= 0 or n_clear <= 0:
        raise ValueError("No valid pixels available for epoch metrics")
    cloud_mse = totals["cloud_sq"] / n_cloud
    baseline_mse = totals["baseline_cloud_sq"] / n_cloud
    # Images are normalized to [0,1], hence the PSNR peak signal is 1.0.
    return {
        "cloud_mae": totals["cloud_abs"] / n_cloud,
        "clear_mae": totals["clear_abs"] / n_clear,
        "baseline_cloud_mae": totals["baseline_cloud_abs"] / n_cloud,
        "cloud_psnr_db": -10.0 * math.log10(max(cloud_mse, 1e-12)),
        "baseline_cloud_psnr_db": -10.0 * math.log10(max(baseline_mse, 1e-12)),
    }


def run_epoch(model, loader, device, optimizer, log_every):
    is_train = optimizer is not None
    model.train(is_train)
    totals = blank_totals()
    torch.set_grad_enabled(is_train)
    try:
        for step, batch in enumerate(loader, 1):
            data = move_batch(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            pred = model(data["input"])
            errors = get_region_errors(
                pred, data["target"], data["input"],
                data["cloud_mask"], data["valid_mask"]
            )
            if is_train:
                # Optimize within the cloud and gently preserve the noncloudy area.
                loss = (errors["cloud_abs"] / errors["cloud_count"]
                        + CLEAR_WEIGHT * errors["clear_abs"] / errors["clear_count"])
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(f"Non-finite loss at batch {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            accumulate(totals, errors)
            if is_train and (step == 1 or step % log_every == 0 or step == len(loader)):
                print(f"  Train batch {step}/{len(loader)}", flush=True)
    finally:
        torch.set_grad_enabled(True)
    return finish_metrics(totals)


def save_checkpoint(path, epoch, model, optimizer, best_val, args):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_cloud_mae": best_val,
        "base_channels": 16,
        "in_channels": 4,
        "out_channels": 3,
        "normalization_divisor": 1023.0,
        "clear_weight": CLEAR_WEIGHT,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dataset_name": "cloud_reconstruction_v2_synthetic",
        "selection_metric": "validation cloud-region MAE",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def save_history(record, resume):
    existing = HISTORY_CSV.is_file()
    if resume and not existing:
        raise FileNotFoundError("Missing training history for resumed experiment")
    if existing:
        with HISTORY_CSV.open("r", newline="", encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        if history and int(history[-1]["epoch"]) >= int(record["epoch"]):
            raise ValueError("History already has this epoch; refusing to duplicate it")
    with HISTORY_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not existing:
            writer.writeheader()
        writer.writerow(record)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    rows = read_rows()

    train_loader = make_loader(rows, "train", args.batch_size, use_cuda)
    val_loader = make_loader(rows, "val", args.batch_size, use_cuda)

    model = CloudUNet(base_channels=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch = 1
    best_val = float("inf")
    if args.resume:
        if not LATEST_PT.is_file():
            raise FileNotFoundError(f"Cannot resume; missing {LATEST_PT}")
        saved = torch.load(LATEST_PT, map_location=device, weights_only=True)
        if (saved["base_channels"] != 16 or saved["batch_size"] != args.batch_size
                or saved["lr"] != args.lr or saved["seed"] != args.seed):
            raise ValueError("Resume settings differ from checkpoint (batch size, lr or seed)")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best_val = float(saved["best_val_cloud_mae"])
        if not HISTORY_CSV.is_file():
            raise FileNotFoundError("Cannot resume: training_history.csv is missing")
        with HISTORY_CSV.open("r", newline="", encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        if not history or int(history[-1]["epoch"]) != start_epoch - 1:
            raise ValueError("History and latest checkpoint refer to different epochs")
    else:
        if LATEST_PT.exists() or BEST_PT.exists() or HISTORY_CSV.exists():
            raise FileExistsError(
                f"Training outputs already exist under {OUTPUT_DIR}. "
                "Use --resume to continue; nothing has been overwritten."
            )

    if start_epoch > args.epochs:
        raise ValueError(f"Already finished {start_epoch-1} epochs; --epochs must be larger")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_JSON.exists():
        with CONFIG_JSON.open("w", encoding="utf-8") as f:
            json.dump({
                "dataset_manifest": MANIFEST.relative_to(ROOT).as_posix(),
                "train_patches": len(train_loader.dataset),
                "validation_patches": len(val_loader.dataset),
                "test_patches_reserved": EXPECTED["test"],
                "base_channels": 16,
                "input_channels": 4,
                "output_channels": 3,
                "dn_divisor": 1023,
                "cloud_loss_weight": 1.0,
                "clear_loss_weight": CLEAR_WEIGHT,
                "optimizer": "AdamW",
                "weight_decay": 1e-4,
                "initial_lr": args.lr,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "fixed_synthetic_cloud_per_source_patch": True,
                "selection_metric": "validation cloud-region MAE",
            }, f, indent=2)

    print("MUMBAI SATELLITE CLOUD RECONSTRUCTION — U-NET TRAINING V1", flush=True)
    print(f"Device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if use_cuda else ""), flush=True)
    print(f"Training: {len(train_loader.dataset)} | Validation: {len(val_loader.dataset)}", flush=True)
    print("Test split: 75 patches RESERVED (not loaded)", flush=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Total target epochs: {args.epochs} | batch size: {args.batch_size} | LR: {args.lr}", flush=True)
    print("Loss: cloud masked L1 + 0.10 * noncloud valid L1", flush=True)
    print("Precision: float32 (first stable baseline); no pandas required", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train = run_epoch(model, train_loader, device, optimizer, args.log_every)
        val = run_epoch(model, val_loader, device, None, args.log_every)
        elapsed = time.perf_counter() - started
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2) if use_cuda else 0.0

        improved = val["cloud_mae"] < best_val
        if improved:
            best_val = val["cloud_mae"]
            save_checkpoint(BEST_PT, epoch, model, optimizer, best_val, args)
        save_checkpoint(LATEST_PT, epoch, model, optimizer, best_val, args)

        record = {
            "epoch": epoch,
            "train_cloud_mae": f"{train['cloud_mae']:.8f}",
            "train_clear_mae": f"{train['clear_mae']:.8f}",
            "val_cloud_mae": f"{val['cloud_mae']:.8f}",
            "val_clear_mae": f"{val['clear_mae']:.8f}",
            "val_cloudy_baseline_mae": f"{val['baseline_cloud_mae']:.8f}",
            "val_cloud_mae_dn": f"{val['cloud_mae'] * 1023:.4f}",
            "val_cloud_psnr_db": f"{val['cloud_psnr_db']:.4f}",
            "val_cloudy_baseline_psnr_db": f"{val['baseline_cloud_psnr_db']:.4f}",
            "epoch_seconds": f"{elapsed:.1f}",
            "peak_cuda_allocated_mib": f"{peak_mib:.1f}",
            "best_val_cloud_mae": f"{best_val:.8f}",
        }
        save_history(record, resume=args.resume)
        print(
            f"  Train cloud MAE: {train['cloud_mae']:.6f} | "
            f"Validation cloud MAE: {val['cloud_mae']:.6f} | "
            f"Cloudy-input baseline: {val['baseline_cloud_mae']:.6f}",
            flush=True,
        )
        print(
            f"  Val cloud PSNR: {val['cloud_psnr_db']:.2f} dB | "
            f"Val baseline PSNR: {val['baseline_cloud_psnr_db']:.2f} dB | "
            f"Val clear MAE: {val['clear_mae']:.6f}",
            flush=True,
        )
        print(
            f"  Time: {elapsed:.1f}s | Peak allocated CUDA: {peak_mib:.1f} MiB | "
            f"Best saved: {'YES' if improved else 'no'}",
            flush=True,
        )

    print("\nTRAINING RUN COMPLETE", flush=True)
    print(f"Best validation cloud MAE: {best_val:.6f}", flush=True)
    print(f"Best model: {BEST_PT}", flush=True)
    print(f"Latest checkpoint: {LATEST_PT}", flush=True)
    print(f"Training history: {HISTORY_CSV}", flush=True)
    print("Test split has NOT been evaluated. Real-cloud performance remains unverified.", flush=True)


if __name__ == "__main__":
    main()
