"""One-batch U-Net forward/backward smoke test on the audited Mumbai dataset.

Put this file, unet_model.py, and test_pytorch_dataloader.py together in src/.
Reads two TRAIN pairs; runs one forward and backward pass; does not train,
modify source data, save checkpoints, or touch validation/test examples.
"""

import csv

import torch
from torch.utils.data import DataLoader

from test_pytorch_dataloader import CloudPairsDataset, MANIFEST
from unet_model import CloudUNet


BATCH_SIZE = 2


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cloud_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute error across all 3 bands of valid synthetic cloud pixels."""
    region = cloud_mask * valid_mask  # [B, 1, H, W], broadcasts to 3 bands
    count = region.sum() * prediction.shape[1]
    if count.item() == 0:
        raise ValueError("This batch contains no valid cloud-covered pixels")
    return ((prediction - target).abs() * region).sum() / count


def main() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest missing: {MANIFEST}")

    with MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("U-NET FORWARD + BACKWARD SMOKE TEST", flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        torch.cuda.reset_peak_memory_stats()

    ds = CloudPairsDataset(rows, "train")
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    batch = next(iter(loader))

    x = batch["input"].to(device)
    target = batch["target"].to(device)
    cloud_mask = batch["cloud_mask"].to(device)
    valid_mask = batch["valid_mask"].to(device)

    model = CloudUNet(base_channels=16).to(device)
    model.train()
    param_count = sum(p.numel() for p in model.parameters())

    prediction = model(x)
    if prediction.shape != target.shape:
        raise AssertionError(
            f"Prediction shape {tuple(prediction.shape)} != target {tuple(target.shape)}"
        )
    if not bool(torch.isfinite(prediction).all().item()):
        raise AssertionError("Non-finite U-Net predictions")
    if not bool(((prediction >= 0) & (prediction <= 1)).all().item()):
        raise AssertionError("Prediction outside normalized [0,1] range")

    loss = masked_l1(prediction, target, cloud_mask, valid_mask)
    cloudy_baseline = masked_l1(x[:, :3], target, cloud_mask, valid_mask)
    if not bool(torch.isfinite(loss).item()):
        raise AssertionError("Non-finite masked L1 loss")

    loss.backward()
    grad_sum = sum(
        float(p.grad.detach().abs().sum().item())
        for p in model.parameters() if p.grad is not None
    )
    if grad_sum <= 0 or not (grad_sum < float("inf")):
        raise AssertionError("Gradients are missing or invalid")

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Patch IDs: {list(batch['patch_id'])}", flush=True)
    print(f"Trainable parameters: {param_count:,}", flush=True)
    print(f"Input shape: {tuple(x.shape)}", flush=True)
    print(f"Output shape: {tuple(prediction.shape)}", flush=True)
    print(f"Valid cloud pixels: {(cloud_mask * valid_mask).sum().item():.0f}", flush=True)
    print(f"Initial masked L1: {loss.item():.6f}", flush=True)
    print(f"Cloudy input masked L1: {cloudy_baseline.item():.6f}", flush=True)
    print(f"Gradient check: passed (sum abs grads={grad_sum:.4f})", flush=True)
    if device.type == "cuda":
        memory_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak allocated CUDA memory: {memory_mib:.1f} MiB", flush=True)
    print("\nU-NET SMOKE TEST PASSED (no training performed).", flush=True)


if __name__ == "__main__":
    main()
