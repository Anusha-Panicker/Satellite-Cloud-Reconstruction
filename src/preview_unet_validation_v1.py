"""Visual quality check of the best U-Net on VALIDATION data only.

Save alongside unet_model.py and test_pytorch_dataloader.py in src/.
Run from the project root:
  .\\.venv\\Scripts\\python.exe -u .\\src\\preview_unet_validation_v1.py

Outputs a PNG comparison sheet and CSV of the selected validation examples.
Never opens the held-out test split, never trains, never changes input arrays.
Uses stdlib CSV, NumPy, Pillow and PyTorch; pandas is not required.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from test_pytorch_dataloader import CloudPairsDataset, MANIFEST, ROOT
from synthetic_clouds_v2 import false_color
from unet_model import CloudUNet

CHECKPOINT = ROOT / "outputs" / "unet_training_v1" / "checkpoints" / "best.pt"
OUTPUT = ROOT / "outputs" / "unet_training_v1" / "validation_preview"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8, help="Validation examples (1..75)")
    args = parser.parse_args()
    if not 1 <= args.samples <= 75:
        parser.error("--samples must be between 1 and 75")
    return args


def load_checkpoint(device):
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {CHECKPOINT}")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if saved.get("in_channels") != 4 or saved.get("out_channels") != 3:
        raise ValueError("Checkpoint input/output channels do not match U-Net V1")
    if float(saved.get("normalization_divisor", 0)) != 1023.0:
        raise ValueError("Checkpoint normalization differs from expected DN/1023")
    model = CloudUNet(base_channels=int(saved["base_channels"]))
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return model.to(device).eval(), int(saved["epoch"])


def sample_selection(rows, number):
    """Coverage-stratified and deterministic so low/high cloud examples appear."""
    by_coverage = sorted(rows, key=lambda r: (float(r["cloud_fraction"]), r["patch_id"]))
    if number == 1:
        return [by_coverage[len(by_coverage) // 2]]
    positions = np.linspace(0, len(by_coverage) - 1, num=number)
    return [by_coverage[int(round(p))] for p in positions]


def model_infer(model, item, device):
    with torch.inference_mode():
        predicted = model(item["input"].unsqueeze(0).to(device))[0].cpu().numpy()
    # All channels below have HWC order [NIR, Red, Green].
    input_dn = item["input"][:3].numpy().transpose(1, 2, 0) * 1023.0
    clear_dn = item["target"].numpy().transpose(1, 2, 0) * 1023.0
    predicted_dn = np.clip(predicted.transpose(1, 2, 0) * 1023.0, 0.0, 1023.0)
    mask = item["cloud_mask"][0].numpy() > 0.5
    valid = item["valid_mask"][0].numpy() > 0.5
    composite_dn = np.where(mask[..., None], predicted_dn, input_dn)
    composite_dn[~valid] = input_dn[~valid]
    if not np.any(mask & valid) or not np.any((~mask) & valid):
        raise ValueError(f"Missing cloud/clear valid region in {item['patch_id']}")
    return input_dn, clear_dn, predicted_dn, composite_dn, mask, valid


def region_mae(pred, target, region):
    return float(np.abs(pred - target)[region].mean() / 1023.0)


def draw_sheet(records):
    ncol, tile, gap, row_h, header = 5, 210, 12, 252, 82
    width = gap + ncol * (tile + gap)
    height = header + len(records) * row_h + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((gap, 8), "Mumbai cloud reconstruction | Best U-Net V1 | VALIDATION ONLY", fill="black", font=font)
    draw.text((gap, 26), "Same false-colour tone mapping for each satellite image panel", fill="black", font=font)
    headings = ["Clear target", "Simulated cloudy", "Raw U-Net output", "Cloud-only composite", "Binary cloud mask"]
    for col, heading in enumerate(headings):
        draw.text((gap + col * (tile + gap), 56), heading, fill="black", font=font)
    for index, item in enumerate(records):
        top = header + index * row_h
        x_data, clear, pred, composite, mask, valid = item["arrays"]
        panels = [
            false_color(clear, valid),
            false_color(x_data, valid),
            false_color(pred, valid),
            false_color(composite, valid),
            np.repeat((mask[..., None].astype(np.uint8) * 255), 3, axis=2),
        ]
        for col, panel in enumerate(panels):
            x = gap + col * (tile + gap)
            frame = Image.fromarray(panel).resize((tile, tile), Image.Resampling.BILINEAR)
            sheet.paste(frame, (x, top))
        caption1 = f"{item['patch_id']} | cloud={item['cloud_fraction']:.1%}"
        caption2 = f"Cloud MAE: model={item['model_cloud_mae']:.4f}  cloudy={item['baseline_cloud_mae']:.4f}"
        draw.text((gap, top + tile + 4), caption1, fill="black", font=font)
        draw.text((gap, top + tile + 18), caption2, fill="black", font=font)
    return sheet


def main():
    args = parse_args()
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    val_rows = [r for r in rows if r["split"].strip() == "val"]
    if len(val_rows) != 75:
        raise ValueError(f"Expected 75 validation samples, found {len(val_rows)}")
    if len({r["patch_id"] for r in val_rows}) != 75:
        raise ValueError("Duplicate validation patch IDs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_checkpoint(device)
    dataset = CloudPairsDataset(rows, "val")
    val_idx = {r["patch_id"]: i for i, r in enumerate(dataset.rows)}
    selected = sample_selection(val_rows, args.samples)
    print("U-NET V1 | VALIDATION RECONSTRUCTION PREVIEW", flush=True)
    print(f"Device: {device} | best checkpoint epoch: {epoch}", flush=True)
    print(f"Selected {len(selected)} of 75 validation patches (test split not loaded)", flush=True)

    evaluated = []
    for r in selected:
        patch_id = r["patch_id"]
        item = dataset[val_idx[patch_id]]
        arrays = model_infer(model, item, device)
        cloudy, clear, raw, composite, mask, valid = arrays
        cloud_region = mask & valid
        clear_region = (~mask) & valid
        record = {
            "patch_id": patch_id,
            "split": "val",
            "cloud_fraction": float(r["cloud_fraction"]),
            "model_cloud_mae": region_mae(raw, clear, cloud_region),
            "baseline_cloud_mae": region_mae(cloudy, clear, cloud_region),
            "raw_prediction_clear_mae": region_mae(raw, clear, clear_region),
            "composite_clear_mae": region_mae(composite, clear, clear_region),
            "arrays": arrays,
        }
        evaluated.append(record)
        print(
            f"  {patch_id}: cloud={record['cloud_fraction']:.1%} | "
            f"U-Net masked MAE={record['model_cloud_mae']:.5f} | "
            f"cloudy baseline={record['baseline_cloud_mae']:.5f} | "
            f"raw clear MAE={record['raw_prediction_clear_mae']:.5f}",
            flush=True,
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure = OUTPUT / "validation_reconstruction_comparison.png"
    metrics = OUTPUT / "selected_validation_metrics.csv"
    draw_sheet(evaluated).save(figure)
    fields = (
        "patch_id", "split", "cloud_fraction", "model_cloud_mae",
        "baseline_cloud_mae", "raw_prediction_clear_mae", "composite_clear_mae"
    )
    with metrics.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in evaluated:
            writer.writerow({name: row[name] for name in fields})

    print("\nVALIDATION PREVIEW COMPLETE", flush=True)
    print(f"Comparison image: {figure}", flush=True)
    print(f"Sample metrics: {metrics}", flush=True)
    print("The composite preserves the original cloudy input OUTSIDE the binary cloud mask.", flush=True)
    print("Uncorrected soft cloud edges/shadows outside the mask may remain.", flush=True)
    print("These are synthetic validation results, not real-cloud evidence.", flush=True)


if __name__ == "__main__":
    main()
