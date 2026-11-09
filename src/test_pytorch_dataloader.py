"""Smoke-test Mumbai LISS-IV synthetic cloud pairs with PyTorch.

Reads files lazily from the audited V2 synthetic dataset. Requires torch and numpy;
no pandas, torchvision, or model training needed.
"""

from collections import Counter
from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "data" / "processed" / "cloud_reconstruction_v2_synthetic"
    / "metadata" / "synthetic_pairs_manifest_v2.csv"
)
EXPECTED = {"train": 499, "val": 75, "test": 75}
BATCH_SIZE = 2
DN_MAX = 1023.0


class CloudPairsDataset(Dataset):
    def __init__(self, rows, split):
        self.rows = [r for r in rows if r["split"].strip() == split]
        self.split = split
        if len(self.rows) != EXPECTED[split]:
            raise ValueError(
                f"{split}: expected {EXPECTED[split]} records, "
                f"found {len(self.rows)}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        record = self.rows[index]
        patch_id = record["patch_id"]

        def load(field):
            path = Path(record[field])
            if not path.is_absolute():
                path = ROOT / path
            return np.load(path, allow_pickle=False)

        cloudy = load("input_path").astype(np.float32)
        clear = load("target_path").astype(np.float32)
        cloud_mask = load("mask_path").astype(np.float32)
        valid_mask = load("valid_mask_path").astype(np.float32)

        if cloudy.shape != (256, 256, 3) or clear.shape != (256, 256, 3):
            raise ValueError(f"Invalid image shape: {patch_id}")
        if cloud_mask.shape != (256, 256) or valid_mask.shape != (256, 256):
            raise ValueError(f"Invalid mask shape: {patch_id}")
        if not (np.isfinite(cloudy).all() and np.isfinite(clear).all()):
            raise ValueError(f"Non-finite image values: {patch_id}")
        if cloudy.min() < 0 or cloudy.max() > DN_MAX or clear.min() < 0 or clear.max() > DN_MAX:
            raise ValueError(f"DN values outside 0..1023: {patch_id}")

        cloudy = cloudy / DN_MAX
        clear = clear / DN_MAX
        cloud_mask = np.clip(cloud_mask, 0.0, 1.0)
        valid_mask = np.clip(valid_mask, 0.0, 1.0)

        # Channels: [NIR, Red, Green, binary cloud mask].
        model_input = np.concatenate(
            [cloudy.transpose(2, 0, 1), cloud_mask[None, :, :]], axis=0
        )
        target = clear.transpose(2, 0, 1)

        return {
            "input": torch.from_numpy(np.ascontiguousarray(model_input)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "cloud_mask": torch.from_numpy(np.ascontiguousarray(cloud_mask[None])),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask[None])),
            "patch_id": patch_id,
        }


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    with MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 649:
        raise ValueError(f"Expected 649 manifest records, found {len(rows)}")
    if len({r["patch_id"] for r in rows}) != 649:
        raise ValueError("Duplicate patch IDs in manifest")
    if any(r["patch_id"] == "C0137" for r in rows):
        raise ValueError("Uncertain patch C0137 must be excluded")

    counts = dict(Counter(r["split"].strip() for r in rows))
    if counts != EXPECTED:
        raise ValueError(f"Unexpected split counts: {counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    for split in ("train", "val", "test"):
        dataset = CloudPairsDataset(rows, split)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,  # Most reliable starting point on Windows
            pin_memory=(device.type == "cuda"),
        )

        batch = next(iter(loader))
        x = batch["input"].to(device, non_blocking=(device.type == "cuda"))
        y = batch["target"].to(device, non_blocking=(device.type == "cuda"))
        valid = batch["valid_mask"].to(device, non_blocking=(device.type == "cuda"))
        mask = batch["cloud_mask"].to(device, non_blocking=(device.type == "cuda"))

        assert x.shape == (BATCH_SIZE, 4, 256, 256)
        assert y.shape == (BATCH_SIZE, 3, 256, 256)
        assert valid.shape == mask.shape == (BATCH_SIZE, 1, 256, 256)
        assert torch.isfinite(x).all() and torch.isfinite(y).all()

        if device.type == "cuda":
            torch.cuda.synchronize()

        print(f"\n{split.upper()}: {len(dataset)} patches", flush=True)
        print(f"  IDs: {list(batch['patch_id'])}", flush=True)
        print(f"  Input: {tuple(x.shape)} | Target: {tuple(y.shape)}", flush=True)
        print(f"  Cloud mask: {tuple(mask.shape)} | Valid mask: {tuple(valid.shape)}", flush=True)
        print(f"  Normalized range: input [{x.min().item():.3f}, {x.max().item():.3f}]", flush=True)
        print(f"  Batch device: {x.device}", flush=True)

    print("\nDATALOADER TEST PASSED: all three splits loaded successfully.", flush=True)


if __name__ == "__main__":
    main()
