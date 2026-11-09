"""Audit the saved Mumbai synthetic-cloud pairs before model training.

No pandas, no rasterio and no regeneration. Requires numpy only.
Checks 649 approved target pairs, their source data, masks and CSV metadata.
Run: .\\.venv\\Scripts\\python.exe -u .\\src\\audit_synthetic_dataset_v2.py
"""

from collections import Counter, defaultdict
from pathlib import Path
import csv

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "cloud_reconstruction_v2_synthetic"
MANIFEST = DATASET / "metadata" / "synthetic_pairs_manifest_v2.csv"
APPROVED = (ROOT / "data" / "processed" / "cloud_reconstruction_v1"
            / "metadata" / "approved_target_manifest.csv")
EXPECTED_SPLITS = {"train": 499, "val": 75, "test": 75}
FIELDS = ("patch_id", "split", "input_path", "target_path", "alpha_path",
          "mask_path", "shadow_path", "valid_mask_path", "source_patch_path",
          "cloud_fraction", "valid_fraction")


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    return rows, fields


def array_from(record, column):
    path = Path(record[column])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"{column}: {path}")
    return np.load(path, allow_pickle=False)


def main():
    rows, columns = read_rows(MANIFEST)
    approved_rows, approved_columns = read_rows(APPROVED)
    missing = set(FIELDS) - columns
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if not {"patch_id", "split", "manual_status"}.issubset(approved_columns):
        raise ValueError("Approved-source manifest has unexpected columns")

    approved_ids = {
        (r["patch_id"].strip(), r["split"].strip())
        for r in approved_rows if r["manual_status"].strip() == "approved"
    }
    record_keys = [(r["patch_id"].strip(), r["split"].strip()) for r in rows]
    ids = [r["patch_id"].strip() for r in rows]
    counts = Counter(r["split"].strip() for r in rows)
    print("SYNTHETIC DATASET V2 — INTEGRITY AUDIT", flush=True)
    print(f"Manifest entries: {len(rows)}; split counts: {dict(counts)}", flush=True)

    errors = []
    if len(rows) != 649:
        errors.append(f"Expected 649 rows, got {len(rows)}")
    if counts != Counter(EXPECTED_SPLITS):
        errors.append(f"Unexpected split counts: {dict(counts)}")
    if len(set(record_keys)) != len(rows) or len(set(ids)) != len(rows):
        errors.append("Duplicate patch IDs or patch ID/split pairs")
    if set(record_keys) != approved_ids:
        errors.append("Synthetic manifest differs from approved-target manifest")
    if "C0137" in ids:
        errors.append("Excluded uncertain patch C0137 is present")
    if errors:
        print("AUDIT FAILED:")
        for err in errors:
            print(" -", err)
        raise SystemExit(1)

    coverage = defaultdict(list)
    foreground = defaultdict(list)
    shadow_nonzero = 0
    invalid_pixel_counts = 0

    for index, record in enumerate(rows, start=1):
        patch_id, split = record_keys[index - 1]
        try:
            target = array_from(record, "target_path")
            source = array_from(record, "source_patch_path")
            cloudy = array_from(record, "input_path")
            alpha = array_from(record, "alpha_path")
            mask = array_from(record, "mask_path")
            shadow = array_from(record, "shadow_path")
            validity = array_from(record, "valid_mask_path")

            valid = validity.astype(bool)
            if any(image.shape != (256, 256, 3) for image in (target, source, cloudy)):
                raise ValueError("Invalid target/source/cloudy shape")
            if any(image.shape != (256, 256) for image in (alpha, mask, shadow, valid)):
                raise ValueError("Invalid alpha/binary/shadow/valid shape")
            if target.dtype != np.uint16 or source.dtype != np.uint16:
                raise ValueError(f"Expected uint16 target/source, got {target.dtype}/{source.dtype}")
            if cloudy.dtype != np.float32 or alpha.dtype != np.float32:
                raise ValueError(f"Unexpected cloudy/alpha dtype: {cloudy.dtype}/{alpha.dtype}")
            if not np.array_equal(target, source):
                raise ValueError("Saved target does not match the original extracted patch")
            if not all(np.isfinite(x).all() for x in (cloudy, alpha, shadow)):
                raise ValueError("NaN or infinity detected")
            if target.max() > 1023 or target.min() < 0:
                raise ValueError("Original target outside 10-bit DN range")
            if cloudy.min() < 0 or cloudy.max() > 1023:
                raise ValueError("Synthetic input outside 10-bit DN range")
            if alpha.min() < 0 or alpha.max() > 1 or shadow.min() < 0 or shadow.max() > 1:
                raise ValueError("Alpha or shadow opacity out of range")
            if not valid.any() or valid.mean() < 0.995:
                raise ValueError("Invalid validity mask / too few valid pixels")
            if not np.all((mask == 0) | (mask == 1)):
                raise ValueError("Cloud mask contains values other than 0/1")
            expected_mask = (alpha >= 0.30) & valid
            if not np.array_equal(mask.astype(bool), expected_mask):
                raise ValueError("Binary mask does not match alpha >= 0.30 on valid pixels")
            if np.any(alpha[~valid] != 0) or np.any(shadow[~valid] != 0):
                raise ValueError("Opacity or shadow present on invalid pixels")
            if not np.array_equal(cloudy[~valid], target[~valid].astype(np.float32)):
                raise ValueError("Synthetic image changed invalid pixels")
            unaffected = valid & (alpha == 0) & (shadow == 0)
            if not np.allclose(cloudy[unaffected], target[unaffected], atol=1e-4, rtol=0):
                raise ValueError("Unclouded, unshadowed valid pixels were modified")

            actual = float(mask.astype(bool).sum() / valid.sum())
            if abs(actual - float(record["cloud_fraction"])) > 1e-5:
                raise ValueError("Reported coverage disagrees with actual mask")
            if abs(float(valid.mean()) - float(record["valid_fraction"])) > 1e-5:
                raise ValueError("Reported validity fraction disagrees with mask")
            if actual < 0.01:
                raise ValueError("Cloud coverage suspiciously low (<1%)")
            if not np.any(np.abs(cloudy - target.astype(np.float32)) > 1e-4):
                raise ValueError("Cloudy and target arrays are identical")

            coverage[split].append(actual)
            foreground[split].append(float((alpha > 0.01).sum() / valid.sum()))
            shadow_nonzero += int(np.any(shadow > 0))
            invalid_pixel_counts += int(np.count_nonzero(~valid))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{patch_id} [{split}]: {exc}")
        if index == 1 or index % 100 == 0 or index == len(rows):
            print(f"Checked {index}/{len(rows)} pairs", flush=True)

    print("\nQUALITY SUMMARY")
    for split in ("train", "val", "test"):
        vals = np.asarray(coverage[split], dtype=np.float64)
        if len(vals):
            print(f"{split:<5}: {len(vals):3d} valid pairs | "
                  f"mask coverage min={vals.min():.1%}, mean={vals.mean():.1%}, "
                  f"max={vals.max():.1%}")
    print(f"Pairs with at least some synthetic shadow: {shadow_nonzero}/{len(rows)}")
    print(f"Total invalid source pixels (preserved): {invalid_pixel_counts}")

    if errors:
        print(f"\nAUDIT FAILED: {len(errors)} issue(s). First 15:")
        for error in errors[:15]:
            print(" -", error)
        raise SystemExit(1)

    print("\nAUDIT PASSED: all 649 pairs and masks are consistent with approved targets.")
    print("This verifies file integrity, not atmospheric realism or model accuracy.")


if __name__ == "__main__":
    main()
