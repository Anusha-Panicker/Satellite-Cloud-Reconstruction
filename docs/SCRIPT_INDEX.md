# Script index and safe source organization

All Python scripts currently stay in **`src/`**. This is deliberate: several scripts import sibling modules by bare name (e.g. `from train_unet_residual_v3 import ...`). Moving them into nested directories without adjusting imports and testing would break the working pipeline. Categorization here documents the code safely.

## Satellite imagery and data preparation

- `preview_mumbai_scenes.py`, `inspect_sample.py`, `preview_sample.py`, `inspect_radiometry.py` — exploration.
- `crop_apr_jun_overlap.py`, `prepare_overlap_masks.py`, `cloud_mask_preview.py`, `inspect_aligned_patches.py` — overlap and mask inspection.
- `select_clear_patches.py`, `extract_clear_patch_candidates.py`, `review_clear_patches.py`, `extract_approved_patches.py` — preliminary patch selection and review.
- `build_large_patch_catalog.py`, `audit_patch_quality.py`, `audit_patch_distribution.py`, `review_flagged_qc_patches.py`, `generate_full_qc_sheets.py`, `finalize_qc_manifest.py`, `extract_final_patches.py`, `rebalance_patch_splits.py`, `finalize_extracted_dataset.py`, `check_spatial_split_separation.py` — expanded dataset QC and finalization.

## Synthetic generation and training

- `synthetic_clouds.py` — original simulator, historical baseline.
- `synthetic_clouds_v2.py` — updated simulator.
- `generate_synthetic_cloud_pairs_v2.py` and `audit_synthetic_dataset_v2.py` — finalized synthetic pairs and audit.
- `unet_model.py` — shared U-Net architecture.
- `train_unet_v1.py`, `train_unet_detail_v2.py`, `train_unet_residual_v3.py` — three recorded model experiments.
- `test_pytorch_dataloader.py`, `test_unet_forward.py` — foundational smoke checks.

## Synthetic evaluation

- `diagnose_unet_validation_v1.py`, `preview_unet_validation_v1.py` — V1 diagnostics.
- `evaluate_v3_cloud_boundaries.py`, `evaluate_v3_structure.py` — boundary and structure checks.
- `evaluate_v3_locked_test.py` — frozen V3 synthetic test; do not tune against test.

## Multi-temporal feasibility (no approved June reconstruction)

- `audit_april_june_temporal_reference.py`, `diagnose_april_june_radiometry.py`, `review_april_june_cloud_and_alignment.py`, `verify_april_june_landmark_registration.py`.
- `auto_register_april_june_v1.py`, `verify_april_june_green_band.py`, `audit_april_june_registration_bias.py`, `audit_within_scene_band_geometry.py`, `verify_june_band_offsets_on_independent_tiles.py`.
- `preview_real_temporal_transfer_unregistered.py` — illustrative unregistered transfer, NOT validated.
- `evaluate_partial_temporal_reference_v1.py` — T041/T058 feasibility gate; NO reconstruction approved.

## June real-image experiments (June-only proxy, no April transfer)

- `preview_real_june_unet_v3.py` — qualitative inference.
- `compare_real_june_masks_v3_no_gui.py` — automatic four-mask comparison without editor window.
- **`export_experimental_real_june_v3_geotiff_v3.py`** — latest verified experimental GeoTIFF exporter; intentionally retains the v3 suffix because two earlier exporter implementations had CRS-verification failures.

## Why some experimental scripts are retained

Registration and real-cloud failures are scientifically meaningful, so their code should be preserved and explicitly labelled. Do not discard or rename scripts solely for aesthetic reasons. Legacy versions should be archived only after dependency and reproducibility testing.

## Missing files in an ordinary clone

The raw imagery, synthetic pairs, source manifests, checkpoints and large figures are local-only (`data/`, `outputs/`). See [Reproducibility](REPRODUCIBILITY.md).
