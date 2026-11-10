# Reproducibility and local environment

## Platform

Initial prototype: Windows PowerShell, VS Code, Python 3.11 virtual environment, NVIDIA RTX 4050 (6GB), PyTorch with CUDA, NumPy, Pillow, Rasterio, Matplotlib and SciPy. A pandas native extension was blocked by Windows Smart App Control; core current scripts should not require pandas. Some legacy exploratory scripts may need additional dependencies.

## Install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# Install a compatible PyTorch build for CPU/CUDA according to pytorch.org.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The original environment reported `torch 2.11.0+cu128`; the new `requirements.txt` lists needed packages but is **not** a pinned lockfile. For exact reproduction, record a tested version list plus hardware and CUDA versions.

## Local data paths

- `data/raw/mumbai_scenes/2025-04-16/` with source `BAND4.tif`, `BAND3.tif`, `BAND2.tif`.
- `data/raw/mumbai_scenes/2025-06-03/` with corresponding bands.
- `data/processed/cloud_reconstruction_v1/metadata/approved_target_manifest.csv` and associated target arrays.
- `data/processed/cloud_reconstruction_v2_synthetic/` with approved pairs and dataset manifests.
- `outputs/` with model checkpoints and reports, generated locally.

The repository does not include data or checkpoints. Some scripts require a specific QC manifest, exact file naming or the historical V3 model checkpoint. Avoid presenting a fresh clone as one-command fully reproducible without these prerequisites.

## Verified self-tests to run first

```powershell
.\.venv\Scripts\python.exe -u .\src\evaluate_v3_locked_test.py --self-test
.\.venv\Scripts\python.exe -u .\src\evaluate_partial_temporal_reference_v1.py --self-test
.\.venv\Scripts\python.exe -u .\src\preview_real_june_unet_v3.py --self-test
.\.venv\Scripts\python.exe -u .\src\compare_real_june_masks_v3_no_gui.py --self-test
.\.venv\Scripts\python.exe -u .\src\export_experimental_real_june_v3_geotiff_v3.py --check-version
.\.venv\Scripts\python.exe -u .\src\export_experimental_real_june_v3_geotiff_v3.py --self-test
```

The real-data run requires both source imagery and the frozen V3 checkpoint:

```powershell
.\.venv\Scripts\python.exe -u .\src\export_experimental_real_june_v3_geotiff_v3.py
```

Expected experiments: T041 and T058, four GeoTIFFs each, plus a preview, CSV and provenance JSON under `outputs/real_june_v3_experimental_geotiff/`. This is **unvalidated cloud removal**, even when the exporter verifies exact shape, data/mask values and spatial placement.

## Integrity

Do not change or tune based on the held-out 75 synthetic test examples. Preserve original satellite bands and source metadata. Record the exact model checkpoint SHA-256 and generator/version if publishing results. Do not distribute satellite raster products unless permitted under their licensing terms.

## Windows Git workflow

Run `git status`, inspect `git diff --cached --name-status` and verify `.gitignore` before any push. Never use `git push --force` in this cleanup. The optional ZIP helper provided with the handover stages only code/docs, not data and outputs.
