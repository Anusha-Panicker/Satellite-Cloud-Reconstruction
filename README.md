# Satellite Cloud Removal & Reconstruction

**A Mumbai-focused research prototype for reconstructing cloud-obscured multispectral satellite imagery using deep learning and multi-temporal observations.**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Framework](https://img.shields.io/badge/PyTorch-CUDA-orange) ![Study%20area](https://img.shields.io/badge/Study%20area-Mumbai-green) ![Status](https://img.shields.io/badge/Status-Research%20prototype-yellow)

> **Project status (November 2026):** The single-image synthetic-cloud pipeline and its held-out evaluation are complete. April–June multi-temporal registration and real-cloud reconstruction are **in progress**. Reported benchmark scores are on *synthetically clouded* satellite patches, **not** ground-truth real-cloud reconstructions.

## Overview

Optical satellite scenes often have clouds that obscure information about roads, vegetation, water and other surface features. This project studies two approaches:

1. **Single-image reconstruction:** A lightweight U-Net predicts missing spectral values from a synthetically clouded image and its cloud mask.
2. **Multi-temporal reconstruction (ongoing):** Aligned observations from different dates may supply actual earlier surface information where a later image is cloudy.

**Important limitation:** If thick cloud completely blocks a surface in one optical image, a single-image neural network can only *estimate* the hidden values. Plausible-looking output should not be mistaken for verified recovery of ground truth.

## Data and study area

| Item | Description |
| --- | --- |
| Study area | Mumbai region, Maharashtra, India |
| Satellite / sensor | Resourcesat-2A / LISS-IV |
| Acquisitions examined | 27 February, 16 April, 3 June 2025 |
| Primary source for paired synthetic experiments | 16 April 2025 |
| Spectral order in model arrays | Band 4 (NIR), Band 3 (Red), Band 2 (Green) |
| Pixel spacing | 5 m |
| Reference CRS | WGS 84 / UTM zone 43N (EPSG:32643) |
| Patch size | 256 × 256 × 3 (clear target) |
| Input to single-image model | 3 spectral channels + 1 binary cloud-mask channel |
| Nominal sensor encoding | 10-bit digital numbers stored in `uint16`; normalization by 1023 |

### Dataset quality control

After patch extraction and manual visual review, **649 patches were approved** for exploratory paired training, and one uncertain patch was excluded.

| Split | Patches |
| --- | ---: |
| Training | 499 |
| Validation | 75 |
| Held-out test | 75 |
| **Total** | **649** |

The pairs use synthetic cloud opacity, texture, and shadows. Dataset integrity was audited, and the held-out test split was reserved until final model selection. The visual QC does **not** certify every original patch as physically cloud-free.

## Methodology

```text
Mumbai LISS-IV GeoTIFF scenes
            |
            v
Geospatial + radiometric inspection
            |
            v
Patch extraction + spatial split + manual QC
            |
            v
649 approved clear targets
            |
            v
Synthetic cloud generation + cloud masks
            |
            v
Single-image U-Net experiments
    |               |             |
    V1              V2            V3
standard        detail-aware    residual U-Net
    |               |             |
    +---------------+-------------+
                    |
                    v
Validation + structural diagnostics
                    |
                    v
Locked held-out synthetic test

Parallel ongoing branch:
April 16 + June 03 images
            |
            v
Cloud screening + radiometry diagnostics
            |
            v
Local geographic registration (IN PROGRESS)
            |
            v
Multi-temporal reconstruction (PLANNED)
```

## Model experiments and results

Three models have been trained and compared on the same fixed synthetic validation set:

| Experiment | Validation cloud-core MAE ↓ | Main finding |
| --- | ---: | --- |
| U-Net V1 | 0.01613 | Low spectral error, oversmoothed structures |
| Detail-aware U-Net V2 | 0.01597 | Slight MAE improvement; texture still weak |
| Residual U-Net V3 | **0.01300** | Best cloud-core validation accuracy |

### Held-out synthetic test: frozen Residual U-Net V3

| Metric | Result |
| --- | ---: |
| Test patches | 75 |
| Cloud-core MAE | **0.01218** |
| Interior 7×7 SSIM | **0.9312** |
| Cloud-core MAE of visible-mean baseline | 0.01486 |
| Gradient correlation | 0.0395 |

V3 reduces cloud-core MAE by approximately **18%** relative to a simple visible-mean fill on the held-out *synthetic* test. However, **gradient correlation remains low**: the model generally does not recover detailed geographic features hidden beneath thick clouds.

A 3-pixel cloud-mask dilation with 3-pixel feathering reduced synthetic cloud-halo error but increased cloud-core error, so it remains an **optional postprocessing comparison**, not a universally superior model.

## Current multi-temporal work

We are examining whether **16 April 2025** can provide useful reference information for locations obscured on **3 June 2025**. Both scenes use EPSG:32643 and 5 m pixel spacing, but the June scene has different brightness distributions and extensive cloud-like regions.

Cloud screening has been performed, and preliminary automatic/manual registration has been tested. **Local alignment is not yet validated**: the first manual landmark check on a clear-looking tile showed inconsistent matches. No claim of accurate real-scene reconstruction has been made.

## Repository layout

```text
Satellite-Cloud-Reconstruction/
├── src/                     # Data preparation, training, evaluation and temporal QC scripts
├── .gitignore               # Excludes raw imagery and generated artifacts
├── requirements.txt         # Python package requirements (excluding CUDA wheel setup)
└── README.md                # This project overview

Local-only working folders (not included in Git):
├── data/raw/                # Original satellite GeoTIFFs
├── data/processed/          # Extracted patches, masks and manifests
└── outputs/                 # Model checkpoints, audit reports, plots and previews
```

Git does not track empty directories. Datasets, checkpoints and bulk generated outputs are intentionally kept off the public repository; add lightweight, distributable illustrations under a future `docs/images/` folder if suitable permissions exist.

## Setup (Windows PowerShell)

Python 3.11 is used for this project. From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For NVIDIA CUDA training on the tested Windows configuration, install the appropriate PyTorch wheel separately (the prototype was tested with **PyTorch 2.11.0 + CUDA 12.8**):

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> Original satellite data are **not included**, so scripts that read scenes or saved patch manifests require those files to be prepared locally. Some scripts expect output from previous pipeline stages.

### Selected reproducible commands

Run these from the project root **after the necessary local data and intermediate files exist**:

```powershell
# Verify saved synthetic image pairs
.\.venv\Scripts\python.exe -u .\src\audit_synthetic_dataset_v2.py

# Test PyTorch dataset loading and CUDA execution
.\.venv\Scripts\python.exe -u .\src\test_pytorch_dataloader.py

# Structural evaluation on the fixed validation split
.\.venv\Scripts\python.exe -u .\src\evaluate_v3_structure.py

# Multi-temporal April–June quality-control screening
.\.venv\Scripts\python.exe -u .\src\review_april_june_cloud_and_alignment.py
```

The final test evaluation was performed with a frozen V3 checkpoint. **Do not use the held-out test split to tune further models.**

## Reproducibility and usage notes

- The original image digital numbers have **not** been converted into calibrated surface reflectance.
- April and June acquisitions are separated by 48 days: seasonal changes and registration errors can make naive temporal pixel transfer misleading.
- Cloud masks inferred from intensity are screening proxies, not independently verified cloud labels.
- Model results are specific to the processed study area and the synthetic augmentation procedure; generalization to other sensors, seasons, and real thick clouds remains unproven.
- Satellite products and derivatives should only be shared when permitted by the data provider's license or terms. No full raw imagery is redistributed here.

## Next milestones

1. Verify robust local registration on suitable clear-looking landmarks.
2. Review April reference suitability and June cloud masks.
3. Build a geographically aligned multi-temporal prototype.
4. Report qualitative real-scene results with explicit uncertainty.
5. Add representative figures, technical documentation, and a reproducible demonstration.

---

**Primary stack:** Python, NumPy, Rasterio, Pillow, PyTorch, Matplotlib
