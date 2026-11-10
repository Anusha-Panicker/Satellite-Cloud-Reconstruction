# Satellite Cloud Removal & Reconstruction

**Mumbai case study | Resourcesat-2A LISS-IV | PyTorch Residual U-Net | Research prototype**

> **Research status, November 2026:** A reproducible *synthetic-cloud* benchmark and an experimental real-June GeoTIFF export pipeline are implemented. Real cloud removal has **not been validated**. Independent April–June registration checks failed to establish a trustworthy cross-date alignment; no temporal pixel transfer is approved.

## What this project does

This research prototype investigates masked-region reconstruction in 5 m, three-band (Band 4 NIR, Band 3 Red, Band 2 Green) Mumbai satellite imagery. It constructs paired synthetic cloudy/reference patches from visually screened imagery, trains three U-Net variants, evaluates a frozen synthetic test split, and runs exploratory inference on actual June 2025 clouds. The real-scene outputs are deliberately labelled **experimental predictions**, not confirmed views of the hidden terrain.

| Item | Details |
| --- | --- |
| Study area | Mumbai, India |
| Sensor | Resourcesat-2A LISS-IV / L4FX |
| Dates examined | 27 Feb, 16 Apr, 3 Jun 2025 |
| Main synthetic reference | 16 Apr 2025; visually screened, **not certified cloud-free** |
| Real cloudy evaluation | 3 Jun 2025 |
| Bands | B4 NIR / B3 Red / B2 Green |
| Pixel spacing / CRS | 5 m / WGS 84 UTM zone 43N |
| Synthetic patch size | 256 × 256 |
| Approved reference patches | **649** (499 train, 75 val, 75 held-out test) |
| Model family | U-Net V1, detail-aware U-Net V2, Residual U-Net V3 |
| Selected synthetic baseline | Residual U-Net V3 |

## Results: synthetic versus real

### Fixed validation set (synthetic clouds)

| Model | Normalized cloud-core MAE ↓ |
| --- | ---: |
| U-Net V1 | 0.01613 |
| Detail-aware U-Net V2 | 0.01597 |
| Residual U-Net V3 | **0.01300** |

### Frozen held-out test: Residual U-Net V3 only (synthetic clouds)

| Metric | Value |
| --- | ---: |
| Test samples | 75 |
| Normalized cloud-core MAE | **0.01218** |
| Interior SSIM | **0.9312** |
| Structural gradient correlation | **0.0395** |
| Simple visible mean-fill MAE | 0.01486 |

V3 achieved approximately 18% lower normalized cloud-core MAE than mean fill on this *synthetic* test. The weak gradient correlation means fine geographical structures are **not** reliably recovered. The test split is frozen and must not be used for tuning.

### Actual June imagery: experimental only

Two June tiles were tested with a June-only brightness-based, **unvalidated** cloud proxy. V3 produced smooth and often implausible cloud-region fills; mask-expansion experiments did not resolve this problem. Do not report the within-mask DN change as accuracy.

| Real June tile | Selected by mask proxy | Mean absolute change within mask (DN; **not accuracy**) |
| --- | ---: | ---: |
| T041 | 25.8% | 227.87 |
| T058 | 27.5% | 84.10 |

**Eight GeoTIFFs** (original, experimental prediction, proxy, validity for each of two tiles) were successfully written and verified with `src/export_experimental_real_june_v3_geotiff_v3.py`. Exported spatial coordinates were checked for legacy-WKT/EPSG equivalence. Successful georeferencing does **not** establish successful cloud removal.

## Repository organization

```text
Satellite-Cloud-Reconstruction/
├── src/                  # Original runnable scripts; intentionally kept flat for stable imports
│   ├── unet_model.py
│   ├── train_unet_v1.py
│   ├── train_unet_detail_v2.py
│   ├── train_unet_residual_v3.py
│   ├── evaluate_v3_locked_test.py
│   ├── preview_real_june_unet_v3.py
│   └── export_experimental_real_june_v3_geotiff_v3.py
├── docs/
│   ├── PROJECT_STATUS.md
│   ├── EXPERIMENTS_AND_RESULTS.md
│   ├── SCRIPT_INDEX.md
│   ├── REPRODUCIBILITY.md
│   └── AFTER_EXAMS_ROADMAP.md
├── requirements.txt
├── .gitignore
└── README.md
```

**Local-only (excluded from Git):** `data/raw/` original satellite products, `data/processed/` generated patches and masks, `outputs/` checkpoints and bulk plots, `.venv/`. A fresh clone contains **source code and documentation, not the data or trained checkpoint**. Observe the satellite data provider's redistribution rules.

Files remain in `src/` rather than being moved between packages because the existing training/evaluation scripts rely on same-directory imports (e.g. `from unet_model import CloudUNet`). Reorganizing source into packages would require coordinated imports, tests and checkpoint-compatibility review.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# Install a compatible PyTorch build for your hardware; see https://pytorch.org/get-started/locally/
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Run commands from the repository root. Most training and evaluation scripts require the local source data, processed manifests and/or trained checkpoint that are **not** included in Git.

```powershell
# Synthetic-pair audit (requires local processed dataset)
.\.venv\Scripts\python.exe -u .\src\audit_synthetic_dataset_v2.py

# Synthetic locked-test check (uses local data/checkpoint; do not tune on test)
.\.venv\Scripts\python.exe -u .\src\evaluate_v3_locked_test.py --self-test

# Real June inference self-test (temporary data; no real scene access)
.\.venv\Scripts\python.exe -u .\src\preview_real_june_unet_v3.py --self-test

# Real June georeferenced exporter self-test
.\.venv\Scripts\python.exe -u .\src\export_experimental_real_june_v3_geotiff_v3.py --self-test

# Run real export ONLY with required local June scene + V3 checkpoint
.\.venv\Scripts\python.exe -u .\src\export_experimental_real_june_v3_geotiff_v3.py
```

See [Reproducibility](docs/REPRODUCIBILITY.md) for prerequisites and limitations. The `requirements.txt` is a dependency declaration, not a fully locked environment; preserve a verified `pip freeze` before claiming exact recreation of the original experiment.

## Documentation

- [Current project status and scientific limitations](docs/PROJECT_STATUS.md)
- [Experiments and numerical results](docs/EXPERIMENTS_AND_RESULTS.md)
- [Source-script workflow index](docs/SCRIPT_INDEX.md)
- [Data requirements and reproducibility](docs/REPRODUCIBILITY.md)
- [Roadmap after exams](docs/AFTER_EXAMS_ROADMAP.md)

## What is deferred

No frontend, GAN, Transformer, diffusion model, or validated multi-temporal fusion has been implemented. The **frontend is postponed until after the developer's exams**. Current priorities are code organization, honest documentation, and a reproducible research-prototype handover.

## Scientific safety note

Generated cloud-removed-looking images may be *plausible but wrong*. Never use the current outputs as verified ground observations, for remote-sensing measurements requiring physical accuracy, or for change-detection conclusions. Cloud masks, cross-date alignment and source-target correspondence need independent validation before operational use.
