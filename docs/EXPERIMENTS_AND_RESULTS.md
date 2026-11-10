# Experiments and results

**Evidence type is essential:** synthetic metrics must never be presented as accuracy on real June imagery.

## Dataset

Resourcesat-2A LISS-IV, Mumbai, April 16, 2025; NIR/Red/Green at nominal 5 m, DN pixels. Original approved patches were **visually** screened rather than certified physically cloud-free. Synthetic-pair generator V2 produced 649 pairs: 499 training, 75 validation, 75 locked test. Mean synthetic mask coverage was 27.1%, 28.4%, 27.3% respectively.

## Synthetic validation (same original validation split)

| Model | Cloud-core normalized MAE ↓ | Patches beating visible mean-fill | Texture std ratio | Gradient ratio |
| --- | ---: | ---: | ---: | ---: |
| Basic U-Net V1 | 0.01613 | 37/75 | 0.431 | 0.478 |
| Detail-aware U-Net V2 | 0.01597 | 47/75 | 0.284 | 0.218 |
| Residual U-Net V3 | **0.01300** | **66/75** | 0.474 | 0.227 |

Do not interpret an improved MAE as automatically improved structural fidelity. Training ran for three epochs in the initial study.

## Frozen synthetic held-out V3 test — 75 samples

| Metric | Value |
| --- | ---: |
| Cloud-core normalized MAE | **0.01218** |
| Interior SSIM | **0.9312** |
| Gradient correlation | **0.0395** |
| Visible mean-fill normalized MAE | 0.01486 |

V3 MAE is ~18% lower than the simple baseline on the *synthetic test*, while structural correlations remain very poor. The test set was frozen for reporting; no tuning against it.

## Boundary postprocessing trade-off

Validation V3 cloud-core MAE 0.01300, halo MAE 0.07320 and overall MAE 0.01082. With 3px dilation + 3px feather: core MAE **0.01804** (worse), halo MAE **0.02786** (better), overall MAE 0.00826. Treat as a distinct comparison, never quietly replace the core baseline.

## April–June reference feasibility

Dates April 16 vs June 3 are **48 days** apart. Independent registration checks did not approve a global or local cross-date correction. Partial-tile test: T041 25.4% candidate, 7.8% stable Red controls, shift (-4,-1), held-out correlation gain +0.5822 **provisional**; T058 27.0% candidate, 2.3% stable Red controls and **insufficient** evaluation. No validated multi-temporal reconstruction exists.

## Real June V3 inference (unvalidated)

| Tile | June-only mask-proxy fraction | Mean absolute change within proxy (DN, NOT accuracy) |
| --- | ---: | ---: |
| T041 | 25.8% | 227.87 |
| T058 | 27.5% | 84.10 |

Four automatic proxy variants were compared. For T041: original 25.8%, expanded 3px 30.8%, expanded 8px 39.5%, looser nearby 35.3%; for T058: 27.5%, 36.7%, 45.7%, 51.1%. Larger masks did not recover convincing surface structure. The real-cloud domain gap remains unresolved.

## Experimental export (not validated cloud-free)

`src/export_experimental_real_june_v3_geotiff_v3.py` successfully generated and verified four GeoTIFFs for each tile (original, V3 predicted, proxy, validity). Source legacy WKT versus EPSG:32643 verification checks spatial equivalence, not just WKT string equality. Output is research-only. The data and checkpoint are not distributed in Git.

## Recommended Phase 2 metrics

PSNR, spectral angle, edge/structure preservation, cloud-mask IoU against audited labels, boundary-region metrics, out-of-distribution performance, inference memory/runtime, uncertainties and a geographic/date-held-out evaluation. These are **planned**, not reported accomplishments.
