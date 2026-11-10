# Project status and limitations — 10 November 2026

## Verified engineering milestones

- Read Resourcesat-2A LISS-IV Mumbai imagery, inspected NoData, bands and georeferencing.
- Reviewed 649 visually suitable original image patches: 499 train / 75 validation / 75 held-out synthetic test, 256×256 pixels.
- Created V2 synthetic cloudy/reference pairs and audited all 649 pairs.
- Trained U-Net V1, detail-aware U-Net V2 and Residual U-Net V3; selected V3 for the frozen synthetic test.
- Tested synthetic image reconstruction; metrics are in `EXPERIMENTS_AND_RESULTS.md`.
- Conducted extensive April/June radiometry and registration experiments; results do **not** approve transferring April image pixels into June.
- Performed qualitative V3 inference on genuine June clouds, tested multiple mask-proxy variants, and exported **eight verified but unvalidated** GeoTIFFs for T041 and T058.

## Unresolved scientific limitations

1. **No validated ground truth for actual June cloud-covered pixels.** April is 48 days earlier and must not be used as June truth.
2. **Synthetic-to-real domain gap.** Real clouds create poor, overly smooth V3 predictions; experiments with larger proxies did not fix it.
3. **Weak structural recovery.** Held-out synthetic gradient correlation is 0.0395 even when masked SSIM is high.
4. **Unvalidated real cloud and shadow masks.** Brightness screening misses haze and may select bright ground.
5. **Unverified April–June registration and possible band effects.** Independent spectral checks rejected seemingly promising local shifts. Do not globally warp images.
6. **Raw DN radiometry and limited scene diversity.** Atmospheric/reflectance calibration and geographically independent validation were not established.
7. **Spatial generalization and patch leakage risks.** Split control exists, but new geographic/date holdouts are needed.
8. **No uncertainty calibration, physically grounded spectral checks or validated full-scene inference.** Current georeferenced exports are small experimental tiles.

## Claims we may make

A repeatable synthetic-cloud supervised benchmark was built, three models were compared on validation, and frozen V3 synthetic-test performance was recorded. Experimental June inference and georeferenced export were demonstrated.

## Claims we must not make

Do not claim accurate real-cloud removal, validated geographic recovery, approved April-to-June transfer, 0.9312 SSIM on real clouds, or that mean DN change is model accuracy.

## Current scope

Code organization and documentation now. Browser frontend and any new model training **after exams**. No GAN, Transformer or diffusion models implemented.
