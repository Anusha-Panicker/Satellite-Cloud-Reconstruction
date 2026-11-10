# Roadmap — after exams

**Scope freeze (November 2026):** Finish GitHub source organization, documentation and reproducibility checks first. The frontend is explicitly **postponed until after exams**. No current frontend work is required.

## After exams: frontend demonstration

Build a browser-based UI (Flask or another lightweight Python backend, HTML/CSS/JS) for curated 256×256 images: synthetic input/ground truth/result, unvalidated real June input/mask/output, metrics with correct dataset labels, provenance and GeoTIFF downloads. No desktop image-mask editor. Maintain an explicit warning that the real-June output is experimental.

## Later scientific Phase 2

1. Obtain better real-cloud training/evaluation samples and expert-audited masks.
2. Improve synthetic realism and domain adaptation; quantify generalization to independent dates/locations.
3. Improve structural reconstruction using multiscale/edge/spectral losses with honest validation.
4. Revisit geometric/within-scene band alignment with independent controls and physically grounded radiometric calibration.
5. Only after reliable reference selection, test multi-temporal fusion.
6. Consider GAN, Transformer, diffusion or SAR assistance as comparative research, not claimed existing capabilities.
7. Scale to full-scene streaming inference, uncertainty, and reproducible benchmark reporting.

No Phase 2 architecture or frontend should be represented as already implemented.
