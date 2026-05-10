# Phase 04 — Build the Deterministic Morphometric Measurements Engine

## Objective
Replace the feature extraction stage so it computes objective morphometric measurements instead of subjective deep-only scorer features.

## Inputs
- `data/processed/preprocessed_manifest.parquet`
- saved landmarks / pose
- optional 3D outputs

## Core measurement groups
### A. Linear distances
Examples:
- face width
- face height
- bizygomatic width proxy
- jaw width proxy
- mouth width
- nose width / height proxies
- inter-eye distance
- intercanthal distance proxy

### B. Ratios and proportions
Examples:
- width / height ratio
- lower-third / middle-third / upper-third proportions
- nose width / inter-eye distance
- mouth width / face width
- chin width / jaw width

### C. Symmetry metrics
Examples:
- left-right landmark distance asymmetry
- contour asymmetry
- eye/mouth/nose offset from facial midline

### D. Angular metrics
Examples:
- eye line angle
- mouth line angle
- nose deviation angle
- jaw angle proxies

### E. Pose-aware normalized metrics
Adjust or exclude metrics when pose exceeds configured thresholds.

### F. Optional deep support features
Deep embeddings may still be extracted and stored, but only as auxiliary stability features, not as an attractiveness target pathway.

## Tasks for Codex
1. Refactor `scripts/extract_features.py` into a measurement-centric pipeline.
2. Keep output tables modular:
   - `measurements_raw.parquet`
   - `measurements_normalized.parquet`
   - `symmetry_features.parquet`
   - `geometry_features.parquet`
   - `fusion_features.parquet` (optional)
3. Build a reusable measurement library under `src/` or equivalent.
4. Document exact formulas for every metric.
5. Add validation for impossible values and NaNs.
6. Include sample-level feature completeness flags.

## Output artifacts
- `data/features/measurements_raw.parquet`
- `data/features/measurements_normalized.parquet`
- `data/features/symmetry_features.parquet`
- `data/features/geometry_features.parquet`
- optional `data/features/deep_embeddings.parquet`
- `reports/metrics/feature_extraction_summary.json`

## Completion criteria
- A valid preprocessed sample produces deterministic measurements.
- Output tables are non-empty and keyed by sample.
- Formulas are documented.
