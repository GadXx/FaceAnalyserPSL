# Phase 05 — Build the Reference Model and Objective Scoring Layer

## Objective
Repurpose training scripts away from supervised beauty scoring and toward reference-based morphometric modeling.

## Core idea
The project should learn or estimate **reference distributions**, not attractiveness labels.

## Supported reference approaches
### Required baseline
- per-group descriptive statistics:
  - mean
  - standard deviation
  - median
  - IQR
  - percentile tables

### Recommended statistical scoring
- z-scores per metric
- aggregate absolute z-score profile
- Mahalanobis distance in feature space

### Optional compact latent reference
- PCA on normalized geometry features
- distance-to-reference-subspace

### Optional anomaly-style baseline
- one-class model on feature vectors

## Grouping strategy
Reference models must support grouping by configurable combinations such as:
- sex only
- age_group only
- ethnicity only
- sex × age_group
- sex × ethnicity
- age_group × ethnicity
- sex × age_group × ethnicity (when sample counts are sufficient)

## Tasks for Codex
1. Refactor `scripts/train_baseline.py` into `scripts/build_reference_model.py` or equivalent.
2. Refactor `scripts/train_geometry.py` into `scripts/build_geometry_reference.py` or merge logically.
3. Train/store reference statistics only on eligible high-quality samples.
4. Produce per-feature stats by group.
5. Implement inference helpers that transform measurements into:
   - z-scores
   - percentiles
   - Mahalanobis distance
   - optional PCA distance
6. Store model metadata with dataset provenance.
7. Add minimum sample count safeguards before emitting group-specific reference models.

## Output artifacts
- `models/reference/reference_stats.json`
- `models/reference/reference_covariance.pkl`
- `models/reference/pca_reference.pkl` (optional)
- `reports/metrics/reference_model_summary.json`

## Completion criteria
- Reference models can be built without any beauty labels.
- A new sample can be scored against a selected subgroup reference.
