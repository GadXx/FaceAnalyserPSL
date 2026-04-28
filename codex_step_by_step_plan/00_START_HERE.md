# Morphometric Face Analysis — Step-by-Step Plan (v2)

## Goal
Rebuild the project from a subjective face-score pipeline into a **multi-dataset morphometric analysis system**.

The system must:
- **not** predict beauty / attractiveness / PSL;
- compute **objective facial measurements** from landmarks and optional 3D geometry;
- normalize those measurements into **ratios, symmetry indices, angles, z-scores, percentiles**;
- produce either:
  - **measurement-only reports**, or
  - a **reference-deviation / harmony index** based on a configurable reference model;
- support **multiple datasets** instead of being hard-wired to SCUT-FBP5500.

## What is already implemented and must be reused/refactored
Current scripts already exist and should be **refactored**, not deleted blindly:
- `scripts/prepare_dataset.py`
- `scripts/extract_preprocessing_artifacts.py`
- `scripts/extract_features.py`
- `scripts/train_baseline.py`
- `scripts/train_geometry.py`

Existing outputs/folders should also be reused where practical:
- `data/interim/`
- `data/processed/`
- `data/features/`
- `models/`
- `reports/metrics/`

## New target architecture
1. **Dataset adapters** for FairFace, UTKFace, Chicago Face Database, optional SCUT, optional 3D Facial Norms metadata.
2. **Unified manifest schema** independent of any one dataset.
3. **Preprocessing pipeline** for face detection, landmarks, alignment, pose, quality, and optional 3D extraction.
4. **Morphometric engine** that computes deterministic measurements.
5. **Reference modeling** based on group-aware statistics.
6. **Scoring/reporting layer** that outputs objective metrics and optional composite indices.
7. **Robustness and subgroup validation**.
8. **Inference CLI and API**.

## Recommended execution order
1. Phase 01 — Re-scope project and rename artifacts
2. Phase 02 — Generalize dataset manifest to multi-dataset input
3. Phase 03 — Upgrade preprocessing for landmarks/pose/quality
4. Phase 04 — Build morphometric measurements engine
5. Phase 05 — Build reference model and objective scoring
6. Phase 06 — Run 2D baseline end-to-end
7. Phase 07 — Add optional 3D extension
8. Phase 08 — Reporting and composite indices
9. Phase 09 — Robustness evaluation
10. Phase 10 — Subgroup validation
11. Phase 11 — Inference CLI
12. Phase 12 — API
13. Phase 13 — Tests, docs, and final cleanup

## Datasets by role
### Core train / development
- FairFace
- UTKFace

### Standardized external holdout
- Chicago Face Database

### Optional legacy / compatibility dataset
- SCUT-FBP5500

### Optional normative 3D support
- 3D Facial Norms Database (metadata / summary or controlled data if available)

## Non-goals
- No attractiveness prediction.
- No supervised target called `score`, `beauty_score`, `psl_score`, `appeal`, or similar.
- No ranking by subjective labels.
- No hard-coded golden-ratio-only logic.

## Deliverables after full completion
- Multi-dataset manifest and dataset adapters.
- Preprocessing artifacts for 2D and optional 3D.
- Morphometric feature tables.
- Reference statistics per subgroup.
- Objective inference report for a single image.
- CLI and API.
- Tests, metrics, and documentation.
