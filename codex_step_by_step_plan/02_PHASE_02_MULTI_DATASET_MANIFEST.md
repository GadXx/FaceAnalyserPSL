# Phase 02 — Generalize Dataset Preparation to a Multi-Dataset Manifest

## Objective
Refactor `scripts/prepare_dataset.py` so it no longer assumes SCUT-FBP5500 and instead builds a unified manifest from multiple supported datasets.

## Supported datasets
- FairFace
- UTKFace
- Chicago Face Database
- SCUT-FBP5500 (optional compatibility)
- user-provided custom datasets via generic folder adapter

## Unified manifest schema
The manifest must contain at least:
- `dataset_name`
- `sample_id`
- `subject_id` (nullable)
- `image_path`
- `split`
- `sex_label_raw`
- `age_label_raw`
- `ethnicity_label_raw`
- `sex_group_normalized`
- `age_group_normalized`
- `ethnicity_group_normalized`
- `is_frontal_hint`
- `has_landmarks_source`
- `license_or_usage_notes`
- `quality_gate_status`
- `adapter_version`

Optional fields:
- `landmarks_path`
- `metadata_json_path`
- `source_score_columns_present`
- `notes`

## Tasks for Codex
1. Replace any SCUT-specific folder assumptions with adapter classes/functions.
2. Implement dataset adapters:
   - `FairFaceAdapter`
   - `UTKFaceAdapter`
   - `ChicagoFaceDatabaseAdapter`
   - `SCUTAdapter` (legacy compatibility only)
   - `GenericImageFolderAdapter`
3. Normalize labels into a common schema without inventing missing labels.
4. Preserve raw labels and normalized labels side by side.
5. Add a CLI flag to choose dataset roots from config or command line.
6. Write non-empty manifest rows only for files that truly exist.
7. Save the combined manifest to `data/interim/dataset_manifest.parquet`.
8. Save dataset summary stats to `reports/metrics/dataset_manifest_summary.json`.

## Constraints
- Do not use any beauty / score labels as targets.
- Do not drop dataset identity.
- Do not silently coerce all ethnicity labels into one universal taxonomy unless mapping is explicit and documented.

## Output artifacts
- `data/interim/dataset_manifest.parquet`
- `reports/metrics/dataset_manifest_summary.json`

## Completion criteria
- Running `scripts/prepare_dataset.py` yields a combined manifest across available datasets.
- Zero-row outputs only occur when the dataset roots are actually empty.
- Summary JSON reports per-dataset row counts.
