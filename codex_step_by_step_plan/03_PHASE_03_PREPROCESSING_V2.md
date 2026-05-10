# Phase 03 — Upgrade Preprocessing for Objective Morphometric Analysis

## Objective
Refactor `scripts/extract_preprocessing_artifacts.py` to support robust preprocessing for morphometric analysis.

## Required outputs per valid image
- face detection bounding box
- 2D landmarks
- optional 3D landmarks / mesh coefficients
- aligned crop
- head pose (`yaw`, `pitch`, `roll`)
- quality flags
- transformation metadata

## Recommended tool priority
### Primary path
- InsightFace for detection/alignment/embeddings-compatible preprocessing

### Optional alternative / fallback
- MediaPipe Face Landmarker for dense landmarks and transformation matrices

### Optional advanced path
- 3DDFA_V2 for 3D dense alignment and pose

## Tasks for Codex
1. Refactor preprocessing to consume the new unified manifest.
2. Store preprocessing back into a new parquet, not only side-effect image files.
3. Implement quality gating:
   - no face detected
   - multiple faces
   - strong blur
   - extreme pose
   - severe occlusion (best-effort heuristic)
4. Save aligned crops only for samples that pass minimum quality.
5. Save landmark arrays and pose values in machine-readable form.
6. Persist per-sample processing status.
7. Keep support for local ONNX models and configurable paths.
8. If no models are present, fail with a clear dependency error instead of silent bad output.

## Output artifacts
- `data/processed/preprocessed_manifest.parquet`
- `data/processed/aligned_faces/...`
- `data/processed/landmarks/...`
- `reports/metrics/preprocessing_summary.json`

## Completion criteria
- Non-empty input images produce non-empty preprocessing rows.
- Quality-gated samples are explicitly marked.
- Landmarks and pose are available for downstream measurement extraction.
