# Codex Requests — Morphometric Face Analysis (v2)

Use these prompts **in order**. Wait for each step to finish, inspect the diff, and only then move to the next prompt.

---

## Global instruction prefix for every prompt
Before each request, prepend this instruction block:

```text
You are modifying an existing repository.
Work incrementally and preserve existing working code whenever possible.
Do not delete working scripts unless replacing them with a clearly better equivalent.
The project must now implement objective morphometric face analysis, not beauty / attractiveness / PSL scoring.
Never introduce or preserve targets named beauty_score, attractiveness_score, psl_score, appeal_score, or similar.
Prefer refactoring existing files over creating parallel duplicates unless duplication is necessary.
At the end of the task, print:
1) files changed,
2) commands to run,
3) expected artifacts,
4) any external models or datasets still required.
```

---

<!-- ## Request 01 — Re-scope and rename
```text
Audit the repository and refactor project vocabulary from subjective face scoring to objective morphometric analysis.

Tasks:
- Replace or deprecate subjective naming such as beauty, attractiveness, psl, appeal, ranking, score targets.
- Introduce neutral terminology: morphometric, measurements, reference_model, reference_deviation, symmetry_index, proportion_profile, harmony_index.
- Update README, config comments, docstrings, and active code paths.
- Ensure the project description clearly states that the system computes objective facial measurements and reference-based indices.
- Keep imports and script entry points working after the rename.

Deliverables:
- Updated README/config/docstrings
- Working imports
- No active code path depending on subjective labels
``` -->

<!-- ## Request 02 — Generalize dataset manifest
```text
Refactor scripts/prepare_dataset.py into a multi-dataset manifest builder.

Requirements:
- Do not assume SCUT-FBP5500.
- Support adapters for FairFace, UTKFace, Chicago Face Database, optional SCUT compatibility, and a generic image-folder adapter.
- Create a unified manifest schema with dataset_name, sample_id, subject_id, image_path, split, raw labels, normalized labels, quality_gate_status, and adapter_version.
- Preserve raw labels and normalized labels separately.
- Save output to data/interim/dataset_manifest.parquet.
- Save summary stats to reports/metrics/dataset_manifest_summary.json.
- Do not use any beauty/score columns as targets.

Also add tests for at least one adapter and one generic folder case.
``` -->

<!-- ## Request 03 — Upgrade preprocessing
```text
Refactor scripts/extract_preprocessing_artifacts.py to support morphometric preprocessing on top of the unified manifest.

Requirements:
- Consume data/interim/dataset_manifest.parquet.
- Produce face detection, landmarks, aligned crop, pose (yaw/pitch/roll), quality flags, and processing status.
- Save machine-readable outputs and a parquet manifest at data/processed/preprocessed_manifest.parquet.
- Save aligned crops under data/processed/aligned_faces/.
- Add quality gates for no face, multiple faces, blur, extreme pose, and severe occlusion where feasible.
- Keep support for local InsightFace model paths.
- If required models are missing, fail clearly with a helpful dependency message.
- Do not silently generate invalid outputs.

Also add a preprocessing summary JSON in reports/metrics/.
``` -->

<!-- ## Request 04 — Build measurement engine
```text
Refactor scripts/extract_features.py into a deterministic morphometric measurement pipeline.

Requirements:
- Input: preprocessed manifest plus saved landmarks/pose.
- Compute raw measurements, normalized ratios, symmetry metrics, angle metrics, and geometry feature tables.
- Save outputs to:
  - data/features/measurements_raw.parquet
  - data/features/measurements_normalized.parquet
  - data/features/symmetry_features.parquet
  - data/features/geometry_features.parquet
- Document formulas in code comments or docs.
- Validate NaN/impossible values.
- Keep optional deep embeddings only as auxiliary features if already present, but not as a subjective scoring path.
- Save feature extraction summary JSON.

Also add unit tests for at least several measurement formulas.
``` -->

## Request 05 — Build reference model
```text
Replace subjective training with reference-model construction.

Requirements:
- Refactor scripts/train_baseline.py into a reference model builder, or add a new script and preserve compatibility wrappers if needed.
- Remove dependence on beauty labels.
- Build per-group descriptive statistics from normalized morphometric features:
  - mean, std, median, IQR, percentiles.
- Implement z-score computation and Mahalanobis distance helpers.
- Support grouping by configurable combinations of sex_group_normalized, age_group_normalized, ethnicity_group_normalized.
- Enforce minimum sample count safeguards.
- Save artifacts under models/reference/ and metrics under reports/metrics/reference_model_summary.json.

Also update docs to explain how reference groups are selected.
```

## Request 06 — Deliver 2D end-to-end baseline
```text
Make the full 2D morphometric pipeline runnable end-to-end.

Requirements:
- Wire together manifest -> preprocessing -> measurements -> reference model -> single-sample report.
- Add a simple runbook command sequence in README or docs.
- Produce reports/examples/sample_report.json and sample_report.md.
- The report must include quality status, raw measurements, normalized ratios, symmetry metrics, and reference deviation outputs.
- No subjective labels or beauty-style scores anywhere.

Also add a small smoke test or dry-run check for the end-to-end 2D path.
```

## Request 07 — Add optional 3D extension
```text
Add an optional 3D branch without breaking the 2D path.

Requirements:
- Support an optional 3D preprocessing branch controlled by config.
- Store 3D outputs separately under data/processed/3d_artifacts/.
- Add geometry_3d_features.parquet if 3D is enabled.
- Keep graceful fallback when 3D models are not installed.
- Document required external models and exact expected paths.

Do not make 3D mandatory for the baseline pipeline.
```

## Request 08 — Reporting and composite indices
```text
Implement objective reporting utilities.

Requirements:
- Support JSON and Markdown report generation.
- Include image quality, preprocessing status, raw measurements, normalized ratios, symmetry metrics, angle metrics, reference comparison, and limitations.
- Optional composite indices may include symmetry_index, proportion_consistency_index, reference_deviation_index, morphometric_harmony_index.
- Do not introduce attractiveness, beauty, or PSL scores.
- Make formulas and reference-group selection transparent.

Also save a reporting summary JSON under reports/metrics/.
```

## Request 09 — Robustness evaluation
```text
Add a robustness evaluation script for the objective pipeline.

Requirements:
- Test perturbations such as crop shift, resize, JPEG compression, mild blur, mild brightness change, and small rotation.
- Re-run preprocessing and measurement extraction on perturbed images.
- Compute per-feature and per-index deltas versus the original.
- Save results to reports/metrics/robustness_summary.json and a per-feature parquet table.

Keep this evaluation strictly about stability of objective measurements.
```

## Request 10 — Subgroup validation
```text
Add subgroup validation and coverage reporting.

Requirements:
- Report preprocessing success rate by dataset source, sex group, age group, ethnicity group, and quality bucket.
- Report feature completeness by subgroup.
- Report reference-model coverage and sample counts by subgroup.
- Warn when subgroup support is weak.
- Keep this as transparency/coverage analysis, not a fairness marketing claim.

Save outputs under reports/metrics/subgroup_validation.json and reports/metrics/reference_group_coverage.json.
```

## Request 11 — Inference CLI
```text
Implement a single-image inference CLI.

Requirements:
- Add scripts/run_inference.py or equivalent.
- Accept image path, reference-group selection, output format, and optional 3D flag.
- Produce a JSON report and optionally a Markdown report.
- Save reports to reports/examples/ by default.
- Fail clearly when required models are missing.

Also document exact example commands.
```

## Request 12 — API layer
```text
Implement or refactor an API for the final morphometric pipeline.

Requirements:
- Add endpoints such as POST /analyze, GET /health, GET /reference-groups.
- Reuse the same service layer as the CLI.
- Return only objective morphometric outputs and metadata.
- Add request/response schemas and clear dependency errors.

Also update docs with a minimal API example.
```

## Request 13 — Tests, docs, and dependency checklist
```text
Finish the project with tests, docs, and setup guidance.

Requirements:
- Add tests for dataset adapters, preprocessing smoke path, measurement formulas, reference model, CLI smoke path, and API smoke path.
- Update README with setup instructions, required model downloads, supported datasets, expected folder structure, and limitations.
- Add a dependency checklist describing exactly which external models are needed for:
  - 2D baseline
  - optional 3D path
- Ensure all scripts and docs are consistent with the morphometric scope.

At the end, provide a concise runbook from empty repo state to first successful 2D inference.
```
