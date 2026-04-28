# Phase 08 — Implement Objective Scoring and Human-Readable Reports

## Objective
Turn raw/reference outputs into usable reports without introducing subjective judgments.

## Required outputs
### Per-sample report sections
- image quality
- preprocessing status
- raw measurements
- normalized ratios
- symmetry metrics
- angle metrics
- reference comparison
- optional composite indices
- warnings / limitations

## Composite indices (optional)
Allowed indices:
- `symmetry_index`
- `proportion_consistency_index`
- `reference_deviation_index`
- `morphometric_harmony_index`

Not allowed:
- attractiveness score
- beauty score
- PSL score

## Tasks for Codex
1. Implement report generation utilities.
2. Ensure reports explain which reference group was used.
3. Add caution text when group sample size is low.
4. Support both JSON and Markdown output.
5. Keep formulas transparent and reproducible.

## Output artifacts
- `reports/examples/*.json`
- `reports/examples/*.md`
- `reports/metrics/reporting_summary.json`

## Completion criteria
- Reports are objective, reproducible, and free of subjective labels.
