# Phase 06 — Deliver the 2D Morphometric Baseline End-to-End

## Objective
Make the pipeline fully usable in 2D before any 3D extension.

## End-to-end path
1. manifest creation
2. preprocessing
3. measurement extraction
4. reference model build
5. single-image objective report

## Tasks for Codex
1. Add a one-command local runbook for the full 2D path.
2. Ensure all intermediate artifacts are saved deterministically.
3. Add report generation for one sample and for batch evaluation.
4. Produce a concise markdown/JSON report with:
   - quality status
   - raw measurements
   - normalized ratios
   - symmetry indices
   - z-scores vs selected reference group
   - optional composite indices
5. Add command-line arguments to select reference group.

## Output artifacts
- `reports/examples/sample_report.json`
- `reports/examples/sample_report.md`
- `reports/metrics/pipeline_2d_summary.json`

## Completion criteria
- A single valid image can be processed end-to-end using only the 2D path.
- The report contains only objective morphometric outputs.
