# Phase 09 — Evaluate Robustness

## Objective
Measure how stable the morphometric outputs are under nuisance variation.

## Perturbations to test
- crop shift
- resize / resample
- JPEG compression
- mild brightness change
- mild blur
- small rotation
- small pose variation where possible

## Metrics to compute
- per-feature absolute delta
- per-index absolute delta
- stability summaries by perturbation type
- failure rate under perturbations

## Tasks for Codex
1. Add a robustness evaluation script.
2. Re-run preprocessing and measurement extraction on perturbed copies.
3. Compare outputs against the original image.
4. Save tables and plots if plotting exists.

## Output artifacts
- `reports/metrics/robustness_summary.json`
- `reports/metrics/robustness_per_feature.parquet`

## Completion criteria
- The project can quantify stability of objective metrics under controlled perturbations.
