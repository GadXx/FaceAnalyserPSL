# Phase 10 — Validate Across Subgroups

## Objective
Ensure the system behaves transparently across demographic subgroups and dataset sources.

## Validation dimensions
- dataset source
- sex group
- age group
- ethnicity group
- quality bucket

## Tasks for Codex
1. Add subgroup metric reporting for preprocessing success rates.
2. Add subgroup metric reporting for measurement completeness.
3. Add subgroup reporting for reference model coverage and sample counts.
4. Add warnings when a subgroup has weak support.
5. Keep this as transparency/coverage analysis, not a fairness marketing claim.

## Output artifacts
- `reports/metrics/subgroup_validation.json`
- `reports/metrics/reference_group_coverage.json`

## Completion criteria
- The project can show where it has strong vs weak reference support.
