# Phase 01 — Re-scope the Project and Rename Subjective Concepts

## Objective
Convert the codebase vocabulary and interfaces from a subjective score pipeline into an objective morphometric analysis pipeline.

## Inputs
- Existing repository state
- Existing scripts and configs

## Tasks for Codex
1. Audit the repository for subjective naming and logic:
   - `beauty`
   - `score`
   - `psl`
   - `attractiveness`
   - `ranking`
   - `baseline` where it semantically implies subjective supervision
2. Replace naming with neutral morphometric terms where appropriate:
   - `morphometric`
   - `measurements`
   - `reference_model`
   - `deviation_score`
   - `symmetry_index`
   - `proportion_profile`
   - `harmony_index`
3. Update README / module docstrings / config comments to reflect the new objective.
4. Keep backward-compatible wrappers only if they reduce churn and do not preserve subjective semantics.
5. Ensure there is no remaining training path that expects beauty labels.

## Required code changes
- Introduce a single source of truth in config for project mode:
  - `task_type: morphometric_reference_analysis`
- Introduce output terminology:
  - `raw_measurements`
  - `normalized_measurements`
  - `reference_deviation`
  - `composite_indices`

## Artifacts to produce
- Updated README section describing the new task
- Updated config file(s)
- Refactored imports / names / docstrings
- Migration notes in `docs/` or `reports/`

## Completion criteria
- No subjective labels remain in active code paths.
- The project description clearly states the task is objective morphometric analysis.
- Existing scripts still import and execute after renaming.
