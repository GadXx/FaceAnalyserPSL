# Phase 07 — Add Optional 3D Extension

## Objective
Introduce optional 3D geometry for better pose robustness and richer morphometric measurements.

## Recommended tools
- 3DDFA_V2 as the first optional 3D path
- optional DECA integration later if needed

## Tasks for Codex
1. Add an optional 3D preprocessing branch controlled by config.
2. Store 3D coefficients / landmarks / mesh-derived metadata separately.
3. Extend the measurement engine with 3D-capable metrics where feasible.
4. Keep 2D and 3D artifacts distinguishable.
5. Add graceful fallback so the project works without 3D models.

## Output artifacts
- `data/processed/3d_artifacts/...`
- `data/features/geometry_3d_features.parquet`
- `reports/metrics/geometry_3d_summary.json`

## Completion criteria
- Enabling 3D adds extra measurements without breaking the 2D path.
