# Phase 11 — Build the Inference CLI

## Objective
Provide a practical command-line entry point for single-image analysis.

## Expected CLI behavior
Input:
- image path
- selected reference group or auto selection policy
- output format
- optional 3D enabled flag

Output:
- JSON report
- optional Markdown report

## Tasks for Codex
1. Implement `scripts/run_inference.py` or equivalent.
2. Add argument validation.
3. Support model path overrides.
4. Fail clearly when required models are missing.
5. Save report to `reports/examples/` by default.

## Completion criteria
- A user can run one command and obtain a full objective morphometric report.
