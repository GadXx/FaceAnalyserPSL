# Phase 12 — Add the API Layer

## Objective
Expose the completed inference pipeline over an API.

## Suggested endpoints
- `POST /analyze`
- `GET /health`
- `GET /reference-groups`

## Response shape
The API must return only objective morphometric outputs and metadata.

## Tasks for Codex
1. Implement or refactor a minimal API app.
2. Add request validation.
3. Add response schema definitions.
4. Reuse the same service layer as the CLI.
5. Add clear error messages for model/dataset dependency failures.

## Completion criteria
- The API reproduces the same analysis as the CLI for the same image and settings.
