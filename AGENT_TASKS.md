# Agent Tasks
Project Brief reference: all specialists must follow `PROJECT_BRIEF.md` (Layer 0-2 locked source of truth).

## Requirements
- Map Layer 0-2 constraints into explicit, testable requirements in `REQUIREMENTS.md`.
- Preserve required headings and deterministic acceptance criteria.

## Designer
- Define UX flows for block palette, workspace sequencing, run feedback, and retry loop.
- Specify tutorial clarity for Level 1 (`Move`) and Level 2 (`Jump`) with simple narrative text.

## Frontend
- Implement block editor and Run/Reset controls with current-block highlighting during deterministic execution.
- Ensure keyboard-operable core interactions and visible focus states.
- Integrate with backend endpoints for level retrieval/progression.

## Backend
- Implement GET /health, GET /api/levels, GET /api/levels/:id.
- Implement SQLite persistence for level data and unlocked progress restoration.
- Keep execution and endpoint behavior deterministic for offline tests.

## QA
- Author deterministic tests for execution, endpoint contracts, and persistence behavior.
- Maintain test commands in `TEST.md` and enforce deterministic exit code `0`.

## Release
- Maintain release-facing docs (`README.md`, `RUNBOOK.md`, `TEST.md`) with deterministic, offline validation commands.
- Maintain local-first startup in repo-root `docker-compose.yml` for frontend + backend + SQLite workflow.
- Ensure `.env.example` contains only local-safe defaults and no secrets.
- Ensure `.gitignore` excludes local env files, SQLite artifacts, and dependency directories.
