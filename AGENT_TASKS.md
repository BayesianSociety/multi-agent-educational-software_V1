# Agent Tasks
Project Brief reference: all specialists must follow locked constraints in `PROJECT_BRIEF.md` and `PROJECT_BRIEF.yaml`.

## Requirements
- Keep acceptance criteria aligned to Layer 1 outcomes and Layer 2 backend-required architecture.
- Ensure requirement wording remains deterministic and testable.

## Designer
- Produce `/design` content for palette/workspace/play-area layout and snap/connect rules.
- Keep Move/Jump distances aligned with `background.png` scale/layout.

## Frontend
- Implement editor and deterministic step runner in `/frontend`.
- Integrate frontend runtime calls to backend endpoints for level retrieval.

## Backend
- Implement API endpoints (`/health`, `/api/levels`, `/api/levels/:id`) in `/backend`.
- Use SQLite-based persistence and deterministic level/progress access.

## QA
- Maintain deterministic test commands and validation logic under `/tests` and `TEST.md`.
- Verify acceptance criteria and frontend-backend integration behavior.
