# Agent Tasks
Project Brief reference: All specialists must follow PROJECT_BRIEF.md (locked after bootstrap).

## Requirements
- Map Layer 0-2 constraints into explicit functional and non-functional requirements.
- Keep acceptance criteria deterministic and testable.

## Designer
- Define /design UX flows for palette/workspace/play area and single-sequence snapping.
- Specify Level 1/2 tutorial progression and narrative text aligned with Project Brief.

## Frontend
- Implement editor, run/reset controls, block highlight during execution.
- Integrate frontend runtime calls to backend level endpoints from Project Brief.

## Backend
- Implement GET /health, GET /api/levels, GET /api/levels/:id.
- Implement SQLite persistence for levels and unlocked progress.

## QA
- Author deterministic tests for execution, endpoint contracts, and persistence behavior.
- Maintain TEST.md commands and ensure deterministic exit code 0.
