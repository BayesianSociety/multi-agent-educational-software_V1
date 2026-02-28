# Overview
Web-based educational block-coding game for ages 7-12 where players program a dog to reach a bone.

# Scope
- MVP with Move and Jump blocks only.
- Deterministic execution/highlighting loop.
- 10 levels with early tutorial progression.
- Local backend + SQLite persistence for levels and unlocked progress.

# Non-Goals
- Multiplayer
- Accounts/cloud saves
- Complex physics
- Variables/custom blocks in MVP

# Acceptance Criteria
- Deterministic block execution order with current-block highlight.
- Frontend calls backend endpoints for level data.
- Backend serves `GET /health`, `GET /api/levels`, `GET /api/levels/:id`.
- Progress persistence is restored after refresh.
- Test commands in TEST.md run deterministically and return exit code 0.

# Risks
- Frontend and backend contracts drifting without integration tests.
- Misalignment between animation movement and required `background.png` scale.
- Docker/SQLite local setup instability across environments.
