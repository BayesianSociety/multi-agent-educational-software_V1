# Overview
Build a deterministic, kid-safe (ages 7-12) web MVP where players use Scratch-like Move and Jump blocks to guide a dog to a bone. The system must run with a required backend and SQLite persistence.

# Scope
- Frontend in Next.js (TypeScript) under /frontend.
- Backend in Node.js API (TypeScript) under /backend.
- SQLite as source of truth for levels and progress.
- Required endpoints: GET /health, GET /api/levels, GET /api/levels/:id.
- 10 short levels with tutorial progression (Move then Jump).
- Friendly visual tone with background.png as mandatory Barbie-themed background alignment reference.

# Non-Goals
- Multiplayer.
- Accounts or cloud saves.
- Complex physics.
- Advanced Scratch features like variables or custom blocks in MVP.

# Acceptance Criteria
- Deterministic block execution highlights current block while running.
- Level 1 solvable using Move blocks; Level 2 introduces Jump requirement.
- Unlocked progress persists in SQLite and restores on refresh.
- Frontend runtime calls backend API endpoints.
- Tests are deterministic and exit with code 0.

# Risks
- Coordinate/scale mismatch against background.png causing incorrect move/jump distances.
- Drift between frontend behavior and backend level/progress contracts.
- Non-deterministic timing in animations affecting perceived execution order.
