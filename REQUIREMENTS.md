# Overview
Build a deterministic, kid-safe web game MVP (ages 7-12) where players assemble `Move` and `Jump` blocks to guide a dog to a bone, run the program, observe current-block highlighting, and debug by retrying.

# Scope
MVP scope is limited to:
- Web target (modern browsers) with frontend in Next.js (TypeScript) under `/frontend`.
- Required backend in Node.js API (TypeScript) under `/backend`.
- SQLite as source of truth for levels and unlocked progress, with local-only persistence.
- Required endpoints: `GET /health`, `GET /api/levels`, `GET /api/levels/:id`.
- Deterministic execution model: compile blocks to internal instructions, then interpret in deterministic order.
- Gameplay loop: build sequence from `Move(1)` and `Jump`, press Run, animate dog, highlight current block, retry on failure.
- Content: exactly 10 short levels; Level 1 teaches `Move`, Level 2 teaches `Jump`; one sentence of narrative progression per level.
- Accessibility baseline: keyboard-operable core interactions, visible focus, readable text size, no color-only meaning, narration text available on screen.
- Safety/privacy baseline: no accounts, no PII collection, no external network calls required to play levels.
- Local development and integration through repo-root `docker-compose.yml`.
- Deterministic offline tests with a command documented in `TEST.md` that exits with code `0`.

# Non-Goals
- Multiplayer.
- User accounts or cloud saves.
- Complex physics simulation.
- Advanced Scratch features in MVP (variables, custom blocks).

# Acceptance Criteria
- Deterministic block execution highlights current block while running.
- Level 1 is solvable with a short `Move`/`Jump` sequence that reaches the bone.
- Frontend integrates with backend level endpoints.
- Unlocked progress persists in SQLite and restores on refresh.
- Core interactions are keyboard operable with visible focus states.
- Narration/help text for each level is available as on-screen text.
- Deterministic tests run offline and exit with code `0`.

# Risks
- Coordinate/scale mismatch against `background.png` causing incorrect move/jump distances.
- Drift between frontend behavior and backend level/progress contracts.
- Non-deterministic animation timing affecting perceived execution order.
