# Layer 0 - Non-negotiables
- Target platform: Web (modern browsers)
- Audience / age range: 7-12
- Safety & privacy stance: No accounts, no PII collection, no external network calls required to play levels. Local-only persistence via local backend + SQLite.
- Accessibility baseline: Keyboard operable core interactions, visible focus states, readable font sizing, avoid color-only meaning, narration text available as on-screen text.

# Layer 1 - MVP + Core learning loop
## Core learning objectives
- Sequencing with deterministic execution order.
- Procedural thinking through block planning.
- Debugging through step-by-step execution feedback.

## Core experience / gameplay loop
- Player assembles a program from Move and Jump blocks to guide a dog to a bone.
- Player presses Run.
- Dog animates and current block is highlighted.
- On failure, player edits blocks and retries.

## MVP scope
- 10 short levels.
- Blocks: Move(1), Jump, Run/Start.
- Tutorial: Level 1 teaches Move, Level 2 teaches Jump.
- Narrative: one sentence per level with simple progression text.

## Non-goals
- Multiplayer
- User accounts / cloud saves
- Complex physics simulation
- Advanced Scratch features (variables, custom blocks) in MVP

## Acceptance criteria
- Deterministic block execution with current-block highlighting.
- Level 1 solvable with small Move/Jump sequence to reach bone.
- App stores unlocked levels in SQLite and restores progress after refresh.
- Tests run deterministically and exit code is 0.

# Layer 2 - Architecture constraints
- Backend REQUIRED for MVP.
- Frontend: Next.js (TypeScript) in /frontend.
- Backend: Node.js API (TypeScript) in /backend.
- Database: SQLite source of truth.
- Local development uses repo-root docker-compose.yml.
- Data access / migrations: Prisma schema in /backend/prisma/schema.prisma.
- Level format: SQLite only.
- Execution model: compile blocks into internal instructions, interpret deterministically.
- Persistence: user progress in SQLite.
- Testing: offline deterministic tests with deterministic command in TEST.md.
- Security: no secrets committed, include .env.example.
- Frontend-backend integration required.
- Required endpoints: GET /health, GET /api/levels, GET /api/levels/:id.
