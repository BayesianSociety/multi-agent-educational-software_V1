# Layer 0 - Non-negotiables
- **Target platform:** Web (modern browsers)
- **Audience / age range:** 7-12
- **Safety & privacy stance:** No accounts, no PII collection, no external network calls required to play levels. Local-only persistence via the app's local backend + SQLite storage.
- **Accessibility baseline:** Keyboard operable for core interactions; visible focus states; readable font sizing; avoid color-only meaning; narration text available as on-screen text.

# Layer 1 - MVP + Core learning loop
## Core learning objectives
- **Sequencing:** blocks run top-to-bottom / left-to-right in a deterministic order
- **Procedural thinking:** plan steps to reach a goal
- **Debugging:** observe step-by-step execution and adjust blocks

## Core experience / gameplay loop
- Player assembles a small program from blocks (at MVP: **Move**, **Jump**) to guide a dog to a bone.
- Player presses **Run**.
- The dog animates while the UI highlights the current block.
- If the dog fails (falls into a gap, hits obstacle, does not reach bone), player edits blocks and tries again.

## MVP scope
- 10 short levels.
- Blocks:
  - **Move(1)**: move dog forward by 1 tile
  - **Jump**: leap over a single-tile gap/obstacle
  - **Run/Start**: executes block sequence
- Minimal tutorial: Level 1 teaches Move; Level 2 teaches Jump.
- Narrative: one sentence per level + small story progression between worlds.

## Non-goals
- Multiplayer
- User accounts / cloud saves
- Complex physics simulation
- Advanced Scratch features (variables, custom blocks) in MVP

## Acceptance criteria (MVP)
- Level runner executes blocks deterministically and highlights current block during execution.
- Level 1 can be solved with a small block sequence to reach the bone.
- The app stores unlocked levels in SQLite and restores progress on refresh.
- Tests run deterministically and exit with code 0.

# Layer 2 - Architecture constraints
- **Backend REQUIRED** for MVP. The `/backend` directory MUST contain server logic.
- **Frontend framework:** Next.js (TypeScript). The Next.js app MUST live in `/frontend`.
- **Backend framework:** Node.js API (TypeScript) in `/backend`.
- **Database:** SQLite is the source of truth. Local development MUST use Docker Compose (`docker-compose.yml`).
- **Data access / migrations:** Use Prisma (or explicitly choose another migration tool).
- **Level format:** Store levels in SQLite only.
- **Execution model:** Compile blocks into a simple internal instruction list, then interpret step-by-step deterministically.
- **Persistence:** Store user progress in SQLite (no accounts required).
- **Testing:** must run offline and deterministically.
- **Security:** No secrets committed. Provide `.env.example`.
- **Frontend-backend integration:** frontend runtime flow MUST call backend API endpoint(s).
- **Required backend endpoints:** `GET /health`, `GET /api/levels`, `GET /api/levels/:id`.
- **Run documentation:** deliver exact startup commands for frontend, backend, tests, and optional docker compose usage in project docs.

# Layer 3 - Content & UX specifics
## Scratch-like editor UX
- Palette on the left, workspace in the middle, play area on the right.
- Blocks snap/connect vertically in a single sequence for MVP.
- Run starts execution; Reset returns dog to start.

## Level 1 reference
- Narrative bubble: "Taffy really wants that bone!"
- Start: dog on left side.
- Goal: bone on right side.
- Level 1: no gap, solvable with Move blocks.
- Level 2+: introduce a gap requiring Jump.

## Visual tone
- Friendly, bright, kid-safe visuals.
- `background.png` is mandatory and actor positioning/sizing plus Move/Jump distances must align to that background scale and layout.
- Simple animations: dog walks, jumps; bone sparkles on success.
