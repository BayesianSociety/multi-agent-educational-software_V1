# Layer 0
## Product identity
This product is a browser-based educational coding puzzle game named Pet Vet Coding Puzzles.

## Target learners
The target audience is beginner learners who can read simple instructions.

## Safety and privacy
No personal data collection is required for gameplay. Analytics are local to this application environment and can be used by a teacher or observer. The application must not require third party authentication for core gameplay.

## Accessibility baseline
The user interface must provide large controls, clear labels, keyboard navigation for major controls, readable contrast, and no critical information that is conveyed only by color.

## Platform constraints
The game must run in a modern desktop browser without requiring native installation.

# Layer 1
## Learning model
The game must teach sequencing, loops, and conditionals through puzzle progression.

## Core loop
1. Open puzzle.
2. Read story and goal.
3. Drag blocks into connected sequence under On Start.
4. Press Play.
5. Observe animation outcome.
6. If failure: show Oops and helpful hint.
7. If success: mark puzzle complete and unlock next puzzle.

## Required puzzle count
Exactly seventeen playable puzzles.

## Scope in this minimum viable product
- Puzzle progression with lock and unlock state.
- Visual block workspace with command library and active code area.
- Deterministic execution engine for movement, actions, loops, and conditionals.
- Hint system tied to explicit failure reasons.
- SQLite telemetry for movement events and all significant user and runtime events.
- Local analytics views for replay and event inspection.

## Non-goals in this minimum viable product
- Multiplayer gameplay.
- Paid accounts and online leaderboards.
- Proprietary brand characters and proprietary logos.

## Acceptance criteria
- Seventeen puzzles are playable.
- Incorrect programs show Oops and at least one helpful hint.
- Correct programs complete goals and unlock progression.
- Movement and execution events are persisted in SQLite without dropping execution or movement steps.
- Analytics pages can replay movement from persisted movement records.

# Layer 2
## System architecture
Frontend and backend are required.
- Frontend: browser application with block workspace and game scene rendering.
- Backend: Hypertext Transfer Protocol service with SQLite persistence and analytics endpoints.

## Required backend endpoints
- GET /health
- GET /api/levels
- GET /api/levels/:id
- POST /api/session/start
- POST /api/session/end
- POST /api/events/batch
- GET /api/analytics/dashboard
- GET /api/analytics/puzzle/:id
- GET /api/analytics/events

## Frontend backend integration
The frontend must use backend endpoints for level retrieval and telemetry submission in runtime flow.

## Data model constraints
Puzzle data must be data-driven and include id, title, storyText, goalText, scene, grid, entities, availableBlocks, constraints, successCriteria, and hintRules.

## Execution constraints
- Execute connected blocks under On Start only.
- Disconnected blocks do not execute and must trigger a warning.
- Apply deterministic loop safety cap.
- Movement collision must fail run with specific failure reason.

## Persistence constraints
SQLite must automatically initialize on first backend start.
Required tables include users, sessions, puzzles, attempts, events, movements, and puzzle_progress.

## Test and validation constraints
- Deterministic local test suite must pass before acceptance.
- Product acceptance must verify persistence, movement tracking, and analytics replay capability.
- Documentation must include exact local startup and test commands.

# Layer 3
## Narrative and visual direction
The theme is a busy veterinary clinic day with original mentor and pet characters.

## User interface structure
- Top bar with puzzle index and progress.
- Main scene area with mentor speech and goal banner.
- Workspace overlay with left command library and right code area.
- Clear Play and Reset controls.

## Puzzle progression details
- Puzzles one through five: sequencing basics.
- Puzzles six through ten: loops.
- Puzzles eleven through seventeen: conditionals and mixed logic.

## Hint quality requirement
Hints must be tailored to deterministic failure reasons, without revealing complete solutions immediately.
