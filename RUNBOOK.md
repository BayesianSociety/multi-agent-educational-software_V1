# Runbook

All commands below are run from repo root unless noted.

## Startup (local processes)
1. Create environment file: `cp .env.example .env`
2. Start backend:
   - `cd backend && npm install && npm run dev`
3. Start frontend:
   - `cd frontend && npm install && npm run dev`
4. Verify backend health:
   - `curl -fsS http://localhost:3001/health`

## Startup (Docker Compose)
- `docker compose up --build`
- Backend is available on `http://localhost:3001`
- Frontend is available on `http://localhost:3000`
- Stop stack: `docker compose down`

## Release validation checklist
1. Run deterministic tests:
   ```bash
   python3 -m unittest discover -s tests -p 'test_*.py'
   ```
   - Expected: exit code `0`
2. Verify endpoint contract:
   - `curl -fsS http://localhost:3001/health`
   - `curl -fsS http://localhost:3001/api/levels`
   - `curl -fsS http://localhost:3001/api/levels/1`
   - Expected: all commands exit with code `0` and return JSON payloads.
   - Frontend must consume backend level APIs at `/api/levels` and `/api/levels/:id`.
3. Verify SQLite-backed persistence:
   - Unlock at least one level in the app.
   - Refresh the browser.
   - Confirm unlocked progress is restored.

## Troubleshooting
- If frontend cannot load levels, verify backend is running and `/api/levels` returns `200`.
- If persistence fails, verify SQLite path, file permissions, and mounted backend volume.
- If tests fail intermittently, remove timing assumptions and enforce deterministic step order.
- Follow deterministic recovery steps below when retrying a failed release step.

## Deterministic recovery
- Re-run orchestrator with verbose mode and inspect `/.orchestrator/runs/<run_id>/steps/*/attempt_*.json`.
- Revert unauthorized changes using `git restore` and delete unauthorized untracked files from the latest run window.
- Re-run failed step with same policy state to preserve deterministic retry behavior.
