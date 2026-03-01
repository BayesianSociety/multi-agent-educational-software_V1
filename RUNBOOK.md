# Runbook

## Startup
- Frontend: `cd frontend && npm install && npm run dev`
- Backend: `cd backend && npm install && npm run dev`
- Tests: `python3 -m unittest discover -s tests -p 'test_*.py'`
- Optional local containers: `docker compose up --build`

## Troubleshooting
- If frontend cannot load levels, verify backend process is running and `/api/levels` returns 200.
- If persistence fails, verify SQLite file path/mount and backend write permissions.
- If tests fail intermittently, remove timing assumptions and enforce deterministic step order.

## deterministic recovery
- Re-run orchestrator with verbose mode and inspect `/.orchestrator/runs/<run_id>/steps/*/attempt_*.json`.
- Revert unauthorized changes using git restore and delete unauthorized untracked files from latest run window.
- Re-run failed step with same policy state to preserve deterministic retry behavior.
