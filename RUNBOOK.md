# Runbook

## Troubleshooting
- If `docker compose up -d` fails, check Docker daemon status and rerun.
- If backend fails health checks, inspect backend logs and verify SQLite path/mount.
- If frontend cannot load levels, confirm backend endpoint availability and CORS policy.

## Deterministic Recovery
- Re-run `orchestrator.py` with `--verbose` to capture deterministic step logs.
- Use orchestrator rollback artifacts in `.orchestrator/runs/<run_id>/steps/*` to identify allowlist or validator failures.
- Restore deterministic state by re-running the failed step under orchestrator-managed revert logic.
