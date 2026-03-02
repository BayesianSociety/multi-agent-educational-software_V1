# How to run tests
Run from repo root:
```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

# Determinism guarantees
- Test suite is offline and does not require internet access.
- Command must exit with code `0`.
- Run with a clean working tree for reproducible validation reports.

# Environments
- Local Linux/macOS shell with Python 3.11+.
- For full app integration, Docker Compose can be used for SQLite-backed backend services.

# Optional integration smoke checks (local, deterministic)
Start app services from repo root:
```bash
docker compose up --build -d
```

Validate required backend endpoints:
```bash
curl -fsS http://localhost:3001/health
curl -fsS http://localhost:3001/api/levels
curl -fsS http://localhost:3001/api/levels/1
```

Stop services:
```bash
docker compose down
```
