# How to run tests
```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

# Environments
- Local Linux/macOS shell with Python 3.11+.
- Tests are deterministic and offline.
- For full app integration, Docker Compose can be used for SQLite-backed backend services.
