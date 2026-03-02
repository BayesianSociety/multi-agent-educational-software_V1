# Multi-Agent Educational Software V1

Deterministic, local-first educational game MVP for ages 7-12.

## Prerequisites
- Node.js 20+
- npm 10+
- Python 3.11+
- Docker + Docker Compose plugin (optional for containerized startup)

## Environment
1. Copy `.env.example` to `.env` at repo root.
2. Keep `DATABASE_URL` pointed at local SQLite (`file:./dev.db`).
3. Do not place secrets in `.env` for this MVP; local-only defaults are expected.

## Local startup
### Backend
```bash
cd backend
npm install
npm run dev
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Endpoint contract
Backend must expose:
- `GET /health`
- `GET /api/levels`
- `GET /api/levels/:id`

Frontend integration requirement:
- Frontend reads level data from backend `GET /api/levels` and level detail from `GET /api/levels/:id`.

## Tests
Run the deterministic offline suite:
```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Docker Compose (optional)
```bash
docker compose up --build
```

## Deterministic validation
Run the required offline validator from repo root:
```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
Expected result: exit code `0`.

Optional endpoint smoke checks (with backend running):
```bash
curl -fsS http://localhost:3001/health
curl -fsS http://localhost:3001/api/levels
curl -fsS http://localhost:3001/api/levels/1
```

## Safety and privacy
- No accounts and no PII collection.
- Local-only persistence via backend + SQLite.
- No external network calls are required to play levels.
