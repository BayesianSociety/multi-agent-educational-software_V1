# Multi-Agent Educational Software V1

## Frontend
- `cd frontend`
- `npm install`
- `npm run dev`

## Backend
- `cd backend`
- `npm install`
- `npm run dev`

## Tests
- `python3 -m unittest discover -s tests -p 'test_*.py'`

## Docker Compose (optional)
- `docker compose up --build`

The frontend must call backend endpoints (`/health`, `/api/levels`, `/api/levels/:id`) and backend uses SQLite.
