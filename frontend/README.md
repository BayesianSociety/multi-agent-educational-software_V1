# Frontend

Next.js + TypeScript frontend for the deterministic scratch-like game MVP.

## Features
- Level loading from backend endpoint `GET /api/levels` with local fallback data.
- Deterministic block execution (`Move`, `Jump`) with current-step highlighting.
- Keyboard-operable controls and visible focus states.
- On-screen narration text and run feedback for debugging/retry loop.

## Run locally
```bash
cd frontend
npm install
npm run dev
```

App runs at `http://localhost:3000`.
