#!/usr/bin/env bash
# One-command start for Resonance Lab: builds the frontend, serves it and the API on http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"

if [ ! -d backend/.venv ]; then
  echo "[start] creating Python environment"
  (cd backend && uv venv .venv --python 3.11 -q 2>/dev/null || python3 -m venv .venv)
fi
if ! backend/.venv/bin/python -c "import fastapi, numpy, yaml" 2>/dev/null; then
  echo "[start] installing backend dependencies"
  (cd backend && (uv pip install -q -e ".[dev]" --python .venv/bin/python 2>/dev/null || .venv/bin/pip install -q -e ".[dev]"))
fi
if [ ! -d frontend/node_modules ]; then
  echo "[start] installing frontend dependencies"
  (cd frontend && npm install --legacy-peer-deps)
fi
if [ ! -f frontend/dist/index.html ] || [ "${REBUILD:-0}" = "1" ]; then
  echo "[start] building frontend"
  (cd frontend && npm run build)
fi
echo "[start] Resonance Lab -> http://localhost:${PORT}  (API docs: http://localhost:${PORT}/docs)"
cd backend && exec .venv/bin/uvicorn resonance.api:app --host 0.0.0.0 --port "${PORT}"
