# Resonance Lab — developer entry points. `./start.sh` is the one documented command for users.
PY=backend/.venv/bin/python
UV=uv

.PHONY: setup backend-setup frontend-setup dev api web build types test test-backend test-frontend smoke lint experiments

setup: backend-setup frontend-setup

backend-setup:
	cd backend && $(UV) venv .venv --python 3.11 -q || true
	cd backend && . .venv/bin/activate && $(UV) pip install -q -e ".[dev]"

frontend-setup:
	cd frontend && npm install --legacy-peer-deps

# Backend API only (port 8000)
api:
	cd backend && . .venv/bin/activate && uvicorn resonance.api:app --reload --port 8000

# Frontend dev server only (port 5173, proxies /api to 8000)
web:
	cd frontend && npm run dev

# Both, for development
dev:
	@trap 'kill 0' INT TERM; \
	( cd backend && . .venv/bin/activate && uvicorn resonance.api:app --reload --port 8000 ) & \
	( cd frontend && npm run dev ) & \
	wait

build:
	cd frontend && npm run build

# Regenerate TypeScript types from the FastAPI OpenAPI document
types:
	cd backend && . .venv/bin/activate && python -c "import json; from resonance.api import app; json.dump(app.openapi(), open('/tmp/openapi.json','w'))"
	cd frontend && npx openapi-typescript /tmp/openapi.json -o src/api/schema.d.ts

test: test-backend test-frontend

test-backend:
	cd backend && . .venv/bin/activate && python -m pytest -q

test-frontend:
	cd frontend && npx vitest run

smoke:
	cd frontend && npx playwright test

lint:
	cd backend && . .venv/bin/activate && ruff check resonance tests
	cd frontend && npx tsc -b --noEmit

experiments:
	cd backend && . .venv/bin/activate && python -m resonance.cli experiments --out ../data/exports
