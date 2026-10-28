# Vehicle Damage AI

Vehicle damage assessment and insurance-report verification. **Phase 1 only**: project foundation, database schema, and inspection CRUD (create/list/get). No image upload, damage detection, report generation or verification exists yet.

See [docs/architecture.md](docs/architecture.md) and [docs/api-design.md](docs/api-design.md).

## Stack

React 19 + TypeScript + Vite + Tailwind 4 · FastAPI + Pydantic 2 · SQLAlchemy 2 + Alembic · PostgreSQL (Docker Compose) · Pytest / Vitest. AI models (later phases) run locally.

## Prerequisites (macOS)

- Python **3.11+** (developed on 3.14)
- Node.js 20+ and npm
- Docker Desktop (for PostgreSQL), or any PostgreSQL 14+ you manage yourself

## Setup and run

Run each in its own terminal, from the repo root.

### 1. Database

```bash
cp .env.example .env          # then set POSTGRES_PASSWORD (and POSTGRES_PORT if 5432 is taken)
docker compose up -d db
docker compose ps             # wait for "healthy"
```

Data persists in the `pgdata` Docker volume. Create the test database once:

```bash
docker compose exec db createdb -U "$POSTGRES_USER" vehicle_damage_test   # or run the SQL via psql
```

### 2. Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # set the password/port in DATABASE_URL and TEST_DATABASE_URL to match step 1
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev                   # http://localhost:5173
```

## Environment variables

| File | Variable | Purpose |
|---|---|---|
| `.env` | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT` | Compose database (user/password required) |
| `backend/.env` | `DATABASE_URL` (required) | `postgresql+psycopg://user:pass@host:port/db` |
| | `TEST_DATABASE_URL` | Disposable DB for pytest; **wiped on every run**; must differ from `DATABASE_URL` |
| | `CORS_ORIGINS` | JSON list of allowed origins |
| | `LOG_LEVEL`, `APP_ENV`, `STORAGE_ROOT` | Optional |
| `frontend/.env` | `VITE_API_BASE_URL` | Backend base URL (default `http://localhost:8000`) |

## Tests and checks

```bash
# Backend (database tests are skipped, with a reason, unless TEST_DATABASE_URL is set)
cd backend && source .venv/bin/activate
pytest -q                     # export TEST_DATABASE_URL first, or put it in your shell env

# Frontend
cd frontend
npm test
npm run typecheck
npm run build
```

Note: `TEST_DATABASE_URL` is read from the process environment (not `backend/.env`), e.g. `export TEST_DATABASE_URL=postgresql+psycopg://...`.

## Git safety

`.gitignore` excludes env files, virtualenvs, `node_modules`, logs, model weight files (`*.pt`, `*.safetensors`, `*.gguf`, …) and everything under `storage/inspections/` except a placeholder.
