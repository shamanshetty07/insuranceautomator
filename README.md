# Vehicle Damage AI

Vehicle damage assessment and insurance-report verification. **Current scope (Phase 3)**: inspections, image capture/upload with validation and local storage, image management, and optional local YOLO damage detection (API and UI). Report generation and claim verification are wired end to end in code: local Qwen-VL adapters (`backend/app/services/vlm/`), a report service that stores each generated claim (`report_claims`, migration 0006), claim-level verification with a deterministic policy, HTTP endpoints (`/api/v1/inspections/{id}/reports`, `/api/v1/reports/{id}`, `/api/v1/reports/{id}/verifications`) and a report section on the inspection page. **Not yet validated:** no model weights are included and real inference has not been run (see `backend/scripts/vlm_smoke.py`), and the database-backed tests (including migration 0006) have not been run against a test database.

See [docs/architecture.md](docs/architecture.md) and [docs/api-design.md](docs/api-design.md).

## Stack

Pillow for image handling · React 19 + TypeScript + Vite + Tailwind 4 · FastAPI + Pydantic 2 · SQLAlchemy 2 + Alembic · PostgreSQL (Docker Compose) · Pytest / Vitest. AI models (later phases) run locally.

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

## Image storage and maintenance

Uploaded images are validated, orientation-normalized and saved under `storage/inspections/` (git-ignored) with server-generated names; see [docs/architecture.md](docs/architecture.md#image-storage-and-ingestion). Browser camera access needs `http://localhost` or HTTPS.

Check that the database and the files agree (read-only, safe to run any time):

```bash
cd backend && source .venv/bin/activate
python -m app.services.reconciliation          # human-readable
python -m app.services.reconciliation --json   # machine-readable
```

It reports missing files, orphaned files, checksum mismatches and unsafe paths, and **never changes or deletes anything**. Exit code 1 means it found problems. To resolve one, inspect the listed relative path under `STORAGE_ROOT` yourself (an orphan can be removed by hand once you are sure it is not needed; a missing file means the row should be deleted through the API or restored from backup).

Detection runs left `running` by a crashed or killed worker are cleared **manually** (nothing schedules this):

```bash
python -m app.services.detection.recovery
```

It fails only runs whose worker has sent no heartbeat for `DETECTOR_STALE_RUN_MINUTES`; a slow run that is still alive keeps beating and is never touched. It is safe to run at any time, repeatedly, and alongside workers or other recovery processes.

## Environment variables

| File | Variable | Purpose |
|---|---|---|
| `.env` | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT` | Compose database (user/password required) |
| `backend/.env` | `DATABASE_URL` (required) | `postgresql+psycopg://user:pass@host:port/db` |
| | `TEST_DATABASE_URL` | Disposable DB for pytest; **wiped on every run**; must differ from `DATABASE_URL` |
| | `CORS_ORIGINS` | JSON list of allowed origins |
| | `STORAGE_ROOT` | Image directory (default `../storage/inspections`) |
| | `MAX_UPLOAD_BYTES`, `MAX_IMAGES_PER_INSPECTION`, `MIN_IMAGE_SIDE_PX`, `MAX_IMAGE_PIXELS` | Server-side upload limits |
| | `LOG_LEVEL`, `APP_ENV` | Optional |
| | `DETECTOR_BACKEND`, `DETECTOR_WEIGHTS_PATH`, `DETECTOR_MODEL_NAME`, `DETECTOR_MODEL_VERSION` | Local detection (`none` disables it; weights are a local `.pt` file, never downloaded) |
| | `DETECTOR_IMAGE_SIZE` | Inference size; must be a positive multiple of 32 (never rounded), and of the loaded model's stride |
| | `DETECTOR_WEIGHTS_SHA256` | Optional pin: weights with any other SHA-256 are refused before loading |
| | `DETECTOR_SAFE_LOAD` | Opt-in restricted checkpoint loading; refuses to load if it cannot be guaranteed (off by default, not yet tried with a real checkpoint) |
| | `VLM_BACKEND`, `VLM_MODEL_PATH`, `VLM_MODEL_NAME`, `VLM_MODEL_VERSION`, `VLM_WEIGHTS_SHA256`, `VLM_DEVICE`, `VLM_DTYPE`, `VLM_TIMEOUT_SECONDS` | Local Qwen-VL (`none` disables it; needs the `vlm` extra and a local safetensors snapshot, never downloaded at runtime) |
| | `DETECTOR_STALE_RUN_MINUTES`, `DETECTOR_HEARTBEAT_SECONDS` | Stale-run recovery threshold and worker heartbeat interval (threshold must be >= 4 heartbeats) |
| `frontend/.env` | `VITE_API_BASE_URL` | Backend base URL (default `http://localhost:8000`) |
| | `VITE_MAX_IMAGE_BYTES`, `VITE_MAX_IMAGES_PER_INSPECTION` | Client-side limits for fast feedback; keep equal to the backend values |

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
