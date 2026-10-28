# API design

Interactive docs: `http://localhost:8000/docs` (OpenAPI JSON at `/openapi.json`).

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Process liveness only. Does **not** check the database. |
| GET | `/api/v1/health` | API liveness. |
| GET | `/api/v1/health/db` | Runs `SELECT 1`. `200 {"status":"ok","database":"reachable"}` or `503 {"status":"unavailable","database":"unreachable"}`. |
| GET | `/api/v1/inspections?limit=20&offset=0` | Newest first. `limit` 1–100. Returns `{items,total,limit,offset}`; `items` is `[]` when empty. |
| POST | `/api/v1/inspections` | Body `{title?, notes?}`; `title` ≤ 200 chars, `notes` ≤ 5000. Whitespace is trimmed, blank → `null`. Unknown fields rejected. `201` with the inspection. |
| GET | `/api/v1/inspections/{id}` | `404` if unknown, `422` if `id` is not a UUID. |

## Errors

All non-2xx responses (except the `/health/db` 503 body above) use:

```json
{"error": {"code": "validation_error", "message": "…", "details": [{"field": "body.title", "message": "…"}]}}
```

Codes: `validation_error` (422), `not_found` (404), `method_not_allowed` (405), `http_error` (other HTTP errors), `internal_error` (500; details are logged, not returned).

## Conventions

- IDs are UUIDs; timestamps are ISO 8601 with a timezone offset (the database session timezone).
- Versioned under `/api/v1`; breaking changes get a new prefix.
- CORS origins come from `CORS_ORIGINS`; only GET/POST and `Content-Type` are allowed for now.
