# Architecture (Phase 1)

```
React + Vite (frontend)  ──HTTP/JSON──▶  FastAPI (backend)  ──SQLAlchemy──▶  PostgreSQL
                                              │
                                              └── local filesystem: storage/inspections/   (Phase 2+)
                                              └── local models (YOLO, Qwen-VL)             (later phases)
```

Everything runs natively except PostgreSQL, which runs via Docker Compose. AI inference is always local; no cloud inference APIs.

## Backend layout (`backend/app`)

| Path | Responsibility |
|---|---|
| `main.py` | App factory, CORS, lifespan, `/health` |
| `core/` | Settings (pydantic-settings), logging, error handlers |
| `db/` | Declarative base, engine/session management |
| `models/` | SQLAlchemy 2.x ORM models |
| `schemas/` | Pydantic request/response schemas |
| `api/` | Versioned router (`/api/v1`), route modules, dependencies |
| `services/` | Interfaces (Protocols) and value types for future AI services; no implementations |

## Data model

- **Inspection** → many **InspectionImage** → many **DamagePrediction**
- **Inspection** → many **GeneratedReport** → many **VerificationResult**
- All foreign keys use `ON DELETE CASCADE`; deleting an inspection removes everything beneath it. (Files on disk must be removed by application code in Phase 2.)
- Enumerations are stored as `VARCHAR` with a `CHECK` constraint (not native PG enums) so adding a value is a simple migration.
- `InspectionImage.storage_path` is relative to `STORAGE_ROOT` and is internal; it must not be exposed to clients.
- `(inspection_id, image_order)` is unique.

### Inspection statuses

Only `created` is assigned in Phase 1; no transitions are implemented and the API does not let clients set status.

| Status | Intended meaning |
|---|---|
| `created` | Record exists, no images yet |
| `images_pending` | Waiting for the required images |
| `ready_for_analysis` | Images present and validated |
| `processing` | Detection / report generation running |
| `review_required` | Output produced; a human should review it |
| `completed` | Reviewed and finalized |
| `failed` | Processing failed; see logs |

### Coordinate conventions (`DamagePrediction`)

- `bounding_box`: `{"x_min","y_min","x_max","y_max"}`, floats in `[0, 1]` normalized to image width/height, origin top-left, y downward, `x_min < x_max`, `y_min < y_max`. Validated by `app.schemas.damage.BoundingBox`.
- `segmentation_data`: list of polygons, each a list of `[x, y]` points in the same normalized space.
- `confidence`: `[0, 1]`, enforced by a DB check constraint.
- Normalized coordinates keep predictions valid if images are resized.

### Verification labels

`supported`, `contradicted`, `wrong_location`, `wrong_damage_type`, `unsupported_severity`, `unverifiable`, `omitted_damage`.

`omitted_damage` differs from the others: it is damage visible in the images but missing from the report, so it is **not** a claim extracted from the report. It needs a separate image-to-report comparison (`ClaimVerificationService.find_omissions`). For such rows `claim_text` describes the omitted damage rather than quoting the report.

## AI service extension points

`app/services/interfaces.py` defines `DamageDetectionService`, `VisionAnalysisService`, `ReportGenerationService`, `ClaimVerificationService` as `typing.Protocol`s, with plain dataclasses in `services/types.py`. There are no implementations, no model loading, and no network calls in Phase 1.

## Frontend layout (`frontend/src`)

`components/` shared UI, `features/<area>/` feature code (only `inspections/` exists yet; `images`, `damage`, `reports`, `verification` are added as those phases arrive), `lib/` API client and helpers, `types/` API types.
