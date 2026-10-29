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

```
Inspection ─┬─ InspectionImage ── DetectionRun ── DamagePrediction
            └─ GeneratedReport ─┬─ ReportPrediction ──▶ DamagePrediction   (exact evidence used)
                                └─ VerificationRun ── VerificationResult
```

- **DetectionRun**: one execution of one detection model/config on one image. Owns the model name/version, `config` (JSONB), optional `weights_sha256`, `status`, `error_message`, `created_at`, `completed_at`. A run may produce zero predictions. Re-running detection with another model or threshold creates a new run, so configurations can be compared on the same image.
- **DamagePrediction** belongs to exactly one run. The image, model and parameters are *not* repeated on it; they come from the run.
- **GeneratedReport** records `model_name`/`model_version`, `evidence_strategy`, `generation_config` (JSONB), and the exact predictions used via **ReportPrediction** rows.
- **ReportPrediction means "this prediction was supplied as evidence to the report-generation model"**, not "the report mentions it". If three predictions are supplied and the text discusses two, all three are linked. Which predictions the text actually covers (report coverage, omitted damage) is a separate, later research concern and must not be encoded in this table.
- **VerificationRun**: one execution of one verifier config over one report (`verifier_model`, `verifier_version`, `config`, `status`, timestamps). **VerificationResult** belongs to exactly one run, so re-verifying never mixes results; the report is reached through the run.
- **InspectionImage.sha256**: optional lowercase hex SHA-256 of the stored file, set by the Phase 2 upload code (image bytes stay on the filesystem, never in PostgreSQL).
- Foreign keys use `ON DELETE CASCADE`, so deleting an inspection removes everything beneath it. **Exception:** `report_predictions.damage_prediction_id` is `NO ACTION` and checked at commit, so a prediction or run a report depends on cannot be deleted on its own, but deleting the whole inspection works.
- Cascades delete database rows only. Files under `STORAGE_ROOT` are not touched, so the Phase 2 delete code must remove them (collect paths, delete rows, then delete files) and a reconciliation job should find orphans in either direction.
- Enumerations (inspection status, run status, evidence strategy, verification label) are `VARCHAR` with a `CHECK`, so adding a value is a simple migration.
- `InspectionImage.storage_path` is relative to `STORAGE_ROOT`, internal, and never exposed to clients. `(inspection_id, image_order)` is unique.
- Service-layer invariants (not expressible as plain FKs), enforced in `app/services/provenance.py`: a prediction belongs to exactly the run that produced it (NOT NULL FK), and a report may link only predictions whose run's image belongs to **the report's own inspection**. A report for inspection A can never reference a prediction from inspection B; mixed batches are rejected atomically.
  - *Considered, not adopted:* a database-level guarantee would need `inspection_id` copied into `detection_runs`, `damage_predictions`, `generated_reports` and `report_predictions` with composite foreign keys (or a constraint trigger). That denormalizes three tables to protect one write path that currently has a single service function, so it was deferred. Revisit if more write paths appear.
- Human ground-truth annotations are intentionally **not** modelled yet. When added they will live in separate tables, never mixed into model-generated `verification_results`.

### Run status

`pending → running → completed | failed`. `completed_at` is set exactly when a run is `completed` or `failed` (CHECK constraint).

### One canonical place per fact

| Fact | Where |
|---|---|
| Detection model name/version, params, weights checksum | `detection_runs` |
| Image a prediction is on | `detection_runs.inspection_image_id` |
| Report model name/version | `generated_reports.model_name/model_version` |
| Report evidence strategy | `generated_reports.evidence_strategy` |
| Predictions a report used | `report_predictions` (not repeated in JSON) |
| Prompt version, sampling, strategy settings | `generated_reports.generation_config` |
| Verifier model/version/params | `verification_runs` |

### JSON configuration shapes

Validated on write by `app/schemas/runs.py` (unknown keys are rejected); the database stores the JSON as given.

- `DetectionConfig`: `confidence_threshold`, `iou_threshold`, `image_size`, `max_detections`, optional `class_filter`, `device`.
- `GenerationConfig`: `prompt_version`, `sampling` (`temperature`, `top_p`, `max_new_tokens`, optional `seed`), `evidence_settings` (`crop_padding` for cropped regions, `mask_opacity` for mask overlays).
- `VerificationConfig`: `prompt_version`, `sampling`.

### Comparing evidence strategies

`evidence_strategy` is one of `full_image`, `cropped_region`, `image_with_segmentation_mask`.

| Strategy | Model input | Needs from detection | Strategy-specific settings |
|---|---|---|---|
| `full_image` | Whole image | Nothing (predictions may still be listed as text) | none |
| `cropped_region` | Crops of predicted boxes (plus optionally the full image) | `bounding_box` | `crop_padding` |
| `image_with_segmentation_mask` | Image with mask overlay | `segmentation_data` | `mask_opacity` |

To compare strategies fairly, generate reports for the same inspection with the same `model_name`/`model_version`, `prompt_version` and `sampling`, differing only in `evidence_strategy`, and link every report to the same `DetectionRun`'s predictions via `report_predictions`. Then verify each report with the same verifier config and group `verification_results` by `generated_reports.evidence_strategy`. For error-propagation studies, generate the same report from predictions of two different detection runs and compare the resulting verification labels.

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

## Image storage and ingestion

**Where.** Files live under `STORAGE_ROOT` (default `storage/inspections/`, git-ignored). The database stores metadata only; image bytes are never in PostgreSQL.

**Paths.** `storage_path` is relative and always server-generated: `<inspection_uuid>/<image_uuid>.<jpg|png|webp>`. The client's filename is stored only as display text (`original_filename`, directory parts and control characters stripped) and is never part of a path. Every path is re-validated on use: it must match that exact pattern, must not be or traverse a symlink, and must resolve inside the root. Files are created with `O_EXCL | O_NOFOLLOW` (never overwrite) and fsynced. API responses never include `storage_path` or absolute paths.

**Validation and normalization** (`image_processing.py`, Pillow). Only decoded content counts; the declared MIME type and extension are ignored (a PNG sent as `.jpg` is stored as `image/png`). Accepted: JPEG, PNG, WebP (HEIC/GIF/BMP are rejected). The header is read first and dimensions are checked *before* pixel decoding (min side `MIN_IMAGE_SIDE_PX`, max `MAX_IMAGE_PIXELS`, decompression bombs rejected). The image is then fully decoded (truncated or corrupt files are rejected), EXIF orientation is **applied to the pixels** (`ImageOps.exif_transpose`), and the result is re-encoded in the same format (JPEG/WebP quality 95, so JPEG/WebP are re-compressed once; PNG is lossless). No EXIF is written, so GPS, camera make/model and timestamps are dropped; only an embedded ICC colour profile is kept. Animated images keep their first frame.

**Orientation and coordinates.** `exif_orientation` stores the original EXIF value (1–8, or NULL when absent or invalid) as metadata only and must never be applied again. `width_px`/`height_px` are the size of the stored, already-rotated image. YOLO will run on this stored image, and bounding boxes are normalized to it: origin top-left, x to the right, y downward, values in [0, 1]. The UI displays the stored image without any CSS rotation, so future overlays line up.

**Checksum.** `sha256` is computed over the exact bytes written to disk (the normalized file), not the upload. It is NULL only for rows that predate Phase 2. `(inspection_id, sha256)` is unique, so the same normalized content cannot be stored twice in an inspection (HTTP 409 `duplicate_image`).

**Limits.** `MAX_UPLOAD_BYTES` (streamed in chunks; rejected with 413 as soon as exceeded) and `MAX_IMAGES_PER_INSPECTION` are enforced by the server; the frontend mirrors them via `VITE_*` variables for quick feedback only. Starlette spools multipart bodies to a temporary file before the handler runs, so also set a request-size limit at your reverse proxy in production.

**Upload workflow.** Check inspection exists (404) → read with size limit → validate/normalize → lock the inspection row (`SELECT … FOR UPDATE`) so concurrent uploads cannot collide on `image_order`, exceed the count limit, or duplicate content → write file → insert row → commit. If the insert or commit fails, the new file is deleted; if that cleanup also fails, an ERROR log with the relative path (`orphan needs reconciliation`) is emitted. Order is `max(image_order)+1`, so deleting an image leaves a gap but never reuses a number.

**Deletion workflow.** Collect paths → delete rows → commit → only then delete files. A failed or rolled-back transaction never deletes a file. A file-delete failure after commit is logged and leaves an orphan the reconciliation report will list. `DELETE …/images/{id}` returns 409 `image_in_use` if a report's provenance references a prediction on that image. `delete_inspection()` exists as a service for cleanup but has no HTTP endpoint.

**Reconciliation** (report-only). `python -m app.services.reconciliation [--json]` lists database rows whose file is missing, files with no row, checksum mismatches, unsafe database paths, skipped symlinks, and rows with no checksum. It never modifies or deletes anything, only scans inside `STORAGE_ROOT`, and never follows symlinks. Exit code 1 means problems were found. See the README for how to act on the output.

## AI service extension points

`app/services/interfaces.py` defines `DamageDetectionService`, `VisionAnalysisService`, `ReportGenerationService`, `ClaimVerificationService` as `typing.Protocol`s, with plain dataclasses in `services/types.py`. Only `DamageDetectionService` has an implementation (Phase 3, below); the others have none yet. All inference runs locally.

## Damage detection (Phase 3)

**Pieces.** `services/detection/ultralytics_detector.py` implements `DamageDetectionService` with a local YOLO detection weights file. `services/detection/provider.py` builds it from settings (`DETECTOR_*`) and caches one instance per process. `services/detection_service.py` runs it on an inspection's images and persists the results. Routes are in `api/routes/detections.py`. The protocol carries `model: ModelInfo` (name, version, `weights_sha256`) and `config: DetectionConfig` (the parameters actually applied), so every run records exactly what produced it.

**Optional dependency and licensing.** Ultralytics is in the optional `detection` extra (`pip install -e ".[detection]"`), pinned to the version verified on Python 3.14 / macOS arm64 (ultralytics 8.4.191 with torch 2.15.0 on CPU, about 1.2 GB installed). It is imported lazily: the app and the test suite run without it, and `GET /api/v1/detector` then reports it as unavailable. Ultralytics is **AGPL-3.0**: running it in a network service you distribute or expose may oblige you to publish your source under the AGPL, unless you hold an Ultralytics enterprise licence. Check this before deployment. Pretrained weights carry their own licences (and their training data's), separate from the code.

**Local only.** Before importing Ultralytics the adapter sets `YOLO_OFFLINE=1` (disables its usage telemetry and online checks) and `YOLO_AUTOINSTALL=0` (no runtime pip installs). The weights path must be an existing regular file, checked before loading, because given a missing path Ultralytics would try to download a model by that name. Weights `.pt` files are Python pickles: **loading one can execute arbitrary code**, so only configure weights from a source you trust. Importing Ultralytics writes its own settings file to its config directory (`YOLO_CONFIG_DIR`, default the user config directory).

**Coordinates.** The adapter decodes the *stored* (already EXIF-normalized) image with Pillow, the same decoder used at upload, and passes the pixels to the model. Boxes come from Ultralytics `boxes.xyxyn`, which is already normalized to that image (origin top-left, x right, y down). They are clamped to [0, 1]; boxes that are empty after clamping or contain non-finite numbers are dropped. Boxes are validated again with `BoundingBox` before they are stored. Class names are stored verbatim from the model as `damage_type` (≤ 100 characters, checked at load). Mapping model classes to a project damage taxonomy is a separate, later decision.

**Run lifecycle.** For each image: insert a `DetectionRun` as `running` and commit; run inference; then commit either `completed` with its `DamagePrediction` rows, or `failed` with no predictions and a short client-safe `error_message`. Detector exceptions are logged server-side with run and image ids, never image contents. Because the `running` row is committed first, a process crash leaves a visible stale `running` run rather than nothing; nothing sweeps these yet. A missing or unsafe file fails that run without calling the model. If the image is deleted while its run is in progress, the run is deleted with it (cascade) and is omitted from the response. A `pending` status exists in the schema but is unused while detection is synchronous.

**Timing.** `created_at` and `completed_at` are set from the application's wall clock (not the database's transaction-start `now()`), so `duration_ms = completed_at − created_at` covers inference plus persistence for that image. On the development machine (Apple M4, CPU), YOLO11n at 640 px took about 1 s to load, about 2.5 s for the first inference (warm-up) and about 25 ms for each later one.

**Concurrency and limits.** Requests are synchronous and sequential per image, so a request takes roughly images × inference time; the image count is bounded by `MAX_IMAGES_PER_INSPECTION` and `image_ids` by 100. Inference is serialized by a process-wide lock because Ultralytics models are not thread-safe; loading is serialized by another, so concurrent first requests load the model once. Concurrent detection requests for the same images are allowed and simply create separate runs. A failed load is not cached, so fixing the weights file takes effect on the next request. For long batches or several workers, move detection to a background job queue; the `pending` status is reserved for that.

**Reproducibility.** Each run stores `model_name`, `model_version`, `weights_sha256` (computed from the weights file at load) and the full `DetectionConfig`. With a fixed weights file, configuration and stored image, CPU inference is expected to be deterministic. GPU/MPS inference may differ in the last digits.

## Frontend layout (`frontend/src`)

`components/` shared UI, `features/<area>/` feature code (only `inspections/` exists yet; `images`, `damage`, `reports`, `verification` are added as those phases arrive), `lib/` API client and helpers, `types/` API types.

## Claim contracts and evidence policy (Sprint 1; no inference yet)

Typed contracts exist for the planned report-verification phase; no report generation, Qwen inference or verification endpoint is implemented.

- `app/schemas/claims.py`: `ClaimType`, `Verdict` (`supported`, `contradicted`, `insufficient_evidence`), `EvidenceReference`, `ExtractedClaim`, `ClaimExtractionResult`. All strict (`extra="forbid"`) and frozen. A `Verdict` never encodes human-review routing. An empty claims list means "nothing extracted", never "no problems". Report text stays in `GeneratedReport.content`.
- `app/services/claim_policy.py`: pure `evaluate_claim()` returning a verdict, a separate `needs_human_review` flag, and a stable `ReasonCode`. First matching rule wins (order in the module docstring). It fails closed: unknown claim types, missing or unchecked evidence, an absent verifier verdict, and hidden-damage, cost, cause and unvalidated-severity claims all yield `insufficient_evidence` with review. Detector confidence is deliberately not an input, and a missing detection never establishes absence. Disagreement between verifier and detection match is never promoted to `supported`.
- Not yet connected to any database table: `VerificationResult` still has the older seven-value `VerificationLabel`; mapping to the three verdicts needs a later migration.

**Detection run metadata.** `DetectionRun.config` now also stores `runtime: {ultralytics, torch}` versions, read from package metadata (never by importing the packages). Server-written only; older rows without it remain valid.

**Stale runs.** Runs left `running` by a hard crash are cleared by `python -m app.services.detection.recovery` (age threshold `DETECTOR_STALE_RUN_MINUTES`, default 30). Age is the only signal, so with several workers a still-active run older than the threshold would also be failed.

**Restricted checkpoint loading.** `DETECTOR_SAFE_LOAD=true` (default off) sets `ULTRALYTICS_SAFE_LOAD=1` before the first Ultralytics import. Read from the installed 8.4.191 source; needs torch >= 2.6 and has not been tried with a real damage checkpoint.

## Claim-level verification (Sprint 2; model-independent)

Implemented and tested with development stubs only. No Qwen or other model is wired in, and there are no HTTP endpoints.

**Schema (migration 0004).** `verification_results` keeps every existing column and adds `claim_id`, `claim_type`, `verdict` (`supported` / `contradicted` / `insufficient_evidence`), `needs_human_review` (routing, never part of the verdict) and `policy_reason_codes`. `verification_label` is now nullable and holds the **raw verifier label**; `verdict` is the **final policy outcome**. `evidence_reference` holds an `EvidenceRecord` (association, integrity, detection match, resolved ids, crops, problem codes). `(verification_run_id, claim_id)` is unique when `claim_id` is set; a partial index serves the review queue. The CHECK value lists are frozen in the migration, and a test fails if they drift from the Python enums.

**Legacy label -> verdict** (backfill and `app.services.verdict_mapping.legacy_verdict`, kept in agreement by a test):

| Legacy label | Verdict |
|---|---|
| `supported` | `supported` |
| `contradicted`, `wrong_location`, `wrong_damage_type` | `contradicted` |
| `unverifiable` | `insufficient_evidence` |
| `unsupported_severity` | none (ambiguous: overstated vs. not established) |
| `omitted_damage` | none (a coverage finding, not a verdict on a report claim; a CHECK forbids giving it one) |

Downgrade is lossy: new columns are dropped and NULL labels become `unverifiable`.

**Evidence association** (`services/evidence_association.py`) is read-only and returns exactly one result per claim. A bad reference makes the claim `unresolved` with a problem code; it never drops or reorders claims. Cross-inspection images and predictions are treated as unknown. Integrity checks file presence, checksum, decodability and stored dimensions. `detection_match` is a signal only: `matches_claim` means the claim cites a real detection on the cited image (citation consistency, not a class or region comparison), `no_detection` needs every cited image to have a completed run with zero predictions, and `conflicts_with_claim` is never produced yet.

**Orchestration** (`services/verification_service.py::verify_report`). The run is committed as `running`, evidence is read and that transaction ended, the verifier is called with no transaction open, and results are written in one short transaction. The verifier sees only the claim text and the cited original images, never the report or detection boxes, and is not called when the policy already decides the claim. A verifier exception fails that claim only; the run is `failed` only if every verifier call failed. Interruption marks the run `failed` and re-raises. Association is checked before integrity in the policy so the reason code names the real cause.

**Stubs** (`services/dev_stubs.py`) are deterministic test doubles, not AI. Model names start with `stub-`.

**Not yet done:** report persistence from structured output, the reconciled report, a human-review table, endpoints, a semantic detection comparison, and the raw verifier's own `evidence_reference` (discarded).
