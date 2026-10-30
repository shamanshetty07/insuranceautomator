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

### Images

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/inspections/{id}/images` | `multipart/form-data` with one `file` field (JPEG, PNG or WebP). One image per request; clients upload several by repeating the call. `201` with image metadata. |
| GET | `/api/v1/inspections/{id}/images` | `{items, total}` ordered by `image_order`. |
| GET | `/api/v1/inspections/{id}/images/{image_id}` | Metadata. `404` if the image belongs to a different inspection. |
| GET | `/api/v1/inspections/{id}/images/{image_id}/content` | The normalized image bytes with its validated `Content-Type`, `nosniff`. |
| DELETE | `/api/v1/inspections/{id}/images/{image_id}` | `204`; `409 image_in_use` if report provenance references it. |

Image metadata: `id, inspection_id, original_filename, media_type, file_size_bytes, sha256, width_px, height_px, exif_orientation, image_order, created_at, content_url`. `width_px`/`height_px` describe the stored, orientation-normalized image.

Upload errors: `404 not_found`, `413 file_too_large`, `415 unsupported_media_type`, `422 empty_file | invalid_image | invalid_dimensions | validation_error`, `409 image_limit_reached | duplicate_image`.

Every request body is capped before parsing at `MAX_UPLOAD_BYTES` + 64 KiB (multipart framing): a larger `Content-Length` gets `413 file_too_large`, and a body without `Content-Length` (chunked) gets `411 length_required`. Multi-picture JPEGs (MPO, from many phone cameras) are stored as a plain JPEG of their first frame; JPEG comments are stripped along with EXIF/XMP.

### Damage detection

| Method | Path | Notes |
|---|---|---|
| GET | `/api/v1/detector` | `{available, backend, model_name, model_version, weights_sha256, classes, config, reason}`. Never loads the model: `weights_sha256` and `classes` are `null` until the first detection request has loaded it. `reason` explains `available: false`. No filesystem paths are returned. |
| POST | `/api/v1/inspections/{id}/detections` | Optional body `{"image_ids": [uuid, …]}` (1–100 ids; omit or `null` for every image, in `image_order`). Synchronous: one detection run per image, run one after another. `200 {items: [run]}` with the runs this call created, each `completed` or `failed`. |
| GET | `/api/v1/inspections/{id}/detections?latest=true` | `latest=true` (default): the newest run per image, in `image_order`. `latest=false`: every run, newest first. |

Run: `{id, inspection_image_id, model_name, model_version, weights_sha256, config, status, error_message, created_at, completed_at, duration_ms, predictions}`; `config` is `{confidence_threshold, iou_threshold, image_size, max_detections, class_filter, device}`; `duration_ms` is `completed_at − created_at` (wall clock, `null` until finished). Prediction: `{id, damage_type, confidence, bounding_box}` sorted by confidence, highest first; `bounding_box` is `{x_min, y_min, x_max, y_max}` normalized to the stored image (see architecture.md).

A failure on one image does not fail the request: that run is `failed` with a short, client-safe `error_message` (`The image file is not available.`, `Detection failed for this image.`, `The detector returned …`); details are only logged. A `completed` run with `predictions: []` means the model found nothing above the threshold.

Detection errors: `404 not_found` (inspection, or an `image_ids` entry that is not an image of this inspection), `409 no_images`, `422 validation_error` (empty `image_ids`, malformed ids, unknown body fields), `503 detector_unavailable` (no detector configured or the model failed to load; no runs are created). Validation (404/409/422) happens before the model is loaded.

### Reports and claim verification

Synchronous, like detection (no job queue exists). All output is a model prediction, not ground truth.

| Method | Path | Notes |
|---|---|---|
| GET | `/api/v1/vlm` | `{available, backend, reason, model_name, model_version, weights_sha256}`. Never loads the model. |
| POST | `/api/v1/inspections/{id}/reports` | Optional body `{}` (no options; unknown fields → 422). Generates one report from every image (in `image_order`) and the latest detection run per image, stores it with its claims exactly as generated, and returns `201` with the report detail (`verification: null`). Nothing is stored if any step fails. Does not verify. |
| GET | `/api/v1/inspections/{id}/reports` | `{items: [summary]}`, newest first. |
| GET | `/api/v1/reports/{id}` | Report detail: the claims as generated, each with the result of the report's **latest** verification run only (`result: null` when not verified). |
| POST | `/api/v1/reports/{id}/verifications` | Optional body `{}`. Verifies the stored claims in a new run and returns `201` with the report detail. The claims are never modified. A run in which every model call failed is still returned (`status: failed`). |

Summary: `{id, inspection_id, content, model_name, model_version, evidence_strategy, created_at, claim_count}`. Detail adds `generation_config` (prompt version, sampling, model weights digest and package versions, limits, and the image / detection-run ids given; no paths), `claims` and `verification`. Claim: `{id, claim_id, position, text, claim_type, damage_type, image_ids, prediction_ids, evidence: [{image_id, prediction_id, crop, note}], result}`; `result`: `{verdict, needs_human_review, policy_reason_codes, verification_label, explanation}` (`verdict` is the final policy outcome, `verification_label` the verifier's raw answer). Verification: `{id, report_id, verifier_model, verifier_version, status, error_message, policy_version, verifier_independent, created_at, completed_at, review_count}`.

Report errors: `404 not_found`; `409 no_images`, `image_unavailable`, `too_many_images`, `too_many_detections`, `uncitable_detection`, `image_unreadable`, `evidence_changed` (an image or prediction was deleted during generation), `no_claims` (verification of a report without claims); `502 vlm_output_rejected` (the message names the reason, e.g. `unknown_alias`), `vlm_inference_error`; `503 vlm_unavailable` (not configured, failed or refused load; also development stubs when `APP_ENV=production`), `vlm_busy`; `504 vlm_timeout`; `500 report_save_failed`. 404/409 checks happen before the model is loaded. `DELETE …/images/{id}` also returns `409 image_in_use` when a stored claim cites the image.

## Errors

All non-2xx responses (except the `/health/db` 503 body above) use:

```json
{"error": {"code": "validation_error", "message": "…", "details": [{"field": "body.title", "message": "…"}]}}
```

Codes: `validation_error` (422), `not_found` (404), `method_not_allowed` (405), `http_error` (other HTTP errors), `internal_error` (500; details are logged, not returned).

## Conventions

- IDs are UUIDs; timestamps are ISO 8601 with a timezone offset (the database session timezone).
- Versioned under `/api/v1`; breaking changes get a new prefix.
- CORS origins come from `CORS_ORIGINS`; only GET/POST/DELETE and `Content-Type` are allowed for now.
