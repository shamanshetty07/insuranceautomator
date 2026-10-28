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
