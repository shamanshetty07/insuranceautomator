# Phase 3 QA and security review (Agent D)

Pass 1 (2026-10-28): an independent review of the Phase 2 image-ingestion code, and a
threat model for the Phase 3 detection boundary, written before the detector code exists.
Pass 2 (a review of the detection implementation) is pending.

Scope reviewed:
- `app/api/routes/images.py`
- `app/services/image_service.py`, `image_processing.py`, `image_storage.py` and `reconciliation.py`
- `app/core/errors.py`, `app/core/config.py` and `app/main.py`
- the existing tests

Environment: Pillow 12.3.0, Starlette 1.10.0, FastAPI 0.162.0, Python 3.14.2. Tests ran against a
disposable local PostgreSQL 18 database (`vda_test_d`); no shared or cloud database was used.

New tests (in `backend/tests/test_qa_processing.py` and `test_qa_ingestion_api.py`): 32 tests
pass, and 3 strict `xfail` tests mark the defects below. Each `xfail` turns into an
XPASS failure once its defect is fixed, which signals that the marker should then be removed.

---

## Part 1: Phase 2 ingestion findings

### Critical

None found. Path handling is robust:
- Paths are server-generated and checked against a regex.
- Writes use `O_EXCL|O_NOFOLLOW`, with symlink checks on the directory and the file and a resolve-inside-root check.
- An unsafe path stored in the database is never served (tested).
- The client filename is display-only (tested with traversal input).

### Important

**I-1. An upload request body has no size cap; the limit is checked only after the whole body is on disk.**
**Status: Fixed by the lead.**
- `app/core/body_limit.py` caps `Content-Length` at `max_upload_bytes` + 64 KiB and refuses chunked bodies with 411.
- The `xfail` marker was removed, and the test now asserts a 413 `file_too_large`.
- `test_chunked_body_without_content_length_is_refused` was added.
- Pass 2 reviews the fix and finds a CORS issue (P2-I-3).
- Where: `app/api/routes/images.py:58-69`; Starlette `formparsers.py` `on_part_data` / `on_headers_finished`.
- Evidence: Starlette's `max_part_size` (1 MB) applies only to non-file fields. File parts are
  written to a `SpooledTemporaryFile` with no limit, up to 1000 file parts per request
  (`max_files=1000`). The route's 413 check runs after parsing has finished and looks only at
  the `file` part.
- Test: `test_request_body_far_larger_than_the_limit_is_rejected` (`xfail`). A request carrying
  an extra file part of three times the limit is accepted with 201.
- Impact: any client can make the server write arbitrarily large temporary files before
  rejecting the request, which can exhaust disk space.
- Recommendation: put a byte cap on the request body before multipart parsing. Either use an
  ASGI middleware that rejects `Content-Length > max_upload_bytes + 64 KiB` and counts streamed
  bytes for chunked bodies, or enforce `client_max_body_size` at the reverse proxy. Also reject
  file parts other than `file`.

**I-2. A JPEG `COM` (comment) segment is copied into the stored file.**
**Status: Fixed by the lead.**
- `normalized.info.pop("comment", None)` now runs before the save.
- The test passes without an `xfail` marker.
- Where: `app/services/image_processing.py:95-108`, and the claim in the comment at lines 106-107.
- Evidence: Pillow keeps `im.info["comment"]` through `exif_transpose()` and `copy()`, and
  `JpegImagePlugin` writes it back on save. XMP, EXIF and PNG text chunks are stripped
  (tested), but a JPEG comment survives with or without an EXIF rotation.
- Test: `test_jpeg_comment_segment_is_not_stored` (`xfail`).
- Impact: the documented guarantee that metadata is stripped is false for free-text comments,
  which may contain names or other identifiers. The bytes are also served back from the content
  endpoint (an image MIME type with `nosniff`, so the XSS risk is low).
- Recommendation: before saving, clear `normalized.info` except for the ICC profile, for
  example `normalized.info = {}`, or pass `comment=b""` for JPEG.

**I-3. Multi-picture JPEGs (MPO) from cameras are rejected with 415.**
**Status: Fixed by the lead.**
- `MPO` is now mapped to `JPEG`, and frame 0 is stored.
- The test passes without an `xfail` marker.
- Where: `app/services/image_processing.py:20-24` and `:67`.
- Evidence: Pillow reports any JPEG that carries an APP2 MP index as format `"MPO"`. Many phone
  and stereo cameras produce these, including photos taken on some Samsung and Fujifilm devices
  and some portrait-mode shots.
- Test: `test_multi_picture_jpeg_from_cameras_is_accepted` (`xfail`).
- Impact: real vehicle photos uploaded from a phone gallery can fail with "unsupported media
  type". Camera capture in the browser is unaffected, because canvas `toBlob` produces a plain
  JPEG.
- Recommendation: treat `"MPO"` as JPEG. Store the primary frame only (frame 0, which is what
  `load()` decodes), and save it with `format="JPEG"`.

**I-4. Each upload can use a lot of memory, and nothing limits how many uploads decode at once.**
- Where: `app/core/config.py:26,29`; `images.py:61-71` (chunks are joined, so the bytes are
  held twice); `image_processing.py:90-108`.
- Evidence: with the defaults (15 MB upload, 40 MP limit), one upload holds the raw bytes twice,
  the decoded pixels (about 160 MB for RGBA at 40 MP), the transposed copy and a possible
  conversion. The peak is roughly 350-500 MB. Sync routes run in Starlette's thread pool
  (40 tokens), so about 40 concurrent large uploads could need many GB.
- Recommendation: put a process-wide `threading.BoundedSemaphore` (for example 2-4) around
  `process_image`, and consider lowering the default `max_image_pixels` to about 25 MP. The same
  semaphore, or a separate one, should guard detector inference (see T-3).

### Optional

**O-1. A CMYK JPEG keeps its CMYK ICC profile after conversion to RGB.**
- Where: `image_processing.py:95-102`.
- `convert("RGB")` is a naive conversion, and the original (CMYK) profile is then embedded in an
  RGB JPEG, which is an invalid pairing that viewers may render wrongly. This is rare for
  vehicle photos.
- Recommendation: when the mode changes, convert through `ImageCms` or drop the profile.

**O-2. Multi-frame PNG and WebP files are stored as their first frame without telling the client.**
- Tested in `test_multi_frame_input_is_stored_as_a_single_frame`.
- This is acceptable, but should be documented in `docs/api-design.md`.

**O-3. A narrow race between deleting a directory and creating a file in it.**
- Where: `image_storage.py:53` (`mkdir`) and `:81` (`rmdir`).
- `delete_image` does not take the inspection row lock. Deleting the last image of an
  inspection can `rmdir` the directory between an upload's `mkdir(exist_ok=True)` and its
  `os.open`. The upload then fails with a 500 and leaves no row and no file, so a retry
  succeeds.
- Recommendation: retry `mkdir` and `open` once on `FileNotFoundError`, or skip the `rmdir`.

**O-4. Images stay cached for up to an hour after deletion.**
- Where: `images.py:112`.
- `Cache-Control: private, max-age=3600` lets a browser keep showing a deleted image for up to an
  hour. This is acceptable for one user, but use `no-cache` with an ETag (the sha256) if deletion
  must take effect at once.

**O-5. Symlinks found in storage do not count as problems.**
- Where: `reconciliation.py:44-47`.
- `skipped_symlinks` does not affect `has_problems`, so the exit code is 0. The app never creates
  symlinks, so one inside the managed root is suspicious and arguably should return exit 1.

**O-6. CORS is checked and correct; no change needed.**
- Configured origins only, no wildcard, no credentials, and no PUT or PATCH. This was not tested
  before; it is now covered by `test_cors_allows_only_configured_origins_and_methods`.
- Unexpected exceptions return a generic 500 body that leaks no paths or OS messages. This is now
  covered by `test_unexpected_storage_failure_returns_generic_500_without_details`.

### Coverage added (all passing unless marked)

| Test | Gap closed |
| --- | --- |
| `test_non_rgb_modes_are_accepted_and_stored_decodable` (8 cases) | RGBA, LA, P, I;16 and 1-bit PNG; RGBA WebP; CMYK and L JPEG. Stored output decodes; the detector will consume these modes. |
| `test_multi_frame_input_is_stored_as_a_single_frame` | Animated PNG and WebP |
| `test_exif_orientation_is_applied_for_png_and_webp` | EXIF rotation was tested for JPEG only |
| `test_pixel_limit_is_inclusive` | Off-by-one on `max_pixels` |
| `test_oversized_jpeg_header_is_rejected_without_decoding` | The header-size check was tested for PNG only; this asserts `load()` is never called |
| `test_jpeg_header_bomb_beyond_pillow_threshold_is_rejected` | A JPEG decompression bomb |
| `test_png_text_chunks_and_xmp_are_not_stored`, `test_xmp_packets_are_not_stored` | Metadata outside EXIF |
| `test_jpeg_comment_segment_is_not_stored` | **xfail**, I-2 |
| `test_multi_picture_jpeg_from_cameras_is_accepted` | **xfail**, I-3 |
| `test_hostile_filename_is_display_only_and_never_shapes_the_path` | Traversal filename through the API |
| `test_clean_filename_strips_paths_and_control_characters` (5 cases) | NUL, ESC, DEL, Windows paths, blank names and length |
| `test_upload_size_limit_is_inclusive` | Exact size boundary |
| `test_request_body_far_larger_than_the_limit_is_rejected` | **xfail**, I-1 |
| `test_content_endpoint_headers` | `Cache-Control`, `nosniff` and an inline disposition |
| `test_unsafe_database_path_is_never_served` | A tampered `storage_path` reaching the content endpoint |
| `test_missing_file_on_disk_is_a_clean_404` | No path leak |
| `test_unexpected_storage_failure_returns_generic_500_without_details` | 500 body, no partial state, no lingering lock |
| `test_cors_allows_only_configured_origins_and_methods` | CORS |
| `test_repeated_delete_is_404_and_keeps_other_images` | Idempotence of delete |
| `test_concurrent_identical_uploads_store_exactly_one` | Truly concurrent duplicate uploads (four threads, separate sessions) |

---

## Part 2: Detection-boundary threat model (pre-implementation)

These items are written against the lead's contract (`GET /detector`, and `POST` and `GET`
`/inspections/{id}/detections`, with a synchronous run per image). Pass 2 will check the
implementation against them.

**T-1. Loading weights means running code (Critical if violated).**
- Ultralytics `.pt` files are PyTorch pickles. `torch.load` on an untrusted file runs arbitrary
  code; `weights_only=True` is not a guarantee, because Ultralytics has historically
  overridden it to load its model classes.
- Requirements:
  - weights come only from `DETECTOR_WEIGHTS_PATH`, set by the operator; there is never an API
    parameter, upload, URL or automatic hub download;
  - the path must be an existing regular file and not a symlink outside an expected directory;
  - the sha256 is computed before loading and recorded in `DetectionRun.weights_sha256`;
  - optionally, an expected-hash setting refuses to load on a mismatch;
  - the model name passed to Ultralytics is a local path, so that a bare name such as
    `yolo11n.pt` does not trigger a download.
- Prefer `.onnx` or `.safetensors` exports where the chosen model allows it.

**T-2. Invalid input reaching the detector.**
- The detector receives stored, normalized files, but their modes vary (see the mode tests:
  I;16, 1, P, LA, RGBA, L).
- Requirements:
  - convert to 3-channel RGB explicitly, or rely on `cv2.imread(IMREAD_COLOR)` and document it;
  - resolve paths only through `ImageStorage.resolve`, never by joining raw strings;
  - a missing or corrupt file becomes a `failed` run, not a 500;
  - `image_ids` in the request body must be UUIDs scoped to the inspection (an unknown id
    gives 404 and creates no runs), deduplicated, and bounded by
    `max_images_per_inspection`; an empty list gives 422.

**T-3. Resource limits.**
- Synchronous inference over up to 20 images can take tens of seconds on CPU, which ties up a
  worker thread and invites client timeouts.
- Requirements:
  - one process-wide lock or semaphore around `predict`, because Ultralytics models are not
    thread-safe and concurrent calls multiply memory;
  - fixed `imgsz`, `max_det` and confidence settings from the server config, never from
    request input;
  - lazy loading exactly once behind a lock, so concurrent first requests do not load the
    model twice;
  - document the expected latency, and consider a per-request cap or a later move to a
    background job.

**T-4. Concurrent detection requests.**
- Two POSTs for the same inspection will both create runs. That is acceptable under the
  append-only design, but `GET ?latest=true` must pick one run per image deterministically
  (`created_at` DESC, then `id`).
- An image deleted mid-run: the run's FK cascades, so the final UPDATE affects 0 rows. That must
  not cause a 500 for the other images.

**T-5. Failure recovery.**
- A run is inserted as `running` and committed before inference. If the process dies, it stays
  `running` forever; the CHECK constraint still holds, because `completed_at` is NULL.
- Requirements:
  - write the final status in a separate transaction, with a wall-clock `completed_at`;
  - catch every exception per image, record it as `failed`, and continue;
  - provide either a reaper that marks stale `running` runs older than N minutes as `failed`,
    or a reconciliation report entry for them;
  - `GET` must show such runs truthfully rather than hide them.

**T-6. Reproducibility.**
- Each run records:
  - `model_name`, `model_version` and `weights_sha256`;
  - the full `DetectionConfig` that was actually applied, validated with `extra="forbid"`
    (including the device);
  - ideally the Ultralytics and torch versions. There is no column for these, and the config is
    strict, so this needs a lead decision (for example a `framework` key added to
    `DetectionConfig`).
- Note that results on CPU and MPS can differ slightly.
- Predictions must store boxes exactly as normalized `xyxyn`, clamped to [0, 1], with
  degenerate boxes dropped, and validated through `BoundingBox` before insert.

**T-7. Information leakage.**
- `error_message` is returned to clients. It must be a fixed, sanitized string such as
  "Inference failed" or "Image file missing", never `str(exc)`, which can contain absolute
  paths, the weights location or CUDA and torch internals. Full details belong only in the
  server logs, logged with the run id.
- `GET /detector` must not expose `DETECTOR_WEIGHTS_PATH` or other filesystem paths (the
  contract already says this).

**T-8. Supply chain and licensing.**
- `ultralytics` is AGPL-3.0, so serving it over a network carries source-disclosure obligations.
- It and torch are large dependencies with frequent releases. Pin exact versions in the optional
  extra, keep detection disabled when the extra is not installed, and make sure no import of
  `ultralytics` or `torch` happens at app startup when `DETECTOR_BACKEND=none`.

---

## Pass 2: review of the detection implementation (Agent A) and of the lead's fixes

Reviewed:
- `app/services/detection/` (`__init__.py`, `provider.py`, `ultralytics_detector.py`)
- `app/services/detection_service.py`, `app/schemas/detection.py` and `app/api/routes/detections.py`
- the edits to `core/config.py`, `api/deps.py`, `api/router.py`, `services/types.py` and `services/interfaces.py`
- Agent A's tests: `test_detection_api.py`, `test_detection_adapter.py` and `test_detection_ultralytics.py`
- the lead's `app/core/body_limit.py`, the `image_processing.py` changes and the `main.py` change

Ultralytics behaviour was checked by reading the source of `ultralytics==8.4.191` in Agent A's
scratch venv. Nothing was installed in `backend/.venv`, and no weights were loaded.

New tests are in `backend/tests/test_qa_detection.py`: 17 pass and 4 are strict `xfail`. The
full suite on `vda_test_d` finished with **213 passed, 1 skipped, 4 xfailed**. The skipped test
is Agent A's real-model test, which needs Ultralytics and weights.

### Threat model status

| Item | Status | Evidence |
| --- | --- | --- |
| T-1 weights | **Partly met** | Weights come only from the operator's path, with an `is_file()` check before import, so no download is possible. The sha256 is recorded on every run. Missing: a file-type allowlist (P2-I-2), an expected-hash pin, and Ultralytics' opt-in safe load (P2-I-4). |
| T-2 invalid input | **Mostly met** | Paths go through `ImageStorage.resolve`. A missing or unsafe file gives a `failed` run. `image_ids` are UUIDs scoped to the inspection, deduplicated, and limited to 1-100. Each stored mode reaches the model as RGB at the stored size (tested), except 16-bit PNG (P2-I-1). |
| T-3 resources | **Met** | One process-wide `_INFERENCE_LOCK`, shared across detector instances (tested with 6 threads, at most 1 active). A single model load under `_load_lock` (tested with 5 threads, 1 instance). Settings come from the server only. |
| T-4 concurrency | **Met** | Concurrent POSTs create independent runs, and `latest` is deterministic (`DISTINCT ON image_order`, then `created_at` and `id` DESC; tested). An image deleted mid-run is dropped without a 500 (Agent A's test). |
| T-5 recovery | **Partly met** | Runs are committed as `running` before inference, and each per-image exception becomes a `failed` run. A stale `running` run is shown truthfully (tested). Missing: a reaper or reconciliation entry (P2-O-1). |
| T-6 reproducibility | **Partly met** | Recorded: the model name and version, `weights_sha256`, the applied `DetectionConfig` including the device, and wall-clock timestamps. Not recorded: the Ultralytics and torch versions (needs a lead decision; no schema change was made). The `image_size` stride issue is P2-O-2. |
| T-7 leakage | **Met** | Client-facing messages are fixed strings; exception text containing paths or CUDA details never reaches the response (tested). `/detector` exposes no paths (Agent A's test and this pass). |
| T-8 supply chain | **Met** | `ultralytics==8.4.191` is pinned in the optional `detection` extra. The import is lazy and does not happen at startup. `YOLO_OFFLINE=1` is honored, because `env_bool` accepts "1", so `ONLINE` is False, which disables GA telemetry and hub calls. `YOLO_AUTOINSTALL=0` is set. Both are tested. The AGPL caveat is in the module docstring. |

### Critical

None found.

### Important

**P2-I-1. 16-bit PNGs reach the model as a saturated, almost white image.**

**Status: Fixed by the lead.** `ultralytics_detector.to_rgb` scales 16-bit greyscale to 8-bit before inference (covers already-stored images).
- Where: `app/services/detection/ultralytics_detector.py:137-138`. The upload path keeps `I;16`
  (`image_processing.py`; see the pass-1 mode test).
- Evidence: Pillow's `I;16 → RGB` conversion clips: a value of 1000 becomes 255, and 32768 becomes
  255. The model therefore sees a blank image, and the run is recorded as `completed` with few or
  no detections. This is a silently wrong result, not a failure.
- Test: `test_sixteen_bit_png_is_not_saturated_before_inference` (`xfail`).
- Recommendation: scale 16-bit data to 8-bit, either before inference
  (`Image.eval(img.convert("I"), lambda v: v >> 8)` and then RGB) or, better, at upload, so that
  the stored file, the UI and the model all see the same 8-bit pixels.

**P2-I-2. A `.yaml` or `.yml` weights path silently creates an untrained model.**

**Status: Fixed by the lead.** Only `.pt` weights are accepted (`is_weights_file`, checked by the provider and the detector).
- Where: `app/services/detection/provider.py:66` (only `is_file()` is checked) and
  `ultralytics_detector.py:116`.
- Evidence: `ultralytics/engine/model.py:135` treats a path ending in `.yaml` or `.yml` as a model
  definition and builds random weights. The run would then record "detections" from a random
  network, which contradicts the project rule of no fake results.
- Test: `test_model_definition_files_are_refused_as_weights[model.yaml|MODEL.YML]` (`xfail`).
- Recommendation: allow only known weight suffixes, compared case-insensitively. For example
  `.pt` and, once supported, `.onnx`; anything else is reported as unavailable with a fixed
  reason.

**P2-I-3. The body-cap rejections (413 and 411) are unreadable to the browser frontend.**

**Status: Fixed by the lead.** `BodySizeLimitMiddleware` is now registered before CORS, so CORS wraps it.
- Where: `app/main.py:39`.
- Evidence: `BodySizeLimitMiddleware` is added after `CORSMiddleware`. Starlette makes the
  last-added middleware the outermost, so its responses bypass CORS, and the 413 carries no
  `Access-Control-Allow-Origin`. An `XMLHttpRequest` from the frontend therefore sees a network
  error, not `file_too_large`.
- The impact is limited: the frontend checks the file size first, and files up to the limit plus
  64 KiB reach the route's own 413, which does carry CORS headers. It surfaces when the
  frontend's `VITE_MAX_IMAGE_BYTES` and the backend's `MAX_UPLOAD_BYTES` differ.
- Test: `test_body_cap_rejection_is_readable_by_the_browser_frontend` (`xfail`).
- Recommendation: add `BodySizeLimitMiddleware` before `CORSMiddleware`, so that CORS stays
  outermost.

**P2-I-4. Ultralytics' opt-in restricted loading is not enabled.**

**Status: Partially mitigated (Sprint 1), opt-in only.** `DETECTOR_SAFE_LOAD=true` sets `ULTRALYTICS_SAFE_LOAD=1` before the first import (checked in the 8.4.191 source; requires torch >= 2.6). It is off by default and has not been tried with a real damage checkpoint, so it is not yet the default. Still open: confirm the approved checkpoint loads under it, then flip the default; add an expected-sha256 pin.
- Where: `ultralytics_detector.py:40-45`.
- Evidence: by default, 8.4.191 loads checkpoints with `torch.load(weights_only=False)`
  (`utils/patches.py` `torch_load`), which unpickles the whole file. The release offers
  `ULTRALYTICS_SAFE_LOAD=1` (`utils/__init__.py:81`, `nn/tasks.py` `_SafeLoad`), which uses
  `weights_only=True` with an allowlist of known classes.
- Weights are operator-provided, so this is defence in depth rather than an exploitable bug.
- Recommendation: set `os.environ.setdefault("ULTRALYTICS_SAFE_LOAD", "1")` in `_import_yolo`, and
  confirm that the chosen damage checkpoint loads under it. Community checkpoints that contain
  custom classes may not. Also consider a `DETECTOR_WEIGHTS_SHA256` pin that refuses to load on a
  mismatch.

### Optional

**P2-O-1. No reaper for runs left in `running`.**

**Status: Mostly fixed (Sprint 1).** BaseException and a failed final commit now leave a `failed` run where possible, and `python -m app.services.detection.recovery` fails runs older than `DETECTOR_STALE_RUN_MINUTES`. Age is the only signal, so with several workers a still-active run past the threshold would also be failed. The DB-backed tests for this could not run (no `TEST_DATABASE_URL`).
- Where: `detection_service.py:136-149`.
- A crash, a kill, or a `BaseException` such as worker cancellation during inference leaves a run
  `running` forever.
- Recommendation: mark runs `failed` after N minutes, either at startup or through a CLI in the
  style of reconciliation, with a fixed message such as "Interrupted".

**P2-O-2. `image_size` is not validated as a multiple of the model stride (32).**
- Where: `config.py:40`.
- Ultralytics rounds it up and logs a warning, so the recorded `DetectionConfig.image_size` can
  differ from what was actually applied.
- Recommendation: validate `% 32 == 0`.

**P2-O-3. A failed model load is retried, and the weights rehashed, on every POST.**
- Where: `provider.py:95-105`.
- This is correct, but a broken multi-hundred-MB checkpoint is hashed and loaded again on each
  request, under `_load_lock`.
- Recommendation: cache the failure briefly, for example for 30-60 seconds.

**P2-O-4. Time-of-check gap between hashing and loading the weights.**
- Where: `ultralytics_detector.py:115-116`.
- The file is hashed, then reopened by Ultralytics. If it is replaced in between, the recorded hash
  does not describe the weights that were actually loaded.
- This is operator-controlled, so the risk is low. Rehashing after the load and refusing on a
  mismatch would close the gap.

**P2-O-5. A synchronous request can wait without limit.**
- Concurrent POSTs serialize on `_INFERENCE_LOCK`, and there is no timeout. A hung `predict` would
  block every detection request indefinitely, although `GET` routes are unaffected.
- This is acceptable for Phase 3, but should be documented, and moving to a background job
  should be considered later.

**P2-O-6. A transaction is held open while the model loads.**
- Where: `detection_service.py:175-178`.
- `_select_images` runs inside the request transaction, and `provider.get()`, which can take
  seconds on the first load, runs before the first commit. The session sits idle in that
  transaction without holding any locks, which is harmless at this scale.
- A `db.rollback()` or `commit()` after `_select_images` would release it.

**Lead fixes, reviewed: correct except P2-I-3.**
- The body cap uses `Content-Length`. The ASGI servers in use (h11 and httptools) refuse to deliver
  more bytes than declared, and refuse requests that carry both `Content-Length` and
  `Transfer-Encoding`, so trusting the declared length is sound.
- The JSON endpoints are covered by the cap as well.
- The MPO and comment fixes are minimal and tested.

### Coverage added in pass 2 (`test_qa_detection.py`)

| Test | Checks |
| --- | --- |
| `test_every_stored_mode_reaches_the_model_as_rgb_at_stored_size` (6 modes) | T-2: the exact input to the model |
| `test_exif_rotated_upload_reaches_the_model_upright` | Boxes stay relative to the stored pixels |
| `test_sixteen_bit_png_is_not_saturated_before_inference` | passing (was xfail), P2-I-1 |
| `test_model_definition_files_are_refused_as_weights` (2) | passing (was xfail), P2-I-2 |
| `test_import_forces_ultralytics_offline_and_no_autoinstall` | T-8 |
| `test_status_never_loads_or_hashes_weights` | `/detector` stays cheap and shows no paths |
| `test_concurrent_detect_calls_never_overlap_in_the_model` | T-3 lock across instances |
| `test_concurrent_first_requests_load_the_model_once` | T-3 single load |
| `test_failure_details_never_reach_the_client` | T-7, in both the POST and GET responses |
| `test_unstorable_class_name_fails_only_that_run` | A DB rejection becomes a `failed` run, with no partial predictions and the other images still processed |
| `test_duplicate_image_ids_create_one_run_per_image` | T-2 |
| `test_stale_running_run_is_reported_truthfully` | T-5 visibility |
| `test_concurrent_detection_requests_are_independent_and_latest_is_deterministic` | T-4 (3 threads, separate sessions) |
| `test_body_cap_counts_framing_allowance_only_once` | Cap boundary through the middleware |
| `test_body_cap_rejection_is_readable_by_the_browser_frontend` | passing (was xfail), P2-I-3 |
