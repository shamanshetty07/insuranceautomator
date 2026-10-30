# Annotation manifest

The annotation manifest is the single source of gold labels for the offline evaluation of claim
verification. It is a JSON file validated by `backend/app/evaluation/manifest.py`
(`load_manifest`). Metrics join system outcomes to gold labels by `claim_key`
(`gold_labels(manifest)` -> `app.evaluation.records.join_labels`).

## Current status

**No real images or human labels exist in the repository; real-world evaluation is blocked on
ground truth.** A search of tracked and untracked files (excluding `node_modules` and `.venv`)
found no image files (`jpg/jpeg/png/webp/heic/...`) and no label or annotation files
(`csv/jsonl/COCO/...`). All tests synthesize images in memory (`backend/tests/image_utils.py`).
No external dataset (e.g. CarDD) is assumed to be available.

The only manifest is `backend/tests/fixtures/evaluation/synthetic_manifest_v0.json`
(`dataset_kind: "synthetic"`). Its labels were written by developers to exercise the pipeline;
they are **not human ground truth**, and any number computed on it must be reported with
`Manifest.result_label` (which says "SYNTHETIC ... not human ground truth"). Synthetic results say
nothing about real-world accuracy.

## Format (schema_version "1")

All objects reject unknown fields. The file is parsed strictly: duplicate JSON keys, `NaN`/`Infinity`
and type coercion (e.g. `"0.1"` for a number) are rejected. Slugs match `^[a-z0-9][a-z0-9_-]{0,63}$`.

### Manifest

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `"1"` | Format version of this document. |
| `dataset_version` | string | Required, e.g. `real-v1`, `synthetic-v0`. Bump on any change (see Versioning). |
| `dataset_kind` | `"synthetic"` \| `"real"` | Required and explicit. Synthetic manifests may only use `synthetic:` sources; real manifests may not use them. |
| `description` | string | What the dataset is, where it came from, known limitations. |
| `created` | date | `YYYY-MM-DD`. |
| `images` | list of ManifestImage | `fixture_id` unique. |
| `claims` | list of ManifestClaim | `claim_key` unique; `inspection_key` must belong to some image. |

### ManifestImage

| Field | Type | Notes |
|---|---|---|
| `fixture_id` | slug | Unique image id within the manifest. |
| `inspection_key` | slug | Groups images of one vehicle inspection; the unit of splitting. |
| `source` | string | `synthetic:<generator>` or a relative POSIX path under the dataset root. Absolute paths, `~`, drive letters, `..`, `.`, empty segments, backslashes and URLs are rejected. |
| `sha256` | 64 lowercase hex \| null | Required for file sources; optional for synthetic generators. |
| `visible_damage` | list of VisibleDamage | Human-labelled damage visible in this image. Empty list = no visible damage. |
| `image_quality` | `ok` \| `blurry` \| `low_resolution` \| `occluded` \| `unknown` | Default `unknown`. |

### VisibleDamage

| Field | Type | Notes |
|---|---|---|
| `damage_type` | string | Normalized like the pipeline (`normalize_damage_type`: lowercase, spaces/hyphens -> `_`). Prefer `dent`, `scratch`, `crack`, `glass_shatter`, `lamp_broken`, `tire_flat`. |
| `location` | string | Free text; prefer the vocabulary: `front bumper`, `rear bumper`, `hood`, `roof`, `trunk`, `windshield`, `rear window`, `front/rear left/right door`, `front/rear left/right fender`, `left/right mirror`, `headlight left/right`, `taillight left/right`, `wheel <position>`, `undercarriage`. |
| `bbox` | BoundingBox \| null | Normalized `[0,1]`, origin top-left, `x_min < x_max`, `y_min < y_max` (same convention as `app.schemas.damage.BoundingBox`); unknown keys rejected. |

### ManifestClaim

| Field | Type | Notes |
|---|---|---|
| `claim_key` | `[A-Za-z0-9_-]{1,64}` | Join key with `ClaimOutcome.claim_key`. |
| `inspection_key` | slug | Inspection the claim is about. |
| `text` | string | The claim as it appears in a report (1-1000 chars). |
| `claim_type` | ClaimType | Same enum as `app.schemas.claims.ClaimType`. |
| `damage_type` | string \| null | Normalized as above. |
| `cited_fixture_ids` | list | Evidence the claim cites. No duplicates. An id not in the manifest requires `adversarial_tag: "fabricated_reference"`; an image of another inspection requires `"cross_inspection_reference"`. |
| `adversarial_tag` | enum \| null | Why the claim is a hard case: `invented_damage`, `wrong_location`, `wrong_damage_type`, `wrong_severity`, `area_not_visible`, `low_quality_evidence`, `ambiguous`, `not_visually_decidable`, `fabricated_reference`, `cross_inspection_reference`. For analysis only; never shown to the system under test. |
| `label` | `supported` \| `contradicted` \| `unverifiable` \| null | Must be null unless status is `draft` or `adjudicated`; required when `adjudicated`. |
| `annotation` | Annotation | See below. |

### Annotation

| Field | Type | Notes |
|---|---|---|
| `annotator` | slug \| null | Pseudonymous id (e.g. `ann-03`). Never an email or real name. Required unless status is `unannotated`. For adjudicated labels, the adjudicator's id. |
| `status` | `unannotated` \| `draft` \| `adjudicated` \| `disputed` | Only `adjudicated` labels count by default. |
| `annotated_at` | datetime \| null | ISO 8601. |
| `notes` | string \| null | Rationale, especially for `contradicted`/`unverifiable` and disputes. |

### Using labels

`gold_labels(manifest, require_status=("adjudicated",))` returns `claim_key -> label`; claims whose
status is not accepted map to `None` and are excluded from metrics, never guessed. Including
drafts (`require_status=("adjudicated", "draft")`) is allowed for development only and must be
stated in any reported result.

## Annotation guidelines

Judge each claim **from the images of its inspection only**, as a careful adjuster looking at the
photos. Do not use the system's detections, the report text around the claim, metadata, or
knowledge outside the photos. Judge the claim, not the citations: look at all images of the
inspection, not only the cited ones.

- **supported**: the images clearly show what the claim states (presence, type, location and,
  where stated, severity). Every factual part of the claim must be visible.
- **contradicted**: the relevant area is clearly visible at adequate quality and shows something
  incompatible with the claim, e.g. the panel is visibly undamaged (invented damage), the damage is
  on a different part (wrong location), the damage is clearly of a different type, or the severity
  is clearly different.
- **unverifiable**: the images do not allow a decision either way, because
  - the relevant area is not visible in any image (out of frame, occluded), or
  - the image quality is insufficient (blurry, low resolution, glare), or
  - the claim type is not visually decidable: repair cost, damage cause, hidden/structural damage,
    most repair recommendations, or
  - the evidence is ambiguous and careful annotators would reasonably disagree.

Rules of thumb: absence of evidence is not contradiction (if you cannot see the area, it is
`unverifiable`, not `contradicted`). A claim with several parts is `contradicted` if any part is
clearly contradicted, otherwise `unverifiable` if any part cannot be checked. A claim citing an
image that does not exist or belongs to another vehicle is still judged on this inspection's own
images; record the problem in `adversarial_tag` and `notes`.

Also label each image: `visible_damage` (type, location, optional bbox) and `image_quality`. Image
labels are made independently of the claims.

## Double annotation and adjudication

1. Two annotators label every claim independently (status `draft`, each in their own copy or
   column). They do not see each other's labels or the system's outputs.
2. Agreement is measured on the drafts (raw agreement and Cohen's kappa) and reported with the
   dataset.
3. Agreeing claims are set to `adjudicated` by the adjudicator (a third person, or the two
   annotators together), with the adjudicator's pseudonymous id.
4. Disagreements are discussed against the guidelines. If resolved, the claim becomes
   `adjudicated` with a note; if not, it becomes `disputed` with `label: null` and is excluded
   from metrics (reported as a count).
5. Guideline changes made during adjudication are recorded in this document and all earlier
   labels are re-checked against them.

## Versioning

- `dataset_version` is bumped on any change to images, claims or labels; published versions are
  never edited in place.
- Every evaluation run records `dataset_version` and `manifest_digest(manifest)` (SHA-256 of the
  canonical, key-sorted JSON). The digest is independent of key order and formatting but changes
  with any content change, including list order.
- `schema_version` changes only when this format changes.

## Leakage rules

- Split by inspection (`split_by_inspection`, deterministic hash of `inspection_key` with a salt):
  all images and claims of one vehicle stay in the same split. Never split by claim or image.
- Fix the split (salt, fraction) before any experiment and record it.
- Never tune prompts, thresholds, policies or model choices on the test split; use dev only. Look
  at test results once per configuration being reported.
- Claims used as examples in prompts must come from dev and must not be in the test split.
- Synthetic fixtures are for pipeline testing, not for tuning or reporting accuracy.

## Privacy

- Committed images must contain no personal data: no readable licence plates, faces, people, VINs,
  addresses or documents. Blur or crop before committing; strip EXIF (GPS, device data).
- Prefer keeping real images outside the repository and referencing them by relative path plus
  `sha256`.
- Annotator ids are pseudonymous slugs; the mapping to people is kept outside the repository.
  The validator rejects emails (`@` is not allowed), but cannot detect names: do not use them.
- Notes must not contain personal data.
- Only use images whose licence or consent permits this use; record the source and licence in
  `description`.
