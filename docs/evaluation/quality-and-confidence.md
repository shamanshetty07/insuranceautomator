# Image quality and detection confidence

Scope: what the system can and cannot say about how usable an evidence image is, and how
much a detector score is worth. Written against `claim-policy-6`; wired in `claim-policy-7` (section 4). Code:
`backend/app/services/evidence_quality.py`. Tests: `backend/tests/test_evidence_quality.py`.

**Summary.** The system has no image-quality gate after upload, and detector confidence is
not calibrated. The new module adds a few checks that only use facts already stored and
limits already configured. Every check can only send a claim to human review. None of them
can make a claim more acceptable. Blur, lighting and occlusion are **not** assessed.

## 1. Signals that already exist

| Signal | Where | What it can show | What it cannot show |
|---|---|---|---|
| Stored `width_px`, `height_px` | `models/inspection_image.py:57-58`; set at upload `services/image_service.py:88-89`; DB check `> 0` (`:37`) | Pixel count of the stored, EXIF-rotated image | Sharpness, exposure, whether the vehicle or damage is in frame |
| Upload minimum side `min_image_side_px=32` | `core/config.py:28`; enforced `services/image_processing.py:70-75` | The image met the configured minimum when it was uploaded | Anything about quality. 32 px is a sanity floor, not a usability standard |
| Upload maximum `max_image_pixels=40M` | `core/config.py:29`; `image_processing.py:76-81` | Decompression-bomb guard | Quality |
| EXIF orientation | `image_processing.py:38-45, 86`; stored as metadata only (`inspection_image.py:59-60`) | The pixels are already upright | Quality |
| Stored-file integrity (sha256 and dimensions vs. file) | `evidence_association.py:226-261` | The file has not changed and decodes | Quality |
| VLM pixel budget `vlm_max_image_pixels=802,816` | `core/config.py:55`; `vlm/adapters.py:55-64` | The size the VLM receives. Images are only **downscaled**, never upscaled | Quality |
| Detection run config (`confidence_threshold`, `iou_threshold`, `image_size`, `max_detections`) | `schemas/runs.py:22-50`; defaults `core/config.py:35-38`; stored per run `detection_service.py:263-268`; applied `ultralytics_detector.py:207-210` | The operating point the run used | Whether that operating point is any good. No validation data exists |
| Per-prediction `confidence` in [0, 1] | `models/damage_prediction.py:20, 28`; finite and in-range check `detection_service.py:158-160`; raw scores clamped `ultralytics_detector.py:119-120, 145` | The model's raw score | Probability of correctness. The score is uncalibrated (`docs/research/phase3-damage-detection.md:128` lists calibration as future work) |
| Manifest `image_quality` label (`ok/blurry/low_resolution/occluded/unknown`) | `evaluation/manifest.py:27, 83` | A human-annotated slice label for offline evaluation | Nothing at runtime. It is not computed, and the default is `unknown` |

Confidence is not carried into association (`PredictionFacts`, `evidence_association.py:80-86`)
or into the claim policy (`ClaimPolicyInput`, `claim_policy.py:69-79`). The experiment protocol
names this as a path to false acceptance (`experiment-protocol.md:222-224`). The generator
prompt shows confidence to the generator (`vlm/prompts.py:64-70`). The verifier gets no
detections at all (`vlm/adapters.py:136-138`).

**What the verifier sees for a tiny image.** It gets the whole stored image, not a crop,
converted to RGB, at its native size, because `load_image` never upscales
(`adapters.py:61-64`; `tests/test_adversarial_verification.py:360-365`). Its prompt states
only the number of photographs and the claim text (`prompts.py:85-90`). It is given no
resolution or quality information. The Hugging Face processor then applies its own resizing
(`transformers_runner.py:135-137`). That resizing is not configured in this repository and
could not be inspected here (`transformers` is not installed in the dev venv). Whatever the
processor does, interpolation adds no information. The current limitation is pinned by
`test_limitation_no_image_quality_gate_blurry_tiny_image_can_be_accepted`
(`test_adversarial_verification.py:373-376`): a 12x9 blurred image that the verifier
"supports" and a detection matches is ACCEPTED. A 12x9 image cannot arrive through upload.
It can only arrive through a non-upload path, legacy rows, or a later increase of
`min_image_side_px`.

## 2. Safeguards implemented (`evidence_quality.py`)

Each finding has a machine-readable `code:subject` and an effect. The effects are ordered:
`INFO` < `REVIEW` < `INSUFFICIENT`.

| Finding | Effect | Rule and justification |
|---|---|---|
| `image_below_min_side:<image_id>` | INSUFFICIENT | `min(width, height) < Settings.min_image_side_px`. This is the system's own configured upload rule (`image_processing.py:70`), so no new threshold is added. A stored image that fails it should never have become evidence. |
| `image_dimensions_unknown:<image_id>` | REVIEW | Width or height is NULL. The minimum-side rule cannot be checked. Upload always records both, so NULL means the row did not come through upload. |
| `image_dimensions_invalid:<image_id>` | REVIEW | The value is not a positive int. This is a data-integrity problem (the DB check normally prevents it). |
| `image_downscaled_for_vlm:<id>:WxH->wxh` | INFO | The verifier sees a downscaled copy (`vlm_max_image_pixels`). This is descriptive only: downscaling is the configured design, not a defect. |
| `region_below_min_side:<subject>` | INFO | A cited box or crop is smaller than `min_image_side_px` in stored pixels. **This threshold is a placeholder**: the upload minimum was set for whole images, not damage regions, so it is reported but has no effect. |
| `detection_confidence_invalid:<prediction_id>` | REVIEW | Confidence is None, non-numeric, bool, NaN, inf, or outside [0, 1]. This is a data-integrity problem, not a quality judgement. |
| `detection_below_run_threshold:<prediction_id>` | REVIEW | Confidence is below the run's own stored `confidence_threshold`, minus a tolerance of 1e-6. The run should have filtered it out (`ultralytics_detector.py:207`), so the stored row is inconsistent with its provenance. The tolerance covers float32 rounding only (1 ulp near 1.0 is about 6e-8). |
| `run_threshold_unknown:<run_id>` | INFO | The run config has no valid `confidence_threshold`, so the check above cannot run. |
| `detections_disagree:<id1>,<id2>` | REVIEW | Two **cited** detections on the same image have different normalized classes and box IoU >= `LOCATION_MATCH_IOU` (0.3, imported from association so "same region" means the same thing in both places, and equally uncalibrated). The detector is unsure what the damage at that spot is. |

`QualityAssessment.evidence_clearly_insufficient` is True **only** for
`image_below_min_side`. That is the only case backed by an existing configured rule.

No high-confidence rule exists. A valid confidence of 0.99 produces no finding. It adds
nothing (tested: `test_valid_confidence_produces_nothing_and_high_confidence_is_not_rewarded`).
No "low confidence" threshold is invented either. The only confidence comparison is against
the run's own recorded threshold.

`escalate_decision(decision, assessment, insufficient_reason=..., review_reason=...)` applies an
assessment to a `ClaimPolicyDecision`:

- `INFO`: the decision is returned unchanged (same object).
- `REVIEW`: the verdict stays the same, `needs_human_review=True`, and `review_reason` is appended.
- `INSUFFICIENT`: the verdict becomes `INSUFFICIENT_EVIDENCE` and `needs_human_review=True`.
  An existing insufficient decision keeps its reasons, with `insufficient_reason` appended.

A test runs every combination of claim type, verdict, error flag, integrity, association,
detection match and independence through `evaluate_claim`, then through each effect. It
asserts that review is never cleared, that the verdict is either kept or becomes
`INSUFFICIENT_EVIDENCE`, and that INFO is a no-op. So no path leads toward acceptance or
toward automatic rejection.

`vlm_input_size(w, h, max_pixels)` mirrors `adapters.load_image`. A test checks that it equals
the real `load_image` output on synthetic images, so the report can state what resolution
the verifier received.

`describe_pixels(image)` returns `PixelStats`: edge variance after PIL `FIND_EDGES`, mean and
stddev of luminance, and the fraction of pixels at or below 4 and at or above 251. It is
marked `calibrated=False`, produces **no findings**, and is not used by any rule. The tests
only check direction on synthetic images (blur lowers edge variance, a flat image scores 0,
black and white are all dark and all bright). These values are scene-dependent. A smooth
panel and a blurred photo can both score low edge variance. Without labelled data no cut-off
can be defended.

## 3. What remains unsolved

- **Blur, lighting, glare, occlusion, wrong framing or missing vehicle region.** None are
  detected. Metadata cannot show them. Pixel statistics are computable with PIL, but there is
  no labelled set and no calibrated threshold. A decision rule needs the failure-slice
  annotations planned in `phase3-damage-detection.md:131-132` and the manifest
  `image_quality` labels, with thresholds fitted on a validation split and frozen before
  testing. A learned quality model is the other option. Neither exists.
- **Confidence calibration.** Scores are raw model outputs. Precision at the deployed
  threshold and the reliability diagram / ECE (`phase3-damage-detection.md:123-128`) are
  unmeasured, so confidence must stay out of acceptance logic. Until calibration exists,
  even a "low confidence means review" rule would be arbitrary.
- **Region size.** Whether a damage region of N pixels is enough for the verifier to judge
  is unknown. `region_below_min_side` is descriptive only.
- **Processor-side resizing** inside the VLM (`transformers_runner.py:135`) is not pinned.
  It should be recorded in `VlmLimitsRecord` or the model record for provenance.
- **Uncited detections** that overlap a cited one are not compared. Only cited detections are
  checked for disagreement, as specified.

## 4. Policy wiring (implemented in claim-policy-7)

The lead wired this **inside** `evaluate_claim` rather than as a post-step, so there is one policy:

- `verification_service.assess_claim_quality` loads the cited `InspectionImage` rows and the cited
  `DamagePrediction` rows with their run's `config`, for visual claim types only, and combines
  `assess_image` and `assess_detections`. Limits come from `Settings` (`QualityLimits.from_settings`)
  in the API route; without settings, `default_quality_limits()` uses the Settings defaults and
  never reads the environment.
- The assessment's effect becomes `ClaimPolicyInput.evidence_quality`: INFO → `adequate`,
  REVIEW → `questionable`, INSUFFICIENT → `insufficient`.
- `insufficient` returns `insufficient_evidence` with review (`evidence_quality_insufficient`)
  right after the integrity rule, so the verifier is not called.
- `questionable` keeps whatever the verdict rules decide and adds review
  (`evidence_quality_questionable`). It never changes the verdict.
- REVIEW and INSUFFICIENT findings are stored in `EvidenceRecord.quality_findings`. INFO findings
  (for example downscaling) are not stored.
- `escalate_decision` was removed: the policy now does this, and a second decision function would
  be a competing policy. Its monotonicity property is tested exhaustively in
  `tests/test_policy_hardening.py`.

Effect on the adversarial suite: only images below `min_image_side_px` are caught. A blurred image
of adequate size is still accepted when the verifier supports it and a detection matches.
