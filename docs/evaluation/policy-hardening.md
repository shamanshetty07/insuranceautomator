# Verification policy hardening (claim-policy-7)

Every change here can only move a claim **toward human review**. None accepts more claims, and
none auto-rejects a claim for insufficient evidence. All checks are deterministic and run without
a model, a database or a download. Tests use synthetic inputs, so they show what the policy does,
not how accurate any model is.

## Rules added to `evaluate_claim`

There is still one policy (`claim_policy.py`). New `ClaimPolicyInput` fields default to the old
behaviour. `verification_service` and the evaluation harness set them.

| Input | Set from | Effect | Reason code |
| --- | --- | --- | --- |
| `evidence_quality = insufficient` | a cited image's stored size is below `min_image_side_px` | `insufficient_evidence` with review, after the integrity rule and before the verifier is called | `evidence_quality_insufficient` |
| `evidence_quality = questionable` | unknown or invalid image dimensions; a detection with an invalid confidence or one below its own run's stored threshold; two cited detections on the same region with different classes | adds review, verdict unchanged | `evidence_quality_questionable` |
| `verifier_consistent = False` | an earlier completed run for the same report claim gave a different raw verdict | adds review, verdict unchanged | `verifier_inconsistent_across_runs` |
| `claim_positive = False` (visual types) | `claim_semantics.assertion_of(text)` is not `positive` | adds review, verdict unchanged | `claim_not_positive_assertion` |

The review-only rules run after the existing verdict rules (`_escalate`). They append their code
and set `needs_human_review`. They never change the verdict, and they never fire before an
earlier rule (claim type, association, integrity, verifier error) has decided.

## Changes outside the policy function

- **Claim semantics** (`claim_semantics.py`, agent A). Each claim text is classified as
  `positive`, `negated`, `uncertain`, `mixed` or `unclear`, with cue strings. The text is never
  rewritten. A claim is `positive` only when no cue is present. A wrong `positive` can let a wrong
  "supported" through, while a wrong non-positive only costs a review, so ambiguity resolves
  toward non-positive. `detection_signal` returns `inconclusive` with details
  `claim_not_positive:<class>` and `claim_cue:<cue>` for non-positive visual claims, so a
  detection never "matches" a claim like "There is no dent".
- **Location consistency** (`evidence_association._location_doubts`, agent B).
  - A `damage_location` claim never reaches `matches_claim`. The location words are never
    machine-checked, and the generator can copy a detector box as its crop
    (`location_text_unchecked`).
  - A match is demoted to `inconclusive` whenever the claim's own crops point somewhere other than
    its cited detections: `location_crop_uncovered`, `location_elsewhere`,
    `location_crops_disagree`, `location_crop_image_mismatch`, `detection_box_absent`.
  - A frozen copy of the previous logic and a grid test over 3000+ configurations prove the change
    only moves MATCH to INCONCLUSIVE.
- **Quality** (`evidence_quality.py`, agent C, wired in `verification_service.assess_claim_quality`).
  See [quality-and-confidence.md](quality-and-confidence.md). Confidence is never a reason to
  accept.
- **Verdict schema and model identity** (agent D).
  - `unsupported_severity` had no defined meaning, so the verifier prompt no longer offers it
    (`OFFERED_LABELS`, prompt `qwen-verify-2`). It stays storable for legacy rows. If a model
    answers with it, the answer is invalid output (`verifier_invalid`).
  - `verifier_is_independent` returns True only when both weight digests are present and differ
    and the model names or versions differ. Display names alone never prove independence. The
    digest-less dev stubs are therefore not independent.
  - `_ask_verifier` now uses `coerce_label`, so a hostile label object is a per-claim outcome, not
    a run failure.
- **Repeated runs** (`verdict_consistency.py`, loaded by `verification_service.load_prior_verdicts`).
  - Raw labels are compared through `legacy_verdict` against earlier **completed** runs of the
    same report.
  - Any difference between usable verdicts counts as disagreement, including
    supported vs. insufficient.
  - Agreement adds nothing: same-model repeats are not independent confirmation.
  - Unusable earlier answers are ignored.

## A deliberate trade-off

Demoting `matches_claim` to `inconclusive` (negation, location) changes "verifier contradicted +
detection matches" from `insufficient_evidence` to `contradicted`. **Both still need review.** In
these cases the detection was never real corroboration: for "There is no dent" a dent detection
contradicts the claim, and box overlap says nothing about the location words. So the verifier's
`contradicted` is the more accurate label.

## What is still not handled

- An independent verifier that wrongly says "supported", together with a detector false positive
  of the right type, is still accepted. Only real labelled data can measure how often this
  happens.
- Some wording is still classified `positive` wrongly: sarcasm, negation through words outside
  the lists ("spared", "escaped damage"), non-English text and typos. See `claim_semantics.py`.
- The location words ("rear bumper") are never machine-checked, because there is no
  part segmentation.
- Uncited detections that overlap a claim's crop are not considered.
- Blur, lighting, occlusion and a missing vehicle region are not assessed; only the stored
  dimensions are. Detector confidence is uncalibrated.
- A fine-tune or re-quantisation of the same base model with different names and digests counts
  as independent.
- Disagreement is checked only against earlier runs of the same report. Nothing re-runs the
  verifier on purpose.
- Parse failures from the verifier are recorded as `verifier_error`, not `verifier_invalid`, so
  `invalid_output_rate` counts only parseable but out-of-schema answers.
- Database-backed paths (`load_prior_verdicts`, `assess_claim_quality` against real rows,
  `associate_claims`) are covered only by fake-session unit tests here. Their database tests are
  skipped without `TEST_DATABASE_URL`.
