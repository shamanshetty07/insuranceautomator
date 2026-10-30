# Experiment protocol: claim verification for generated damage reports

Status: **design only, with no results.** See [Current status](#1-current-status) before reading anything else.

Shared contract: `backend/app/evaluation/records.py` (`Configuration`, `Disposition`, `GoldLabel`,
`ClaimOutcome`, `disposition_of`, `baseline_outcome`). Metric definitions: `backend/app/evaluation/metrics.py`.
Policy under test: `claim-policy-7` (section 9 analyses claim-policy-5/6; the claim-policy-7 changes are listed in [policy-hardening.md](policy-hardening.md)) (`backend/app/services/claim_policy.py:90`). Line references below
point to the tree at commit `2d11550` plus the uncommitted `backend/app/evaluation/`; the claim-policy-6
fixes (section 9, item 2) shifted `evidence_association.py` by 4 lines after line 149 and
`verification_service.py` by 6 lines after line 93.

---

## 1. Current status

- **No real VLM inference has been run** for this experiment. The verifier and generator adapters exist
  (`services/vlm/adapters.py`). The tests use stubs and synthetic fixtures (`backend/tests/image_utils.py`).
- **No human-labelled images exist in the repository.** `git ls-files` lists no image files and no
  annotation files. `storage/inspections/` holds only `.gitkeep`. Human ground truth is not modelled in the
  database yet (`docs/architecture.md:44`).
- **There are therefore no results.** This document does not claim, and must not be read as claiming, that
  the hybrid configuration performs better than the others. Section 9 shows that some of its apparent
  advantages hold *by construction*, so they are not evidence.

## 2. Configurations

All three configurations score **the same generated claims** (§5). They differ only in how a
claim's disposition (`ACCEPTED` / `REJECTED` / `HUMAN_REVIEW`) is derived.

| Id | `Configuration` | Disposition derived from |
|---|---|---|
| B | `BASELINE` | Nothing. Every generated claim is `ACCEPTED` (`records.baseline_outcome`, `records.py:106-108`). |
| S | `SELF_VERIFICATION` | The raw verdict only, from the same VLM, on the original images. |
| H | `HYBRID` | Production path: the raw verdict, evidence association and integrity, the detection match and `evaluate_claim` (claim-policy-6). |

### 2.1 Deriving S without the policy gate

For every claim, the harness calls the verifier directly with the claim text and the claim's cited
images that resolve in its inspection. It does this **even for claims that production would not send to
the verifier** (`verification_service._decided_without_verifier`, `verification_service.py:78-81`), because
otherwise S would have no raw verdict for those claims. The raw label is mapped with the production mapping
`verdict_mapping.legacy_verdict` (`verdict_mapping.py:10-18`):

| Raw verifier output | `ClaimOutcome.verdict` | `needs_human_review` | Disposition | reason_codes |
|---|---|---|---|---|
| `supported` | SUPPORTED | False | ACCEPTED | `verifier_supported` |
| `contradicted`, `wrong_location`, `wrong_damage_type` | CONTRADICTED | False | REJECTED | `verifier_contradicted` |
| `unverifiable` | INSUFFICIENT_EVIDENCE | True | HUMAN_REVIEW | `verifier_insufficient` |
| `unsupported_severity`, `omitted_damage`, unknown label, unparseable JSON | INSUFFICIENT_EVIDENCE | True | HUMAN_REVIEW | `verifier_invalid` |
| exception or timeout | INSUFFICIENT_EVIDENCE | True | HUMAN_REVIEW | `verifier_error` |

S applies no claim-type rules, association, integrity, detection or independence checks.
**In the production code S never occurs.** `verify_report` computes `verifier_is_independent`
(`verification_service.py:261-263`, `model_guard.py:22-33`), and with a single model it is `False`. Claim-policy-5
then sends every *supported* claim to review (`claim_policy.py:205-211`). S is therefore an **offline
ablation** computed from the same raw verdicts that H uses. It answers the question "what if the verifier's
word were final?" It is not a description of deployed behaviour.

### 2.2 Deriving H

`evaluate_claim(ClaimPolicyInput(...))` uses the claim's `ClaimEvidence` from `associate_claims` and the
**same raw `VerifierOutcome`** used for S. `verifier_independent` is computed exactly as in production. For
claims that production does not send to the verifier, the policy decides before reading the verdict, so
reusing the offline verdict gives the production decision. **Replay check (required):** on every inspection
that also has a stored production `VerificationRun`, H's offline decisions must equal the stored
`verdict`/`needs_human_review`/`policy_reason_codes`. Any mismatch invalidates the run.

**H is degenerate when one model both generates and verifies.** In that case every supported claim is
reviewed (`claim_policy.py:205-211`). No visual claim is ever auto-rejected (§9.3). Every `insufficient`
outcome is reviewed (`claim_policy.py:115-116`). As a result, H with a single model sends **100% of claims to
`HUMAN_REVIEW`**: unsupported_acceptance_rate is 0 (or undefined), claim_recall is 0 and claim_precision is
undefined. This is a property of the code, not an empirical finding. Two extra variants make H informative.
Both are reported under `HYBRID` and carry a `variant` tag in the run metadata:

- **H-ind (counterfactual ablation):** the same as H, with `verifier_independent=True` forced. This shows
  what the gate would do if the verifier were independent. It is **not** a real independent verifier and must
  be labelled as a counterfactual.
- **H-ext (optional, real):** H with a *different* verifier model (different weights digest). Its raw
  verdicts differ from S's, so it is paired with S on claims but not on verdicts. Report it separately.

## 3. Research questions (do not conflate)

| Q | Question | Unit | Measured by | Not answered by |
|---|---|---|---|---|
| Q1 | Can the model identify visible damage from an image? | image (manifest `visible_damage`) plus generated claim | Image-level sensitivity and specificity of "≥1 visual claim on this image" against `visible_damage`. Baseline claim_precision is the fraction of generated claims labelled gold-supported. Omission rate (§7). | Any verifier or policy metric |
| Q2 | Can the verifier judge an individual claim? | claim | Verifier–human agreement on **raw** verdicts: percent agreement, a 3×3 confusion matrix (`supported`/`contradicted`/`insufficient` × gold `supported`/`contradicted`/`unverifiable`) and Cohen's κ. Each is given per claim type and with `verifier_invalid`/`verifier_error` reported separately. | Disposition metrics. The policy hides verifier errors behind review. |
| Q3 | Does the full policy keep unsupported claims out of `ACCEPTED`? | claim | **unsupported_acceptance_rate**, together with human_review_rate and the other secondary metrics, for B, S, H, H-ind | Agreement or κ. A verifier can have low κ while the policy still accepts nothing. |

The three questions answer different things and must not be merged. A good Q3 number can come from routing
everything to review, so it says nothing about Q2. A good Q2 number says nothing about Q1, because a
verifier can judge well the claims that a generator wrongly omitted or hallucinated.

## 4. Ground truth

**Images.** Real vehicle photographs, grouped into inspections. The planned sources are the CarDD test split
(research-only licence) and an in-house set that includes clean-car negatives
(`docs/research/phase3-damage-detection.md`, evaluation plan). Deduplicate by perceptual hash across
sources and splits before any split is frozen.

**Image-level labels (manifest).** Each image gets `visible_damage` (yes/no) and, where damage is present,
damage type(s) from the canonical taxonomy and coarse location. The manifest digest (SHA-256 of the
canonical manifest file) is recorded with every evaluation run.

**Claim-level labels.** Each claim is labelled `supported` / `contradicted` / `unverifiable` (`GoldLabel`):

- Annotators judge from **the claim's text and the images only**. They do not see the verifier's label or
  explanation, the detections, the policy outcome, the configuration or other annotators' labels. The
  claim type is shown only if every annotator sees it in the same way.
- The **whole claim** must hold for `supported`. A claim with a correct damage type but a wrong location is
  `contradicted`. Severity, cause, cost and hidden-damage assertions are `unverifiable` unless the photos
  show them as false, in which case they are `contradicted`.
- **Double annotation** of every claim and every image-level label, by two annotators working
  independently.
- **Inter-annotator agreement** is reported as Cohen's κ per label set before adjudication. Pre-registered
  threshold: κ ≥ 0.70 on claim labels. Below 0.60 the guidelines are revised and the affected batch is
  re-annotated, without looking at system outputs. A result between 0.60 and 0.70 is reported as a threat to
  validity (§11).
- **Adjudication:** a third annotator resolves disagreements and the adjudicated label is the gold label.
  Pre-adjudication labels are kept, so the sensitivity of the results to adjudication can be reported.
- A claim with no gold label stays **unlabelled** (`records.join_labels`). It is never guessed and is
  excluded from every denominator, with the exclusion count reported.

## 5. Claim generation and the paired design

1. For each test inspection, the generator runs **once** (`QwenStructuredReportGenerator`, greedy decoding:
   `temperature=0, top_p=1`, `adapters.py:92`, `do_sample=False` in `transformers_runner.py:141`).
2. The resulting claims are frozen and given a stable `claim_key` (inspection id + report id + `claim_id`).
3. The verifier runs **once per claim** (`adapters.py:142-150`, greedy, `adapters.py:132`). Its raw outputs
   are stored.
4. B, S, H and H-ind are all computed **offline from the same claims and the same raw verdicts**. Sampling
   noise therefore cannot differ between configurations, and every comparison is paired by claim.
5. **Determinism check:** re-run generation and verification on a random 10% of inspections. Report the
   fraction of claims whose text changes and the fraction of raw verdicts that flip ("inconsistent verdicts
   across runs", §8). Greedy decoding is not guaranteed bit-identical across hardware or library versions.
6. Record per run: the generator and verifier `weights_sha256`, model name and version, the runtime package
   versions (`ModelRecord.packages`), `GENERATOR_PROMPT_VERSION` / `VERIFIER_PROMPT_VERSION`
   (`prompts.py:9-10`), `POLICY_VERSION`, `LOCATION_MATCH_IOU` (`evidence_association.py:37`), the
   `VlmLimits` record, the detector weights digest and the `DetectionRun` ids, the manifest digest, the split
   file digest and the git commit.

If generation fails for a report (parse rejection is all-or-nothing, `parsing.py:125-136`), the report
has **no claims** in any configuration. The failure is counted in report-level
invalid_output_rejection_rate and in Q1 (as a miss on damaged images). It never silently shrinks the
claim-level denominators.

## 6. Metrics

**Primary (Q3):** `unsupported_acceptance_rate` = accepted ∧ gold ∈ {contradicted, unverifiable} /
gold ∈ {contradicted, unverifiable}.

**Secondary:** claim_precision, claim_recall, false_rejection_rate, human_review_rate,
invalid_output_rejection_rate (verifier `verifier_invalid` and `verifier_missing` at claim level; generator
parse rejections at report level, reported separately), evidence_integrity_failure_rate, and verifier–human
agreement (Q2). Zero denominators give **undefined**, which is never imputed and never reported as 0.

**Rule: unsupported_acceptance_rate is never reported without human_review_rate** next to it, for the same
configuration and slice. A configuration can reach UAR = 0 by reviewing everything. H with a single model
does exactly that (§2.2).

**Report level (secondary):** the fraction of reports with ≥1 accepted unsupported claim, and the fraction of
reports fully accepted with no review. **Slices:** claim type, damage type, detector available or
unavailable, and the image source (CarDD or in-house).

## 7. Scoring rejected and review claims; omissions

`HUMAN_REVIEW` is **neither accepted nor correct**. It is reported separately, not folded into "rejected"
or "correct". Every configuration reports the full table:

| Disposition \ gold | supported | contradicted | unverifiable | unlabelled |
|---|---|---|---|---|
| ACCEPTED | true accept | **unsafe accept** | **unsafe accept** | n |
| REJECTED | false rejection | correct reject | correct reject | n |
| HUMAN_REVIEW | review cost | review (safe) | review (safe) | n |

**Sensitivity analysis on review.** Each configuration is recomputed under two resolutions of `HUMAN_REVIEW`:

- **Oracle:** the reviewer resolves to gold. Supported claims become accepted and the rest rejected. This
  is an upper bound on the value of the whole pipeline with humans in the loop.
- **Worst case:** every reviewed claim resolves against gold. Supported claims become rejected and
  unsupported claims become accepted. This is a lower bound, and it shows how much the safety result
  depends on reviewers being right.

Both are reported next to the "review is separate" primary numbers. Neither replaces them.

**Omissions** (damage visible in the image with no claim about it) cannot appear in any claim-level metric,
because no claim exists to score. They are measured separately against the manifest: an image with
`visible_damage = yes` and no claim citing that image counts as an image-level omission, and instance-level
omissions are counted where the manifest lists instances. Omissions are identical across B/S/H (the same
claims are used), so they bound what every configuration can achieve. `find_omissions` is not
implemented (`adapters.py:165-166`).

## 8. Error taxonomy

Every claim that is gold-unsupported, falsely rejected or reviewed while gold-supported gets one primary code,
plus optional secondary codes, from adjudicated review of the claim, its evidence record and its verifier
explanation.

| Stage | Code | Definition |
|---|---|---|
| Generation | `hallucinated_damage` | Claims damage where none is visible. |
| | `wrong_location` | Damage exists but not where the claim says. |
| | `wrong_type` | Damage exists but is of another type. |
| | `severity_overreach` / `cause_overreach` / `cost_overreach` | Asserts what the photos cannot show. |
| | `unverifiable_asserted` | Any other claim stated as fact that is not decidable from the photos. |
| Evidence | `evidence_ref_fabricated` | Cites an image or detection that does not depict the claimed damage. |
| | `cross_inspection_ref` | Cites evidence outside the inspection (`evidence_unresolved`). |
| | `integrity_failure` | Evidence integrity is missing, mismatched, corrupt or unchecked. |
| Output | `malformed_output` | Generator parse rejection, or verifier `verifier_invalid` / `verifier_error`. |
| Verifier | `verifier_false_support` | Raw `supported`, gold contradicted or unverifiable. |
| | `verifier_false_contradiction` | Raw `contradicted`, gold supported. |
| | `inconsistent_verdict` | Raw verdict differs across repeated runs (§5.5). |
| Policy | `over_escalation` | Gold supported, raw supported, routed to review by a policy rule (record the reason code). |

## 9. Failure analysis of the current code (claim-policy-5, with the claim-policy-6 fix noted)

### 9.1 Ways an unsupported claim can still reach ACCEPTED

`ACCEPTED` is reachable only through `claim_policy.py:212-214`. That requires a visual claim type, an
`ASSOCIATED` association, integrity `OK`, a raw `supported` verdict, `MATCHES_CLAIM`, and
`verifier_independent=True`. Every condition below satisfies all six while the claim is false:

1. **Independent verifier false support combined with a detector false positive.** If the verifier and the
   detector both err in the same direction, the claim is accepted (`claim_policy.py:185-214`). Detector
   confidence is not an input, so a low-confidence false detection counts as fully corroborating.
2. **Presence claims match on any cited detection.** For `visible_damage_presence`, any cited prediction
   gives a match, with no type check and no location text check (`evidence_association.py:149-151`). The
   generator chooses the claim type (`parsing.py:115`), so a claim such as "dent on the rear bumper" typed as
   *presence* and citing a scratch detection on the front door is `MATCHES_CLAIM`.
   **Partly fixed in claim-policy-6:** when the presence claim carries a `damage_type` that differs from the
   cited detection's class, the result is now `inconclusive` (review). A presence claim with no
   `damage_type`, and the location text, are still unchecked.
3. **Location matching compares the generator's crop with the detector's box, not the location text.** For
   `damage_location`, the match is crop-vs-box IoU ≥ 0.3 (`evidence_association.py:117-130, 146-147`). The
   generator is shown the detector boxes (`prompts.py:61-74`), so it can copy a box and reach IoU ≈ 1. The
   stated location ("left front door") is never compared with anything except by the verifier, which
   reads only the text.
4. **Damage-type matching is text-based.** `text_mentions_damage_type` (`evidence_association.py:137-141`)
   checks only for a contiguous word sequence. Negation ("no dent, only a scratch"), compound claims ("dent
   and crack on the hood", where only the `damage_type` field is compared at `:157`) and extra false
   assertions in the same sentence all pass. A detector class mistake propagates directly (`:157-158`).
5. **The verifier never sees the structure that detection matching checks.** The verifier receives only
   `claim.text` (`verification_service.py:87`, `prompts.py:85-90`) and whole images, not crops. Detection
   matching uses only the structured fields `claim_type`, `damage_type` and crop. The two signals can each
   be satisfied by a *different* part of the claim. This "corroboration" is therefore weaker than it looks.
6. **The independence check is a weights-digest or name check, not an error-independence check.**
   `model_guard.py:29-33` falls back to `(name, version)` when a digest is missing. It treats a renamed copy, a
   re-quantised file (different SHA) or a fine-tune of the same base model as independent. Correlated
   errors between related models are not detected.
7. **The integrity check covers bytes, not provenance.** `OK` means only that the file matches its stored
   checksum and dimensions (`evidence_association.py:245-254`). An image that was manipulated before upload
   passes.
8. **Prompt-injection surface.** The verifier is given claim text that the generator wrote. Quoting it as data
   (`prompts.py:88-89`) is a mitigation, not a guarantee.

### 9.2 Ways supported claims are systematically routed to review

1. **The same model generates and verifies.** Every supported claim goes to review (`claim_policy.py:205-211`,
   `model_guard.py:33`). This is the deployed default: one shared model, `docs/architecture.md:225`.
2. **No detector run on a cited image** gives `UNAVAILABLE` (`evidence_association.py:174-176`) and then review
   (`claim_policy.py:121-122, 198-204`). *With no detector, H routes every visual claim to review.* That
   trivially lowers UAR and raises human_review_rate, so the two must always be reported together (§6).
3. **Detector misses** give `NO_DETECTION` and then review (`claim_policy.py:191-197`), so detector recall
   caps H's accept rate. Uncited detections on the image give `INCONCLUSIVE` (`evidence_association.py:177-181`).
4. **Strict type wording.** Plurals and synonyms ("dents", "scrape") or a null `damage_type` give `INCONCLUSIVE`
   (`evidence_association.py:153-156`).
5. **Location claims** with no crop, no stored box, or IoU in (0, 0.3) give `INCONCLUSIVE`
   (`evidence_association.py:120-130, 146-147`). The threshold is uncalibrated (`:36-37`).
6. **A stale prediction** (not from the image's latest completed run) gives `INCONCLUSIVE`
   (`evidence_association.py:186-188`).
7. **An absent stored checksum** gives `UNCHECKED` (`evidence_association.py:246-247`) and then review
   (`claim_policy.py:170-172`).
8. **Non-visual claim types are always reviewed.** Production never passes
   `has_validated_cost_source`, `has_validated_severity_policy` or `human_confirmed`
   (`verification_service.py:65-75`), so rules `claim_policy.py:137-161` always fire. This is by design but
   should be counted.
9. **A legitimate label counted as invalid output.** `unsupported_severity` maps to `None`
   (`verdict_mapping.py:16`). It is then stored as an unrecognised verdict (`verification_service.py:120-121`)
   and becomes `verifier_invalid` (`claim_policy.py:180-181`). This inflates invalid_output_rejection_rate, so
   it is reported as its own row.
10. **Fail-fast after a timeout** makes every remaining claim in the run `verifier_error`
    (`report_service.py:117-124`). These errors cluster by inspection.

### 9.3 Structural properties that make some comparisons uninformative

- **`REJECTED` is unreachable in H.** Only visual types reach the verdict section, because no production input
  passes the type gates for the other types (§9.2.8). For visual types, every `DetectionMatch` value leads to
  review on a contradiction (`claim_policy.py:216-240`, with `NOT_APPLICABLE` mapped to inconclusive at
  `:125-126`). So `:241-246` is not reachable from production inputs. H's false_rejection_rate is 0 by
  construction.
- **H's acceptances are a subset of S's.** H accepts only when the raw verdict is `supported`, and S accepts
  exactly those claims. Because the raw verdicts are shared, UAR(H) ≤ UAR(S) and recall(H) ≤ recall(S) for
  **every** dataset. A significance test of "H accepts fewer unsupported claims than S" therefore tests
  nothing about the policy's *quality*. The informative quantities are the **cost** (supported claims lost to
  review, §9.2) and whether UAR(H-ind) falls below a pre-registered absolute bound.
- **B accepts everything**, so UAR(B) = 1 whenever its denominator is non-zero. B serves as the reference
  for the raw generator error rate (Q1), not as a competitor on Q3.

## 10. Fair comparison, leakage prevention and sample size

**Splits and leakage.**
- Split **by inspection**, never by image or claim (dev / test). Near-duplicate images (perceptual hash) are
  kept in the same split.
- Prompts, `POLICY_VERSION`, `LOCATION_MATCH_IOU` and detector thresholds are **frozen and recorded before any
  test label is seen**. No tuning is done on the test split. Changes discovered on the test split take
  effect only in a new version with a new test set.
- Annotators are blind to system outputs (§4). The manifest and the split file are hashed, and the digests
  are recorded with each run.

**Statistics.**
- Every comparison is paired on `claim_key`.
- **McNemar's exact test** is run on paired accept / not-accept decisions, restricted to gold-unsupported
  claims (for UAR) and to gold-supported claims (for recall). The discordant counts (b, c) are reported, not
  only p. For S vs H the test is one-sided by construction (§9.3) and is reported as descriptive only.
- **Confidence intervals:** a 95% percentile bootstrap that resamples **inspections** (cluster bootstrap,
  ≥ 2,000 resamples), because claims within one inspection share images and errors. Ratios with
  undefined values in a resample are reported with the count of undefined resamples.
- **Pre-registered hypotheses** (to be fixed before test labelling):
  - H1: UAR(S) < UAR(B), which tests whether self-verification catches any generator errors.
  - H2: UAR(H-ind) ≤ 0.05, an absolute safety bound, with the upper end of the 95% CI below 0.05.
  - H3: recall(H-ind) is reported with its CI. This is an estimate with no threshold, giving the cost.
  - H4: κ(verifier, gold) ≥ 0.60 on visual claims (Q2).
  Use Holm correction across H1–H4.

**Sample size (rough estimate, not a power guarantee).** The inputs are assumptions to replace with pilot
data from the dev split.
- *Estimating UAR precisely.* If true UAR ≈ 0.10, a 95% CI half-width of ±0.05 needs n ≈ 1.96² · 0.1 · 0.9 /
  0.05² ≈ 138 gold-unsupported claims. Assume about 5 claims per inspection and an intra-inspection
  correlation of about 0.2. The design effect is then 1 + (5−1)·0.2 = 1.8, giving about 250 unsupported claims.
  If about 30% of generated claims are unsupported, that means about 830 claims, or about 170 inspections.
- *Detecting a paired difference* (for example S vs H-ext, or two verifier models). Take δ = 0.10 difference in
  UAR, a discordant proportion ψ ≈ 0.15, α = 0.05 two-sided and power 0.8. Then n ≈ [1.96·√ψ + 0.84·√(ψ−δ²)]² / δ²
  = [0.759 + 0.314]² / 0.01 ≈ 115 unsupported claims, or about 210 after the design effect.
- If the pilot shows that the unsupported fraction is much lower (a good generator), the required number of
  inspections grows in proportion. In that case, plan for adversarial or held-out hard inspections and report
  them as a **separate** slice, never pooled with the natural distribution.

## 11. Threats to validity

- **Synthetic fixtures are not real images.** No conclusion may be drawn from the stub- or fixture-based
  tests. They test the code paths, not model behaviour.
- **Same-model verification bias.** S uses the generator's own model, which tends to agree with its own
  claims. Errors shared by the generator and the verifier inflate S's acceptances. H-ind is a counterfactual
  that does not remove this bias. Only H-ext, with a different model, starts to address it, and §9.1.6 still
  applies to it.
- **Small n.** The CarDD test split has 374 images, and the in-house set is small. Wide clustered CIs are
  expected and must be shown.
- **Annotator disagreement.** Gold `unverifiable` is the hardest category. Low κ on it bounds every metric
  that uses the unsupported denominator. Report the metrics under pre- and post-adjudication labels.
- **Distribution shift.** CarDD images are web-scraped close-ups, while the product uses phone captures.
  Report each source separately and do not pool them without also showing the per-source numbers.
- **Detector unavailable or weak.** `UNAVAILABLE` or `NO_DETECTION` sends H's visual claims to review (§9.2.2–3).
  This lowers UAR and raises review rate. Always report UAR, human_review_rate and the detector-available
  slice together.
- **Policy tautologies** (§9.3). Do not present `false_rejection_rate(H) = 0` or `UAR(H) ≤ UAR(S)` as findings.
- **Reviewer quality** is not measured. The review sensitivity analysis (§7) bounds it but does not
  estimate it.
- **Model, library and hardware drift.** Recorded digests (§5.6) and the determinism check (§5.5) are the
  only safeguards.
