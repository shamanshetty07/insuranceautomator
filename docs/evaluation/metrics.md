# Evaluation metrics

Code: `backend/app/evaluation/metrics.py`. Tests: `backend/tests/test_evaluation_metrics.py`.
Input records: `backend/app/evaluation/records.py` (`ClaimOutcome`, `GoldLabel`, `LabelledOutcome`).

## Research questions

- **Q1: Does the model identify the damage?** You cannot measure this from claim outcomes alone. It
  needs detection- or image-level ground truth, such as boxes or damage presence per image. None
  of the metrics below answer Q1.
- **Q2: Does the verifier judge a claim correctly?** Answered by verifier agreement with humans.
  This metric uses the raw verifier verdict and ignores the policy.
- **Q3: Does the policy keep unsupported claims out of the accepted state?** Answered mainly by
  the unsupported acceptance rate, and also by the other disposition metrics.

## Conventions

- **Unit:** one claim. Each `ClaimOutcome` is one claim under one configuration.
- **Disposition:** each claim ends up in exactly one disposition: `accepted`, `rejected` or
  `human_review` (see `disposition_of`). A baseline claim with no verdict is `accepted`.
- **Gold labels:** a human labels each claim against the images alone, as `supported`,
  `contradicted` or `unverifiable`. "Positive" means gold `supported`. "Unsupported" means gold
  `contradicted` or `unverifiable`.
- **Labelled vs. all:** metrics that need a gold label use labelled claims only. Unlabelled
  claims are counted (`unlabelled`) and reported, but they are never given a label or imputed.
  Operational rates are computed over all outcomes, as stated below.
- **Zero denominators:** every metric is a `Ratio(numerator, denominator)`. When the denominator
  is 0, `value` is `None` and the ratio prints as `0/0 (undefined)`. The code never substitutes
  0 or 1.
- **Confidence interval:** `Ratio.wilson_95()` gives the Wilson score 95% interval. It returns
  `None` when the denominator is 0.
- **One call, one configuration:** `compute_metrics` raises `ValueError` if `claim_key`s are
  duplicated or if configurations are mixed.
- **Fair comparison:** `compare` requires the same set of `claim_key`s and the same gold label
  for each claim in every configuration. It also requires each outcome to be filed under its own
  configuration. Otherwise it raises `ValueError`.

## Metrics

| Metric | Definition | Population / denominator | Answers |
|---|---|---|---|
| `claim_precision` | accepted ∧ gold supported / accepted | labelled claims that were accepted | Q3 |
| `claim_recall` | accepted ∧ gold supported / gold supported | labelled claims with gold supported | Q3 (cost side) |
| **`unsupported_acceptance_rate`** | accepted ∧ gold ∈ {contradicted, unverifiable} / gold ∈ {contradicted, unverifiable} | labelled unsupported claims | **Q3, primary safety metric** |
| `false_rejection_rate` | rejected ∧ gold supported / gold supported | labelled claims with gold supported | Q3 (cost side) |
| `human_review_rate` | human_review / all outcomes | all outcomes, labelled or not | Q3 (workload) |
| `invalid_output_rate` | verifier called ∧ output invalid / verifier called | outcomes with `verifier_called` | Q2 (operational) |
| `verifier_failure_rate` | verifier called ∧ verifier error / verifier called | outcomes with `verifier_called` | Q2 (operational) |
| `invalid_output_rejection_rate` | output invalid ∧ disposition ≠ accepted / output invalid | all outcomes with invalid verifier output | Q3 (policy invariant, should be 1.0) |
| `evidence_integrity_failure_rate` | integrity failure / all outcomes | all outcomes | Q3 (evidence pipeline) |
| `verifier_agreement` | percent agreement, 3×3 confusion matrix, Cohen's kappa | labelled outcomes with a non-null `raw_verifier_verdict` | **Q2** |

For every metric above, a zero denominator gives `value = None` (undefined).

Definitions of the flags (from `records.py`):

- *Output invalid* means the reason codes include `verifier_invalid` or `verifier_missing`.
- *Verifier error* means `verifier_error`.
- *Integrity failure* means `evidence_unresolved` or any `integrity_*` code other than OK.

### Notes

- **Unsupported acceptance is kept separate from accuracy.** It is the share of claims that
  should not be in the accepted report but are. Precision, recall and any accuracy-style
  number cannot stand in for it. The baseline configuration accepts every claim, so its
  unsupported acceptance rate is 1.0 whenever any unsupported gold exists.
- **Invalid-output and failure rates** count a numerator only when `verifier_called` is true.
  This keeps the ratio within [0, 1], because a claim that never reached the verifier can still
  carry `verifier_missing`.
- **Verifier agreement:** gold labels are mapped to verdicts as supported → `supported`,
  contradicted → `contradicted` and unverifiable → `insufficient_evidence`. The metric compares
  these with the raw verifier verdict, before the policy applies. In the confusion matrix, rows
  are gold labels and columns are verifier verdicts, in the order supported, contradicted,
  insufficient_evidence. Kappa is `(p_o − p_e) / (1 − p_e)`, and it is `None` when no outcomes
  qualify or when `p_e = 1` (both raters used one class only).
- **No policy-level accuracy is reported.** Mapping dispositions to gold labels needs a
  correctness convention, for example whether human review counts as correct for
  `unverifiable`. That convention has not been agreed, so the code leaves this number out.

## Caveat: synthetic fixtures

When outcomes come from synthetic fixtures, the verifier verdicts, detection matches and gold
labels were all written by hand. In that case:

- the results describe only the policy's behaviour, not any model's;
- Q2 numbers say nothing about a real verifier;
- Q1 is never addressed.

Only outcomes from real model runs, labelled independently by humans, support claims about Q2.

## Output

`MetricsReport.to_dict()` produces JSON-ready output:

- each `Ratio` becomes `{numerator, denominator, value}`;
- counts are reported under `counts`, which holds `total`, `labelled`, `unlabelled`,
  `verifier_called`, `by_disposition` and `by_gold`.
