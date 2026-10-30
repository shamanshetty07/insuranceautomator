# Verification evaluation

Offline tooling to measure claim verification. Nothing here runs a model, reads a database or downloads data.

| Document | Contents |
| --- | --- |
| [adversarial-matrix.md](adversarial-matrix.md) | The 12 adversarial cases, the layer each test exercises, actual behaviour, and limitations |
| [metrics.md](metrics.md) | Every metric's definition, denominator, zero-denominator behaviour and research question |
| [annotation-manifest.md](annotation-manifest.md) | The ground-truth manifest format, annotation guidelines, versioning, leakage and privacy rules |
| [experiment-protocol.md](experiment-protocol.md) | Baseline vs self-verification vs hybrid: design, statistics, threats to validity, failure analysis |

## Code

- `backend/app/evaluation/records.py`: shared contract (`ClaimOutcome`, `Disposition`, `GoldLabel`, `Configuration`).
- `backend/app/evaluation/metrics.py`: `compute_metrics`, `compare`, `Ratio` (undefined, never imputed, on a zero denominator).
- `backend/app/evaluation/manifest.py`: strict manifest models, `gold_labels`, `manifest_digest`, `split_by_inspection`.
- `backend/app/evaluation/harness.py`: replays recorded per-claim verifier answers ("trials") through the three
  configurations. The hybrid path reuses production code (`_ask_verifier`, `_decided_without_verifier`,
  `evaluate_claim`), so it scores the policy that actually ships.
- `backend/scripts/evaluate_claims.py`: CLI that prints the labelled JSON result.

```bash
cd backend && source .venv/bin/activate
python scripts/evaluate_claims.py \
  --manifest tests/fixtures/evaluation/synthetic_manifest_v0.json \
  --trials tests/fixtures/evaluation/synthetic_trials_v0.json [--force-independent]
```

## Synthetic versus real evaluation

Three different questions, never to be conflated:

1. **Q1. Can the model identify damage from an image?** Needs image-level human labels (`visible_damage` in a
   real manifest) and real inference. **Not measured.**
2. **Q2. Can the verifier judge an individual claim?** Needs real verifier answers on human-labelled claims
   (`verifier_agreement`). **Not measured.**
3. **Q3. Does the policy keep unsupported claims out of the accepted state?** Measured here **for the policy
   only**, on synthetic inputs: given hand-written verifier answers and evidence states, the tests and
   harness show what the shipped code decides.

The repository has no real vehicle images and no human labels. `synthetic_manifest_v0.json` has
`dataset_kind: "synthetic"` and labels written by the fixture author. `synthetic_trials_v0.json` has
`synthetic: true` and answers written by hand. Every result from them is labelled `SYNTHETIC` (and
`COUNTERFACTUAL` with `--force-independent`). Its numbers show how the software behaves, not how accurate it is.
A passing test establishes that the software follows its policy for that input. It establishes nothing about the
accuracy of Qwen-VL or any damage detector.

Real evaluation needs: a real manifest (`dataset_kind: "real"`, adjudicated human labels), and a trials file
exported from real verification runs (`synthetic: false`). Both are absent.
