"""Compare a claim's raw verifier verdict with the raw verdicts of earlier runs (pure).

Repeated verification can only ever *add* review, never remove it:

- Agreement adds no trust. The result has no field that says "confirmed"; a caller cannot use
  it to relax a decision. Repeating a run with the same model (same weights) and the same greedy
  prompt is expected to agree, so agreement says nothing about whether the claim is true.
- Any disagreement between usable verdicts forces review: supported vs contradicted, and also
  supported or contradicted vs insufficient_evidence. A verdict that flips to "cannot tell" under
  a re-run is not a stable verdict, whatever the policy would do with either answer alone.
- Comparison is on the *raw* verifier label mapped through ``legacy_verdict``, never on the stored
  policy ``verdict`` (that also depends on detection and policy version, which can change).
  Labels that map to the same verdict (``wrong_location`` and ``contradicted``) agree.
- A prior with no usable verdict (verifier error, invalid label, verifier not called) is neither
  agreement nor disagreement; it is counted as ignored. With no usable current verdict there is
  nothing to compare, and the policy already sends that claim to review.

Priors are identified by the caller (same report claim, earlier completed runs, current run
excluded). ``same_model`` counts priors whose verifier is not shown to be a different model
(``model_guard.models_independent``); unknown identity counts as the same model.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from app.models.verification_result import VerificationLabel
from app.schemas.claims import Verdict
from app.services.model_guard import models_independent
from app.services.types import ModelInfo
from app.services.verdict_mapping import coerce_label, legacy_verdict


@dataclass(frozen=True)
class PriorVerdict:
    """The raw label one earlier run stored for the same report claim, and who produced it."""

    run_id: object
    raw_label: VerificationLabel | str | None
    verifier: ModelInfo


@dataclass(frozen=True)
class ConsistencyCheck:
    disagrees: bool
    current: Verdict | None
    compared: int
    ignored: int
    same_model: int
    disagreeing_runs: tuple[str, ...]
    summary: str


def raw_verdict(label: VerificationLabel | str | None) -> Verdict | None:
    """Verdict for a stored or returned raw label; None for anything without one. Never raises."""
    coerced = coerce_label(label)
    return legacy_verdict(coerced) if coerced is not None else None


def check_consistency(
    current_label: VerificationLabel | str | None,
    current_verifier: ModelInfo,
    priors: Iterable[PriorVerdict],
) -> ConsistencyCheck:
    current = raw_verdict(current_label)
    priors = tuple(priors)
    if current is None:
        return ConsistencyCheck(False, None, 0, len(priors), 0, (), "No usable current verdict to compare.")

    compared = ignored = same_model = 0
    disagreeing: list[str] = []
    seen: dict[Verdict, int] = {current: 1}
    for prior in priors:
        verdict = raw_verdict(prior.raw_label)
        if verdict is None:
            ignored += 1
            continue
        compared += 1
        seen[verdict] = seen.get(verdict, 0) + 1
        if not models_independent(current_verifier, prior.verifier):
            same_model += 1
        if verdict is not current:
            disagreeing.append(str(prior.run_id))

    if disagreeing:
        counts = ", ".join(f"{v.value}: {n}" for v, n in sorted(seen.items(), key=lambda kv: kv[0].value))
        summary = (
            f"The verifier's verdict differs across {compared + 1} runs ({counts}); "
            "an unstable verdict needs review."
        )
    elif compared:
        summary = (
            f"{compared} earlier run(s) gave the same verdict ({same_model} from the same model); "
            "agreement is not independent confirmation and adds no trust."
        )
    else:
        summary = "No earlier usable verdict for this claim."
    return ConsistencyCheck(bool(disagreeing), current, compared, ignored, same_model, tuple(disagreeing), summary)
