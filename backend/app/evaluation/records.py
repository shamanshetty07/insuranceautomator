"""Shared evaluation records: one system outcome per claim, joined with an optional human label."""

import enum
from collections.abc import Iterable
from dataclasses import dataclass

from app.schemas.claims import Verdict
from app.services.claim_policy import ClaimPolicyDecision, ReasonCode


class GoldLabel(str, enum.Enum):
    """Human judgement of one claim against the images alone."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class Disposition(str, enum.Enum):
    """Where a claim ends up: in the accepted report, auto-rejected, or with a human."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    HUMAN_REVIEW = "human_review"


class Configuration(str, enum.Enum):
    """The three configurations compared by the experiment protocol."""

    BASELINE = "baseline"
    SELF_VERIFICATION = "self_verification"
    HYBRID = "hybrid"


VERIFIER_OUTPUT_INVALID_CODES = frozenset({ReasonCode.VERIFIER_INVALID, ReasonCode.VERIFIER_MISSING})
VERIFIER_FAILED_CODES = frozenset({ReasonCode.VERIFIER_ERROR})
INTEGRITY_FAILURE_CODES = frozenset(
    {
        ReasonCode.EVIDENCE_UNRESOLVED,
        ReasonCode.INTEGRITY_UNCHECKED,
        ReasonCode.INTEGRITY_MISSING,
        ReasonCode.INTEGRITY_MISMATCHED,
        ReasonCode.INTEGRITY_CORRUPT,
    }
)


def disposition_of(verdict: Verdict | None, needs_human_review: bool) -> Disposition:
    """None verdict means no verification ran (baseline): the claim reaches the report unchecked."""
    if needs_human_review:
        return Disposition.HUMAN_REVIEW
    if verdict is None or verdict is Verdict.SUPPORTED:
        return Disposition.ACCEPTED
    if verdict is Verdict.CONTRADICTED:
        return Disposition.REJECTED
    return Disposition.HUMAN_REVIEW


@dataclass(frozen=True)
class ClaimOutcome:
    """What one configuration did with one claim; claim_key is unique within an evaluation."""

    claim_key: str
    configuration: Configuration
    verdict: Verdict | None
    needs_human_review: bool
    reason_codes: tuple[ReasonCode, ...] = ()
    raw_verifier_verdict: Verdict | None = None
    verifier_called: bool = False

    @property
    def disposition(self) -> Disposition:
        return disposition_of(self.verdict, self.needs_human_review)

    @property
    def verifier_output_invalid(self) -> bool:
        return bool(VERIFIER_OUTPUT_INVALID_CODES.intersection(self.reason_codes))

    @property
    def verifier_failed(self) -> bool:
        return bool(VERIFIER_FAILED_CODES.intersection(self.reason_codes))

    @property
    def integrity_failure(self) -> bool:
        return bool(INTEGRITY_FAILURE_CODES.intersection(self.reason_codes))


def outcome_from_decision(
    claim_key: str,
    configuration: Configuration,
    decision: ClaimPolicyDecision,
    raw_verifier_verdict: Verdict | None = None,
    verifier_called: bool = True,
) -> ClaimOutcome:
    return ClaimOutcome(
        claim_key=claim_key,
        configuration=configuration,
        verdict=decision.verdict,
        needs_human_review=decision.needs_human_review,
        reason_codes=tuple(decision.reason_codes),
        raw_verifier_verdict=raw_verifier_verdict,
        verifier_called=verifier_called,
    )


def baseline_outcome(claim_key: str) -> ClaimOutcome:
    """Baseline: no claim-level verification, so every generated claim is accepted as written."""
    return ClaimOutcome(claim_key, Configuration.BASELINE, None, False)


@dataclass(frozen=True)
class LabelledOutcome:
    outcome: ClaimOutcome
    gold: GoldLabel | None

    @property
    def labelled(self) -> bool:
        return self.gold is not None


def join_labels(outcomes: Iterable[ClaimOutcome], gold: dict[str, GoldLabel | None]) -> list[LabelledOutcome]:
    """Pair outcomes with labels by claim_key; a claim with no entry is unlabelled, never guessed."""
    return [LabelledOutcome(o, gold.get(o.claim_key)) for o in outcomes]
