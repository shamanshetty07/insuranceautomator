"""Mapping from the legacy 7-value verification label to the claim-level Verdict."""

from app.models.verification_result import VerificationLabel
from app.schemas.claims import Verdict

CLAIM_LABELS: frozenset[VerificationLabel] = frozenset(VerificationLabel) - {
    VerificationLabel.OMITTED_DAMAGE
}

_LEGACY: dict[VerificationLabel, Verdict | None] = {
    VerificationLabel.SUPPORTED: Verdict.SUPPORTED,
    VerificationLabel.CONTRADICTED: Verdict.CONTRADICTED,
    VerificationLabel.WRONG_LOCATION: Verdict.CONTRADICTED,
    VerificationLabel.WRONG_DAMAGE_TYPE: Verdict.CONTRADICTED,
    VerificationLabel.UNVERIFIABLE: Verdict.INSUFFICIENT_EVIDENCE,
    VerificationLabel.UNSUPPORTED_SEVERITY: None,
    VerificationLabel.OMITTED_DAMAGE: None,
}


def legacy_verdict(label: VerificationLabel | str) -> Verdict | None:
    """Verdict for a legacy label, or None where no factual verdict can be derived."""
    try:
        return _LEGACY[VerificationLabel(label)]
    except ValueError:
        raise ValueError(f"unknown verification label: {label!r}") from None


def is_claim_label(label: VerificationLabel) -> bool:
    """True if `label` can be stored as the raw verifier answer for a report claim."""
    return label in CLAIM_LABELS
