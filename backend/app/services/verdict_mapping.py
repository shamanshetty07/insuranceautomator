"""Mapping from the legacy 7-value verification label to the claim-level Verdict.

Must agree with the backfill in alembic/versions/0004_claim_level_verification.py;
tests/test_verification_migration.py runs the migration and compares.

Decisions:
  supported            -> supported
  contradicted         -> contradicted
  wrong_location       -> contradicted (stated location conflicts with the evidence)
  wrong_damage_type    -> contradicted (stated damage type conflicts with the evidence)
  unverifiable         -> insufficient_evidence
  unsupported_severity -> None. AMBIGUOUS: may mean overstated (contradicted) or merely
                          not established (insufficient). We do not guess.
  omitted_damage       -> None. A coverage finding about damage missing from the report,
                          not a verdict about a report claim.

Unknown label strings raise ValueError rather than silently returning None, so a new
label cannot be mistaken for "deliberately no verdict".

Claim labels vs. omission findings. Every label except omitted_damage is an answer
about a report claim. omitted_damage is a different kind of row: damage seen in the
images that the report does not mention (ClaimVerificationService.find_omissions). A
verifier that answers omitted_damage when asked about a claim did not answer that
question, so the claim pipeline must not store it as the claim's raw label: a claim row
always gets a verdict, and the database forbids an omitted_damage row with a verdict
(ck_verification_results_omitted_has_no_verdict). CLAIM_LABELS is the single list the
verification service checks before building a row.
"""

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
