"""Pure, deterministic claim-verdict policy."""

import enum
from dataclasses import dataclass

from app.schemas.claims import ClaimType, Verdict


class EvidenceIntegrity(str, enum.Enum):
    UNCHECKED = "unchecked"
    OK = "ok"
    MISSING = "missing"
    MISMATCHED = "mismatched"
    CORRUPT = "corrupt"


class EvidenceAssociation(str, enum.Enum):
    ASSOCIATED = "associated"
    NONE_CITED = "none_cited"
    UNRESOLVED = "unresolved"


class DetectionMatch(str, enum.Enum):
    MATCHES_CLAIM = "matches_claim"
    NO_DETECTION = "no_detection"
    CONFLICTS_WITH_CLAIM = "conflicts_with_claim"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class ReasonCode(str, enum.Enum):
    UNKNOWN_CLAIM_TYPE = "unknown_claim_type"
    HIDDEN_DAMAGE_NOT_VISIBLE = "hidden_damage_not_visible"
    COST_NOT_VALIDATED = "cost_not_validated"
    CAUSE_NOT_ASSERTABLE = "cause_not_assertable"
    SEVERITY_NOT_VALIDATED = "severity_not_validated"
    NOT_VISUALLY_DECIDABLE = "not_visually_decidable"
    INTEGRITY_UNCHECKED = "integrity_unchecked"
    INTEGRITY_MISSING = "integrity_missing"
    INTEGRITY_MISMATCHED = "integrity_mismatched"
    INTEGRITY_CORRUPT = "integrity_corrupt"
    NO_EVIDENCE_CITED = "no_evidence_cited"
    EVIDENCE_UNRESOLVED = "evidence_unresolved"
    VERIFIER_MISSING = "verifier_missing"
    VERIFIER_ERROR = "verifier_error"
    VERIFIER_INVALID = "verifier_invalid"
    DETECTION_CONFLICTS_WITH_SUPPORTED = "detection_conflicts_with_supported"
    DETECTION_MATCHES_CONTRADICTED = "detection_matches_contradicted"
    NO_DETECTION_NOT_ABSENCE = "no_detection_not_absence"
    VERIFIER_SUPPORTED_NO_DETECTION = "verifier_supported_no_detection"
    DETECTION_INCONCLUSIVE = "detection_inconclusive"
    DETECTION_UNAVAILABLE = "detection_unavailable"
    VERIFIER_SUPPORTED = "verifier_supported"
    VERIFIER_CONTRADICTED = "verifier_contradicted"
    DETECTION_CONFLICT_UNCONFIRMED = "detection_conflict_unconfirmed"
    VERIFIER_NOT_INDEPENDENT = "verifier_not_independent"
    VERIFIER_INSUFFICIENT = "verifier_insufficient"


@dataclass(frozen=True)
class VerifierOutcome:
    """verdict None (or an unrecognised value) means the verifier produced nothing usable."""

    verdict: Verdict | str | None = None
    errored: bool = False


@dataclass(frozen=True)
class ClaimPolicyInput:
    claim_type: ClaimType | str | None
    verifier: VerifierOutcome
    integrity: EvidenceIntegrity
    association: EvidenceAssociation
    detection_match: DetectionMatch
    has_validated_cost_source: bool = False
    has_validated_severity_policy: bool = False
    human_confirmed: bool = False
    verifier_independent: bool = True


@dataclass(frozen=True)
class ClaimPolicyDecision:
    verdict: Verdict
    needs_human_review: bool
    reason_codes: tuple[ReasonCode, ...]
    explanation: str


POLICY_VERSION = "claim-policy-6"

_VISUAL_TYPES = frozenset(
    {ClaimType.VISIBLE_DAMAGE_PRESENCE, ClaimType.DAMAGE_TYPE, ClaimType.DAMAGE_LOCATION}
)

_INTEGRITY_REASON = {
    EvidenceIntegrity.UNCHECKED: ReasonCode.INTEGRITY_UNCHECKED,
    EvidenceIntegrity.MISSING: ReasonCode.INTEGRITY_MISSING,
    EvidenceIntegrity.MISMATCHED: ReasonCode.INTEGRITY_MISMATCHED,
    EvidenceIntegrity.CORRUPT: ReasonCode.INTEGRITY_CORRUPT,
}


def _coerce_enum(enum_cls, value):  # noqa: ANN001, ANN202
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            return None
    return None


def _insufficient(reason: ReasonCode, explanation: str, review: bool = True) -> ClaimPolicyDecision:
    return ClaimPolicyDecision(Verdict.INSUFFICIENT_EVIDENCE, review, (reason,), explanation)


def _uncorroborated_reason(ctype: ClaimType, det: DetectionMatch) -> ReasonCode | None:
    """Reason code when detection cannot corroborate the claim (rule 14), else None."""
    if det is DetectionMatch.UNAVAILABLE:
        return ReasonCode.DETECTION_UNAVAILABLE
    if det is DetectionMatch.INCONCLUSIVE:
        return ReasonCode.DETECTION_INCONCLUSIVE
    if det is DetectionMatch.NOT_APPLICABLE and ctype in _VISUAL_TYPES:
        return ReasonCode.DETECTION_INCONCLUSIVE
    return None


def evaluate_claim(inp: ClaimPolicyInput) -> ClaimPolicyDecision:
    """Return exactly one decision for ``inp``; never raises on odd claim types/verdicts."""
    ctype = _coerce_enum(ClaimType, inp.claim_type)

    # 1-6: claim-type policies (a verifier verdict can never override these).
    if ctype is None:
        return _insufficient(ReasonCode.UNKNOWN_CLAIM_TYPE, "Unrecognised claim type; failing closed.")
    if ctype is ClaimType.HIDDEN_STRUCTURAL_DAMAGE:
        return _insufficient(
            ReasonCode.HIDDEN_DAMAGE_NOT_VISIBLE,
            "Hidden structural damage cannot be established from exterior photos.",
        )
    if ctype is ClaimType.REPAIR_COST and not (inp.has_validated_cost_source or inp.human_confirmed):
        return _insufficient(
            ReasonCode.COST_NOT_VALIDATED,
            "Repair cost needs a validated cost source or human confirmation.",
        )
    if ctype is ClaimType.DAMAGE_CAUSE and not inp.human_confirmed:
        return _insufficient(
            ReasonCode.CAUSE_NOT_ASSERTABLE,
            "Damage cause cannot be asserted from appearance alone.",
        )
    if ctype is ClaimType.DAMAGE_SEVERITY and not inp.has_validated_severity_policy:
        return _insufficient(
            ReasonCode.SEVERITY_NOT_VALIDATED,
            "No validated severity policy; a detector class or score is not severity evidence.",
        )
    if ctype in (ClaimType.REPAIR_RECOMMENDATION, ClaimType.OTHER) and not inp.human_confirmed:
        return _insufficient(
            ReasonCode.NOT_VISUALLY_DECIDABLE,
            "This claim type cannot be decided from photos; needs human confirmation.",
        )

    if inp.association is EvidenceAssociation.NONE_CITED:
        return _insufficient(ReasonCode.NO_EVIDENCE_CITED, "No evidence was cited for this claim.")
    if inp.association is not EvidenceAssociation.ASSOCIATED:
        return _insufficient(
            ReasonCode.EVIDENCE_UNRESOLVED, "Cited evidence does not belong to this inspection."
        )

    if inp.integrity is not EvidenceIntegrity.OK:
        reason = _INTEGRITY_REASON.get(inp.integrity, ReasonCode.INTEGRITY_UNCHECKED)
        return _insufficient(reason, "Evidence integrity is not verified OK.")

    v_raw = inp.verifier.verdict
    verdict = _coerce_enum(Verdict, v_raw)
    if inp.verifier.errored:
        return _insufficient(ReasonCode.VERIFIER_ERROR, "Verifier failed.")
    if v_raw is None:
        return _insufficient(ReasonCode.VERIFIER_MISSING, "Verifier produced no verdict.")
    if verdict is None:
        return _insufficient(ReasonCode.VERIFIER_INVALID, "Verifier verdict was not recognised.")

    det = inp.detection_match
    uncorroborated = _uncorroborated_reason(ctype, det)
    if verdict is Verdict.SUPPORTED:
        if det is DetectionMatch.CONFLICTS_WITH_CLAIM:
            return _insufficient(
                ReasonCode.DETECTION_CONFLICTS_WITH_SUPPORTED,
                "Verifier supported the claim but the detection conflicts; not promoting.",
            )
        if det is DetectionMatch.NO_DETECTION:
            return ClaimPolicyDecision(
                Verdict.SUPPORTED,
                True,
                (ReasonCode.VERIFIER_SUPPORTED_NO_DETECTION,),
                "Supported by the verifier alone; no detection corroborates it.",
            )
        if uncorroborated is not None:
            return ClaimPolicyDecision(
                Verdict.SUPPORTED,
                True,
                (uncorroborated,),
                "Supported by the verifier alone; detection could not be compared with the claim.",
            )
        if not inp.verifier_independent:
            return ClaimPolicyDecision(
                Verdict.SUPPORTED,
                True,
                (ReasonCode.VERIFIER_NOT_INDEPENDENT,),
                "Supported, but by the same model that wrote the claim; a human confirms it.",
            )
        return ClaimPolicyDecision(
            Verdict.SUPPORTED, False, (ReasonCode.VERIFIER_SUPPORTED,), "Verifier supports the claim."
        )
    if verdict is Verdict.CONTRADICTED:
        if det is DetectionMatch.MATCHES_CLAIM:
            return _insufficient(
                ReasonCode.DETECTION_MATCHES_CONTRADICTED,
                "Verifier contradicted the claim but a detection matches it; needs review.",
            )
        if det is DetectionMatch.NO_DETECTION:
            return _insufficient(
                ReasonCode.NO_DETECTION_NOT_ABSENCE,
                "A missing detection is not evidence of absence; not contradicting.",
            )
        if uncorroborated is not None:
            return ClaimPolicyDecision(
                Verdict.CONTRADICTED,
                True,
                (uncorroborated,),
                "Contradicted by the verifier alone; detection could not be compared with the claim.",
            )
        if det is DetectionMatch.CONFLICTS_WITH_CLAIM:
            return ClaimPolicyDecision(
                Verdict.CONTRADICTED,
                True,
                (ReasonCode.DETECTION_CONFLICT_UNCONFIRMED,),
                "Verifier contradicts the claim and the cited detection disagrees with its wording; "
                "that disagreement comes from the report itself, so a human confirms it.",
            )
        return ClaimPolicyDecision(
            Verdict.CONTRADICTED,
            False,
            (ReasonCode.VERIFIER_CONTRADICTED,),
            "Verifier contradicts the claim from the photos.",
        )
    return _insufficient(ReasonCode.VERIFIER_INSUFFICIENT, "Verifier reports insufficient evidence.")
