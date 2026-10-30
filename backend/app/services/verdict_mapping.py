"""Mapping from the legacy 7-value verification label to the claim-level Verdict.

Three label sets, from widest to narrowest:

- ``VerificationLabel``: every value the database accepts (frozen CHECK list, migration 0001).
- ``CLAIM_LABELS``: labels that may be stored as the raw verifier answer for a report claim
  (everything except ``omitted_damage``, which is a coverage finding, not an answer about a claim).
- ``OFFERED_LABELS``: labels the verifier prompt offers, i.e. exactly the labels that map to a
  ``Verdict``. ``unsupported_severity`` is storable (legacy rows hold it) but not offered: the
  project never defined whether it means "severity overstated" (contradicted) or "severity not
  established" (insufficient evidence), and severity claims are never sent to the verifier
  (``claim_policy`` decides them as ``severity_not_validated``). A model that answers it anyway
  has left the offered schema; it maps to ``None`` and the policy treats it as invalid output.
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

OFFERED_LABELS: frozenset[VerificationLabel] = frozenset(
    label for label, verdict in _LEGACY.items() if verdict is not None
)


def legacy_verdict(label: VerificationLabel | str) -> Verdict | None:
    """Verdict for a legacy label, or None where no factual verdict can be derived."""
    try:
        return _LEGACY[VerificationLabel(label)]
    except ValueError:
        raise ValueError(f"unknown verification label: {label!r}") from None


def is_claim_label(label: VerificationLabel) -> bool:
    """True if `label` can be stored as the raw verifier answer for a report claim."""
    return label in CLAIM_LABELS


def is_offered_label(label: VerificationLabel) -> bool:
    """True if the verifier prompt offers `label` (it maps to a Verdict)."""
    return label in OFFERED_LABELS


def coerce_label(raw: object) -> VerificationLabel | None:
    """The label for an exact label string, else None. Never raises.

    Only a plain ``str`` (or a ``VerificationLabel``) is accepted, so an object with a
    hostile ``__hash__``/``__eq__`` or a ``str`` subclass cannot raise out of, or steer, the lookup.
    No case folding or whitespace stripping: ``" Supported"`` is not a label.
    """
    if isinstance(raw, VerificationLabel):
        return raw
    if type(raw) is not str:
        return None
    try:
        return VerificationLabel(raw)
    except ValueError:
        return None
