import pytest

from app.models import VerificationLabel
from app.schemas.claims import Verdict
from app.services.verdict_mapping import legacy_verdict

EXPECTED = {
    "supported": Verdict.SUPPORTED,
    "contradicted": Verdict.CONTRADICTED,
    "wrong_location": Verdict.CONTRADICTED,
    "wrong_damage_type": Verdict.CONTRADICTED,
    "unverifiable": Verdict.INSUFFICIENT_EVIDENCE,
    "unsupported_severity": None,
    "omitted_damage": None,
}


def test_every_label_is_covered():
    assert set(EXPECTED) == {m.value for m in VerificationLabel}


@pytest.mark.parametrize("label,verdict", EXPECTED.items())
def test_mapping(label, verdict):
    assert legacy_verdict(label) is verdict
    assert legacy_verdict(VerificationLabel(label)) is verdict


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        legacy_verdict("something_new")


def test_claim_labels_exclude_only_omitted_damage():
    from app.services.verdict_mapping import CLAIM_LABELS, is_claim_label

    assert set(VerificationLabel) - CLAIM_LABELS == {VerificationLabel.OMITTED_DAMAGE}
    assert not is_claim_label(VerificationLabel.OMITTED_DAMAGE)
    assert all(is_claim_label(label) for label in CLAIM_LABELS)


def test_offered_labels_are_exactly_the_labels_with_a_verdict():
    from app.services.verdict_mapping import CLAIM_LABELS, OFFERED_LABELS, is_offered_label

    assert OFFERED_LABELS == {VerificationLabel(v) for v, verdict in EXPECTED.items() if verdict is not None}
    assert OFFERED_LABELS <= CLAIM_LABELS
    assert CLAIM_LABELS - OFFERED_LABELS == {VerificationLabel.UNSUPPORTED_SEVERITY}
    assert not is_offered_label(VerificationLabel.UNSUPPORTED_SEVERITY)
    assert not is_offered_label(VerificationLabel.OMITTED_DAMAGE)


def test_unsupported_severity_stays_storable_but_has_no_verdict():
    from app.services.verdict_mapping import is_claim_label

    assert is_claim_label(VerificationLabel.UNSUPPORTED_SEVERITY)
    assert legacy_verdict("unsupported_severity") is None


class _HostileHash(str):
    def __hash__(self):
        raise RuntimeError("boom")

    def __eq__(self, other):
        raise RuntimeError("boom")


class _Unhashable:
    __hash__ = None


@pytest.mark.parametrize("raw,expected", [
    ("supported", VerificationLabel.SUPPORTED),
    (VerificationLabel.UNVERIFIABLE, VerificationLabel.UNVERIFIABLE),
    ("unsupported_severity", VerificationLabel.UNSUPPORTED_SEVERITY),
    (" supported", None),
    ("Supported", None),
    ("supported\x00", None),
    ("", None),
    (None, None),
    (1, None),
    (True, None),
    (b"supported", None),
    (["supported"], None),
    ({"label": "supported"}, None),
    (_Unhashable(), None),
    (_HostileHash("supported"), None),
])
def test_coerce_label_is_exact_and_never_raises(raw, expected):
    from app.services.verdict_mapping import coerce_label

    assert coerce_label(raw) is expected


def test_hostile_label_object_is_a_per_claim_outcome_not_a_run_failure():
    import uuid
    from pathlib import Path

    from app.schemas.claims import ClaimType, ExtractedClaim
    from app.services import verification_service as vs
    from app.services.claim_policy import DetectionMatch, EvidenceAssociation, EvidenceIntegrity
    from app.services.evidence_association import ClaimEvidence
    from app.services.types import ClaimVerdict, ModelInfo

    image = uuid.UUID(int=1)
    result = ClaimVerdict("dent", "supported", "x")
    object.__setattr__(result, "label", _HostileHash("supported"))

    class Verifier:
        model = ModelInfo("fake", "0")

        def verify(self, claims, evidence):
            return [result]

    claim = ExtractedClaim(claim_id="c1", text="dent", claim_type=ClaimType.VISIBLE_DAMAGE_PRESENCE,
                           image_ids=[image])
    ev = ClaimEvidence("c1", EvidenceAssociation.ASSOCIATED, EvidenceIntegrity.OK, DetectionMatch.MATCHES_CLAIM,
                       ((image, Path("unused.jpg")),), (), (), ())
    raw = vs._ask_verifier(Verifier(), claim, ev)
    assert raw.label is None and raw.outcome.verdict not in (Verdict.SUPPORTED, "supported")
