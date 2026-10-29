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
