import pytest

from app.evaluation.records import (
    Configuration,
    Disposition,
    GoldLabel,
    baseline_outcome,
    disposition_of,
    join_labels,
    outcome_from_decision,
)
from app.schemas.claims import ClaimType, Verdict
from app.services.claim_policy import (
    ClaimPolicyInput,
    DetectionMatch,
    EvidenceAssociation,
    EvidenceIntegrity,
    ReasonCode,
    VerifierOutcome,
    evaluate_claim,
)


@pytest.mark.parametrize(
    "verdict,review,expected",
    [
        (None, False, Disposition.ACCEPTED),
        (Verdict.SUPPORTED, False, Disposition.ACCEPTED),
        (Verdict.CONTRADICTED, False, Disposition.REJECTED),
        (Verdict.INSUFFICIENT_EVIDENCE, False, Disposition.HUMAN_REVIEW),
        (Verdict.SUPPORTED, True, Disposition.HUMAN_REVIEW),
        (Verdict.CONTRADICTED, True, Disposition.HUMAN_REVIEW),
        (None, True, Disposition.HUMAN_REVIEW),
    ],
)
def test_disposition_of(verdict, review, expected):
    assert disposition_of(verdict, review) is expected


def _decide(verdict=Verdict.SUPPORTED, integrity=EvidenceIntegrity.OK, association=EvidenceAssociation.ASSOCIATED,
            errored=False):
    return evaluate_claim(
        ClaimPolicyInput(ClaimType.DAMAGE_TYPE, VerifierOutcome(verdict, errored), integrity, association,
                         DetectionMatch.MATCHES_CLAIM)
    )


def test_outcome_from_decision_copies_the_policy_decision():
    decision = _decide()
    o = outcome_from_decision("k1", Configuration.HYBRID, decision, Verdict.SUPPORTED)
    assert (o.verdict, o.needs_human_review, o.reason_codes) == (
        decision.verdict, decision.needs_human_review, decision.reason_codes
    )
    assert o.disposition is Disposition.ACCEPTED
    assert not (o.integrity_failure or o.verifier_output_invalid or o.verifier_failed)


@pytest.mark.parametrize(
    "kwargs,flag",
    [
        ({"verdict": "garbage"}, "verifier_output_invalid"),
        ({"verdict": None}, "verifier_output_invalid"),
        ({"errored": True}, "verifier_failed"),
        ({"association": EvidenceAssociation.UNRESOLVED}, "integrity_failure"),
        ({"integrity": EvidenceIntegrity.MISMATCHED}, "integrity_failure"),
        ({"integrity": EvidenceIntegrity.UNCHECKED}, "integrity_failure"),
    ],
)
def test_failure_flags_follow_reason_codes(kwargs, flag):
    o = outcome_from_decision("k", Configuration.HYBRID, _decide(**kwargs))
    assert getattr(o, flag)
    assert o.disposition is Disposition.HUMAN_REVIEW


def test_no_evidence_cited_is_not_an_integrity_failure():
    o = outcome_from_decision("k", Configuration.HYBRID, _decide(association=EvidenceAssociation.NONE_CITED))
    assert o.reason_codes == (ReasonCode.NO_EVIDENCE_CITED,)
    assert not o.integrity_failure


def test_baseline_accepts_every_claim_unverified():
    o = baseline_outcome("k")
    assert o.configuration is Configuration.BASELINE
    assert o.disposition is Disposition.ACCEPTED
    assert o.verdict is None and not o.verifier_called


def test_join_labels_never_guesses_missing_labels():
    joined = join_labels([baseline_outcome("a"), baseline_outcome("b")], {"a": GoldLabel.CONTRADICTED})
    assert [(j.outcome.claim_key, j.gold, j.labelled) for j in joined] == [
        ("a", GoldLabel.CONTRADICTED, True),
        ("b", None, False),
    ]
