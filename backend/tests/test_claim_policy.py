import dataclasses
import itertools
import re
from pathlib import Path

import pytest

from app.schemas.claims import ClaimType, Verdict
from app.services import claim_policy as cp
from app.services.claim_policy import (
    ClaimPolicyInput,
    DetectionMatch,
    EvidenceAssociation,
    EvidenceIntegrity,
    ReasonCode,
    VerifierOutcome,
    evaluate_claim,
)

VISUAL = {
    ClaimType.VISIBLE_DAMAGE_PRESENCE,
    ClaimType.DAMAGE_TYPE,
    ClaimType.DAMAGE_LOCATION,
}
OK = EvidenceIntegrity.OK
ASSOC = EvidenceAssociation.ASSOCIATED
UNCORROBORATED = {
    DetectionMatch.NO_DETECTION,
    DetectionMatch.INCONCLUSIVE,
    DetectionMatch.UNAVAILABLE,
    DetectionMatch.NOT_APPLICABLE,
}


def make(ctype=ClaimType.VISIBLE_DAMAGE_PRESENCE, verdict=Verdict.SUPPORTED, integrity=OK,
         association=ASSOC, det=DetectionMatch.MATCHES_CLAIM, errored=False, **kw):
    return ClaimPolicyInput(ctype, VerifierOutcome(verdict, errored), integrity, association, det, **kw)


ALL = list(
    itertools.product(
        list(ClaimType) + ["garbage", "", None],
        [*Verdict, None],
        EvidenceIntegrity,
        EvidenceAssociation,
        DetectionMatch,
    )
)


@pytest.mark.parametrize("ctype,verdict,integ,assoc,det", ALL)
def test_global_invariants(ctype, verdict, integ, assoc, det):
    inp = make(ctype, verdict, integ, assoc, det)
    d = evaluate_claim(inp)
    assert isinstance(d.verdict, Verdict)
    assert isinstance(d.needs_human_review, bool)
    assert d.reason_codes and all(isinstance(r, ReasonCode) for r in d.reason_codes)
    assert d.explanation
    assert evaluate_claim(inp) == d
    if d.verdict is Verdict.SUPPORTED:
        assert integ is OK and assoc is ASSOC and ctype in VISUAL
        assert verdict is Verdict.SUPPORTED and det is not DetectionMatch.CONFLICTS_WITH_CLAIM
    if det is DetectionMatch.NO_DETECTION:
        assert d.verdict is not Verdict.CONTRADICTED
    if d.verdict is Verdict.CONTRADICTED:
        assert integ is OK and assoc is ASSOC and ctype in VISUAL
    if verdict is None:
        assert d.verdict is Verdict.INSUFFICIENT_EVIDENCE and d.needs_human_review
    if d.verdict is Verdict.INSUFFICIENT_EVIDENCE:
        assert d.needs_human_review
    if d.verdict is Verdict.SUPPORTED and not d.needs_human_review:
        assert det is DetectionMatch.MATCHES_CLAIM
    if d.verdict is Verdict.CONTRADICTED and ctype in VISUAL:
        assert d.needs_human_review
    if d.verdict is not Verdict.INSUFFICIENT_EVIDENCE and det in UNCORROBORATED:
        assert d.needs_human_review


def test_inputs_frozen_and_not_mutated():
    inp = make()
    before = dataclasses.asdict(inp)
    evaluate_claim(inp)
    assert dataclasses.asdict(inp) == before
    with pytest.raises(dataclasses.FrozenInstanceError):
        inp.integrity = EvidenceIntegrity.MISSING  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        evaluate_claim(inp).verdict = Verdict.SUPPORTED  # type: ignore[misc]


def test_no_confidence_input():
    assert not any("confidence" in f.name for f in dataclasses.fields(ClaimPolicyInput))


@pytest.mark.parametrize("ctype", sorted(VISUAL, key=lambda c: c.value))
def test_supported_passthrough(ctype):
    d = evaluate_claim(make(ctype, det=DetectionMatch.MATCHES_CLAIM))
    assert (d.verdict, d.needs_human_review, d.reason_codes) == (
        Verdict.SUPPORTED, False, (ReasonCode.VERIFIER_SUPPORTED,))


@pytest.mark.parametrize("ctype", sorted(VISUAL, key=lambda c: c.value))
def test_contradicted_with_conflicting_detection_still_needs_review(ctype):
    d = evaluate_claim(make(ctype, verdict=Verdict.CONTRADICTED, det=DetectionMatch.CONFLICTS_WITH_CLAIM))
    assert (d.verdict, d.needs_human_review, d.reason_codes) == (
        Verdict.CONTRADICTED, True, (ReasonCode.DETECTION_CONFLICT_UNCONFIRMED,))


def test_contradicted_non_visual_claim_cleared_by_its_gate_is_unchanged():
    d = evaluate_claim(make(ClaimType.DAMAGE_SEVERITY, verdict=Verdict.CONTRADICTED,
                            det=DetectionMatch.NOT_APPLICABLE, has_validated_severity_policy=True))
    assert (d.verdict, d.needs_human_review) == (Verdict.CONTRADICTED, False)


def test_regression_off_vocabulary_damage_type_never_finalises_a_contradiction():
    import uuid

    from app.schemas.claims import ExtractedClaim
    from app.services.evidence_association import PredictionFacts, detection_signal

    img, run, pred = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    claim = ExtractedClaim(claim_id="c1", text="The headlight shows shattered glass", claim_type="damage_type",
                           damage_type="shattered glass", image_ids=[img], prediction_ids=[pred])
    det, _ = detection_signal(claim, [img], {pred: PredictionFacts(img, run, "glass shatter", None)},
                              {img: run}, {img: 1})
    assert det is DetectionMatch.CONFLICTS_WITH_CLAIM
    d = evaluate_claim(make(ClaimType.DAMAGE_TYPE, verdict=Verdict.CONTRADICTED, det=det))
    assert d.verdict is Verdict.CONTRADICTED and d.needs_human_review


@pytest.mark.parametrize("ctype", sorted(VISUAL, key=lambda c: c.value))
@pytest.mark.parametrize("v", [Verdict.SUPPORTED, Verdict.CONTRADICTED])
@pytest.mark.parametrize("det,reason", [
    (DetectionMatch.UNAVAILABLE, ReasonCode.DETECTION_UNAVAILABLE),
    (DetectionMatch.INCONCLUSIVE, ReasonCode.DETECTION_INCONCLUSIVE),
    (DetectionMatch.NOT_APPLICABLE, ReasonCode.DETECTION_INCONCLUSIVE),
])
def test_uncorroborated_visual_claim_keeps_verifier_verdict_but_needs_review(ctype, v, det, reason):
    d = evaluate_claim(make(ctype, verdict=v, det=det))
    assert (d.verdict, d.needs_human_review, d.reason_codes) == (v, True, (reason,))


@pytest.mark.parametrize("det", [DetectionMatch.UNAVAILABLE, DetectionMatch.INCONCLUSIVE])
def test_missing_detection_never_less_scrutiny_than_zero_detections(det):
    zero = evaluate_claim(make(det=DetectionMatch.NO_DETECTION))
    missing = evaluate_claim(make(det=det))
    assert zero.needs_human_review and missing.needs_human_review
    assert missing.verdict is zero.verdict is Verdict.SUPPORTED


def test_non_visual_claim_cleared_by_its_gate_is_not_reviewed_for_detection():
    d = evaluate_claim(make(ClaimType.DAMAGE_SEVERITY, det=DetectionMatch.NOT_APPLICABLE,
                            has_validated_severity_policy=True))
    assert (d.verdict, d.needs_human_review) == (Verdict.SUPPORTED, False)


@pytest.mark.parametrize("flags", [
    {"has_validated_cost_source": True},
    {"has_validated_severity_policy": True},
    {"human_confirmed": True},
    {"has_validated_cost_source": True, "has_validated_severity_policy": True, "human_confirmed": True},
])
@pytest.mark.parametrize("ctype", list(ClaimType))
@pytest.mark.parametrize("det", list(DetectionMatch))
def test_gates_never_bypass_hidden_damage_or_evidence_checks(flags, ctype, det):
    d = evaluate_claim(make(ctype, det=det, integrity=EvidenceIntegrity.MISMATCHED, **flags))
    assert d.verdict is Verdict.INSUFFICIENT_EVIDENCE and d.needs_human_review
    d = evaluate_claim(make(ClaimType.HIDDEN_STRUCTURAL_DAMAGE, det=det, **flags))
    assert d.reason_codes == (ReasonCode.HIDDEN_DAMAGE_NOT_VISIBLE,)


def test_policy_version_bumped_for_review_routing_change():
    assert cp.POLICY_VERSION == "claim-policy-5"


@pytest.mark.parametrize("det", list(DetectionMatch))
@pytest.mark.parametrize("ctype", sorted(VISUAL))
def test_verifier_insufficient_always_needs_review(ctype, det):
    d = evaluate_claim(make(ctype, verdict=Verdict.INSUFFICIENT_EVIDENCE, det=det))
    assert d.verdict is Verdict.INSUFFICIENT_EVIDENCE and d.needs_human_review
    assert d.reason_codes == (ReasonCode.VERIFIER_INSUFFICIENT,)


def assert_review(d, reason):
    assert d.verdict is Verdict.INSUFFICIENT_EVIDENCE and d.needs_human_review
    assert reason in d.reason_codes


@pytest.mark.parametrize("v", [Verdict.SUPPORTED, Verdict.CONTRADICTED])
def test_inv1_no_detection_not_absence(v):
    d = evaluate_claim(make(verdict=v, det=DetectionMatch.NO_DETECTION))
    assert d.verdict is not Verdict.CONTRADICTED
    if v is Verdict.CONTRADICTED:
        assert_review(d, ReasonCode.NO_DETECTION_NOT_ABSENCE)
    else:
        assert d.needs_human_review


def test_inv3_hidden():
    assert_review(evaluate_claim(make(ClaimType.HIDDEN_STRUCTURAL_DAMAGE, human_confirmed=True)),
                  ReasonCode.HIDDEN_DAMAGE_NOT_VISIBLE)


@pytest.mark.parametrize("flag", [{}, {"has_validated_severity_policy": True}])
def test_inv4_cost_requires_source(flag):
    assert_review(evaluate_claim(make(ClaimType.REPAIR_COST, **flag)), ReasonCode.COST_NOT_VALIDATED)


@pytest.mark.parametrize("flag", ["has_validated_cost_source", "human_confirmed"])
def test_inv4_cost_allowed(flag):
    d = evaluate_claim(make(ClaimType.REPAIR_COST, **{flag: True}))
    assert d.verdict is Verdict.SUPPORTED


def test_inv5_cause():
    assert_review(evaluate_claim(make(ClaimType.DAMAGE_CAUSE, has_validated_cost_source=True)),
                  ReasonCode.CAUSE_NOT_ASSERTABLE)
    assert evaluate_claim(make(ClaimType.DAMAGE_CAUSE, human_confirmed=True)).verdict is Verdict.SUPPORTED


@pytest.mark.parametrize("det", list(DetectionMatch))
def test_inv6_severity_not_from_detection(det):
    d = evaluate_claim(make(ClaimType.DAMAGE_SEVERITY, det=det))
    assert_review(d, ReasonCode.SEVERITY_NOT_VALIDATED)


def test_inv6_severity_validated_policy_allows_verifier():
    d = evaluate_claim(make(ClaimType.DAMAGE_SEVERITY, has_validated_severity_policy=True))
    assert d.verdict is Verdict.SUPPORTED
    d = evaluate_claim(make(ClaimType.DAMAGE_SEVERITY, verdict=Verdict.INSUFFICIENT_EVIDENCE,
                            has_validated_severity_policy=True))
    assert d.verdict is Verdict.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize("integ,reason", [
    (EvidenceIntegrity.UNCHECKED, ReasonCode.INTEGRITY_UNCHECKED),
    (EvidenceIntegrity.MISSING, ReasonCode.INTEGRITY_MISSING),
    (EvidenceIntegrity.MISMATCHED, ReasonCode.INTEGRITY_MISMATCHED),
    (EvidenceIntegrity.CORRUPT, ReasonCode.INTEGRITY_CORRUPT),
])
@pytest.mark.parametrize("v", [Verdict.SUPPORTED, Verdict.CONTRADICTED])
def test_inv7_integrity(integ, reason, v):
    assert_review(evaluate_claim(make(verdict=v, integrity=integ)), reason)


@pytest.mark.parametrize("assoc,reason", [
    (EvidenceAssociation.NONE_CITED, ReasonCode.NO_EVIDENCE_CITED),
    (EvidenceAssociation.UNRESOLVED, ReasonCode.EVIDENCE_UNRESOLVED),
])
def test_association(assoc, reason):
    assert_review(evaluate_claim(make(association=assoc)), reason)


def test_inv8_verifier_missing_error_invalid():
    assert_review(evaluate_claim(make(verdict=None)), ReasonCode.VERIFIER_MISSING)
    assert_review(evaluate_claim(make(verdict=Verdict.SUPPORTED, errored=True)), ReasonCode.VERIFIER_ERROR)
    assert_review(evaluate_claim(make(verdict="bogus")), ReasonCode.VERIFIER_INVALID)
    assert evaluate_claim(make(verdict="supported")).verdict is Verdict.SUPPORTED


@pytest.mark.parametrize("bad", ["garbage", "", "VISIBLE_DAMAGE_PRESENCE ", None, 5, object()])
def test_inv9_unknown_claim_type(bad):
    assert_review(evaluate_claim(make(bad)), ReasonCode.UNKNOWN_CLAIM_TYPE)


def test_string_claim_type_coerced():
    assert evaluate_claim(make("damage_type")).verdict is Verdict.SUPPORTED


def test_inv10_one_decision_per_input():
    assert all(isinstance(evaluate_claim(make(c)), cp.ClaimPolicyDecision) for c in ClaimType)


def test_inv11_disagreement():
    assert_review(evaluate_claim(make(det=DetectionMatch.CONFLICTS_WITH_CLAIM)),
                  ReasonCode.DETECTION_CONFLICTS_WITH_SUPPORTED)
    assert_review(evaluate_claim(make(verdict=Verdict.CONTRADICTED, det=DetectionMatch.MATCHES_CLAIM)),
                  ReasonCode.DETECTION_MATCHES_CONTRADICTED)


@pytest.mark.parametrize("ctype", [ClaimType.HIDDEN_STRUCTURAL_DAMAGE, ClaimType.REPAIR_COST,
                                   ClaimType.DAMAGE_CAUSE, ClaimType.DAMAGE_SEVERITY,
                                   ClaimType.REPAIR_RECOMMENDATION, ClaimType.OTHER])
@pytest.mark.parametrize("v", list(Verdict))
def test_inv12_policy_beats_verifier(ctype, v):
    d = evaluate_claim(make(ctype, verdict=v))
    assert d.verdict is Verdict.INSUFFICIENT_EVIDENCE and d.needs_human_review


def test_rule_order_type_before_integrity():
    d = evaluate_claim(make(ClaimType.HIDDEN_STRUCTURAL_DAMAGE, integrity=EvidenceIntegrity.CORRUPT))
    assert d.reason_codes == (ReasonCode.HIDDEN_DAMAGE_NOT_VISIBLE,)
    d = evaluate_claim(make(integrity=EvidenceIntegrity.CORRUPT, association=EvidenceAssociation.UNRESOLVED,
                            verdict=None))
    assert d.reason_codes == (ReasonCode.EVIDENCE_UNRESOLVED,)
    d = evaluate_claim(make(integrity=EvidenceIntegrity.UNCHECKED, association=EvidenceAssociation.NONE_CITED,
                            verdict=None))
    assert d.reason_codes == (ReasonCode.NO_EVIDENCE_CITED,)
    d = evaluate_claim(make(integrity=EvidenceIntegrity.CORRUPT, verdict=None))
    assert d.reason_codes == (ReasonCode.INTEGRITY_CORRUPT,)
    d = evaluate_claim(make(association=EvidenceAssociation.UNRESOLVED, verdict=None))
    assert d.reason_codes == (ReasonCode.EVIDENCE_UNRESOLVED,)


def test_reason_codes_stable_strings():
    assert all(re.fullmatch(r"[a-z_]+", r.value) for r in ReasonCode)


def test_module_is_pure_imports():
    src = Path(cp.__file__).read_text()
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", src, re.M)
    for banned in ("sqlalchemy", "torch", "ultralytics", "requests", "httpx", "logging", "os"):
        assert not any(i == banned or i.startswith(banned + ".") for i in imports)
    assert not any(i.startswith("app.models") or i.startswith("app.db") for i in imports)


@pytest.mark.parametrize("ctype", sorted(VISUAL))
def test_supported_by_a_non_independent_verifier_needs_review(ctype):
    d = evaluate_claim(make(ctype, det=DetectionMatch.MATCHES_CLAIM, verifier_independent=False))
    assert (d.verdict, d.needs_human_review, d.reason_codes) == (
        Verdict.SUPPORTED, True, (ReasonCode.VERIFIER_NOT_INDEPENDENT,))
    assert not evaluate_claim(make(ctype, det=DetectionMatch.MATCHES_CLAIM)).needs_human_review


@pytest.mark.parametrize("ctype", list(ClaimType))
@pytest.mark.parametrize("verdict", [*Verdict, None])
@pytest.mark.parametrize("det", list(DetectionMatch))
def test_non_independent_verifier_never_yields_an_unreviewed_outcome(ctype, verdict, det):
    d = evaluate_claim(make(ctype, verdict=verdict, det=det, verifier_independent=False))
    assert d.needs_human_review
