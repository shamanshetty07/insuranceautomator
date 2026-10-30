"""claim-policy-7 rules and their wiring. Synthetic inputs: these show policy behaviour, not model accuracy."""

import dataclasses
import itertools
import types
import uuid
from pathlib import Path

import pytest

from app.evaluation.harness import TrialSet, run_configurations
from app.evaluation.manifest import load_manifest
from app.evaluation.records import Configuration, Disposition, disposition_of
from app.models.verification_result import VerificationLabel
from app.schemas.claims import ClaimType, EvidenceRecord, ExtractedClaim, Verdict
from app.services import verification_service as vs
from app.services.claim_policy import (
    ClaimPolicyInput,
    DetectionMatch,
    EvidenceAssociation,
    EvidenceIntegrity,
    EvidenceQuality,
    ReasonCode,
    VerifierOutcome,
    evaluate_claim,
)
from app.services.evidence_association import ClaimEvidence
from app.services.evidence_quality import QualityLimits
from app.services.types import ModelInfo

FIXTURES = Path(__file__).parent / "fixtures" / "evaluation"
IMG, PRED, RUN = uuid.UUID(int=1), uuid.UUID(int=2), uuid.UUID(int=3)
LIMITS = QualityLimits(min_side_px=32, vlm_max_image_pixels=802_816)

ALL = list(itertools.product(
    list(ClaimType) + ["garbage", None],
    [*Verdict, "garbage", None],
    EvidenceIntegrity,
    EvidenceAssociation,
    DetectionMatch,
    [True, False],
))


def _inp(ctype, verdict, integ, assoc, det, independent, **kw):
    return ClaimPolicyInput(ctype, VerifierOutcome(verdict), integ, assoc, det, verifier_independent=independent, **kw)


@pytest.mark.parametrize("flags", [
    {"claim_positive": False},
    {"verifier_consistent": False},
    {"evidence_quality": EvidenceQuality.QUESTIONABLE},
    {"claim_positive": False, "verifier_consistent": False, "evidence_quality": EvidenceQuality.QUESTIONABLE},
])
def test_escalations_only_ever_add_review(flags):
    for args in ALL:
        base = evaluate_claim(_inp(*args))
        hardened = evaluate_claim(_inp(*args, **flags))
        assert hardened.verdict is base.verdict
        assert hardened.needs_human_review >= base.needs_human_review
        assert base.reason_codes == hardened.reason_codes[: len(base.reason_codes)]
        assert disposition_of(hardened.verdict, hardened.needs_human_review) in (
            Disposition.HUMAN_REVIEW, disposition_of(base.verdict, base.needs_human_review)
        )
        if hardened.needs_human_review is False:
            assert hardened == base


def test_insufficient_quality_never_reaches_the_verifier_and_never_rejects():
    for args in ALL:
        d = evaluate_claim(_inp(*args, evidence_quality=EvidenceQuality.INSUFFICIENT))
        assert d.verdict is not Verdict.SUPPORTED or d.needs_human_review
        assert d.verdict is not Verdict.CONTRADICTED or d.needs_human_review
    claim = ExtractedClaim(claim_id="c1", text="There is a dent.", claim_type=ClaimType.DAMAGE_TYPE, damage_type="dent")
    assert vs._decided_without_verifier(claim, _ev(), EvidenceQuality.INSUFFICIENT)
    assert not vs._decided_without_verifier(claim, _ev(), EvidenceQuality.QUESTIONABLE)
    d = evaluate_claim(vs._policy_input(claim, _ev(), VerifierOutcome(Verdict.SUPPORTED),
                                        quality=EvidenceQuality.INSUFFICIENT))
    assert (d.verdict, d.needs_human_review, d.reason_codes) == (
        Verdict.INSUFFICIENT_EVIDENCE, True, (ReasonCode.EVIDENCE_QUALITY_INSUFFICIENT,)
    )


def test_corroborated_independent_support_is_still_the_only_unreviewed_acceptance():
    accepted = [args for args in ALL if (d := evaluate_claim(_inp(*args))).verdict is Verdict.SUPPORTED
                and not d.needs_human_review]
    assert accepted and all(
        a[1] is Verdict.SUPPORTED and a[2] is EvidenceIntegrity.OK and a[3] is EvidenceAssociation.ASSOCIATED
        and a[4] is DetectionMatch.MATCHES_CLAIM and a[5] is True for a in accepted
    )


def _ev(det=DetectionMatch.MATCHES_CLAIM):
    return ClaimEvidence("c1", EvidenceAssociation.ASSOCIATED, EvidenceIntegrity.OK, det,
                         ((IMG, Path("unused.jpg")),), (PRED,), (), ())


@pytest.mark.parametrize("text", [
    "There is no dent on the door.",
    "The bumper appears undamaged.",
    "No visible scratch is present.",
    "The windshield may be cracked.",
    "The door is damaged, but the bumper is not.",
])
@pytest.mark.parametrize("verdict", [Verdict.SUPPORTED, Verdict.CONTRADICTED])
def test_non_positive_claims_are_reviewed_without_changing_the_verdict(text, verdict):
    claim = ExtractedClaim(claim_id="c1", text=text, claim_type=ClaimType.VISIBLE_DAMAGE_PRESENCE)
    inp = vs._policy_input(claim, _ev(), VerifierOutcome(verdict))
    d = evaluate_claim(inp)
    assert not inp.claim_positive and d.needs_human_review
    assert d.verdict is evaluate_claim(dataclasses.replace(inp, claim_positive=True)).verdict
    assert ReasonCode.CLAIM_NOT_POSITIVE_ASSERTION in d.reason_codes


def test_plain_positive_claim_is_unaffected():
    claim = ExtractedClaim(claim_id="c1", text="There is a dent on the door.", claim_type=ClaimType.DAMAGE_TYPE,
                           damage_type="dent")
    d = evaluate_claim(vs._policy_input(claim, _ev(), VerifierOutcome(Verdict.SUPPORTED)))
    assert (d.verdict, d.needs_human_review, d.reason_codes) == (
        Verdict.SUPPORTED, False, (ReasonCode.VERIFIER_SUPPORTED,)
    )


def test_non_visual_claims_ignore_claim_polarity():
    claim = ExtractedClaim(claim_id="c1", text="No repair is needed.", claim_type=ClaimType.REPAIR_RECOMMENDATION)
    d = evaluate_claim(vs._policy_input(claim, _ev(DetectionMatch.NOT_APPLICABLE), VerifierOutcome(Verdict.SUPPORTED)))
    assert d.reason_codes == (ReasonCode.NOT_VISUALLY_DECIDABLE,)


class _FakeDb:
    def __init__(self, images=(), predictions=(), prior_rows=()):
        self.images, self.predictions, self.prior_rows = images, predictions, prior_rows
        self.queries = 0

    def scalars(self, stmt):
        self.queries += 1
        return list(self.images)

    def execute(self, stmt):
        self.queries += 1
        return list(self.prior_rows if "verification_results" in str(stmt) else self.predictions)


def _image(w, h):
    return types.SimpleNamespace(id=IMG, width_px=w, height_px=h)


def _pred(confidence=0.9, threshold=0.25):
    pred = types.SimpleNamespace(id=PRED, detection_run_id=RUN, damage_type="dent", confidence=confidence,
                                 bounding_box={"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5})
    return (pred, IMG, {"confidence_threshold": threshold})


def _claim(ctype=ClaimType.DAMAGE_TYPE):
    return ExtractedClaim(claim_id="c1", text="There is a dent.", claim_type=ctype, damage_type="dent")


@pytest.mark.parametrize("image,pred,quality,finding", [
    (_image(640, 480), _pred(), EvidenceQuality.ADEQUATE, None),
    (_image(4000, 3000), _pred(), EvidenceQuality.ADEQUATE, None),
    (_image(12, 9), _pred(), EvidenceQuality.INSUFFICIENT, "image_below_min_side"),
    (_image(None, None), _pred(), EvidenceQuality.QUESTIONABLE, "image_dimensions_unknown"),
    (_image(640, 480), _pred(confidence=0.1), EvidenceQuality.QUESTIONABLE, "detection_below_run_threshold"),
    (_image(640, 480), _pred(confidence=0.999, threshold=0.25), EvidenceQuality.ADEQUATE, None),
])
def test_assess_claim_quality(image, pred, quality, finding):
    [q] = vs.assess_claim_quality(_FakeDb([image], [pred]), [_claim()], [_ev()], LIMITS)
    assert q.quality is quality
    assert (finding is None) == (q.findings == ())
    if finding:
        assert q.findings[0].startswith(finding)


def test_quality_is_not_assessed_for_non_visual_claims_and_needs_no_query_without_evidence():
    [q] = vs.assess_claim_quality(_FakeDb([_image(1, 1)]), [_claim(ClaimType.REPAIR_COST)], [_ev()], LIMITS)
    assert q == vs.ClaimQuality()
    db = _FakeDb()
    empty = ClaimEvidence("c1", EvidenceAssociation.NONE_CITED, EvidenceIntegrity.UNCHECKED,
                          DetectionMatch.INCONCLUSIVE, (), (), (), ())
    assert vs.assess_claim_quality(db, [_claim()], [empty], LIMITS) == [vs.ClaimQuality()]
    assert db.queries == 0


def test_default_quality_limits_come_from_settings_defaults():
    assert vs.default_quality_limits() == QualityLimits(32, 802_816)


def test_load_prior_verdicts_maps_rows_back_to_claim_ids():
    row_id, run_id = uuid.uuid4(), uuid.uuid4()
    rows = [(row_id, VerificationLabel.SUPPORTED, run_id, "qwen", "1", {"model": {"weights_sha256": "a" * 64}})]
    priors = vs.load_prior_verdicts(_FakeDb(prior_rows=rows), uuid.uuid4(), {"c1": row_id})
    [p] = priors["c1"]
    assert (p.run_id, p.raw_label, p.verifier) == (run_id, VerificationLabel.SUPPORTED, ModelInfo("qwen", "1", "a" * 64))
    assert vs.load_prior_verdicts(_FakeDb(), uuid.uuid4(), {}) == {}


def test_old_evidence_records_still_validate_and_quality_findings_are_stored():
    old = {"association": "associated", "integrity": "ok", "detection_match": "matches_claim"}
    assert EvidenceRecord.model_validate(old).quality_findings == ()
    claim = _claim()
    raw = vs._Raw(VerificationLabel.SUPPORTED, VerifierOutcome(Verdict.SUPPORTED), "x", True, False)
    decision = evaluate_claim(vs._policy_input(claim, _ev(), raw.outcome, quality=EvidenceQuality.QUESTIONABLE))
    row = vs._build_row(uuid.uuid4(), claim, _ev(), raw, decision, None,
                        vs.ClaimQuality(EvidenceQuality.QUESTIONABLE, ("detection_below_run_threshold:x",)))
    assert row.evidence_reference["quality_findings"] == ["detection_below_run_threshold:x"]
    assert row.needs_human_review and ReasonCode.EVIDENCE_QUALITY_QUESTIONABLE.value in row.policy_reason_codes


def _trials(**changes):
    trials = TrialSet.model_validate_json((FIXTURES / "synthetic_trials_v0.json").read_text())
    data = trials.model_dump(mode="json")
    data["trials"]["alpha-c01"].update(changes)
    return TrialSet.model_validate(data)


def _hybrid(trials, key="alpha-c01"):
    manifest = load_manifest(FIXTURES / "synthetic_manifest_v0.json")
    out = run_configurations(manifest, trials, force_independent=True)[Configuration.HYBRID]
    return next(o for o in out if o.claim_key == key)


def test_harness_baseline_claim_is_accepted_under_the_counterfactual():
    assert _hybrid(_trials()).disposition is Disposition.ACCEPTED


@pytest.mark.parametrize("priors,accepted", [
    (["supported", "supported"], True),
    (["contradicted"], False),
    (["unverifiable"], False),
    (["wrong_location"], False),
    ([None, "garbage"], True),
])
def test_repeated_runs_that_disagree_force_review_and_agreement_adds_nothing(priors, accepted):
    out = _hybrid(_trials(prior_raw_labels=priors))
    assert (out.disposition is Disposition.ACCEPTED) is accepted
    if not accepted:
        assert out.reason_codes[-1] is ReasonCode.VERIFIER_INCONSISTENT_ACROSS_RUNS
        assert out.verdict is Verdict.SUPPORTED


@pytest.mark.parametrize("quality,reason", [
    ("questionable", ReasonCode.EVIDENCE_QUALITY_QUESTIONABLE),
    ("insufficient", ReasonCode.EVIDENCE_QUALITY_INSUFFICIENT),
])
def test_harness_applies_evidence_quality(quality, reason):
    out = _hybrid(_trials(evidence_quality=quality))
    assert out.disposition is Disposition.HUMAN_REVIEW and reason in out.reason_codes
    assert out.verifier_called is (quality != "insufficient")
