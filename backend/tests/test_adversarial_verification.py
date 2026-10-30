"""SYNTHETIC POLICY TESTS: these establish software behaviour for given inputs and say nothing about Qwen-VL or detector accuracy."""

import itertools
import json
import uuid
from pathlib import Path

import pytest
from PIL import Image, ImageFilter

from app.evaluation.records import Configuration, Disposition, outcome_from_decision
from app.schemas.claims import ClaimExtractionResult, ClaimType, ExtractedClaim, StructuredReport, Verdict
from app.services import verification_service as vs
from app.services.claim_policy import (
    ClaimPolicyInput,
    DetectionMatch as D,
    EvidenceAssociation as A,
    EvidenceIntegrity as I,
    ReasonCode as R,
    VerifierOutcome,
    evaluate_claim,
)
from app.services.evidence_association import (
    ClaimEvidence,
    PredictionFacts,
    detection_signal,
    text_mentions_damage_type,
)
from app.services.evidence_quality import ImageQualityFacts, QualityLimits, assess_image
from app.services.report_service import check_citations
from app.services.types import ClaimVerdict, Detection, EvidenceBundle, ImageRef, ModelInfo
from app.services.vlm.adapters import QwenClaimVerifier, QwenStructuredReportGenerator, VlmLimits
from app.services.vlm.errors import VlmOutputError
from app.services.vlm.parsing import parse_structured_report, parse_verdict
from app.services.vlm.prompts import build_alias_map
from tests.test_vlm import FakeRunner

IMG, IMG2, FOREIGN_IMG = uuid.UUID(int=1), uuid.UUID(int=2), uuid.UUID(int=99)
P1, P2, FOREIGN_P = uuid.UUID(int=11), uuid.UUID(int=12), uuid.UUID(int=98)
RUN, OLD_RUN = uuid.UUID(int=21), uuid.UUID(int=22)
BOX = {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}
FAR = {"x_min": 0.6, "y_min": 0.6, "x_max": 0.9, "y_max": 0.9}
LIMITS = VlmLimits(timeout_seconds=5, max_claims=5, max_images=2, max_image_pixels=200_000)
VISUAL = (ClaimType.VISIBLE_DAMAGE_PRESENCE, ClaimType.DAMAGE_TYPE, ClaimType.DAMAGE_LOCATION)
QUALITY_LIMITS = QualityLimits(min_side_px=32, vlm_max_image_pixels=802_816)


def make_claim(ctype=ClaimType.DAMAGE_TYPE, text="There is a dent on the front door.", damage_type="dent",
               images=(IMG,), preds=(P1,), crop=None):
    evidence = [{"image_id": str(images[0]), "prediction_id": str(preds[0]), "crop": crop}] if crop else []
    return ExtractedClaim(claim_id="c1", text=text, claim_type=ctype, damage_type=damage_type,
                          image_ids=list(images), prediction_ids=list(preds), evidence=evidence)


def make_evidence(det=D.MATCHES_CLAIM, assoc=A.ASSOCIATED, integ=I.OK):
    return ClaimEvidence("c1", assoc, integ, det, ((IMG, Path("unused.jpg")),), (P1,), (), ())


def decide(ctype, verdict, det, *, assoc=A.ASSOCIATED, integ=I.OK, independent=True, errored=False):
    return evaluate_claim(ClaimPolicyInput(
        claim_type=ctype, verifier=VerifierOutcome(verdict=verdict, errored=errored), integrity=integ,
        association=assoc, detection_match=det, verifier_independent=independent,
    ))


def disposition(decision):
    return outcome_from_decision("k", Configuration.HYBRID, decision).disposition


class FakeVerifier:
    model = ModelInfo("fake-verifier", "0")

    def __init__(self, *labels, raises=None, results=None):
        self.labels = itertools.cycle(labels) if labels else None
        self.raises = raises
        self.results = results
        self.calls = 0

    def verify(self, claims, evidence):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        if self.results is not None:
            return self.results
        return [ClaimVerdict(c.text, next(self.labels), "synthetic") for c in claims]


def escalate(verifier, claim, ev, independent=True):
    if vs._decided_without_verifier(claim, ev):
        raw = vs._Raw(None, VerifierOutcome(verdict=None), "", False, False)
    else:
        raw = vs._ask_verifier(verifier, claim, ev)
    decision = evaluate_claim(vs._policy_input(claim, ev, raw.outcome, independent))
    return raw, decision, disposition(decision)


def facts(damage_type="dent", box=BOX, run=RUN, image=IMG):
    return PredictionFacts(image_id=image, run_id=run, damage_type=damage_type, bounding_box=box)


def signal(c, preds=None, latest_run=None, latest_count=None):
    preds = {P1: facts()} if preds is None else preds
    latest_run = {IMG: RUN} if latest_run is None else latest_run
    latest_count = {IMG: max(len(preds), 1)} if latest_count is None else latest_count
    return detection_signal(c, list(c.image_ids), preds, latest_run, latest_count)[0]


def save_image(path, size, blur=False):
    img = Image.new("RGB", size, (180, 180, 180))
    img.paste((20, 20, 20), (0, 0, max(1, size[0] // 2), max(1, size[1] // 2)))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=4))
    img.save(path)
    return path


# Layer 1: application policy (evaluate_claim) -------------------------------------------------


class TestPolicyLayer:
    @pytest.mark.parametrize("case,ctype,verdict,det,independent,expected,review,reason", [
        ("real_damage_independent", ClaimType.VISIBLE_DAMAGE_PRESENCE, Verdict.SUPPORTED, D.MATCHES_CLAIM, True,
         Verdict.SUPPORTED, False, R.VERIFIER_SUPPORTED),
        ("real_damage_self_verified", ClaimType.VISIBLE_DAMAGE_PRESENCE, Verdict.SUPPORTED, D.MATCHES_CLAIM, False,
         Verdict.SUPPORTED, True, R.VERIFIER_NOT_INDEPENDENT),
        ("wrong_location_verifier_contradicts", ClaimType.DAMAGE_LOCATION, Verdict.CONTRADICTED,
         D.CONFLICTS_WITH_CLAIM, True, Verdict.CONTRADICTED, True, R.DETECTION_CONFLICT_UNCONFIRMED),
        ("wrong_location_verifier_supports", ClaimType.DAMAGE_LOCATION, Verdict.SUPPORTED, D.CONFLICTS_WITH_CLAIM,
         True, Verdict.INSUFFICIENT_EVIDENCE, True, R.DETECTION_CONFLICTS_WITH_SUPPORTED),
        ("invented_no_detection_verifier_contradicts", ClaimType.VISIBLE_DAMAGE_PRESENCE, Verdict.CONTRADICTED,
         D.NO_DETECTION, True, Verdict.INSUFFICIENT_EVIDENCE, True, R.NO_DETECTION_NOT_ABSENCE),
        ("invented_no_detection_verifier_wrongly_supports", ClaimType.VISIBLE_DAMAGE_PRESENCE, Verdict.SUPPORTED,
         D.NO_DETECTION, True, Verdict.SUPPORTED, True, R.VERIFIER_SUPPORTED_NO_DETECTION),
        ("wrong_type_detection_conflicts_verifier_supports", ClaimType.DAMAGE_TYPE, Verdict.SUPPORTED,
         D.CONFLICTS_WITH_CLAIM, True, Verdict.INSUFFICIENT_EVIDENCE, True, R.DETECTION_CONFLICTS_WITH_SUPPORTED),
        ("wrong_type_detection_conflicts_verifier_contradicts", ClaimType.DAMAGE_TYPE, Verdict.CONTRADICTED,
         D.CONFLICTS_WITH_CLAIM, True, Verdict.CONTRADICTED, True, R.DETECTION_CONFLICT_UNCONFIRMED),
        ("area_not_visible", ClaimType.DAMAGE_LOCATION, Verdict.INSUFFICIENT_EVIDENCE, D.MATCHES_CLAIM, True,
         Verdict.INSUFFICIENT_EVIDENCE, True, R.VERIFIER_INSUFFICIENT),
        ("ambiguous_detection_supported", ClaimType.DAMAGE_TYPE, Verdict.SUPPORTED, D.INCONCLUSIVE, True,
         Verdict.SUPPORTED, True, R.DETECTION_INCONCLUSIVE),
        ("verifier_contradicts_matching_detection", ClaimType.DAMAGE_TYPE, Verdict.CONTRADICTED, D.MATCHES_CLAIM,
         True, Verdict.INSUFFICIENT_EVIDENCE, True, R.DETECTION_MATCHES_CONTRADICTED),
        ("detection_unavailable_supported", ClaimType.DAMAGE_TYPE, Verdict.SUPPORTED, D.UNAVAILABLE, True,
         Verdict.SUPPORTED, True, R.DETECTION_UNAVAILABLE),
    ])
    def test_adversarial_case_policy_decision(self, case, ctype, verdict, det, independent, expected, review,
                                              reason):
        d = decide(ctype, verdict, det, independent=independent)
        assert (d.verdict, d.needs_human_review, d.reason_codes) == (expected, review, (reason,))

    @pytest.mark.parametrize("assoc,integ,reason", [
        (A.UNRESOLVED, I.OK, R.EVIDENCE_UNRESOLVED),
        (A.NONE_CITED, I.UNCHECKED, R.NO_EVIDENCE_CITED),
        (A.ASSOCIATED, I.MISMATCHED, R.INTEGRITY_MISMATCHED),
        (A.ASSOCIATED, I.CORRUPT, R.INTEGRITY_CORRUPT),
    ])
    def test_supported_verdict_and_matching_detection_cannot_override_bad_evidence(self, assoc, integ, reason):
        d = decide(ClaimType.DAMAGE_TYPE, Verdict.SUPPORTED, D.MATCHES_CLAIM, assoc=assoc, integ=integ)
        assert (d.verdict, d.needs_human_review, d.reason_codes) == (Verdict.INSUFFICIENT_EVIDENCE, True, (reason,))

    @pytest.mark.parametrize("raw,errored,reason", [
        (None, False, R.VERIFIER_MISSING),
        ("unrecognised", False, R.VERIFIER_INVALID),
        ("SUPPORTED", False, R.VERIFIER_INVALID),
        ("unsupported_severity", False, R.VERIFIER_INVALID),
        ("", False, R.VERIFIER_INVALID),
        (Verdict.SUPPORTED, True, R.VERIFIER_ERROR),
    ])
    def test_unusable_verifier_output_fails_closed(self, raw, errored, reason):
        d = decide(ClaimType.DAMAGE_TYPE, raw, D.MATCHES_CLAIM, errored=errored)
        assert (d.verdict, d.needs_human_review, d.reason_codes) == (Verdict.INSUFFICIENT_EVIDENCE, True, (reason,))

    def test_exactly_one_pipeline_input_combination_is_accepted_without_review(self):
        accepted = set()
        for ctype, v, det, assoc, integ, indep in itertools.product(
            ClaimType, [*Verdict, None], D, A, I, (True, False)
        ):
            if disposition(decide(ctype, v, det, assoc=assoc, integ=integ, independent=indep)) is Disposition.ACCEPTED:
                accepted.add((v, det, assoc, integ, indep))
                assert ctype in VISUAL
        assert accepted == {(Verdict.SUPPORTED, D.MATCHES_CLAIM, A.ASSOCIATED, I.OK, True)}

    def test_pipeline_inputs_never_reach_auto_rejection(self):
        for ctype, v, det, assoc, integ, indep in itertools.product(
            ClaimType, [*Verdict, None], D, A, I, (True, False)
        ):
            d = decide(ctype, v, det, assoc=assoc, integ=integ, independent=indep)
            assert disposition(d) is not Disposition.REJECTED


# Layer 2: evidence integrity -----------------------------------------------------------------


@pytest.fixture
def bundle(tmp_path):
    return EvidenceBundle(
        images=[ImageRef(IMG, save_image(tmp_path / "1.png", (64, 48))),
                ImageRef(IMG2, save_image(tmp_path / "2.png", (64, 48)))],
        detections=[Detection(IMG, "dent", 0.8, BOX, prediction_id=P1),
                    Detection(IMG2, "scratch", 0.5, FAR, prediction_id=P2)],
    )


def _report(**claim):
    base = {"text": "There is a dent.", "type": "damage_type", "damage_type": "dent", "images": ["img1"]}
    return json.dumps({"claims": [{**base, **claim}]})


class TestEvidenceIntegrityLayer:
    @pytest.mark.parametrize("images,detections", [
        (["img3"], []),
        (["IMG1"], []),
        ([" img1"], []),
        (["img01"], []),
        ([str(IMG)], []),
        (["det1"], []),
        (["img1"], ["det9"]),
        (["img1"], ["det01"]),
        (["img1"], [str(P1)]),
        (["img1"], ["img1"]),
    ])
    def test_fabricated_alias_rejects_the_whole_report(self, bundle, images, detections):
        with pytest.raises(VlmOutputError) as exc:
            parse_structured_report(_report(images=images, detections=detections), build_alias_map(bundle), 5)
        assert exc.value.code == "unknown_alias"

    def test_fabricated_evidence_alias_rejects_the_whole_report(self, bundle):
        raw = _report(evidence=[{"image": "img7", "detection": None}])
        with pytest.raises(VlmOutputError) as exc:
            parse_structured_report(raw, build_alias_map(bundle), 5)
        assert exc.value.code == "unknown_alias"

    def test_one_fabricated_alias_discards_valid_sibling_claims(self, bundle):
        good = {"text": "There is a dent.", "type": "damage_type", "damage_type": "dent", "images": ["img1"]}
        bad = {**good, "images": ["img9"]}
        runner = FakeRunner([json.dumps({"claims": [good, bad]})])
        with pytest.raises(VlmOutputError):
            QwenStructuredReportGenerator(runner, LIMITS).generate_structured(bundle)

    @pytest.mark.parametrize("claim", [
        {"images": ["img1"], "detections": ["det2"]},
        {"images": ["img1", "img2"], "detections": ["det1"], "evidence": [{"image": "img2", "detection": "det1"}]},
    ])
    def test_detection_paired_with_the_wrong_image_is_rejected(self, bundle, claim):
        with pytest.raises(VlmOutputError) as exc:
            parse_structured_report(_report(**claim), build_alias_map(bundle), 5)
        assert exc.value.code == "detection_image_mismatch"

    def test_parsed_reports_always_cite_inside_the_bundle(self, bundle):
        raw = _report(images=["img1", "img2"], detections=["det1"],
                      evidence=[{"image": "img1", "detection": "det1", "crop": [0.1, 0.1, 0.5, 0.5]}])
        check_citations(parse_structured_report(raw, build_alias_map(bundle), 5), bundle)

    @pytest.mark.parametrize("images,preds", [
        ((FOREIGN_IMG,), ()),
        ((IMG, FOREIGN_IMG), ()),
        ((IMG,), (FOREIGN_P,)),
        ((IMG,), (P1, FOREIGN_P)),
    ])
    def test_citation_of_evidence_from_another_inspection_is_rejected(self, bundle, images, preds):
        c = ExtractedClaim(claim_id="c1", text="There is a dent.", claim_type=ClaimType.DAMAGE_TYPE,
                           damage_type="dent", image_ids=list(images), prediction_ids=list(preds))
        report = StructuredReport(report_text="x", claims=ClaimExtractionResult(claims=[c]))
        with pytest.raises(VlmOutputError) as exc:
            check_citations(report, bundle)
        assert exc.value.code == "uncited_evidence"

    @pytest.mark.parametrize("assoc", [A.UNRESOLVED, A.NONE_CITED])
    def test_unresolved_or_uncited_evidence_is_decided_before_the_verifier(self, assoc):
        assert vs._decided_without_verifier(make_claim(), make_evidence(assoc=assoc))

    def test_unresolved_evidence_is_recorded_with_its_problems(self):
        ev = ClaimEvidence("c1", A.UNRESOLVED, I.OK, D.NOT_APPLICABLE, (), (), (),
                           (f"image_not_in_inspection:{FOREIGN_IMG}",))
        rec = ev.to_record()
        assert rec.association == "unresolved" and rec.problems == (f"image_not_in_inspection:{FOREIGN_IMG}",)

    @pytest.mark.parametrize("crop,expected", [
        (BOX, D.INCONCLUSIVE),
        (FAR, D.CONFLICTS_WITH_CLAIM),
        ({"x_min": 0.4, "y_min": 0.4, "x_max": 0.8, "y_max": 0.8}, D.INCONCLUSIVE),
    ])
    def test_wrong_vehicle_location_against_detection_box(self, crop, expected):
        c = make_claim(ClaimType.DAMAGE_LOCATION, "Damage on the lower left of the door.", None, crop=crop)
        assert signal(c) is expected

    @pytest.mark.parametrize("text,damage_type,detected,expected", [
        ("There is a scratch on the door.", "scratch", "dent", D.CONFLICTS_WITH_CLAIM),
        ("There is a scratch on the door.", "dent", "dent", D.INCONCLUSIVE),
        ("There are dents on the door.", "dent", "dent", D.INCONCLUSIVE),
        ("There is a dent on the door.", "dent", "dent", D.MATCHES_CLAIM),
    ])
    def test_damage_type_inconsistent_with_detection(self, text, damage_type, detected, expected):
        assert signal(make_claim(text=text, damage_type=damage_type), {P1: facts(detected)}) is expected

    def test_negated_claim_text_still_mentions_the_type_but_never_matches(self):
        text = "There is no dent on the door."
        assert text_mentions_damage_type(text, "dent")
        assert signal(make_claim(text=text)) is D.INCONCLUSIVE

    @pytest.mark.parametrize("damage_type,detected,expected", [
        ("dent", "scratch", D.INCONCLUSIVE),
        ("Dent", "dent", D.MATCHES_CLAIM),
        (None, "scratch", D.MATCHES_CLAIM),
    ])
    def test_presence_claim_with_a_damage_type_must_agree_with_the_detection_class(self, damage_type, detected,
                                                                                   expected):
        c = make_claim(ClaimType.VISIBLE_DAMAGE_PRESENCE, "There is damage on the door.", damage_type)
        assert signal(c, {P1: facts(detected)}) is expected

    @pytest.mark.parametrize("preds,latest_run,latest_count,expected", [
        ({}, {IMG: RUN}, {IMG: 0}, D.NO_DETECTION),
        ({}, {IMG: RUN}, {IMG: 2}, D.INCONCLUSIVE),
        ({P1: facts()}, {}, {}, D.UNAVAILABLE),
        ({P1: facts(run=OLD_RUN)}, {IMG: RUN}, {IMG: 1}, D.INCONCLUSIVE),
        ({P1: facts("dent"), P2: facts("scratch")}, {IMG: RUN}, {IMG: 2}, D.INCONCLUSIVE),
    ])
    def test_invented_or_ambiguous_detection_evidence(self, preds, latest_run, latest_count, expected):
        c = make_claim(preds=tuple(preds) or ())
        assert signal(c, preds, latest_run, latest_count) is expected


# Layer 3: escalation (final disposition via _ask_verifier + evaluate_claim) ---------------------


def _qwen(raws):
    return QwenClaimVerifier(FakeRunner(raws), LIMITS)


def _qwen_evidence(path, det=D.MATCHES_CLAIM):
    return ClaimEvidence("c1", A.ASSOCIATED, I.OK, det, ((IMG, path),), (P1,), (), ())


class TestEscalationLayer:
    @pytest.mark.parametrize("label,det,independent,expected,reason", [
        ("supported", D.MATCHES_CLAIM, True, Disposition.ACCEPTED, R.VERIFIER_SUPPORTED),
        ("supported", D.MATCHES_CLAIM, False, Disposition.HUMAN_REVIEW, R.VERIFIER_NOT_INDEPENDENT),
        ("wrong_location", D.CONFLICTS_WITH_CLAIM, True, Disposition.HUMAN_REVIEW, R.DETECTION_CONFLICT_UNCONFIRMED),
        ("wrong_location", D.MATCHES_CLAIM, True, Disposition.HUMAN_REVIEW, R.DETECTION_MATCHES_CONTRADICTED),
        ("contradicted", D.NO_DETECTION, True, Disposition.HUMAN_REVIEW, R.NO_DETECTION_NOT_ABSENCE),
        ("supported", D.NO_DETECTION, True, Disposition.HUMAN_REVIEW, R.VERIFIER_SUPPORTED_NO_DETECTION),
        ("wrong_damage_type", D.CONFLICTS_WITH_CLAIM, True, Disposition.HUMAN_REVIEW,
         R.DETECTION_CONFLICT_UNCONFIRMED),
        ("supported", D.CONFLICTS_WITH_CLAIM, True, Disposition.HUMAN_REVIEW, R.DETECTION_CONFLICTS_WITH_SUPPORTED),
        ("unverifiable", D.MATCHES_CLAIM, True, Disposition.HUMAN_REVIEW, R.VERIFIER_INSUFFICIENT),
        ("supported", D.INCONCLUSIVE, True, Disposition.HUMAN_REVIEW, R.DETECTION_INCONCLUSIVE),
        ("contradicted", D.INCONCLUSIVE, True, Disposition.HUMAN_REVIEW, R.DETECTION_INCONCLUSIVE),
    ])
    def test_adversarial_case_final_disposition(self, label, det, independent, expected, reason):
        _, d, disp = escalate(FakeVerifier(label), make_claim(), make_evidence(det), independent)
        assert (disp, d.reason_codes) == (expected, (reason,))

    @pytest.mark.parametrize("assoc", [A.UNRESOLVED, A.NONE_CITED])
    def test_foreign_or_missing_evidence_goes_to_review_without_calling_the_verifier(self, assoc):
        verifier = FakeVerifier("supported")
        raw, d, disp = escalate(verifier, make_claim(), make_evidence(assoc=assoc))
        assert verifier.calls == 0 and not raw.attempted and disp is Disposition.HUMAN_REVIEW
        assert d.verdict is Verdict.INSUFFICIENT_EVIDENCE

    @pytest.mark.parametrize("size", [(8, 6), (1, 1), (32, 24)])
    def test_tiny_image_is_never_upscaled_and_unverifiable_goes_to_review(self, tmp_path, size):
        verifier = _qwen(['{"label": "unverifiable", "explanation": "too small"}'])
        raw, d, disp = escalate(verifier, make_claim(), _qwen_evidence(save_image(tmp_path / "t.png", size)))
        assert verifier._runner.requests[0].images[0].size == size
        assert (raw.label.value, disp, d.reason_codes) == ("unverifiable", Disposition.HUMAN_REVIEW,
                                                           (R.VERIFIER_INSUFFICIENT,))

    def test_large_image_is_only_downscaled(self, tmp_path):
        verifier = _qwen(['{"label": "unverifiable"}'])
        escalate(verifier, make_claim(), _qwen_evidence(save_image(tmp_path / "big.png", (2000, 1000))))
        w, h = verifier._runner.requests[0].images[0].size
        assert w * h <= LIMITS.max_image_pixels and w < 2000 and h < 1000

    def test_tiny_image_is_insufficient_once_its_stored_size_is_assessed(self, tmp_path):
        path = save_image(tmp_path / "blur.png", (12, 9), blur=True)
        ev = _qwen_evidence(path)
        quality = vs.QUALITY_OF_EFFECT[assess_image(ImageQualityFacts(IMG, 12, 9), QUALITY_LIMITS).effect]
        assert vs._decided_without_verifier(make_claim(), ev, quality)
        d = evaluate_claim(vs._policy_input(make_claim(), ev, VerifierOutcome(Verdict.SUPPORTED), quality=quality))
        assert (disposition(d), d.reason_codes) == (Disposition.HUMAN_REVIEW, (R.EVIDENCE_QUALITY_INSUFFICIENT,))

    def test_limitation_blurry_image_of_adequate_size_can_still_be_accepted(self, tmp_path):
        path = save_image(tmp_path / "blur.png", (64, 48), blur=True)
        quality = vs.QUALITY_OF_EFFECT[assess_image(ImageQualityFacts(IMG, 64, 48), QUALITY_LIMITS).effect]
        _, d, disp = escalate(_qwen(['{"label": "supported"}']), make_claim(), _qwen_evidence(path))
        assert quality is vs.EvidenceQuality.ADEQUATE
        assert disp is Disposition.ACCEPTED and d.reason_codes == (R.VERIFIER_SUPPORTED,)

    @pytest.mark.parametrize("raw", [
        "not json",
        "Sure: {\"label\": \"supported\"}",
        '{"label": "supported", "confidence": 0.99}',
        '{"label": "supported", "label": "contradicted"}',
        '{"label": "supported", "explanation": NaN}',
        '{"label": "supported", "score": Infinity}',
        "x" * 64_001,
        json.dumps({"label": "supported", "explanation": "x" * 2001}),
        json.dumps({"label": "s" * 65}),
        "```json\nnot json at all\n```",
        "```json\n{\"label\": \"supported\"}\n```\nand also this",
        '[{"label": "supported"}]',
        '{"label": ["supported"]}',
        '{"label": ""}',
        "",
    ])
    def test_malformed_verifier_output_errors_and_goes_to_review(self, tmp_path, raw):
        with pytest.raises(VlmOutputError):
            parse_verdict(raw)
        verifier = _qwen([raw])
        out, d, disp = escalate(verifier, make_claim(), _qwen_evidence(save_image(tmp_path / "a.png", (64, 48))))
        assert out.errored and out.label is None
        assert (disp, d.reason_codes) == (Disposition.HUMAN_REVIEW, (R.VERIFIER_ERROR,))

    def test_whitespace_label_is_rejected_by_the_parser_as_a_verifier_error(self, tmp_path):
        verifier = _qwen([json.dumps({"label": "   "})])
        out, d, disp = escalate(verifier, make_claim(), _qwen_evidence(save_image(tmp_path / "a.png", (64, 48))))
        assert out.errored
        assert (disp, d.reason_codes) == (Disposition.HUMAN_REVIEW, (R.VERIFIER_ERROR,))

    @pytest.mark.parametrize("label", ["Supported", "maybe", "omitted_damage", "unsupported_severity"])
    def test_parsable_but_unrecognised_label_is_invalid_not_accepted(self, tmp_path, label):
        verifier = _qwen([json.dumps({"label": label})])
        out, d, disp = escalate(verifier, make_claim(), _qwen_evidence(save_image(tmp_path / "a.png", (64, 48))))
        assert not out.errored
        assert (disp, d.reason_codes) == (Disposition.HUMAN_REVIEW, (R.VERIFIER_INVALID,))

    @pytest.mark.parametrize("label", [None, 1, "maybe", "omitted_damage", "SUPPORTED"])
    def test_fake_verifier_unrecognised_label_is_invalid(self, label):
        result = ClaimVerdict("c", "supported", "x")
        object.__setattr__(result, "label", label)
        _, d, disp = escalate(FakeVerifier(results=[result]), make_claim(), make_evidence())
        assert (disp, d.reason_codes) == (Disposition.HUMAN_REVIEW, (R.VERIFIER_INVALID,))

    @pytest.mark.parametrize("results", [
        [],
        [ClaimVerdict("c", "supported", "x")] * 2,
        (ClaimVerdict("c", "supported", "x"),),
        None,
        "supported",
    ])
    def test_wrong_result_count_or_shape_is_a_verifier_error(self, results):
        verifier = FakeVerifier()
        verifier.verify = lambda claims, evidence: results
        _, d, disp = escalate(verifier, make_claim(), make_evidence())
        assert (disp, d.reason_codes) == (Disposition.HUMAN_REVIEW, (R.VERIFIER_ERROR,))

    @pytest.mark.parametrize("exc", [RuntimeError("boom"), VlmOutputError("bad", code="schema"), ValueError()])
    def test_verifier_raising_is_a_verifier_error(self, exc):
        raw, d, disp = escalate(FakeVerifier(raises=exc), make_claim(), make_evidence())
        assert raw.errored and (disp, d.reason_codes) == (Disposition.HUMAN_REVIEW, (R.VERIFIER_ERROR,))

    def test_limitation_wrongly_supporting_verifier_is_accepted_when_detection_matches(self):
        invented = make_claim(text="There is a dent on the roof.")
        _, d, disp = escalate(FakeVerifier("supported"), invented, make_evidence(D.MATCHES_CLAIM))
        assert disp is Disposition.ACCEPTED and d.reason_codes == (R.VERIFIER_SUPPORTED,)

    @pytest.mark.parametrize("det,independent,reason", [
        (D.CONFLICTS_WITH_CLAIM, True, R.DETECTION_CONFLICTS_WITH_SUPPORTED),
        (D.NO_DETECTION, True, R.VERIFIER_SUPPORTED_NO_DETECTION),
        (D.INCONCLUSIVE, True, R.DETECTION_INCONCLUSIVE),
        (D.UNAVAILABLE, True, R.DETECTION_UNAVAILABLE),
        (D.MATCHES_CLAIM, False, R.VERIFIER_NOT_INDEPENDENT),
    ])
    def test_wrongly_supporting_verifier_is_caught_by_detection_or_independence(self, det, independent, reason):
        _, d, disp = escalate(FakeVerifier("supported"), make_claim(), make_evidence(det), independent)
        assert (disp, d.reason_codes) == (Disposition.HUMAN_REVIEW, (reason,))

    @pytest.mark.parametrize("det", [D.MATCHES_CLAIM, D.INCONCLUSIVE])
    def test_negated_claim_with_wrong_support_goes_to_review(self, det):
        c = make_claim(text="There is no dent on the door.")
        _, d, disp = escalate(FakeVerifier("supported"), c, make_evidence(det))
        assert disp is Disposition.HUMAN_REVIEW and R.CLAIM_NOT_POSITIVE_ASSERTION in d.reason_codes

    def test_presence_typed_claim_no_longer_escapes_the_damage_type_check(self):
        c = make_claim(ClaimType.VISIBLE_DAMAGE_PRESENCE, "There is a dent on the door.", "dent")
        det = signal(c, {P1: facts("scratch")})
        _, d, disp = escalate(FakeVerifier("supported"), c, make_evidence(det))
        assert det is D.INCONCLUSIVE
        assert (disp, d.reason_codes) == (Disposition.HUMAN_REVIEW, (R.DETECTION_INCONCLUSIVE,))

    @pytest.mark.parametrize("det,expected", [
        (D.MATCHES_CLAIM, [Disposition.ACCEPTED, Disposition.HUMAN_REVIEW] * 3),
        (D.NO_DETECTION, [Disposition.HUMAN_REVIEW] * 6),
    ])
    def test_limitation_flip_flopping_verifier_runs_are_decided_independently(self, det, expected):
        verifier = FakeVerifier("supported", "contradicted")
        runs = [escalate(verifier, make_claim(), make_evidence(det)) for _ in range(6)]
        assert [disp for _, _, disp in runs] == expected
        assert [raw.outcome.verdict for raw, _, _ in runs] == [Verdict.SUPPORTED, Verdict.CONTRADICTED] * 3
        assert verifier.calls == 6
        assert all(d.reason_codes == runs[i % 2][1].reason_codes for i, (_, d, _) in enumerate(runs))

    def test_exception_while_reading_a_result_is_a_per_claim_verifier_error(self):
        class Broken:
            label = "supported"

            @property
            def explanation(self):
                raise RuntimeError("broken result object")

        raw, d, disp = escalate(FakeVerifier(results=[Broken()]), make_claim(), make_evidence())
        assert raw.errored and raw.attempted
        assert (disp, d.reason_codes) == (Disposition.HUMAN_REVIEW, (R.VERIFIER_ERROR,))
