import types
import uuid
from datetime import UTC, datetime

from app.models.run_status import RunStatus
from app.schemas.assessment import NOT_PROVIDED
from app.services.assessment_service import (
    COST_NOTE,
    DISCLAIMER,
    build_assessment,
    claim_status,
    vehicle_info,
)

IMG1, IMG2 = uuid.uuid4(), uuid.uuid4()
ORDER = {IMG1: 0, IMG2: 1}


def prediction(damage_type="dent", confidence=0.8, image_id=IMG1, box=None):
    return types.SimpleNamespace(
        id=uuid.uuid4(),
        damage_type=damage_type,
        confidence=confidence,
        bounding_box=box or {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5},
        detection_run=types.SimpleNamespace(inspection_image_id=image_id),
    )


def claim_row(claim_id, text, prediction_ids=(), claim_type="damage_type"):
    return types.SimpleNamespace(
        id=uuid.uuid4(), claim_id=claim_id, text=text, claim_type=claim_type, damage_type="dent",
        image_ids=[IMG1], prediction_ids=list(prediction_ids), evidence=[],
    )


def result(row, verdict, review=False, decision=None, label="supported"):
    return types.SimpleNamespace(
        report_claim_id=row.id, claim_id=row.claim_id, verdict=verdict, needs_human_review=review,
        policy_reason_codes=["verifier_not_independent"] if review else [], verification_label=label,
        explanation="because", review_decision=decision,
    )


def run_of(results, status=RunStatus.COMPLETED):
    return types.SimpleNamespace(results=results, status=status)


def report_of(rows):
    return types.SimpleNamespace(
        id=uuid.uuid4(), inspection_id=uuid.uuid4(), created_at=datetime.now(UTC),
        model_name="qwen", model_version="v1", claims=rows,
    )


def inspection(**kw):
    base = dict(title="T", vehicle_make=None, vehicle_model=None, vehicle_year=None,
                vehicle_registration=None, vehicle_color=None)
    return types.SimpleNamespace(**{**base, **kw})


def test_missing_vehicle_information_is_marked_not_provided_and_never_invented():
    info = vehicle_info(inspection(vehicle_make="  ", vehicle_year=None))
    assert info.make == info.model == info.year == info.registration_number == info.color == NOT_PROVIDED
    assert vehicle_info(inspection(vehicle_make="Honda", vehicle_year=2019)).year == "2019"


def test_no_detections_yields_no_findings_and_no_clean_bill_of_health():
    report = build_assessment(report_of([]), None, inspection(), [], ORDER)
    assert report.findings == [] and report.claims == []
    assert report.header.assessment_status == "no_findings"
    assert "not a finding that the vehicle is undamaged" in report.executive_summary
    assert "no damage was detected" in report.recommendations[0].lower()
    assert report.cost_estimate == COST_NOTE and report.header.report_id


def test_findings_come_only_from_stored_detections_with_fixed_unknown_fields():
    p1, p2 = prediction("dent", 0.9), prediction("scratch", 0.3, image_id=IMG2)
    report = build_assessment(report_of([]), None, inspection(), [p2, p1], ORDER)
    assert [f.damage_id for f in report.findings] == ["D1", "D2"]
    first, second = report.findings
    assert (first.damage_type, first.image_label, first.detection_confidence) == ("dent", "Photo 1", 0.9)
    assert first.component == "Not identified by detector" and first.severity == "Not assessed"
    assert first.bounding_box.x_max == 0.5
    assert any("Low detection confidence" in line for line in second.additional_inspection)
    assert first.repair_considerations and "repair" in first.repair_considerations[0].lower()


def test_unverified_report_is_awaiting_verification_and_claims_are_uncertain():
    p = prediction()
    row = claim_row("c1", "There is a dent.", [p.id])
    report = build_assessment(report_of([row]), None, inspection(), [p], ORDER)
    assert report.header.assessment_status == "awaiting_verification"
    assert report.claims[0].status == "uncertain" and report.claims[0].explanation == "Not verified."
    assert report.human_review.verification_status == "not_run"
    assert report.human_review.review_status == "not_started"
    assert report.findings[0].observations[0].status == "uncertain"


def test_incomplete_verification_missing_one_result_is_uncertain_not_supported():
    p = prediction()
    r1, r2 = claim_row("c1", "a", [p.id]), claim_row("c2", "b", [p.id])
    run = run_of([result(r1, "supported")])
    report = build_assessment(report_of([r1, r2]), run, inspection(), [p], ORDER)
    assert [c.status for c in report.claims] == ["supported", "uncertain"]


def test_contradicted_claim_is_unsupported_and_withholds_repair_suggestions():
    p = prediction("dent")
    row = claim_row("c1", "Large dent.", [p.id])
    report = build_assessment(report_of([row]), run_of([result(row, "contradicted", label="contradicted")]),
                              inspection(), [p], ORDER)
    assert report.claims[0].status == "unsupported"
    assert report.findings[0].repair_considerations == []
    assert any("contradicted" in line for line in report.findings[0].additional_inspection)
    assert "1 unsupported" in report.executive_summary


def test_supported_but_flagged_for_review_is_not_presented_as_supported():
    row = claim_row("c1", "x")
    assert claim_status(result(row, "supported", review=True)) == "uncertain"
    assert claim_status(result(row, "supported", review=False)) == "supported"
    assert claim_status(result(row, "insufficient_evidence")) == "uncertain"
    assert claim_status(None) == "uncertain"


def test_human_review_pending_then_complete_with_reviewer_decision():
    p = prediction()
    row = claim_row("c1", "x", [p.id])
    pending = build_assessment(report_of([row]), run_of([result(row, "supported", review=True)]),
                               inspection(), [p], ORDER)
    assert pending.human_review.review_status == "pending"
    assert pending.header.assessment_status == "pending_human_review"
    assert pending.human_review.items[0].outcome is None

    decision = types.SimpleNamespace(outcome="approved", reviewer_note="checked", reviewed_at=datetime.now(UTC))
    done = build_assessment(report_of([row]), run_of([result(row, "supported", review=True, decision=decision)]),
                            inspection(), [p], ORDER)
    item = done.human_review.items[0]
    assert done.human_review.review_status == "complete" and done.header.assessment_status == "ready"
    assert (item.outcome, item.reviewer_note) == ("approved", "checked")


def test_no_review_required_and_failed_verification_states():
    row = claim_row("c1", "x")
    clean = build_assessment(report_of([row]), run_of([result(row, "supported")]), inspection(), [], ORDER)
    assert clean.human_review.review_status == "not_required"
    failed = build_assessment(report_of([row]), run_of([], status=RunStatus.FAILED), inspection(), [], ORDER)
    assert failed.header.assessment_status == "verification_failed"
    assert failed.human_review.verification_status == "failed"


def test_disclaimer_and_cost_note_are_always_present():
    report = build_assessment(report_of([]), None, inspection(), [], ORDER)
    assert report.disclaimer == DISCLAIMER and "not a final insurance claim approval" in report.disclaimer
    assert "no pricing data" in report.cost_estimate
