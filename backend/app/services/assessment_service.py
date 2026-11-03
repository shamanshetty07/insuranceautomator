import uuid
from collections import Counter
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.errors import AppError
from app.models.damage_prediction import DamagePrediction
from app.models.inspection import Inspection
from app.models.inspection_image import InspectionImage
from app.models.report_prediction import ReportPrediction
from app.schemas.assessment import (
    NOT_PROVIDED,
    AssessmentHeaderRead,
    AssessmentReportRead,
    ClaimVerificationRead,
    DamageFindingRead,
    HumanReviewSummaryRead,
    ObservationRead,
    ReviewItemRead,
    VehicleInfoRead,
)
from app.schemas.damage import BoundingBox
from app.services import report_service

INSPECTION_TYPE = "AI-assisted visual damage assessment"
COMPONENT_UNKNOWN = "Not identified by detector"
SEVERITY_UNKNOWN = "Not assessed"
COST_NOTE = "No repair cost estimate is available: this system has no pricing data."
EVIDENCE_NOTICE = (
    "Damage detections are model output with a detection confidence, not the probability that an assessment "
    "is correct. Observations are AI-generated claims; only claims marked Supported were backed by the verifier, "
    "and only claims with a recorded human decision are human-reviewed."
)
DISCLAIMER = (
    "This report is an AI-assisted preliminary assessment based on the submitted images. Visual assessment may "
    "contain inaccuracies, and internal or mechanical damage cannot necessarily be identified from images. "
    "It is not a final insurance claim approval; final decisions require appropriate human assessment."
)
LOW_CONFIDENCE = 0.5

_REPAIR_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("scratch", "scrape", "scuff", "paint"), "Possible refinishing or repainting of the affected area"),
    (("dent", "bump", "crush", "deform"), "Possible panel or bumper repair, with repainting if the finish is affected"),
    (("crack", "broken", "shatter", "glass", "lamp", "light"), "Possible replacement of the affected part"),
)


def _text(value: Any) -> str:
    if value is None:
        return NOT_PROVIDED
    text = str(value).strip()
    return text or NOT_PROVIDED


def vehicle_info(inspection: Any) -> VehicleInfoRead:
    return VehicleInfoRead(
        make=_text(getattr(inspection, "vehicle_make", None)),
        model=_text(getattr(inspection, "vehicle_model", None)),
        year=_text(getattr(inspection, "vehicle_year", None)),
        registration_number=_text(getattr(inspection, "vehicle_registration", None)),
        color=_text(getattr(inspection, "vehicle_color", None)),
    )


def claim_status(result: Any) -> str:
    if result is None:
        return "uncertain"
    verdict = getattr(result.verdict, "value", result.verdict)
    if verdict == "contradicted":
        return "unsupported"
    if verdict == "supported" and not result.needs_human_review:
        return "supported"
    return "uncertain"


def _repair_considerations(damage_type: str, contradicted: bool) -> list[str]:
    if contradicted:
        return []
    name = damage_type.lower()
    for keywords, hint in _REPAIR_HINTS:
        if any(keyword in name for keyword in keywords):
            return [hint]
    return []


def _bbox(value: Any) -> BoundingBox | None:
    if not value:
        return None
    try:
        return BoundingBox.model_validate(value)
    except ValueError:
        return None


def _ordered_predictions(predictions: Sequence[Any], image_order: dict[uuid.UUID, int]) -> list[Any]:
    def key(p: Any) -> tuple[int, float]:
        image_id = p.detection_run.inspection_image_id
        return (image_order.get(image_id, 10_000), -p.confidence)

    return sorted(predictions, key=key)


def _findings(
    predictions: Sequence[Any],
    image_order: dict[uuid.UUID, int],
    claims: list[ClaimVerificationRead],
    claim_texts: dict[str, str],
) -> list[DamageFindingRead]:
    findings: list[DamageFindingRead] = []
    ids = {}
    ordered = _ordered_predictions(predictions, image_order)
    for number, prediction in enumerate(ordered, start=1):
        ids[prediction.id] = f"D{number}"
    claims_by_prediction: dict[uuid.UUID, list[ClaimVerificationRead]] = {}
    prediction_for = {did: pid for pid, did in ids.items()}
    for claim in claims:
        for damage_id in claim.damage_ids:
            claims_by_prediction.setdefault(prediction_for[damage_id], []).append(claim)
    for prediction in ordered:
        image_id = prediction.detection_run.inspection_image_id
        linked = claims_by_prediction.get(prediction.id, [])
        contradicted = any(c.status == "unsupported" for c in linked)
        extra = ["Physical inspection of the affected area", "Check adjacent components for damage not visible in photos"]
        if prediction.confidence < LOW_CONFIDENCE:
            extra.append("Low detection confidence: confirm the damage is actually present in the photo")
        if contradicted:
            extra.append("An AI claim about this damage was contradicted by the verifier: human review needed")
        order = image_order.get(image_id)
        findings.append(
            DamageFindingRead(
                damage_id=ids[prediction.id],
                prediction_id=prediction.id,
                image_id=image_id,
                image_label=f"Photo {order + 1}" if order is not None else "Unknown photo",
                component=COMPONENT_UNKNOWN,
                damage_type=prediction.damage_type,
                severity=SEVERITY_UNKNOWN,
                detection_confidence=prediction.confidence,
                bounding_box=_bbox(prediction.bounding_box),
                observations=[
                    ObservationRead(claim_id=c.claim_id, text=claim_texts[c.claim_id], status=c.status)
                    for c in linked
                ],
                repair_considerations=_repair_considerations(prediction.damage_type, contradicted),
                additional_inspection=extra,
            )
        )
    return findings


def _claims(report: Any, run: Any, finding_ids: dict[uuid.UUID, str]) -> list[ClaimVerificationRead]:
    by_row: dict[Any, Any] = {}
    by_claim_id: dict[str, Any] = {}
    if run is not None:
        for result in run.results:
            if result.report_claim_id is not None:
                by_row[result.report_claim_id] = result
            elif result.claim_id is not None:
                by_claim_id[result.claim_id] = result
    claims = []
    for row in report.claims:
        result = by_row.get(row.id) or by_claim_id.get(row.claim_id)
        label = getattr(result.verification_label, "value", result.verification_label) if result else None
        claims.append(
            ClaimVerificationRead(
                claim_id=row.claim_id,
                claim_text=row.text,
                claim_type=getattr(row.claim_type, "value", row.claim_type),
                damage_ids=[finding_ids[pid] for pid in row.prediction_ids if pid in finding_ids],
                image_ids=list(row.image_ids),
                status=claim_status(result),
                verifier_label=label,
                explanation=(result.explanation if result else "") or ("" if result else "Not verified."),
                needs_human_review=bool(result.needs_human_review) if result else False,
                review_reason_codes=list(result.policy_reason_codes) if result else [],
            )
        )
    return claims


def _human_review(report: Any, run: Any, claims: list[ClaimVerificationRead]) -> HumanReviewSummaryRead:
    if run is None:
        return HumanReviewSummaryRead(
            ai_assessment_status="AI-generated and not yet verified",
            verification_status="not_run",
            review_status="not_started",
            items=[],
        )
    run_status = getattr(run.status, "value", run.status)
    verification = run_status if run_status in {"running", "completed", "failed"} else "running"
    row_for = {r.claim_id: r for r in report.claims}
    by_row = {r.report_claim_id: r for r in run.results if r.report_claim_id is not None}
    by_claim_id = {r.claim_id: r for r in run.results if r.report_claim_id is None and r.claim_id is not None}
    items = []
    for claim in claims:
        row = row_for[claim.claim_id]
        result = by_row.get(row.id) or by_claim_id.get(claim.claim_id)
        decision = getattr(result, "review_decision", None) if result is not None else None
        if not claim.needs_human_review and decision is None:
            continue
        items.append(
            ReviewItemRead(
                claim_id=claim.claim_id,
                required=claim.needs_human_review,
                outcome=getattr(decision.outcome, "value", decision.outcome) if decision else None,
                reviewer_note=decision.reviewer_note if decision else None,
                reviewed_at=decision.reviewed_at if decision else None,
            )
        )
    required = [i for i in items if i.required]
    if verification != "completed":
        review = "not_started"
    elif not required:
        review = "not_required"
    elif any(i.outcome is None for i in required):
        review = "pending"
    else:
        review = "complete"
    ai_status = {
        "completed": "AI-generated, verified by the verifier model",
        "failed": "AI-generated; verification failed",
    }.get(verification, "AI-generated; verification in progress")
    return HumanReviewSummaryRead(
        ai_assessment_status=ai_status, verification_status=verification, review_status=review, items=items
    )


def _state(findings: list, claims: list, review: HumanReviewSummaryRead) -> str:
    if review.verification_status == "failed":
        return "verification_failed"
    if claims and review.verification_status != "completed":
        return "awaiting_verification"
    if review.review_status == "pending":
        return "pending_human_review"
    if not findings and not claims:
        return "no_findings"
    return "ready"


def _summary(
    findings: list[DamageFindingRead],
    claims: list[ClaimVerificationRead],
    photo_count: int,
    review: HumanReviewSummaryRead,
) -> str:
    parts = [f"AI-assisted review of {photo_count} submitted photo(s)."]
    if findings:
        counts = Counter(f.damage_type for f in findings)
        listed = ", ".join(f"{n} x {name}" for name, n in counts.most_common())
        low, high = min(f.detection_confidence for f in findings), max(f.detection_confidence for f in findings)
        photos = sorted({f.image_label for f in findings})
        parts.append(
            f"The detector reported {len(findings)} damage instance(s): {listed} "
            f"(detection confidence {round(low * 100)}-{round(high * 100)}%) in {', '.join(photos)}."
        )
    else:
        parts.append("The detector reported no damage instances; this is not a finding that the vehicle is undamaged.")
    if claims:
        supported = sum(1 for c in claims if c.status == "supported")
        unsupported = sum(1 for c in claims if c.status == "unsupported")
        uncertain = len(claims) - supported - unsupported
        if review.verification_status == "not_run":
            parts.append(f"{len(claims)} AI-generated claim(s) have not been verified yet.")
        else:
            parts.append(
                f"Of {len(claims)} AI-generated claim(s): {supported} supported, {uncertain} uncertain, "
                f"{unsupported} unsupported."
            )
    if review.review_status == "pending":
        pending = sum(1 for i in review.items if i.required and i.outcome is None)
        parts.append(f"{pending} claim(s) still await a human decision.")
    flagged = [f for f in findings if f.detection_confidence < LOW_CONFIDENCE]
    if findings:
        parts.append("Overall vehicle condition cannot be concluded from detections alone; physical inspection is required.")
    if flagged:
        parts.append(f"{len(flagged)} detection(s) have low confidence and need confirmation.")
    return " ".join(parts)


def _recommendations(findings: list[DamageFindingRead]) -> list[str]:
    if not findings:
        return ["No damage was detected; if damage is suspected, request a physical inspection."]
    recommendations = ["Arrange a physical inspection to confirm the detected damage."]
    seen = set(recommendations)
    for finding in findings:
        for hint in finding.repair_considerations:
            line = f"{hint} ({finding.damage_id}, {finding.damage_type})"
            if line not in seen:
                seen.add(line)
                recommendations.append(line)
    recommendations.append("Inspect adjacent components for damage that is not visible in photos.")
    return recommendations


def build_assessment(
    report: Any,
    run: Any,
    inspection: Any,
    predictions: Sequence[Any],
    image_order: dict[uuid.UUID, int],
) -> AssessmentReportRead:
    ordered = _ordered_predictions(predictions, image_order)
    finding_ids = {p.id: f"D{n}" for n, p in enumerate(ordered, start=1)}
    claims = _claims(report, run, finding_ids)
    claim_texts = {c.claim_id: c.claim_text for c in claims}
    findings = _findings(predictions, image_order, claims, claim_texts)
    review = _human_review(report, run, claims)
    state = _state(findings, claims, review)
    return AssessmentReportRead(
        header=AssessmentHeaderRead(
            report_id=report.id,
            inspection_id=report.inspection_id,
            inspection_title=getattr(inspection, "title", None),
            generated_at=report.created_at,
            inspection_type=INSPECTION_TYPE,
            assessment_status=state,
            report_model=f"{report.model_name} {report.model_version}",
        ),
        vehicle=vehicle_info(inspection),
        executive_summary=_summary(findings, claims, len(image_order), review),
        findings=findings,
        claims=claims,
        recommendations=_recommendations(findings),
        cost_estimate=COST_NOTE,
        human_review=review,
        evidence_notice=EVIDENCE_NOTICE,
        disclaimer=DISCLAIMER,
    )


def load_assessment(db: Session, report_id: uuid.UUID) -> AssessmentReportRead:
    report, run = report_service.get_report(db, report_id)
    inspection = db.get(Inspection, report.inspection_id)
    if inspection is None:
        raise AppError(404, "not_found", "Inspection not found.")
    images = db.scalars(select(InspectionImage).where(InspectionImage.inspection_id == inspection.id)).all()
    image_order = {i.id: i.image_order for i in images}
    predictions = list(
        db.scalars(
            select(DamagePrediction)
            .join(ReportPrediction, ReportPrediction.damage_prediction_id == DamagePrediction.id)
            .where(ReportPrediction.generated_report_id == report_id)
            .options(selectinload(DamagePrediction.detection_run))
        )
    )
    return build_assessment(report, run, inspection, predictions, image_order)
