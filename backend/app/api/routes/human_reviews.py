"""Human decisions for claims deliberately routed out of the automatic policy."""

import uuid

from fastapi import APIRouter, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession
from app.core.errors import AppError
from app.models.generated_report import GeneratedReport
from app.models.human_review_decision import HumanReviewDecision
from app.models.inspection import Inspection, InspectionStatus
from app.models.verification_result import VerificationResult
from app.models.verification_run import VerificationRun
from app.schemas.common import ErrorResponse
from app.schemas.human_review import HumanReviewDecisionCreate, HumanReviewDecisionRead

router = APIRouter(tags=["human reviews"])


def _result_or_404(db: DbSession, result_id: uuid.UUID) -> VerificationResult:
    result = db.scalars(
        select(VerificationResult)
        .where(VerificationResult.id == result_id)
        .options(
            selectinload(VerificationResult.review_decision),
            selectinload(VerificationResult.verification_run).selectinload(VerificationRun.generated_report),
        )
    ).one_or_none()
    if result is None:
        raise AppError(404, "not_found", "Verification result not found.")
    return result


def _refresh_inspection_review_status(db: DbSession, inspection_id: uuid.UUID) -> None:
    """Complete only when every review-required result has a human disposition."""
    inspection = db.get(Inspection, inspection_id)
    if inspection is None:  # Referential integrity makes this defensive only.
        return
    outstanding = db.scalar(
        select(VerificationResult.id)
        .join(VerificationRun, VerificationResult.verification_run_id == VerificationRun.id)
        .join(GeneratedReport, VerificationRun.generated_report_id == GeneratedReport.id)
        .outerjoin(HumanReviewDecision, HumanReviewDecision.verification_result_id == VerificationResult.id)
        .where(
            GeneratedReport.inspection_id == inspection_id,
            VerificationResult.needs_human_review.is_(True),
            HumanReviewDecision.id.is_(None),
        )
        .limit(1)
    )
    inspection.status = InspectionStatus.REVIEW_REQUIRED if outstanding is not None else InspectionStatus.COMPLETED


@router.get(
    "/verification-results/{verification_result_id}/review-decision",
    response_model=HumanReviewDecisionRead,
    responses={404: {"model": ErrorResponse}},
)
def get_review_decision(verification_result_id: uuid.UUID, db: DbSession) -> HumanReviewDecision:
    result = _result_or_404(db, verification_result_id)
    if result.review_decision is None:
        raise AppError(404, "review_decision_not_found", "No human review decision has been recorded.")
    return result.review_decision


@router.post(
    "/verification-results/{verification_result_id}/review-decision",
    response_model=HumanReviewDecisionRead,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def create_review_decision(
    verification_result_id: uuid.UUID, payload: HumanReviewDecisionCreate, db: DbSession
) -> HumanReviewDecision:
    result = _result_or_404(db, verification_result_id)
    if not result.needs_human_review:
        raise AppError(409, "review_not_required", "This verification result was not routed to human review.")
    if result.review_decision is not None:
        raise AppError(409, "review_decision_exists", "A human review decision is already recorded and cannot be overwritten.")

    decision = HumanReviewDecision(
        verification_result_id=result.id, outcome=payload.outcome, reviewer_note=payload.reviewer_note
    )
    db.add(decision)
    try:
        db.flush()
    except IntegrityError:
        # The database unique constraint is the authority if two reviewers submit simultaneously.
        db.rollback()
        raise AppError(409, "review_decision_exists", "A human review decision is already recorded and cannot be overwritten.") from None
    inspection_id = result.verification_run.generated_report.inspection_id
    _refresh_inspection_review_status(db, inspection_id)
    db.commit()
    db.refresh(decision)
    return decision
