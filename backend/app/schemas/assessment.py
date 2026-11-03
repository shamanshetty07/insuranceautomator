import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.schemas.damage import BoundingBox

ClaimStatus = Literal["supported", "uncertain", "unsupported"]
VerificationState = Literal["not_run", "running", "completed", "failed"]
ReviewState = Literal["not_started", "not_required", "pending", "complete"]
AssessmentState = Literal[
    "no_findings", "awaiting_verification", "verification_failed", "pending_human_review", "ready"
]

NOT_PROVIDED = "Not Provided"


class VehicleInfoRead(BaseModel):
    make: str = NOT_PROVIDED
    model: str = NOT_PROVIDED
    year: str = NOT_PROVIDED
    registration_number: str = NOT_PROVIDED
    color: str = NOT_PROVIDED


class AssessmentHeaderRead(BaseModel):
    report_id: uuid.UUID
    inspection_id: uuid.UUID
    inspection_title: str | None
    generated_at: datetime
    inspection_type: str
    assessment_status: AssessmentState
    report_model: str


class ObservationRead(BaseModel):
    claim_id: str
    text: str
    status: ClaimStatus


class DamageFindingRead(BaseModel):
    damage_id: str
    prediction_id: uuid.UUID
    image_id: uuid.UUID | None
    image_label: str
    component: str
    damage_type: str
    severity: str
    detection_confidence: float
    bounding_box: BoundingBox | None
    observations: list[ObservationRead]
    repair_considerations: list[str]
    additional_inspection: list[str]


class ClaimVerificationRead(BaseModel):
    claim_id: str
    claim_text: str
    claim_type: str
    damage_ids: list[str]
    image_ids: list[uuid.UUID]
    status: ClaimStatus
    verifier_label: str | None
    explanation: str
    needs_human_review: bool
    review_reason_codes: list[str]


class ReviewItemRead(BaseModel):
    claim_id: str
    required: bool
    outcome: str | None
    reviewer_note: str | None
    reviewed_at: datetime | None


class HumanReviewSummaryRead(BaseModel):
    ai_assessment_status: str
    verification_status: VerificationState
    review_status: ReviewState
    items: list[ReviewItemRead]


class AssessmentReportRead(BaseModel):
    header: AssessmentHeaderRead
    vehicle: VehicleInfoRead
    executive_summary: str
    findings: list[DamageFindingRead]
    claims: list[ClaimVerificationRead]
    recommendations: list[str]
    cost_estimate: str
    human_review: HumanReviewSummaryRead
    evidence_notice: str
    disclaimer: str
