"""Request and response shapes for recorded human claim-review decisions."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.human_review_decision import ReviewOutcome


class HumanReviewDecisionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: ReviewOutcome
    reviewer_note: str | None = Field(default=None, max_length=2_000)


class HumanReviewDecisionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    verification_result_id: uuid.UUID
    outcome: ReviewOutcome
    reviewer_note: str | None
    reviewed_at: datetime
