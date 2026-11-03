"""Immutable human decisions for verification results that require review."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.verification_result import VerificationResult


class ReviewOutcome(str, enum.Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class HumanReviewDecision(Base):
    """A single, immutable human disposition of one review-required result."""

    __tablename__ = "human_review_decisions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    verification_result_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("verification_results.id", ondelete="CASCADE"), unique=True, index=True
    )
    outcome: Mapped[ReviewOutcome] = mapped_column(
        Enum(ReviewOutcome, native_enum=False, length=16, create_constraint=True,
             name="human_review_outcome", values_callable=lambda e: [m.value for m in e])
    )
    reviewer_note: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    verification_result: Mapped["VerificationResult"] = relationship(back_populates="review_decision")
