import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, String, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.detection_run import DetectionRun


class DamagePrediction(Base):
    """One detected damage instance on one image (populated from the detection phase)."""

    __tablename__ = "damage_predictions"
    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_damage_confidence"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    detection_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("detection_runs.id", ondelete="CASCADE"), index=True
    )
    damage_type: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float)
    bounding_box: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    segmentation_data: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    detection_run: Mapped["DetectionRun"] = relationship(back_populates="predictions")
