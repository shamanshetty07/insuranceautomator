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
    """One detected damage instance on one image (populated from the detection phase).

    bounding_box (nullable JSON), normalized to the image, origin top-left:
        {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6}
    with 0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1.
    segmentation_data (nullable JSON): list of polygons, each a list of
        [x, y] points in the same normalized coordinates.
    Validation lives in app.schemas.damage; the DB stores the JSON as given.
    The image, model name/version and parameters are not repeated here: they are
    those of `detection_run`.
    """

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
