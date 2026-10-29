import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, Uuid, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.run_status import (
    COMPLETED_AT_CHECK,
    SHA256_HEX_CHECK,
    RunStatus,
    run_status_type,
)

if TYPE_CHECKING:
    from app.models.damage_prediction import DamagePrediction
    from app.models.inspection_image import InspectionImage


class DetectionRun(Base):
    """One execution of one detection model/configuration on one image.

    Canonical home of the model identity and parameters for its predictions; a
    run may produce zero predictions (a valid "nothing detected" result).
    `config` is validated on write by app.schemas.runs.DetectionConfig.
    """

    __tablename__ = "detection_runs"
    __table_args__ = (
        CheckConstraint(COMPLETED_AT_CHECK, name="ck_detection_runs_completed_at"),
        CheckConstraint(SHA256_HEX_CHECK.format(col="weights_sha256"), name="ck_detection_runs_sha256"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    inspection_image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inspection_images.id", ondelete="CASCADE"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(100))
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    # Lowercase hex SHA-256 of the weights file, when known.
    weights_sha256: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[RunStatus] = mapped_column(
        run_status_type("detection_run_status"),
        default=RunStatus.PENDING,
        server_default=RunStatus.PENDING.value,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Last sign of life from the worker holding a `running` run (refreshed periodically);
    # stale-run recovery uses it, falling back to created_at when NULL (older rows).
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    inspection_image: Mapped["InspectionImage"] = relationship(back_populates="detection_runs")
    predictions: Mapped[list["DamagePrediction"]] = relationship(
        back_populates="detection_run", cascade="all, delete-orphan", passive_deletes=True
    )
