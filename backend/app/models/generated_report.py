import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text, Uuid, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.inspection import Inspection
    from app.models.report_claim import ReportClaim
    from app.models.report_prediction import ReportPrediction
    from app.models.verification_run import VerificationRun


class EvidenceStrategy(str, enum.Enum):
    """How visual evidence was presented to the report generator (see docs/architecture.md)."""

    FULL_IMAGE = "full_image"
    CROPPED_REGION = "cropped_region"
    IMAGE_WITH_SEGMENTATION_MASK = "image_with_segmentation_mask"


class GeneratedReport(Base):
    """A report plus everything needed to reproduce it."""

    __tablename__ = "generated_reports"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    inspection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), index=True
    )
    content: Mapped[str] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(100))
    evidence_strategy: Mapped[EvidenceStrategy] = mapped_column(
        Enum(
            EvidenceStrategy,
            native_enum=False,
            length=32,
            create_constraint=True,
            name="evidence_strategy",
            values_callable=lambda e: [m.value for m in e],
        ),
        index=True,
    )
    generation_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    inspection: Mapped["Inspection"] = relationship(back_populates="reports")
    prediction_links: Mapped[list["ReportPrediction"]] = relationship(
        back_populates="generated_report", cascade="all, delete-orphan", passive_deletes=True
    )
    verification_runs: Mapped[list["VerificationRun"]] = relationship(
        back_populates="generated_report", cascade="all, delete-orphan", passive_deletes=True
    )
    claims: Mapped[list["ReportClaim"]] = relationship(
        back_populates="generated_report",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ReportClaim.position",
    )
