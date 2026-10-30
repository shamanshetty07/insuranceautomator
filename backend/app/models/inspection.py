import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.generated_report import GeneratedReport
    from app.models.inspection_image import InspectionImage


class InspectionStatus(str, enum.Enum):
    """Lifecycle labels. See docs/architecture.md for meanings."""

    CREATED = "created"
    IMAGES_PENDING = "images_pending"
    READY_FOR_ANALYSIS = "ready_for_analysis"
    PROCESSING = "processing"
    REVIEW_REQUIRED = "review_required"
    COMPLETED = "completed"
    FAILED = "failed"


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text)
    status: Mapped[InspectionStatus] = mapped_column(
        Enum(
            InspectionStatus,
            native_enum=False,
            length=32,
            create_constraint=True,
            name="inspection_status",
            values_callable=lambda e: [m.value for m in e],
        ),
        default=InspectionStatus.CREATED,
        server_default=InspectionStatus.CREATED.value,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    images: Mapped[list["InspectionImage"]] = relationship(
        back_populates="inspection", cascade="all, delete-orphan", passive_deletes=True
    )
    reports: Mapped[list["GeneratedReport"]] = relationship(
        back_populates="inspection", cascade="all, delete-orphan", passive_deletes=True
    )
