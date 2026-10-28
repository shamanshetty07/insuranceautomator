import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.inspection import Inspection
    from app.models.verification_result import VerificationResult


class GeneratedReport(Base):
    __tablename__ = "generated_reports"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    inspection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), index=True
    )
    content: Mapped[str] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    inspection: Mapped["Inspection"] = relationship(back_populates="reports")
    verification_results: Mapped[list["VerificationResult"]] = relationship(
        back_populates="generated_report", cascade="all, delete-orphan", passive_deletes=True
    )
