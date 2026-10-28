import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, Uuid, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.run_status import COMPLETED_AT_CHECK, RunStatus, run_status_type

if TYPE_CHECKING:
    from app.models.generated_report import GeneratedReport
    from app.models.verification_result import VerificationResult


class VerificationRun(Base):
    """One execution of one verifier configuration over one report.

    Re-verifying a report creates a new run; results always belong to exactly one
    run, so different runs never mix. `config` is validated on write by
    app.schemas.runs.VerificationConfig.
    """

    __tablename__ = "verification_runs"
    __table_args__ = (CheckConstraint(COMPLETED_AT_CHECK, name="ck_verification_runs_completed_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    generated_report_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("generated_reports.id", ondelete="CASCADE"), index=True
    )
    verifier_model: Mapped[str] = mapped_column(String(100))
    verifier_version: Mapped[str] = mapped_column(String(100))
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    status: Mapped[RunStatus] = mapped_column(
        run_status_type("verification_run_status"),
        default=RunStatus.PENDING,
        server_default=RunStatus.PENDING.value,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    generated_report: Mapped["GeneratedReport"] = relationship(back_populates="verification_runs")
    results: Mapped[list["VerificationResult"]] = relationship(
        back_populates="verification_run", cascade="all, delete-orphan", passive_deletes=True
    )
