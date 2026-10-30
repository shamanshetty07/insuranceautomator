import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, Enum, ForeignKey, Index, String, Text, Uuid, false, func, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.schemas.claims import ClaimType, Verdict

if TYPE_CHECKING:
    from app.models.verification_run import VerificationRun


class VerificationLabel(str, enum.Enum):
    """Outcome of checking one claim against visual evidence."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    WRONG_LOCATION = "wrong_location"
    WRONG_DAMAGE_TYPE = "wrong_damage_type"
    UNSUPPORTED_SEVERITY = "unsupported_severity"
    UNVERIFIABLE = "unverifiable"
    OMITTED_DAMAGE = "omitted_damage"


class VerificationResult(Base):
    """One labelled claim from one verification run (the report and verifier come from the run)."""

    __tablename__ = "verification_results"
    __table_args__ = (
        CheckConstraint(
            "verification_label IS DISTINCT FROM 'omitted_damage' OR verdict IS NULL",
            name="ck_verification_results_omitted_has_no_verdict",
        ),
        Index(
            "uq_verification_results_run_claim",
            "verification_run_id",
            "claim_id",
            unique=True,
            postgresql_where=text("claim_id IS NOT NULL"),
        ),
        Index(
            "ix_verification_results_review",
            "verification_run_id",
            postgresql_where=text("needs_human_review"),
        ),        Index(
            "uq_verification_results_run_report_claim",
            "verification_run_id",
            "report_claim_id",
            unique=True,
            postgresql_where=text("report_claim_id IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    verification_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("verification_runs.id", ondelete="CASCADE"), index=True
    )
    claim_text: Mapped[str] = mapped_column(Text)
    verification_label: Mapped[VerificationLabel | None] = mapped_column(
        Enum(
            VerificationLabel,
            native_enum=False,
            length=32,
            create_constraint=True,
            name="verification_label",
            values_callable=lambda e: [m.value for m in e],
        ),
        index=True,
        nullable=True,
    )
    report_claim_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("report_claims.id", ondelete="CASCADE"), index=True
    )
    claim_id: Mapped[str | None] = mapped_column(String(64))
    claim_type: Mapped[ClaimType | None] = mapped_column(
        Enum(
            ClaimType,
            native_enum=False,
            length=32,
            create_constraint=True,
            name="claim_type",
            values_callable=lambda e: [m.value for m in e],
        )
    )
    verdict: Mapped[Verdict | None] = mapped_column(
        Enum(
            Verdict,
            native_enum=False,
            length=32,
            create_constraint=True,
            name="verdict",
            values_callable=lambda e: [m.value for m in e],
        )
    )
    needs_human_review: Mapped[bool] = mapped_column(
        Boolean, server_default=false(), default=False
    )
    policy_reason_codes: Mapped[list[str]] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb"), default=list
    )
    explanation: Mapped[str] = mapped_column(Text)
    evidence_reference: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    verification_run: Mapped["VerificationRun"] = relationship(back_populates="results")
