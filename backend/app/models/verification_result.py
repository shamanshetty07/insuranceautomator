import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.verification_run import VerificationRun


class VerificationLabel(str, enum.Enum):
    """Outcome of checking one claim against visual evidence.

    OMITTED_DAMAGE is different in kind: it describes damage present in the
    images but absent from the report, so it is not a claim extracted from the
    report text. Such rows come from a separate image-to-report comparison;
    claim_text then describes the omitted damage rather than quoting the report.
    """

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

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    verification_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("verification_runs.id", ondelete="CASCADE"), index=True
    )
    claim_text: Mapped[str] = mapped_column(Text)
    verification_label: Mapped[VerificationLabel] = mapped_column(
        Enum(
            VerificationLabel,
            native_enum=False,
            length=32,
            create_constraint=True,
            name="verification_label",
            values_callable=lambda e: [m.value for m in e],
        ),
        index=True,
    )
    explanation: Mapped[str] = mapped_column(Text)
    # Free-form pointer to supporting evidence, e.g. {"image_ids": [...], "prediction_ids": [...]}.
    evidence_reference: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    verification_run: Mapped["VerificationRun"] = relationship(back_populates="results")
