import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint, DateTime, Enum, ForeignKey, Integer, String, Text, UniqueConstraint, Uuid, func,
)
from sqlalchemy import text as sql_text  # `text` is also a column name below
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.schemas.claims import ClaimType

if TYPE_CHECKING:
    from app.models.generated_report import GeneratedReport

CLAIM_ID_CHECK = "claim_id ~ '^[A-Za-z0-9_-]{1,64}$'"
JSON_ARRAYS_CHECK = (
    "jsonb_typeof(image_ids) = 'array' AND jsonb_typeof(prediction_ids) = 'array' "
    "AND jsonb_typeof(evidence) = 'array'"
)


class ReportClaim(Base):
    """One atomic claim exactly as the generator produced it (app.schemas.claims.ExtractedClaim)."""

    __tablename__ = "report_claims"
    __table_args__ = (
        UniqueConstraint("generated_report_id", "claim_id", name="uq_report_claims_report_claim"),
        UniqueConstraint("generated_report_id", "position", name="uq_report_claims_report_position"),
        CheckConstraint(CLAIM_ID_CHECK, name="ck_report_claims_claim_id"),
        CheckConstraint("char_length(text) > 0", name="ck_report_claims_text"),
        CheckConstraint("position >= 0", name="ck_report_claims_position"),
        CheckConstraint(JSON_ARRAYS_CHECK, name="ck_report_claims_json_arrays"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    generated_report_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("generated_reports.id", ondelete="CASCADE")
    )
    position: Mapped[int] = mapped_column(Integer)
    claim_id: Mapped[str] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[ClaimType] = mapped_column(
        Enum(
            ClaimType,
            native_enum=False,
            length=32,
            create_constraint=True,
            name="report_claim_type",
            values_callable=lambda e: [m.value for m in e],
        )
    )
    damage_type: Mapped[str | None] = mapped_column(String(100))
    image_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default=sql_text("'[]'::jsonb"))
    prediction_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default=sql_text("'[]'::jsonb"))
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default=sql_text("'[]'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    generated_report: Mapped["GeneratedReport"] = relationship(back_populates="claims")
    image_links: Mapped[list["ReportClaimImage"]] = relationship(
        back_populates="report_claim", cascade="all, delete-orphan", passive_deletes=True
    )


class ReportClaimImage(Base):
    """An image a claim cites (only images inside the report's inspection get a row)."""

    __tablename__ = "report_claim_images"

    report_claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("report_claims.id", ondelete="CASCADE"), primary_key=True
    )
    inspection_image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inspection_images.id", deferrable=True, initially="DEFERRED"),
        primary_key=True,
        index=True,
    )

    report_claim: Mapped["ReportClaim"] = relationship(back_populates="image_links")
