import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.damage_prediction import DamagePrediction
    from app.models.generated_report import GeneratedReport


class ReportPrediction(Base):
    """The exact predictions a report was generated from (canonical provenance)."""

    __tablename__ = "report_predictions"

    generated_report_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("generated_reports.id", ondelete="CASCADE"), primary_key=True
    )
    damage_prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("damage_predictions.id", deferrable=True, initially="DEFERRED"),
        primary_key=True,
        index=True,
    )

    generated_report: Mapped["GeneratedReport"] = relationship(back_populates="prediction_links")
    damage_prediction: Mapped["DamagePrediction"] = relationship()
