import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.damage_prediction import DamagePrediction
    from app.models.generated_report import GeneratedReport


class ReportPrediction(Base):
    """The exact predictions a report was generated from (canonical provenance).

    Deleting a report removes its links. Deleting a prediction (or its run) that a
    report still uses is refused at commit (the FK is DEFERRABLE INITIALLY DEFERRED,
    NO ACTION), so provenance cannot silently vanish. Deleting the whole inspection
    works because the reports and links go in the same transaction, in any order.
    Invariant for the service layer (not expressible as a simple FK): linked
    predictions must belong to the report's own inspection.
    """

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
