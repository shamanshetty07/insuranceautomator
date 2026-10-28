import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.damage_prediction import DamagePrediction
    from app.models.inspection import Inspection


class InspectionImage(Base):
    """Metadata for an image stored on the local filesystem (populated from Phase 2)."""

    __tablename__ = "inspection_images"
    __table_args__ = (
        UniqueConstraint("inspection_id", "image_order", name="uq_inspection_images_order"),
        CheckConstraint("file_size_bytes >= 0", name="ck_inspection_images_size"),
        CheckConstraint("image_order >= 0", name="ck_inspection_images_order"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    inspection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), index=True
    )
    # Path relative to STORAGE_ROOT; never exposed directly to clients.
    storage_path: Mapped[str] = mapped_column(String(500), unique=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(100))
    file_size_bytes: Mapped[int] = mapped_column(BigInteger)
    image_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    inspection: Mapped["Inspection"] = relationship(back_populates="images")
    damage_predictions: Mapped[list["DamagePrediction"]] = relationship(
        back_populates="inspection_image", cascade="all, delete-orphan", passive_deletes=True
    )
