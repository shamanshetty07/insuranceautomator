import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Index,
    SmallInteger,
    text,
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
    from app.models.detection_run import DetectionRun
    from app.models.inspection import Inspection


class InspectionImage(Base):
    """Metadata for an image stored on the local filesystem (populated from Phase 2)."""

    __tablename__ = "inspection_images"
    __table_args__ = (
        UniqueConstraint("inspection_id", "image_order", name="uq_inspection_images_order"),
        CheckConstraint("file_size_bytes >= 0", name="ck_inspection_images_size"),
        CheckConstraint("image_order >= 0", name="ck_inspection_images_order"),
        CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="ck_inspection_images_sha256"),
        CheckConstraint("width_px > 0 AND height_px > 0", name="ck_inspection_images_dimensions"),
        CheckConstraint("exif_orientation BETWEEN 1 AND 8", name="ck_inspection_images_exif_orientation"),
        # The same normalized content cannot be stored twice in one inspection.
        Index(
            "uq_inspection_images_sha256",
            "inspection_id",
            "sha256",
            unique=True,
            postgresql_where=text("sha256 IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    inspection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), index=True
    )
    # Server-generated "<inspection_id>/<image_id>.<ext>" relative to STORAGE_ROOT;
    # never exposed to clients and never derived from the client's filename.
    storage_path: Mapped[str] = mapped_column(String(500), unique=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(100))
    file_size_bytes: Mapped[int] = mapped_column(BigInteger)
    # Lowercase hex SHA-256 of the exact normalized bytes on disk. Nullable only for
    # records that predate checksumming; every newly stored image has one.
    sha256: Mapped[str | None] = mapped_column(String(64))
    # Pixel size of the stored (EXIF-normalized) image: the coordinate space for
    # normalized bounding boxes. Nullable only for records that predate Phase 2.
    width_px: Mapped[int | None] = mapped_column(Integer)
    height_px: Mapped[int | None] = mapped_column(Integer)
    # EXIF orientation (1-8) of the original upload. Metadata only: the stored
    # pixels are already rotated, so this must never be applied again.
    exif_orientation: Mapped[int | None] = mapped_column(SmallInteger)
    image_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    inspection: Mapped["Inspection"] = relationship(back_populates="images")
    detection_runs: Mapped[list["DetectionRun"]] = relationship(
        back_populates="inspection_image", cascade="all, delete-orphan", passive_deletes=True
    )
