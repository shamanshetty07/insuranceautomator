"""Image upload and deletion with explicit handling of partial failures."""

import hashlib
import logging
import re
import uuid

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import AppError
from app.models.inspection import Inspection
from app.models.inspection_image import InspectionImage
from app.services.image_processing import ImageLimits, process_image
from app.services.image_storage import ImageStorage, make_relative_path

logger = logging.getLogger(__name__)

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def clean_filename(name: str | None) -> str:
    """Display-only copy of the client's filename; never used to build a path."""
    base = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    base = _CONTROL_CHARS.sub("", base).strip()
    return base[:255] or "unnamed"


def _delete_files(storage: ImageStorage, paths: list[str]) -> None:
    for path in paths:
        try:
            storage.delete(path)
        except Exception:
            logger.exception("Failed to delete stored file; orphan needs reconciliation path=%s", path)


def upload_image(
    db: Session,
    storage: ImageStorage,
    settings: Settings,
    inspection_id: uuid.UUID,
    raw: bytes,
    original_filename: str | None,
) -> InspectionImage:
    if db.get(Inspection, inspection_id) is None:
        raise AppError(404, "not_found", "Inspection not found.")

    processed = process_image(
        raw, ImageLimits(min_side_px=settings.min_image_side_px, max_pixels=settings.max_image_pixels)
    )
    sha256 = hashlib.sha256(processed.data).hexdigest()

    if db.get(Inspection, inspection_id, with_for_update=True, populate_existing=True) is None:
        raise AppError(404, "not_found", "Inspection not found.")
    count, max_order = db.execute(
        select(func.count(), func.max(InspectionImage.image_order)).where(
            InspectionImage.inspection_id == inspection_id
        )
    ).one()
    if count >= settings.max_images_per_inspection:
        raise AppError(
            409,
            "image_limit_reached",
            f"An inspection can have at most {settings.max_images_per_inspection} images.",
        )
    duplicate = db.scalar(
        select(InspectionImage.id).where(
            InspectionImage.inspection_id == inspection_id, InspectionImage.sha256 == sha256
        )
    )
    if duplicate is not None:
        raise AppError(409, "duplicate_image", "This image has already been uploaded to this inspection.")

    image_id = uuid.uuid4()
    relative_path = make_relative_path(inspection_id, image_id, processed.extension)
    storage.write(relative_path, processed.data)

    image = InspectionImage(
        id=image_id,
        inspection_id=inspection_id,
        storage_path=relative_path,
        original_filename=clean_filename(original_filename),
        media_type=processed.media_type,
        file_size_bytes=len(processed.data),
        sha256=sha256,
        width_px=processed.width_px,
        height_px=processed.height_px,
        exif_orientation=processed.exif_orientation,
        image_order=0 if max_order is None else max_order + 1,
    )
    try:
        db.add(image)
        db.commit()
    except BaseException:
        db.rollback()
        try:
            storage.delete(relative_path)
        except Exception:
            logger.exception(
                "Database insert failed and cleanup of the new file also failed; "
                "orphan needs reconciliation path=%s",
                relative_path,
            )
        raise
    db.refresh(image)
    return image


def delete_image(
    db: Session, storage: ImageStorage, inspection_id: uuid.UUID, image_id: uuid.UUID
) -> None:
    image = db.get(InspectionImage, image_id)
    if image is None or image.inspection_id != inspection_id:
        raise AppError(404, "not_found", "Image not found.")
    path = image.storage_path
    try:
        db.delete(image)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise AppError(
            409,
            "image_in_use",
            "This image is used as evidence by a report (its detections or a claim) and cannot be deleted.",
        ) from None
    except BaseException:
        db.rollback()
        raise
    _delete_files(storage, [path])


def delete_inspection(db: Session, storage: ImageStorage, inspection_id: uuid.UUID) -> None:
    """Delete an inspection and everything under it, then its files (no HTTP endpoint yet)."""
    inspection = db.get(Inspection, inspection_id)
    if inspection is None:
        raise AppError(404, "not_found", "Inspection not found.")
    paths = list(
        db.scalars(
            select(InspectionImage.storage_path).where(InspectionImage.inspection_id == inspection_id)
        )
    )
    try:
        db.delete(inspection)
        db.commit()
    except BaseException:
        db.rollback()
        raise
    _delete_files(storage, paths)
