import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, File, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select

from app.api.deps import AppSettings, DbSession, Storage
from app.core.errors import AppError
from app.models.inspection import Inspection
from app.models.inspection_image import InspectionImage
from app.schemas.common import ErrorResponse
from app.schemas.image import ImageList, ImageRead
from app.services import image_service
from app.services.image_storage import StorageError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/inspections/{inspection_id}/images", tags=["images"])

_errors = {404: {"model": ErrorResponse}}
_READ_CHUNK = 1024 * 1024


def _require_inspection(db: DbSession, inspection_id: uuid.UUID) -> None:
    if db.get(Inspection, inspection_id) is None:
        raise AppError(404, "not_found", "Inspection not found.")


def _get_image(db: DbSession, inspection_id: uuid.UUID, image_id: uuid.UUID) -> InspectionImage:
    image = db.get(InspectionImage, image_id)
    if image is None or image.inspection_id != inspection_id:
        raise AppError(404, "not_found", "Image not found.")
    return image


@router.post(
    "",
    response_model=ImageRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def upload_image(
    inspection_id: uuid.UUID,
    file: Annotated[UploadFile, File(description="One JPEG, PNG or WebP image.")],
    db: DbSession,
    storage: Storage,
    settings: AppSettings,
) -> InspectionImage:
    """Upload one image. The client uploads several images with one request each."""
    _require_inspection(db, inspection_id)
    chunks: list[bytes] = []
    total = 0
    while chunk := file.file.read(_READ_CHUNK):
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise AppError(
                413,
                "file_too_large",
                f"File exceeds the maximum size of {settings.max_upload_bytes} bytes.",
            )
        chunks.append(chunk)
    return image_service.upload_image(
        db, storage, settings, inspection_id, b"".join(chunks), file.filename
    )


@router.get("", response_model=ImageList, responses=_errors)
def list_images(inspection_id: uuid.UUID, db: DbSession) -> ImageList:
    _require_inspection(db, inspection_id)
    rows = db.scalars(
        select(InspectionImage)
        .where(InspectionImage.inspection_id == inspection_id)
        .order_by(InspectionImage.image_order)
    ).all()
    total = db.scalar(
        select(func.count()).select_from(InspectionImage).where(
            InspectionImage.inspection_id == inspection_id
        )
    )
    return ImageList(items=[ImageRead.model_validate(r) for r in rows], total=total or 0)


@router.get("/{image_id}", response_model=ImageRead, responses=_errors)
def get_image(inspection_id: uuid.UUID, image_id: uuid.UUID, db: DbSession) -> InspectionImage:
    return _get_image(db, inspection_id, image_id)


@router.get("/{image_id}/content", response_class=FileResponse, responses=_errors)
def get_image_content(
    inspection_id: uuid.UUID, image_id: uuid.UUID, db: DbSession, storage: Storage
) -> FileResponse:
    image = _get_image(db, inspection_id, image_id)
    try:
        path = storage.resolve(image.storage_path)
    except StorageError:
        logger.error("Unsafe storage path in database image_id=%s", image.id)
        raise AppError(404, "image_file_missing", "The image file is not available.") from None
    if not path.is_file():
        logger.error("Image file missing on disk image_id=%s path=%s", image.id, image.storage_path)
        raise AppError(404, "image_file_missing", "The image file is not available.")
    return FileResponse(
        path,
        media_type=image.media_type,
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=3600"},
    )


@router.delete(
    "/{image_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def delete_image(
    inspection_id: uuid.UUID, image_id: uuid.UUID, db: DbSession, storage: Storage
) -> Response:
    image_service.delete_image(db, storage, inspection_id, image_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
