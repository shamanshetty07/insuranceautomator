import uuid
from typing import Annotated

from fastapi import APIRouter, Body, Query

from app.api.deps import DbSession, Detector, Storage
from app.schemas.common import ErrorResponse
from app.schemas.detection import (
    DetectionRequest,
    DetectionRunList,
    DetectionRunRead,
    DetectorStatusRead,
)
from app.services import detection_service

router = APIRouter(tags=["detections"])


@router.get("/detector", response_model=DetectorStatusRead)
def detector_status(provider: Detector) -> DetectorStatusRead:
    """Whether damage detection is available, and with which model and parameters."""
    return provider.status()


@router.post(
    "/inspections/{inspection_id}/detections",
    response_model=DetectionRunList,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def run_detection(
    inspection_id: uuid.UUID,
    db: DbSession,
    storage: Storage,
    provider: Detector,
    body: Annotated[DetectionRequest | None, Body()] = None,
) -> DetectionRunList:
    """Run detection synchronously, one run per image (omit image_ids for all images).

    A per-image failure is returned as a `failed` run; the request still succeeds.
    """
    runs = detection_service.run_detection(
        db, storage, provider, inspection_id, body.image_ids if body else None
    )
    return DetectionRunList(items=[DetectionRunRead.from_run(r) for r in runs])


@router.get(
    "/inspections/{inspection_id}/detections",
    response_model=DetectionRunList,
    responses={404: {"model": ErrorResponse}},
)
def list_detections(
    inspection_id: uuid.UUID,
    db: DbSession,
    latest: Annotated[bool, Query(description="Only the newest run per image.")] = True,
) -> DetectionRunList:
    runs = detection_service.list_detection_runs(db, inspection_id, latest)
    return DetectionRunList(items=[DetectionRunRead.from_run(r) for r in runs])
