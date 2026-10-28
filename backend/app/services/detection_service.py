"""Run damage detection on an inspection's images and persist the results.

One DetectionRun per image, run sequentially. Each run is committed as `running`
before inference, so a crash leaves a visible (stale) running run instead of
nothing. A failure on one image becomes a `failed` run with a client-safe message
and never aborts the remaining images. Timestamps are wall-clock (not the
transaction-start time of now()) so completed_at - created_at is the real duration.
"""

import logging
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.errors import AppError
from app.models.damage_prediction import DamagePrediction
from app.models.detection_run import DetectionRun
from app.models.inspection import Inspection
from app.models.inspection_image import InspectionImage
from app.models.run_status import RunStatus
from app.schemas.damage import BoundingBox
from app.services.detection.provider import DetectorProvider, DetectorUnavailable
from app.services.image_storage import ImageStorage, StorageError
from app.services.interfaces import DamageDetectionService
from app.services.types import Detection, ImageRef

logger = logging.getLogger(__name__)


class _RunFailure(Exception):
    """An expected per-image failure; the message is stored on the run."""


def _now() -> datetime:
    return datetime.now(UTC)


def _require_inspection(db: Session, inspection_id: uuid.UUID) -> None:
    if db.get(Inspection, inspection_id) is None:
        raise AppError(404, "not_found", "Inspection not found.")


@dataclass(frozen=True)
class _Target:
    """What a run needs from an image, read up front: a rollback expires ORM objects,
    and the image may be deleted concurrently while earlier images are processed."""

    image_id: uuid.UUID
    storage_path: str


def _select_images(
    db: Session, inspection_id: uuid.UUID, image_ids: list[uuid.UUID] | None
) -> list[_Target]:
    rows = db.execute(
        select(InspectionImage.id, InspectionImage.storage_path)
        .where(InspectionImage.inspection_id == inspection_id)
        .order_by(InspectionImage.image_order)
    ).all()
    targets = [_Target(image_id=row.id, storage_path=row.storage_path) for row in rows]
    if image_ids is not None:
        wanted = set(image_ids)
        if not wanted <= {t.image_id for t in targets}:  # wrong inspection looks like 404
            raise AppError(404, "not_found", "Image not found.")
        return [t for t in targets if t.image_id in wanted]
    if not targets:
        raise AppError(409, "no_images", "This inspection has no images to analyse.")
    return targets


def _to_prediction(run_id: uuid.UUID, detection: Detection, image_id: uuid.UUID) -> DamagePrediction:
    if detection.image_id != image_id:
        raise _RunFailure("The detector returned a result for a different image.")
    confidence = float(detection.confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise _RunFailure("The detector returned an invalid confidence score.")
    try:
        box = BoundingBox(**detection.bounding_box) if detection.bounding_box is not None else None
    except (TypeError, ValidationError):
        raise _RunFailure("The detector returned an invalid bounding box.") from None
    # Not attached to the run yet: nothing reaches the session unless the run succeeds.
    return DamagePrediction(
        detection_run_id=run_id,
        damage_type=detection.damage_type,
        confidence=confidence,
        bounding_box=box.model_dump() if box else None,
    )


def _infer(
    run_id: uuid.UUID, target: _Target, storage: ImageStorage, detector: DamageDetectionService
) -> list[DamagePrediction]:
    try:
        path = storage.resolve(target.storage_path)
    except StorageError:
        logger.error("Unsafe storage path in database image_id=%s", target.image_id)
        raise _RunFailure("The image file is not available.") from None
    if not path.is_file():
        logger.error(
            "Image file missing on disk image_id=%s path=%s", target.image_id, target.storage_path
        )
        raise _RunFailure("The image file is not available.")
    detections = detector.detect([ImageRef(image_id=target.image_id, path=path)])
    return [_to_prediction(run_id, d, target.image_id) for d in detections]


def _commit(db: Session, run_id: uuid.UUID, what: str) -> bool:
    """Commit; False (rolled back, logged) if that failed, e.g. the image was deleted."""
    try:
        db.commit()
        return True
    except Exception:
        db.rollback()
        logger.exception("Could not %s detection run run_id=%s", what, run_id)
        return False


def _finish(
    db: Session, run: DetectionRun, run_id: uuid.UUID, status: RunStatus, error: str | None
) -> bool:
    run.status = status
    run.error_message = error
    run.completed_at = _now()
    return _commit(db, run_id, "record the result of")


def _run_one(
    db: Session, storage: ImageStorage, detector: DamageDetectionService, target: _Target
) -> uuid.UUID | None:
    """Run one image; returns the run id, or None if its image disappeared (and the run with it)."""
    run_id = uuid.uuid4()
    run = DetectionRun(
        id=run_id,
        inspection_image_id=target.image_id,
        model_name=detector.model.name,
        model_version=detector.model.version,
        weights_sha256=detector.model.weights_sha256,
        config=detector.config.model_dump(mode="json"),
        status=RunStatus.RUNNING,
        created_at=_now(),
    )
    db.add(run)
    if not _commit(db, run_id, "create"):
        return None

    try:
        predictions = _infer(run_id, target, storage, detector)
    except _RunFailure as exc:
        error: str | None = str(exc)
    except Exception:
        # Details (never image contents) stay in the server log.
        logger.exception("Detection failed run_id=%s image_id=%s", run_id, target.image_id)
        error = "Detection failed for this image."
    else:
        db.add_all(predictions)
        if _finish(db, run, run_id, RunStatus.COMPLETED, None):
            return run_id
        error = "The detection results could not be saved."

    return run_id if _finish(db, run, run_id, RunStatus.FAILED, error) else None


def run_detection(
    db: Session,
    storage: ImageStorage,
    provider: DetectorProvider,
    inspection_id: uuid.UUID,
    image_ids: list[uuid.UUID] | None,
) -> list[DetectionRun]:
    _require_inspection(db, inspection_id)
    targets = _select_images(db, inspection_id, image_ids)
    try:
        detector = provider.get()
    except DetectorUnavailable as exc:
        raise AppError(503, "detector_unavailable", str(exc)) from None

    run_ids = [run_id for t in targets if (run_id := _run_one(db, storage, detector, t))]
    return _load_runs(db, run_ids)


def _load_runs(db: Session, run_ids: list[uuid.UUID]) -> list[DetectionRun]:
    if not run_ids:
        return []
    runs = {
        run.id: run
        for run in db.scalars(
            select(DetectionRun)
            .where(DetectionRun.id.in_(run_ids))
            .options(selectinload(DetectionRun.predictions))
            .execution_options(populate_existing=True)
        )
    }
    return [runs[run_id] for run_id in run_ids if run_id in runs]


def list_detection_runs(db: Session, inspection_id: uuid.UUID, latest: bool) -> list[DetectionRun]:
    """latest=True: newest run per image, in image order; False: all runs, newest first."""
    _require_inspection(db, inspection_id)
    query = (
        select(DetectionRun)
        .join(InspectionImage, DetectionRun.inspection_image_id == InspectionImage.id)
        .where(InspectionImage.inspection_id == inspection_id)
        .options(selectinload(DetectionRun.predictions))
    )
    if latest:
        query = query.distinct(InspectionImage.image_order).order_by(
            InspectionImage.image_order, DetectionRun.created_at.desc(), DetectionRun.id.desc()
        )
    else:
        query = query.order_by(DetectionRun.created_at.desc(), DetectionRun.id.desc())
    return list(db.scalars(query))
