"""Run damage detection on an inspection's images and persist the results."""

import logging
import math
import threading
import uuid
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pydantic import ValidationError
from sqlalchemy import Engine, func, select, update
from sqlalchemy.orm import Session, selectinload

from app.core.errors import AppError
from app.models.damage_prediction import DamagePrediction
from app.models.detection_run import DetectionRun
from app.models.inspection import Inspection
from app.models.inspection_image import InspectionImage
from app.models.run_status import RunStatus
from app.schemas.damage import BoundingBox
from app.services.detection.provider import DetectorProvider, DetectorUnavailable
from app.services.detection.runtime import runtime_versions
from app.services.image_storage import ImageStorage, StorageError
from app.services.interfaces import DamageDetectionService
from app.services.types import Detection, ImageRef

logger = logging.getLogger(__name__)


class _RunFailure(Exception):
    """An expected per-image failure; the message is stored on the run."""


INTERRUPTED_MESSAGE = "Detection was interrupted before it finished."
DEFAULT_HEARTBEAT_SECONDS = 30.0


def _now() -> datetime:
    return datetime.now(UTC)


def stale_cutoff(now: datetime, max_age: timedelta) -> datetime:
    """Runs last seen alive strictly before this instant count as stale."""
    if max_age <= timedelta(0):
        raise ValueError("max_age must be positive")
    return now - max_age


def fail_stale_runs(db: Session, max_age: timedelta, now: datetime | None = None) -> int:
    """Mark `running` runs with no sign of life for max_age as failed; return how many."""
    moment = now or _now()
    last_seen = func.coalesce(DetectionRun.heartbeat_at, DetectionRun.created_at)
    result = db.execute(
        update(DetectionRun)
        .where(DetectionRun.status == RunStatus.RUNNING, last_seen < stale_cutoff(moment, max_age))
        .values(status=RunStatus.FAILED, error_message=INTERRUPTED_MESSAGE, completed_at=moment)
        .execution_options(synchronize_session=False)
    )
    db.commit()
    return result.rowcount or 0


class RunHeartbeat:
    """Refresh one running run's heartbeat_at from a daemon thread until the block exits."""

    def __init__(self, engine: Engine, run_id: uuid.UUID, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._engine = engine
        self._run_id = run_id
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name=f"detection-heartbeat-{run_id}", daemon=True
        )

    def __enter__(self) -> "RunHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=self._interval)
        if self._thread.is_alive():
            logger.warning("Heartbeat thread did not stop in time run_id=%s", self._run_id)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.beat()

    def beat(self) -> None:
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    update(DetectionRun)
                    .where(DetectionRun.id == self._run_id, DetectionRun.status == RunStatus.RUNNING)
                    .values(heartbeat_at=_now())
                )
        except Exception:
            logger.warning("Could not record heartbeat run_id=%s", self._run_id, exc_info=True)


HeartbeatFactory = Callable[[uuid.UUID], AbstractContextManager[object]]


def _fail_if_running(db: Session, run_id: uuid.UUID) -> None:
    """Best-effort, atomic: fail this run only if it is still `running`."""
    try:
        db.rollback()
        db.execute(
            update(DetectionRun)
            .where(DetectionRun.id == run_id, DetectionRun.status == RunStatus.RUNNING)
            .values(status=RunStatus.FAILED, error_message=INTERRUPTED_MESSAGE, completed_at=_now())
            .execution_options(synchronize_session=False)
        )
        db.commit()
    except Exception:
        logger.exception("Could not mark interrupted detection run run_id=%s", run_id)


def _require_inspection(db: Session, inspection_id: uuid.UUID) -> None:
    if db.get(Inspection, inspection_id) is None:
        raise AppError(404, "not_found", "Inspection not found.")


@dataclass(frozen=True)
class _Target:
    """What a run needs from an image, read up front: a rollback expires ORM objects, and the image may be deleted concurrently while earlier images are processed."""

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
        if not wanted <= {t.image_id for t in targets}:
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
    db: Session,
    run: DetectionRun,
    run_id: uuid.UUID,
    status: RunStatus,
    error: str | None,
    predictions: Sequence[DamagePrediction] = (),
) -> bool:
    """Record the outcome, only while the run is still `running`; True if the run is now final."""
    try:
        db.refresh(run, with_for_update=True)
        current = run.status
        if current is not RunStatus.RUNNING:
            db.rollback()
            logger.warning(
                "Detection run run_id=%s was already %s; keeping that outcome", run_id, current.value
            )
            return True
        db.add_all(predictions)
        run.status = status
        run.error_message = error
        run.completed_at = _now()
    except Exception:
        logger.exception("Could not update detection run run_id=%s", run_id)
        db.rollback()
        return False
    return _commit(db, run_id, "record the result of")


def _run_one(
    db: Session,
    storage: ImageStorage,
    detector: DamageDetectionService,
    target: _Target,
    heartbeat: HeartbeatFactory,
) -> uuid.UUID | None:
    """Run one image; returns the run id, or None if its image disappeared (and the run with it)."""
    run_id = uuid.uuid4()
    created = _now()
    run = DetectionRun(
        id=run_id,
        inspection_image_id=target.image_id,
        model_name=detector.model.name,
        model_version=detector.model.version,
        weights_sha256=detector.model.weights_sha256,
        config=_run_config(detector),
        status=RunStatus.RUNNING,
        created_at=created,
        heartbeat_at=created,
    )
    db.add(run)
    if not _commit(db, run_id, "create"):
        return None

    try:
        with heartbeat(run_id):
            return _execute_run(db, storage, detector, target, run, run_id)
    except BaseException:
        _fail_if_running(db, run_id)
        raise


def _run_config(detector: DamageDetectionService) -> dict:
    """Stored config: the detector's parameters plus server-recorded package versions."""
    config = detector.config.model_dump(mode="json")
    config.pop("runtime", None)
    config["runtime"] = runtime_versions().model_dump(mode="json")
    return config


def _execute_run(
    db: Session,
    storage: ImageStorage,
    detector: DamageDetectionService,
    target: _Target,
    run: DetectionRun,
    run_id: uuid.UUID,
) -> uuid.UUID | None:
    try:
        predictions = _infer(run_id, target, storage, detector)
    except _RunFailure as exc:
        error: str | None = str(exc)
    except Exception:
        logger.exception("Detection failed run_id=%s image_id=%s", run_id, target.image_id)
        error = "Detection failed for this image."
    else:
        if _finish(db, run, run_id, RunStatus.COMPLETED, None, predictions):
            return run_id
        error = "The detection results could not be saved."

    return run_id if _finish(db, run, run_id, RunStatus.FAILED, error) else None


def run_detection(
    db: Session,
    storage: ImageStorage,
    provider: DetectorProvider,
    inspection_id: uuid.UUID,
    image_ids: list[uuid.UUID] | None,
    *,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
) -> list[DetectionRun]:
    _require_inspection(db, inspection_id)
    targets = _select_images(db, inspection_id, image_ids)
    try:
        detector = provider.get()
    except DetectorUnavailable as exc:
        raise AppError(503, "detector_unavailable", str(exc)) from None

    engine = db.get_bind().engine

    def heartbeat(run_id: uuid.UUID) -> RunHeartbeat:
        return RunHeartbeat(engine, run_id, heartbeat_seconds)

    run_ids = [run_id for t in targets if (run_id := _run_one(db, storage, detector, t, heartbeat))]
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
