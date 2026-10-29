"""Run damage detection on an inspection's images and persist the results.

One DetectionRun per image, run sequentially. Each run is committed as `running`
before inference, so a crash leaves a visible (stale) running run instead of
nothing. Ordinary exceptions and BaseException (cancellation, KeyboardInterrupt) both
leave the run `failed`; only a hard process crash or a failed final commit can leave
`running`, which `fail_stale_runs` / `python -m app.services.detection.recovery` clears.
A failure on one image becomes a `failed` run with a client-safe message
and never aborts the remaining images. Timestamps are wall-clock (not the
transaction-start time of now()) so completed_at - created_at is the real duration.

Liveness and recovery (status transitions: running -> completed | failed, once):
- While a worker holds a run (waiting for the inference lock, then inferring), a
  RunHeartbeat thread refreshes `heartbeat_at`. Recovery fails only runs whose last
  heartbeat (created_at for rows without one) is older than the stale threshold, so
  a slow but live run is never failed just because it started long ago.
- Every transition out of `running` is conditional on the run still being `running`:
  recovery and the interruption marker use a conditional UPDATE; the worker's final
  write locks the row (SELECT ... FOR UPDATE) and re-checks. Whoever commits first
  wins; the other changes nothing. A run that recovery failed stays failed, without
  predictions, so no row is ever completed with a stale interruption error.
"""

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
    """Mark `running` runs with no sign of life for max_age as failed; return how many.

    Last sign of life is heartbeat_at, or created_at for rows written before heartbeats
    existed. One atomic conditional UPDATE (status still 'running' AND last seen before
    the cutoff): Postgres re-checks both conditions on rows a concurrent transaction
    changed, so a run a worker just finished, a run another recovery just failed, and a
    run whose heartbeat just advanced are all left alone. Idempotent. Sets
    completed_at, as COMPLETED_AT_CHECK requires for failed runs.

    max_age must be several heartbeat intervals (Settings enforces >= 4x), so a single
    missed beat cannot make a live run look abandoned.
    """
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
    """Refresh one running run's heartbeat_at from a daemon thread until the block exits.

    Each beat is its own short transaction on its own pooled connection (never the
    request's Session, which is not thread-safe) and only touches a run that is still
    `running`. A failed beat is logged and retried at the next interval. If beats keep
    failing for longer than the stale threshold (e.g. the database is unreachable),
    recovery may fail the run; the worker's final write then sees that and keeps the
    failure. A hung inference keeps beating: liveness of the process, not progress.
    """

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
        self._thread.join(timeout=self._interval)  # a beat in flight is one short UPDATE
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
    """Best-effort, atomic: fail this run only if it is still `running`. Never raises
    an ordinary exception (it runs while another error is propagating)."""
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
    db: Session,
    run: DetectionRun,
    run_id: uuid.UUID,
    status: RunStatus,
    error: str | None,
    predictions: Sequence[DamagePrediction] = (),
) -> bool:
    """Record the outcome, only while the run is still `running`; True if the run is now final.

    The row is locked and re-read first. If something else (recovery) already ended
    the run, its state is kept, `predictions` are discarded and True is returned: the
    run exists and is final. False means nothing could be recorded (logged, rolled back).
    """
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
    except Exception:  # e.g. the row was deleted with its image
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
        # Cancellation / KeyboardInterrupt / SystemExit: record the run as failed, then
        # let it propagate (never swallowed). Atomic, so a finished run is not touched.
        _fail_if_running(db, run_id)
        raise


def _run_config(detector: DamageDetectionService) -> dict:
    """Stored config: the detector's parameters plus server-recorded package versions."""
    config = detector.config.model_dump(mode="json")
    config.pop("runtime", None)  # only the server writes this
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
        # Details (never image contents) stay in the server log.
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

    engine = db.get_bind().engine  # the Engine even if the session is bound to a Connection

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
