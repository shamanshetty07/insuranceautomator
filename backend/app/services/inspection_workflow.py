"""The explicit, synchronous inspection workflow composed from existing services."""

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.errors import AppError
from app.models.inspection import Inspection, InspectionStatus
from app.schemas.detection import DetectionRunRead
from app.services import detection_service, report_service
from app.services.detection.provider import DetectorProvider
from app.services.evidence_quality import QualityLimits
from app.services.image_storage import ImageStorage


@dataclass(frozen=True)
class WorkflowResult:
    inspection: Inspection
    detections: list[DetectionRunRead]
    report_id: uuid.UUID


def _set_status(db: Session, inspection_id: uuid.UUID, status: InspectionStatus) -> Inspection:
    inspection = db.get(Inspection, inspection_id)
    if inspection is None:
        raise AppError(404, "not_found", "Inspection not found.")
    inspection.status = status
    db.commit()
    db.refresh(inspection)
    return inspection


def run(
    db: Session,
    storage: ImageStorage,
    detector_provider: DetectorProvider,
    vlm_provider: report_service.ReportModelProvider,
    inspection_id: uuid.UUID,
    *,
    heartbeat_seconds: float,
    quality_limits: QualityLimits,
) -> WorkflowResult:
    """Run detection, report generation and verification without changing their semantics.

    Individual services retain their durable runs and stable errors. This coordinator
    only makes the inspection lifecycle visible; it never retries or overwrites a
    prior report.
    """
    _set_status(db, inspection_id, InspectionStatus.PROCESSING)
    try:
        runs = detection_service.run_detection(
            db, storage, detector_provider, inspection_id, None, heartbeat_seconds=heartbeat_seconds
        )
        report_id = report_service.generate_report(db, storage, vlm_provider, inspection_id)
        report, _ = report_service.get_report(db, report_id)
        if report.claims:
            report_service.verify_stored_report(db, storage, vlm_provider, report_id, quality_limits)
        report, verification = report_service.get_report(db, report_id)
        review_required = bool(
            verification and any(result.needs_human_review for result in verification.results)
        )
        inspection = _set_status(
            db,
            inspection_id,
            InspectionStatus.REVIEW_REQUIRED if review_required else InspectionStatus.COMPLETED,
        )
        return WorkflowResult(
            inspection=inspection,
            detections=[DetectionRunRead.from_run(run) for run in runs],
            report_id=report.id,
        )
    except BaseException:
        # Stage-specific runs remain persisted; make the top-level interruption clear.
        try:
            _set_status(db, inspection_id, InspectionStatus.FAILED)
        except Exception:
            db.rollback()
        raise
