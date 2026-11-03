import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from app.api.deps import AppSettings, DbSession, Detector, Storage, Vlm
from app.models.inspection import Inspection
from app.schemas.common import ErrorResponse
from app.schemas.inspection import InspectionCreate, InspectionList, InspectionRead, InspectionWorkflowRead
from app.schemas.report import report_detail
from app.services import inspection_workflow, report_service
from app.services.evidence_quality import QualityLimits

router = APIRouter(prefix="/inspections", tags=["inspections"])


@router.get("", response_model=InspectionList)
def list_inspections(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> InspectionList:
    total = db.scalar(select(func.count()).select_from(Inspection)) or 0
    rows = db.scalars(
        select(Inspection)
        .order_by(Inspection.created_at.desc(), Inspection.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return InspectionList(
        items=[InspectionRead.model_validate(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=InspectionRead, status_code=status.HTTP_201_CREATED)
def create_inspection(payload: InspectionCreate, db: DbSession) -> Inspection:
    inspection = Inspection(**payload.model_dump())
    db.add(inspection)
    db.commit()
    db.refresh(inspection)
    return inspection


@router.get(
    "/{inspection_id}",
    response_model=InspectionRead,
    responses={404: {"model": ErrorResponse}},
)
def get_inspection(inspection_id: uuid.UUID, db: DbSession) -> Inspection:
    inspection = db.get(Inspection, inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail="Inspection not found.")
    return inspection


@router.post(
    "/{inspection_id}/run",
    response_model=InspectionWorkflowRead,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
def run_inspection(
    inspection_id: uuid.UUID,
    db: DbSession,
    storage: Storage,
    detector: Detector,
    vlm: Vlm,
    settings: AppSettings,
) -> InspectionWorkflowRead:
    """Run configured local models and the existing claim-policy pipeline."""
    result = inspection_workflow.run(
        db,
        storage,
        detector,
        vlm,
        inspection_id,
        heartbeat_seconds=settings.detector_heartbeat_seconds,
        quality_limits=QualityLimits.from_settings(settings),
    )
    report, verification = report_service.get_report(db, result.report_id)
    return InspectionWorkflowRead(
        inspection=InspectionRead.model_validate(result.inspection),
        detections=result.detections,
        report=report_detail(report, verification),
    )
