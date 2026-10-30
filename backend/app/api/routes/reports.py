import uuid
from typing import Annotated

from fastapi import APIRouter, Body, status

from app.api.deps import AppSettings, DbSession, Storage, Vlm
from app.schemas.common import ErrorResponse
from app.schemas.report import (
    ReportDetailRead,
    ReportGenerateRequest,
    ReportList,
    ReportVerifyRequest,
    VlmStatusRead,
    report_detail,
    report_summary,
)
from app.services import report_service
from app.services.evidence_quality import QualityLimits

router = APIRouter(tags=["reports"])

_MODEL_ERRORS = {
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}


@router.get("/vlm", response_model=VlmStatusRead)
def vlm_status(provider: Vlm) -> VlmStatusRead:
    """Whether the local vision-language model is configured (never loads it)."""
    return VlmStatusRead.from_status(provider.status())


@router.post(
    "/inspections/{inspection_id}/reports",
    response_model=ReportDetailRead,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, **_MODEL_ERRORS},
)
def generate_report(
    inspection_id: uuid.UUID,
    db: DbSession,
    storage: Storage,
    provider: Vlm,
    body: Annotated[ReportGenerateRequest | None, Body()] = None,
) -> ReportDetailRead:
    """Generate a report from the inspection's images and latest detections (synchronous)."""
    report_id = report_service.generate_report(db, storage, provider, inspection_id)
    return report_detail(*report_service.get_report(db, report_id))


@router.get(
    "/inspections/{inspection_id}/reports",
    response_model=ReportList,
    responses={404: {"model": ErrorResponse}},
)
def list_reports(inspection_id: uuid.UUID, db: DbSession) -> ReportList:
    return ReportList(items=[report_summary(r) for r in report_service.list_reports(db, inspection_id)])


@router.get("/reports/{report_id}", response_model=ReportDetailRead, responses={404: {"model": ErrorResponse}})
def get_report(report_id: uuid.UUID, db: DbSession) -> ReportDetailRead:
    """The report, its claims as generated, and per-claim results of its latest verification run."""
    return report_detail(*report_service.get_report(db, report_id))


@router.post(
    "/reports/{report_id}/verifications",
    response_model=ReportDetailRead,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, **_MODEL_ERRORS},
)
def verify_report(
    report_id: uuid.UUID,
    db: DbSession,
    storage: Storage,
    provider: Vlm,
    settings: AppSettings,
    body: Annotated[ReportVerifyRequest | None, Body()] = None,
) -> ReportDetailRead:
    """Verify the report's stored claims in a new run (synchronous); claims are never changed."""
    report_service.verify_stored_report(db, storage, provider, report_id, QualityLimits.from_settings(settings))
    return report_detail(*report_service.get_report(db, report_id))
