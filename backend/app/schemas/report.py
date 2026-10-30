"""API shapes for generated reports, their stored claims and claim-level verification."""

import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.generated_report import EvidenceStrategy
from app.models.run_status import RunStatus
from app.models.verification_result import VerificationLabel
from app.schemas.claims import ClaimType, EvidenceReference, Verdict


class _Empty(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReportGenerateRequest(_Empty):
    """No options yet (evidence strategy is full_image; settings come from the server)."""


class ReportVerifyRequest(_Empty):
    """No options yet (the verifier and its settings come from the server)."""


class VlmStatusRead(BaseModel):
    available: bool
    backend: str
    reason: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    weights_sha256: str | None = None

    @classmethod
    def from_status(cls, status: object) -> "VlmStatusRead":
        return cls.model_validate(asdict(status) if is_dataclass(status) else status, from_attributes=True)


class ClaimResultRead(BaseModel):
    """Final policy outcome plus the verifier's raw label (kept apart on purpose)."""

    model_config = ConfigDict(from_attributes=True)

    verdict: Verdict | None
    needs_human_review: bool
    policy_reason_codes: list[str]
    verification_label: VerificationLabel | None
    explanation: str


class ReportClaimRead(BaseModel):
    id: uuid.UUID
    claim_id: str
    position: int
    text: str
    claim_type: ClaimType
    damage_type: str | None
    image_ids: list[uuid.UUID]
    prediction_ids: list[uuid.UUID]
    evidence: list[EvidenceReference]
    result: ClaimResultRead | None = None


class VerificationRunRead(BaseModel):
    id: uuid.UUID
    report_id: uuid.UUID
    verifier_model: str
    verifier_version: str
    status: RunStatus
    error_message: str | None
    policy_version: str | None
    verifier_independent: bool | None
    created_at: datetime
    completed_at: datetime | None
    review_count: int


class ReportSummaryRead(BaseModel):
    id: uuid.UUID
    inspection_id: uuid.UUID
    content: str
    model_name: str
    model_version: str
    evidence_strategy: EvidenceStrategy
    created_at: datetime
    claim_count: int


class ReportDetailRead(ReportSummaryRead):
    generation_config: dict[str, Any]
    claims: list[ReportClaimRead]
    verification: VerificationRunRead | None = None


class ReportList(BaseModel):
    items: list[ReportSummaryRead] = Field(default_factory=list)


def report_summary(report: Any) -> ReportSummaryRead:
    return ReportSummaryRead(
        id=report.id,
        inspection_id=report.inspection_id,
        content=report.content,
        model_name=report.model_name,
        model_version=report.model_version,
        evidence_strategy=report.evidence_strategy,
        created_at=report.created_at,
        claim_count=len(report.claims),
    )


def verification_run_read(run: Any) -> VerificationRunRead:
    config = run.config or {}
    return VerificationRunRead(
        id=run.id,
        report_id=run.generated_report_id,
        verifier_model=run.verifier_model,
        verifier_version=run.verifier_version,
        status=run.status,
        error_message=run.error_message,
        policy_version=config.get("policy_version"),
        verifier_independent=config.get("verifier_independent"),
        created_at=run.created_at,
        completed_at=run.completed_at,
        review_count=sum(1 for r in run.results if r.needs_human_review),
    )


def report_detail(report: Any, run: Any | None) -> ReportDetailRead:
    """`run` is the report's latest verification run (with results) or None."""
    by_claim_row: dict[uuid.UUID, Any] = {}
    by_claim_id: dict[str, Any] = {}
    if run is not None:
        for result in run.results:
            if result.report_claim_id is not None:
                by_claim_row[result.report_claim_id] = result
            elif result.claim_id is not None:
                by_claim_id[result.claim_id] = result
    claims = []
    for row in report.claims:
        result = by_claim_row.get(row.id) or by_claim_id.get(row.claim_id)
        claims.append(
            ReportClaimRead(
                id=row.id,
                claim_id=row.claim_id,
                position=row.position,
                text=row.text,
                claim_type=row.claim_type,
                damage_type=row.damage_type,
                image_ids=row.image_ids,
                prediction_ids=row.prediction_ids,
                evidence=row.evidence,
                result=ClaimResultRead.model_validate(result) if result is not None else None,
            )
        )
    summary = report_summary(report)
    return ReportDetailRead(
        **summary.model_dump(),
        generation_config=report.generation_config or {},
        claims=claims,
        verification=verification_run_read(run) if run is not None else None,
    )
