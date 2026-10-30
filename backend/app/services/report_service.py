"""Generate and verify inspection reports by composing the existing services."""

import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from app.core.errors import AppError
from app.models.generated_report import EvidenceStrategy, GeneratedReport
from app.models.inspection import Inspection
from app.models.inspection_image import InspectionImage
from app.models.report_claim import ReportClaim, ReportClaimImage
from app.models.run_status import RunStatus
from app.models.verification_run import VerificationRun
from app.schemas.claims import StructuredReport
from app.schemas.runs import EvidenceInputs, GenerationConfig, ModelRecord, StoredGenerationConfig
from app.services.detection_service import list_detection_runs
from app.services.evidence_bundle import evidence_bundle
from app.services.evidence_quality import QualityLimits
from app.services.image_storage import ImageStorage, StorageError
from app.services.model_guard import refuse_stub_in_production
from app.services.provenance import ProvenanceError, attach_predictions_to_report
from app.services.report_claims import claim_row_values
from app.services.types import ClaimVerdict, EvidenceBundle, ReportClaim as VerifierClaim
from app.services.verification_service import verify_report
from app.services.vlm.errors import (
    VlmBusy,
    VlmError,
    VlmInferenceError,
    VlmInputError,
    VlmOutputError,
    VlmTimeout,
    VlmUnavailable,
)

logger = logging.getLogger(__name__)


class ReportModelProvider(Protocol):
    """What the report service needs from a model provider (vlm.SettingsVlmProvider)."""

    def get_generator(self): ...  # noqa: ANN201 - StructuredReportGenerationService + generation_config

    def get_verifier(self): ...  # noqa: ANN201 - ClaimVerificationService + verification_config


def vlm_error_to_app_error(exc: VlmError) -> AppError:
    if isinstance(exc, VlmUnavailable):
        return AppError(503, "vlm_unavailable", str(exc))
    if isinstance(exc, VlmBusy):
        return AppError(503, "vlm_busy", str(exc))
    if isinstance(exc, VlmTimeout):
        return AppError(504, "vlm_timeout", str(exc))
    if isinstance(exc, VlmInputError):
        return AppError(409, exc.code, str(exc))
    if isinstance(exc, VlmOutputError):
        return AppError(502, "vlm_output_rejected", f"The model output was rejected ({exc.code}).")
    if isinstance(exc, VlmInferenceError):
        return AppError(502, "vlm_inference_error", str(exc))
    return AppError(502, "vlm_error", "The vision-language model failed.")


def check_citations(report: StructuredReport, bundle: EvidenceBundle) -> None:
    """Every image and prediction a claim cites must have been given to the generator."""
    images = {ref.image_id for ref in bundle.images}
    predictions = {d.prediction_id for d in bundle.detections if d.prediction_id is not None}
    for claim in report.claims.claims:
        cited_images = set(claim.image_ids) | {ref.image_id for ref in claim.evidence}
        cited_predictions = set(claim.prediction_ids) | {
            ref.prediction_id for ref in claim.evidence if ref.prediction_id is not None
        }
        if not cited_images <= images or not cited_predictions <= predictions:
            raise VlmOutputError("A claim cites evidence the generator was not given.", code="uncited_evidence")


def supplied_prediction_ids(bundle: EvidenceBundle) -> list[uuid.UUID]:
    """Provenance links every prediction SUPPLIED to the generator, cited or not."""
    return list(dict.fromkeys(d.prediction_id for d in bundle.detections if d.prediction_id is not None))


@dataclass(frozen=True)
class EvidenceSet:
    bundle: EvidenceBundle
    detection_run_ids: tuple[uuid.UUID, ...]


def stored_generation_config(generator: object, evidence: EvidenceSet) -> StoredGenerationConfig:
    config = getattr(generator, "generation_config", None)
    if not isinstance(config, GenerationConfig):
        raise TypeError("the generator must expose a GenerationConfig as generation_config")
    model = generator.model  # type: ignore[attr-defined]
    return StoredGenerationConfig(
        **config.model_dump(),
        model=ModelRecord(weights_sha256=model.weights_sha256, packages=dict(getattr(generator, "runtime", None) or {})),
        limits=getattr(generator, "limits_record", None),
        evidence=EvidenceInputs(
            image_ids=tuple(ref.image_id for ref in evidence.bundle.images),
            detection_run_ids=evidence.detection_run_ids,
        ),
    )


class FailFastVerifier:
    """Wraps a verifier: after a timeout or busy model, the remaining claims fail at once."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self._stopped = False

    def __getattr__(self, name: str):  # noqa: ANN204 - model, verification_config, runtime, IS_STUB, ...
        return getattr(self._inner, name)

    def verify(self, claims: list[VerifierClaim], evidence: EvidenceBundle) -> list[ClaimVerdict]:
        if self._stopped:
            raise VlmBusy("Skipped: the model stopped responding earlier in this run.", code="skipped")
        try:
            return self._inner.verify(claims, evidence)  # type: ignore[attr-defined]
        except (VlmTimeout, VlmBusy):
            self._stopped = True
            raise


def _require_inspection(db: Session, inspection_id: uuid.UUID) -> None:
    if db.get(Inspection, inspection_id) is None:
        raise AppError(404, "not_found", "Inspection not found.")


def load_evidence(db: Session, storage: ImageStorage, inspection_id: uuid.UUID) -> EvidenceSet:
    """Images in report order with their stored files, plus the latest run per image."""
    _require_inspection(db, inspection_id)
    rows = db.execute(
        select(InspectionImage.id, InspectionImage.storage_path)
        .where(InspectionImage.inspection_id == inspection_id)
        .order_by(InspectionImage.image_order)
    ).all()
    if not rows:
        raise AppError(409, "no_images", "This inspection has no images to report on.")
    images = []
    for image_id, storage_path in rows:
        try:
            path = storage.resolve(storage_path)
        except StorageError:
            path = None
        if path is None or not path.is_file():
            logger.error("Image file unavailable for report image_id=%s", image_id)
            raise AppError(409, "image_unavailable", "An image file of this inspection is not available.")
        images.append((image_id, path))
    runs = list_detection_runs(db, inspection_id, latest=True)
    bundle = evidence_bundle(images, runs)
    completed = tuple(r.id for r in runs if r.status is RunStatus.COMPLETED)
    return EvidenceSet(bundle=bundle, detection_run_ids=completed)


def _claim_image_ids(claim) -> Iterable[uuid.UUID]:  # noqa: ANN001
    return dict.fromkeys([*claim.image_ids, *(ref.image_id for ref in claim.evidence)])


def _save_report(
    db: Session,
    inspection_id: uuid.UUID,
    generator: object,
    structured: StructuredReport,
    evidence: EvidenceSet,
) -> uuid.UUID:
    model = generator.model  # type: ignore[attr-defined]
    report = GeneratedReport(
        id=uuid.uuid4(),
        inspection_id=inspection_id,
        content=structured.report_text,
        model_name=model.name,
        model_version=model.version,
        evidence_strategy=EvidenceStrategy.FULL_IMAGE,
        generation_config=stored_generation_config(generator, evidence).model_dump(mode="json"),
    )
    try:
        db.add(report)
        db.flush()
        attach_predictions_to_report(db, report, supplied_prediction_ids(evidence.bundle))
        for position, claim in enumerate(structured.claims.claims):
            row = ReportClaim(id=uuid.uuid4(), generated_report_id=report.id, **claim_row_values(claim, position))
            row.image_links = [ReportClaimImage(inspection_image_id=i) for i in _claim_image_ids(claim)]
            db.add(row)
        db.commit()
    except (ProvenanceError, IntegrityError):
        db.rollback()
        logger.warning("Evidence changed during report generation inspection_id=%s", inspection_id, exc_info=True)
        raise AppError(409, "evidence_changed", "The inspection's evidence changed during generation; try again.") from None
    except Exception:
        db.rollback()
        logger.exception("Could not save generated report inspection_id=%s", inspection_id)
        raise AppError(500, "report_save_failed", "The generated report could not be saved.") from None
    return report.id


def generate_report(
    db: Session, storage: ImageStorage, provider: ReportModelProvider, inspection_id: uuid.UUID
) -> uuid.UUID:
    """Generate, validate and store one report; returns its id. Never verifies."""
    evidence = load_evidence(db, storage, inspection_id)
    db.rollback()
    try:
        generator = provider.get_generator()
        refuse_stub_in_production(generator, "vlm_unavailable")
        structured = generator.generate_structured(evidence.bundle)
        check_citations(structured, evidence.bundle)
    except VlmError as exc:
        raise vlm_error_to_app_error(exc) from None
    return _save_report(db, inspection_id, generator, structured, evidence)


def verify_stored_report(
    db: Session,
    storage: ImageStorage,
    provider: ReportModelProvider,
    report_id: uuid.UUID,
    quality_limits: QualityLimits | None = None,
) -> uuid.UUID:
    """Verify a stored report's claims in a new run; returns the run id."""
    if db.get(GeneratedReport, report_id) is None:
        raise AppError(404, "not_found", "Report not found.")
    count = db.scalar(select(func.count()).where(ReportClaim.generated_report_id == report_id))
    if not count:
        # A completed run with no results must never read as "verified".
        raise AppError(409, "no_claims", "This report has no claims to verify.")
    db.rollback()
    try:
        verifier = provider.get_verifier()
    except VlmError as exc:
        raise vlm_error_to_app_error(exc) from None
    return verify_report(db, storage, FailFastVerifier(verifier), report_id, quality_limits).id


def get_report(db: Session, report_id: uuid.UUID) -> tuple[GeneratedReport, VerificationRun | None]:
    """The report with its claims, and its latest verification run (with results) or None."""
    report = db.scalars(
        select(GeneratedReport)
        .where(GeneratedReport.id == report_id)
        .options(selectinload(GeneratedReport.claims))
        .execution_options(populate_existing=True)
    ).one_or_none()
    if report is None:
        raise AppError(404, "not_found", "Report not found.")
    run = db.scalars(
        select(VerificationRun)
        .where(VerificationRun.generated_report_id == report_id)
        .order_by(VerificationRun.created_at.desc(), VerificationRun.id.desc())
        .limit(1)
        .options(selectinload(VerificationRun.results))
        .execution_options(populate_existing=True)
    ).first()
    return report, run


def list_reports(db: Session, inspection_id: uuid.UUID) -> list[GeneratedReport]:
    """Reports of one inspection, newest first, with claims loaded (for claim counts)."""
    _require_inspection(db, inspection_id)
    return list(
        db.scalars(
            select(GeneratedReport)
            .where(GeneratedReport.inspection_id == inspection_id)
            .order_by(GeneratedReport.created_at.desc(), GeneratedReport.id.desc())
            .options(selectinload(GeneratedReport.claims))
        )
    )
