"""Persist claim-level verification of one report (model-independent orchestration)."""

import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session, selectinload

from app.core.config import Settings
from app.core.errors import AppError
from app.models.damage_prediction import DamagePrediction
from app.models.detection_run import DetectionRun
from app.models.generated_report import GeneratedReport
from app.models.inspection_image import InspectionImage
from app.models.run_status import RunStatus
from app.models.verification_result import VerificationLabel, VerificationResult
from app.models.verification_run import VerificationRun
from app.schemas.claims import MAX_REFERENCES_PER_CLAIM, ClaimType, ExtractedClaim, Verdict
from app.schemas.runs import ModelRecord, StoredVerificationConfig, VerificationConfig
from app.services.claim_policy import (
    POLICY_VERSION,
    ClaimPolicyDecision,
    ClaimPolicyInput,
    EvidenceQuality,
    ReasonCode,
    VerifierOutcome,
    evaluate_claim,
)
from app.services.claim_semantics import assertion_of
from app.services.evidence_association import ClaimEvidence, associate_claims
from app.services.evidence_quality import (
    DetectionQualityFacts,
    ImageQualityFacts,
    QualityEffect,
    QualityLimits,
    assess_detections,
    assess_image,
    combine,
)
from app.services.image_storage import ImageStorage
from app.services.interfaces import ClaimVerificationService
from app.services.model_guard import refuse_stub_in_production, verifier_is_independent
from app.services.report_claims import claims_from_rows
from app.services.types import EvidenceBundle, ImageRef, ModelInfo, ReportClaim
from app.services.verdict_consistency import PriorVerdict, check_consistency
from app.services.verdict_mapping import coerce_label, is_claim_label, legacy_verdict

logger = logging.getLogger(__name__)

INTERRUPTED_MESSAGE = "Verification was interrupted before it finished."
ALL_FAILED_MESSAGE = "The verifier failed for every claim it was asked to check."
SAVE_FAILED_MESSAGE = "The verification results could not be saved."
_MAX_EXPLANATION_CHARS = 2000
_UNRECOGNISED = "unrecognised"
_VISUAL_TYPES = frozenset({ClaimType.VISIBLE_DAMAGE_PRESENCE, ClaimType.DAMAGE_TYPE, ClaimType.DAMAGE_LOCATION})
QUALITY_OF_EFFECT = {
    QualityEffect.INFO: EvidenceQuality.ADEQUATE,
    QualityEffect.REVIEW: EvidenceQuality.QUESTIONABLE,
    QualityEffect.INSUFFICIENT: EvidenceQuality.INSUFFICIENT,
}


def _now() -> datetime:
    return datetime.now(UTC)


class _Raw:
    """What the verifier produced for one claim, before the policy sees it."""

    def __init__(
        self,
        label: VerificationLabel | None,
        outcome: VerifierOutcome,
        explanation: str,
        attempted: bool,
        errored: bool,
    ) -> None:
        self.label = label
        self.outcome = outcome
        self.explanation = explanation
        self.attempted = attempted
        self.errored = errored


@dataclass(frozen=True)
class ClaimQuality:
    quality: EvidenceQuality = EvidenceQuality.ADEQUATE
    findings: tuple[str, ...] = ()


def _policy_input(
    claim: ExtractedClaim,
    ev: ClaimEvidence,
    outcome: VerifierOutcome,
    verifier_independent: bool = True,
    verifier_consistent: bool = True,
    quality: EvidenceQuality = EvidenceQuality.ADEQUATE,
) -> ClaimPolicyInput:
    return ClaimPolicyInput(
        claim_type=claim.claim_type,
        verifier=outcome,
        integrity=ev.integrity,
        association=ev.association,
        detection_match=ev.detection_match,
        verifier_independent=verifier_independent,
        claim_positive=assertion_of(claim.text).is_positive,
        verifier_consistent=verifier_consistent,
        evidence_quality=quality,
    )


def _decided_without_verifier(
    claim: ExtractedClaim, ev: ClaimEvidence, quality: EvidenceQuality = EvidenceQuality.ADEQUATE
) -> bool:
    """True if a mandatory policy rule fires before the verifier's answer could matter."""
    decision = evaluate_claim(_policy_input(claim, ev, VerifierOutcome(verdict=None), quality=quality))
    return ReasonCode.VERIFIER_MISSING not in decision.reason_codes


def assess_claim_quality(
    db: Session, claims: Sequence[ExtractedClaim], evidence: Sequence[ClaimEvidence], limits: QualityLimits
) -> list[ClaimQuality]:
    """Metadata checks on each visual claim's cited images and detections; findings never add trust."""
    image_ids = {i for ev in evidence for i, _ in ev.image_paths}
    pred_ids = {p for ev in evidence for p in ev.prediction_ids}
    images = (
        {img.id: img for img in db.scalars(select(InspectionImage).where(InspectionImage.id.in_(image_ids)))}
        if image_ids else {}
    )
    preds: dict[uuid.UUID, DetectionQualityFacts] = {}
    if pred_ids:
        rows = db.execute(
            select(DamagePrediction, DetectionRun.inspection_image_id, DetectionRun.config)
            .join(DetectionRun, DamagePrediction.detection_run_id == DetectionRun.id)
            .where(DamagePrediction.id.in_(pred_ids))
        )
        for pred, image_id, run_config in rows:
            preds[pred.id] = DetectionQualityFacts(
                pred.id, image_id, pred.detection_run_id, pred.damage_type, pred.confidence,
                pred.bounding_box, run_config,
            )
    out: list[ClaimQuality] = []
    for claim, ev in zip(claims, evidence, strict=True):
        if claim.claim_type not in _VISUAL_TYPES:
            out.append(ClaimQuality())
            continue
        assessment = combine(
            *(assess_image(ImageQualityFacts.from_model(images[i]), limits) for i, _ in ev.image_paths if i in images),
            assess_detections([preds[p] for p in ev.prediction_ids if p in preds]),
        )
        findings = tuple(str(f) for f in assessment.findings if f.effect >= QualityEffect.REVIEW)
        out.append(ClaimQuality(QUALITY_OF_EFFECT[assessment.effect], findings))
    return out


def load_prior_verdicts(
    db: Session, report_id: uuid.UUID, report_claim_ids: Mapping[str, uuid.UUID]
) -> dict[str, list[PriorVerdict]]:
    """Raw labels earlier completed runs stored for these report claims, keyed by claim_id."""
    if not report_claim_ids:
        return {}
    by_row = {row_id: claim_id for claim_id, row_id in report_claim_ids.items()}
    rows = db.execute(
        select(
            VerificationResult.report_claim_id,
            VerificationResult.verification_label,
            VerificationRun.id,
            VerificationRun.verifier_model,
            VerificationRun.verifier_version,
            VerificationRun.config,
        )
        .join(VerificationRun, VerificationResult.verification_run_id == VerificationRun.id)
        .where(
            VerificationRun.generated_report_id == report_id,
            VerificationRun.status == RunStatus.COMPLETED,
            VerificationResult.report_claim_id.in_(list(by_row)),
        )
    )
    priors: dict[str, list[PriorVerdict]] = {}
    for row_id, label, run_id, name, version, config in rows:
        model = (config or {}).get("model") or {}
        digest = model.get("weights_sha256") if isinstance(model, Mapping) else None
        priors.setdefault(by_row[row_id], []).append(PriorVerdict(run_id, label, ModelInfo(name, version, digest)))
    return priors


def _ask_verifier(verifier: ClaimVerificationService, claim: ExtractedClaim, ev: ClaimEvidence) -> _Raw:
    bundle = EvidenceBundle(images=[ImageRef(image_id=i, path=p) for i, p in ev.image_paths])
    try:
        verdicts = verifier.verify([ReportClaim(text=claim.text)], bundle)
    except Exception:
        logger.exception("Verifier failed claim_id=%s", claim.claim_id)
        return _Raw(None, VerifierOutcome(verdict=None, errored=True), "The verifier failed on this claim.", True, True)
    if not isinstance(verdicts, list) or len(verdicts) != 1:
        logger.error("Verifier returned an unexpected result count claim_id=%s", claim.claim_id)
        return _Raw(
            None,
            VerifierOutcome(verdict=None, errored=True),
            "The verifier returned an unexpected number of results.",
            True,
            True,
        )
    result = verdicts[0]
    try:
        explanation = str(getattr(result, "explanation", "") or "")[:_MAX_EXPLANATION_CHARS]
        raw_label = getattr(result, "label", None)
    except Exception:
        logger.exception("Verifier result could not be read claim_id=%s", claim.claim_id)
        return _Raw(None, VerifierOutcome(verdict=None, errored=True), "The verifier result could not be read.",
                    True, True)
    label = coerce_label(raw_label)
    if label is None:
        return _Raw(
            None,
            VerifierOutcome(verdict=_UNRECOGNISED),
            "The verifier returned an unrecognised label.",
            True,
            False,
        )
    if not is_claim_label(label):
        return _Raw(
            None,
            VerifierOutcome(verdict=_UNRECOGNISED),
            f"The verifier answered '{label.value}', which is not a verdict on a report claim.",
            True,
            False,
        )
    mapped = legacy_verdict(label.value)
    outcome = VerifierOutcome(verdict=mapped if mapped is not None else label.value)
    return _Raw(label, outcome, explanation or "No explanation given.", True, False)


def default_quality_limits() -> QualityLimits:
    """The Settings defaults, without reading the environment."""
    fields = Settings.model_fields
    return QualityLimits(fields["min_image_side_px"].default, fields["vlm_max_image_pixels"].default)


def _build_row(
    run_id: uuid.UUID,
    claim: ExtractedClaim,
    ev: ClaimEvidence,
    raw: _Raw,
    decision: ClaimPolicyDecision,
    report_claim_id: uuid.UUID | None = None,
    quality: ClaimQuality = ClaimQuality(),
) -> VerificationResult:
    record = ev.to_record()
    if quality.findings:
        record = record.model_copy(update={"quality_findings": quality.findings[:MAX_REFERENCES_PER_CLAIM]})
    return VerificationResult(
        verification_run_id=run_id,
        report_claim_id=report_claim_id,
        claim_id=claim.claim_id,
        claim_text=claim.text,
        claim_type=claim.claim_type,
        verification_label=raw.label,
        explanation=raw.explanation,
        verdict=decision.verdict,
        needs_human_review=decision.needs_human_review,
        policy_reason_codes=[code.value for code in decision.reason_codes],
        evidence_reference=record.model_dump(mode="json"),
    )


def _fail_if_running(db: Session, run_id: uuid.UUID, message: str) -> None:
    """Best-effort and atomic: fail the run only if it is still `running`; never raises."""
    try:
        db.rollback()
        db.execute(
            update(VerificationRun)
            .where(VerificationRun.id == run_id, VerificationRun.status == RunStatus.RUNNING)
            .values(status=RunStatus.FAILED, error_message=message, completed_at=_now())
            .execution_options(synchronize_session=False)
        )
        db.commit()
    except Exception:
        logger.exception("Could not mark verification run failed run_id=%s", run_id)


def _load_run(db: Session, run_id: uuid.UUID) -> VerificationRun:
    return db.scalars(
        select(VerificationRun)
        .where(VerificationRun.id == run_id)
        .options(selectinload(VerificationRun.results))
        .execution_options(populate_existing=True)
    ).one()


def _execute(
    db: Session,
    storage: ImageStorage,
    verifier: ClaimVerificationService,
    run_id: uuid.UUID,
    inspection_id: uuid.UUID,
    claims: list[ExtractedClaim],
    report_claim_ids: Mapping[str, uuid.UUID] | None = None,
    verifier_independent: bool = True,
    priors: Mapping[str, Sequence[PriorVerdict]] | None = None,
    quality_limits: QualityLimits | None = None,
) -> VerificationRun:
    evidence = associate_claims(db, storage, inspection_id, claims)
    if len(evidence) != len(claims):
        raise RuntimeError("evidence association returned a different number of claims")
    qualities = assess_claim_quality(db, claims, evidence, quality_limits or default_quality_limits())
    db.rollback()

    rows: list[VerificationResult] = []
    attempted = errored = 0
    for claim, ev, quality in zip(claims, evidence, qualities, strict=True):
        if _decided_without_verifier(claim, ev, quality.quality):
            raw = _Raw(None, VerifierOutcome(verdict=None), "", False, False)
            consistent = True
        else:
            raw = _ask_verifier(verifier, claim, ev)
            check = check_consistency(raw.label, verifier.model, (priors or {}).get(claim.claim_id, ()))
            consistent = not check.disagrees
        attempted += raw.attempted
        errored += raw.errored
        decision = evaluate_claim(
            _policy_input(claim, ev, raw.outcome, verifier_independent, consistent, quality.quality)
        )
        if not raw.attempted:
            raw.explanation = f"Verifier not called: {decision.explanation}"
        claim_row_id = (report_claim_ids or {}).get(claim.claim_id)
        rows.append(_build_row(run_id, claim, ev, raw, decision, claim_row_id, quality))

    all_failed = attempted > 0 and errored == attempted
    status = RunStatus.FAILED if all_failed else RunStatus.COMPLETED
    try:
        db.add_all(rows)
        db.execute(
            update(VerificationRun)
            .where(VerificationRun.id == run_id, VerificationRun.status == RunStatus.RUNNING)
            .values(
                status=status,
                error_message=ALL_FAILED_MESSAGE if all_failed else None,
                completed_at=_now(),
            )
            .execution_options(synchronize_session=False)
        )
        db.commit()
    except Exception:
        logger.exception("Could not save verification results run_id=%s", run_id)
        _fail_if_running(db, run_id, SAVE_FAILED_MESSAGE)
    return _load_run(db, run_id)


def run_config(
    verifier: ClaimVerificationService, verifier_independent: bool
) -> StoredVerificationConfig:
    """The config recorded on a run, built from the verifier itself (never from a client)."""
    config = getattr(verifier, "verification_config", None)
    if not isinstance(config, VerificationConfig):
        raise TypeError("the verifier must expose a VerificationConfig as verification_config")
    limits = getattr(verifier, "limits_record", None)
    return StoredVerificationConfig(
        **config.model_dump(exclude={"policy_version"}),
        policy_version=POLICY_VERSION,
        model=ModelRecord(
            weights_sha256=verifier.model.weights_sha256,
            packages=dict(getattr(verifier, "runtime", None) or {}),
        ),
        limits=limits,
        verifier_independent=verifier_independent,
    )


def verify_report(
    db: Session,
    storage: ImageStorage,
    verifier: ClaimVerificationService,
    report_id: uuid.UUID,
    quality_limits: QualityLimits | None = None,
) -> VerificationRun:
    """Verify the stored claims of one report; returns the VerificationRun with its results."""
    report = db.scalars(
        select(GeneratedReport)
        .where(GeneratedReport.id == report_id)
        .options(selectinload(GeneratedReport.claims))
    ).one_or_none()
    if report is None:
        raise AppError(404, "not_found", "Report not found.")
    refuse_stub_in_production(verifier, "vlm_unavailable")
    claims = claims_from_rows(report.id, report.claims)
    claim_ids = {row.claim_id: row.id for row in report.claims}
    inspection_id = report.inspection_id
    independent = verifier_is_independent(
        report.model_name, report.model_version, report.generation_config, verifier.model
    )
    config = run_config(verifier, independent)
    priors = load_prior_verdicts(db, report.id, claim_ids)

    run_id = uuid.uuid4()
    db.add(
        VerificationRun(
            id=run_id,
            generated_report_id=report_id,
            verifier_model=verifier.model.name,
            verifier_version=verifier.model.version,
            config=config.model_dump(mode="json"),
            status=RunStatus.RUNNING,
        )
    )
    db.commit()

    try:
        return _execute(
            db, storage, verifier, run_id, inspection_id, list(claims.claims), claim_ids, independent,
            priors, quality_limits,
        )
    except BaseException:
        _fail_if_running(db, run_id, INTERRUPTED_MESSAGE)
        raise


__all__ = ["Verdict", "verify_report"]
