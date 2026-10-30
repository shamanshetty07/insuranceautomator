"""Persist claim-level verification of one report (model-independent orchestration)."""

import logging
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session, selectinload

from app.core.errors import AppError
from app.models.generated_report import GeneratedReport
from app.models.run_status import RunStatus
from app.models.verification_result import VerificationLabel, VerificationResult
from app.models.verification_run import VerificationRun
from app.schemas.claims import ExtractedClaim, Verdict
from app.schemas.runs import ModelRecord, StoredVerificationConfig, VerificationConfig
from app.services.claim_policy import (
    POLICY_VERSION,
    ClaimPolicyDecision,
    ClaimPolicyInput,
    ReasonCode,
    VerifierOutcome,
    evaluate_claim,
)
from app.services.evidence_association import ClaimEvidence, associate_claims
from app.services.image_storage import ImageStorage
from app.services.interfaces import ClaimVerificationService
from app.services.model_guard import refuse_stub_in_production, verifier_is_independent
from app.services.report_claims import claims_from_rows
from app.services.types import EvidenceBundle, ImageRef, ReportClaim
from app.services.verdict_mapping import is_claim_label, legacy_verdict

logger = logging.getLogger(__name__)

INTERRUPTED_MESSAGE = "Verification was interrupted before it finished."
ALL_FAILED_MESSAGE = "The verifier failed for every claim it was asked to check."
SAVE_FAILED_MESSAGE = "The verification results could not be saved."
_MAX_EXPLANATION_CHARS = 2000
_UNRECOGNISED = "unrecognised"


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


def _policy_input(
    claim: ExtractedClaim, ev: ClaimEvidence, outcome: VerifierOutcome, verifier_independent: bool = True
) -> ClaimPolicyInput:
    return ClaimPolicyInput(
        claim_type=claim.claim_type,
        verifier=outcome,
        integrity=ev.integrity,
        association=ev.association,
        detection_match=ev.detection_match,
        verifier_independent=verifier_independent,
    )


def _decided_without_verifier(claim: ExtractedClaim, ev: ClaimEvidence) -> bool:
    """True if a mandatory policy rule fires before the verifier's answer could matter."""
    decision = evaluate_claim(_policy_input(claim, ev, VerifierOutcome(verdict=None)))
    return ReasonCode.VERIFIER_MISSING not in decision.reason_codes


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
    try:
        label = VerificationLabel(raw_label)
    except ValueError:
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


def _build_row(
    run_id: uuid.UUID,
    claim: ExtractedClaim,
    ev: ClaimEvidence,
    raw: _Raw,
    decision: ClaimPolicyDecision,
    report_claim_id: uuid.UUID | None = None,
) -> VerificationResult:
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
        evidence_reference=ev.to_record().model_dump(mode="json"),
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
) -> VerificationRun:
    evidence = associate_claims(db, storage, inspection_id, claims)
    if len(evidence) != len(claims):
        raise RuntimeError("evidence association returned a different number of claims")
    db.rollback()

    rows: list[VerificationResult] = []
    attempted = errored = 0
    for claim, ev in zip(claims, evidence, strict=True):
        if _decided_without_verifier(claim, ev):
            raw = _Raw(None, VerifierOutcome(verdict=None), "", False, False)
        else:
            raw = _ask_verifier(verifier, claim, ev)
        attempted += raw.attempted
        errored += raw.errored
        decision = evaluate_claim(_policy_input(claim, ev, raw.outcome, verifier_independent))
        if not raw.attempted:
            raw.explanation = f"Verifier not called: {decision.explanation}"
        claim_row_id = (report_claim_ids or {}).get(claim.claim_id)
        rows.append(_build_row(run_id, claim, ev, raw, decision, claim_row_id))

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
            db, storage, verifier, run_id, inspection_id, list(claims.claims), claim_ids, independent
        )
    except BaseException:
        _fail_if_running(db, run_id, INTERRUPTED_MESSAGE)
        raise


__all__ = ["Verdict", "verify_report"]
