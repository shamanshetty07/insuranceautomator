"""Persist claim-level verification of one report (model-independent orchestration).

For each structured claim: associate it with inspection evidence (evidence_association),
ask the verifier about the ORIGINAL stored images it cites, run the deterministic claim
policy, and store one VerificationResult. Nothing here knows about Qwen or YOLO; the
verifier is any ClaimVerificationService.

- Every input claim yields exactly one VerificationResult. A claim that cannot be checked
  is stored as insufficient_evidence with a reason code, never dropped.
- The raw verifier label/explanation (`verification_label`, `explanation`) are stored
  apart from the final policy outcome (`verdict`, `needs_human_review`,
  `policy_reason_codes`). The verifier never sees the report text or detection boxes,
  only the claim text and the cited original images, so it cannot grade its own report.
  Only claim labels (verdict_mapping.CLAIM_LABELS) are stored as a claim's raw label;
  an unrecognised label or omitted_damage is stored as NULL with the reason in the
  explanation, and the policy fails that claim closed (insufficient_evidence, review).
  Every row built here therefore satisfies the table's CHECK constraints.
- Results are written in ONE transaction: if that write still fails, no result is kept
  and the run is marked `failed` with SAVE_FAILED_MESSAGE (never a partial run).
- The verifier is not called when the policy already decides the claim without it
  (unusable evidence, hidden damage, cost, ...), so no inference is wasted.
- No database transaction is open while the verifier runs: the run is committed as
  `running`, evidence is read and the read transaction ended, then results are written
  in one short transaction. A verifier exception fails that claim only; the run is
  `failed` only if every verifier call that was made failed.
- The original GeneratedReport is never modified.
"""

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session, selectinload

from app.core.errors import AppError
from app.models.generated_report import GeneratedReport
from app.models.run_status import RunStatus
from app.models.verification_result import VerificationLabel, VerificationResult
from app.models.verification_run import VerificationRun
from app.schemas.claims import ClaimExtractionResult, ExtractedClaim, Verdict
from app.schemas.runs import VerificationConfig
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
from app.services.types import EvidenceBundle, ImageRef, ReportClaim
from app.services.verdict_mapping import is_claim_label, legacy_verdict

logger = logging.getLogger(__name__)

INTERRUPTED_MESSAGE = "Verification was interrupted before it finished."
ALL_FAILED_MESSAGE = "The verifier failed for every claim it was asked to check."
SAVE_FAILED_MESSAGE = "The verification results could not be saved."
_MAX_EXPLANATION_CHARS = 2000
_UNRECOGNISED = "unrecognised"  # stands in for an untrusted verifier label we will not store


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


def _policy_input(claim: ExtractedClaim, ev: ClaimEvidence, outcome: VerifierOutcome) -> ClaimPolicyInput:
    return ClaimPolicyInput(
        claim_type=claim.claim_type,
        verifier=outcome,
        integrity=ev.integrity,
        association=ev.association,
        detection_match=ev.detection_match,
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
    explanation = str(getattr(result, "explanation", "") or "")[:_MAX_EXPLANATION_CHARS]
    try:
        label = VerificationLabel(getattr(result, "label", None))
    except ValueError:
        return _Raw(
            None,
            VerifierOutcome(verdict=_UNRECOGNISED),
            "The verifier returned an unrecognised label.",
            True,
            False,
        )
    if not is_claim_label(label):
        # omitted_damage describes damage missing from the report, not this claim. It is
        # not stored as the claim's raw label (a claim row always has a verdict, and an
        # omitted_damage row must not); the explanation records what the verifier said.
        return _Raw(
            None,
            VerifierOutcome(verdict=_UNRECOGNISED),
            f"The verifier answered '{label.value}', which is not a verdict on a report claim.",
            True,
            False,
        )
    mapped = legacy_verdict(label.value)
    # A label with no factual verdict (e.g. unsupported_severity) goes to the policy as an
    # unrecognised value, so it fails closed instead of being guessed.
    outcome = VerifierOutcome(verdict=mapped if mapped is not None else label.value)
    return _Raw(label, outcome, explanation or "No explanation given.", True, False)


def _build_row(
    run_id: uuid.UUID,
    claim: ExtractedClaim,
    ev: ClaimEvidence,
    raw: _Raw,
    decision: ClaimPolicyDecision,
) -> VerificationResult:
    return VerificationResult(
        verification_run_id=run_id,
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
) -> VerificationRun:
    evidence = associate_claims(db, storage, inspection_id, claims)
    if len(evidence) != len(claims):  # association must never drop a claim
        raise RuntimeError("evidence association returned a different number of claims")
    db.rollback()  # end the read transaction: none is open while the verifier runs

    rows: list[VerificationResult] = []
    attempted = errored = 0
    for claim, ev in zip(claims, evidence, strict=True):
        if _decided_without_verifier(claim, ev):
            raw = _Raw(None, VerifierOutcome(verdict=None), "", False, False)
        else:
            raw = _ask_verifier(verifier, claim, ev)
        attempted += raw.attempted
        errored += raw.errored
        decision = evaluate_claim(_policy_input(claim, ev, raw.outcome))
        if not raw.attempted:
            raw.explanation = f"Verifier not called: {decision.explanation}"
        rows.append(_build_row(run_id, claim, ev, raw, decision))

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


def verify_report(
    db: Session,
    storage: ImageStorage,
    verifier: ClaimVerificationService,
    report_id: uuid.UUID,
    claims: ClaimExtractionResult,
    config: VerificationConfig,
) -> VerificationRun:
    """Verify `claims` of one stored report; returns the VerificationRun with its results."""
    report = db.get(GeneratedReport, report_id)
    if report is None:
        raise AppError(404, "not_found", "Report not found.")
    if claims.report_id is not None and claims.report_id != report_id:
        raise AppError(422, "report_mismatch", "The claims belong to a different report.")
    inspection_id = report.inspection_id

    run_id = uuid.uuid4()
    run_config = config.model_copy(update={"policy_version": POLICY_VERSION})
    db.add(
        VerificationRun(
            id=run_id,
            generated_report_id=report_id,
            verifier_model=verifier.model.name,
            verifier_version=verifier.model.version,
            config=run_config.model_dump(mode="json"),
            status=RunStatus.RUNNING,
        )
    )
    db.commit()  # visible as `running` before any verifier work, like detection runs

    try:
        return _execute(db, storage, verifier, run_id, inspection_id, list(claims.claims))
    except BaseException:
        # Includes cancellation and KeyboardInterrupt: record the failure, never swallow it.
        _fail_if_running(db, run_id, INTERRUPTED_MESSAGE)
        raise


__all__ = ["Verdict", "verify_report"]
