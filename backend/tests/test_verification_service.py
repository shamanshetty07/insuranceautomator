"""Verification orchestration with the development stubs (no models, no network)."""

import importlib.util
import typing
import uuid

import pytest
from sqlalchemy import select

from app.core.errors import AppError
from app.models import GeneratedReport, Inspection, VerificationLabel, VerificationResult
from app.models.run_status import RunStatus
from app.schemas.claims import (
    ClaimExtractionResult,
    ClaimType,
    EvidenceRecord,
    ExtractedClaim,
    Verdict,
)
from app.schemas.runs import SamplingParams, VerificationConfig
from app.services import claim_policy
from app.services.claim_policy import POLICY_VERSION, ReasonCode
from app.services.dev_stubs import StubClaimVerifier, StubStructuredReportGenerator
from app.services.types import Detection, EvidenceBundle, ImageRef, ModelInfo
from app.services.verification_service import (
    ALL_FAILED_MESSAGE,
    INTERRUPTED_MESSAGE,
    verify_report,
)
from tests.conftest import BACKEND_DIR, requires_db
from tests.factories import make_report
from tests.test_evidence_association import add_image, add_run

CONFIG = VerificationConfig(
    prompt_version="stub-1", sampling=SamplingParams(temperature=0, top_p=1, max_new_tokens=1)
)
BOX = {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}


# ---- pure tests (no database) ------------------------------------------------------------


def test_evidence_record_literals_match_policy_enums():
    hints = typing.get_type_hints(EvidenceRecord)
    pairs = {
        "association": claim_policy.EvidenceAssociation,
        "integrity": claim_policy.EvidenceIntegrity,
        "detection_match": claim_policy.DetectionMatch,
    }
    for field, enum_cls in pairs.items():
        assert set(typing.get_args(hints[field])) == {m.value for m in enum_cls}


def test_frozen_migration_check_lists_match_current_enums():
    path = BACKEND_DIR / "alembic" / "versions" / "0004_claim_level_verification.py"
    spec = importlib.util.spec_from_file_location("mig0004", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CLAIM_TYPES == tuple(m.value for m in ClaimType)
    assert module.VERDICTS == tuple(m.value for m in Verdict)


# ---- database-backed orchestration -------------------------------------------------------


class World:
    def __init__(self, db, storage):
        self.db, self.storage = db, storage
        self.inspection = Inspection(title="a")
        self.other = Inspection(title="b")
        db.add_all([self.inspection, self.other])
        db.flush()
        self.image = add_image(db, storage, self.inspection)
        _, (self.prediction,) = add_run(db, self.image, preds=1)
        self.report = make_report(self.inspection)
        self.report.content = "original report text"
        db.add(self.report)
        db.commit()

    def claim(self, cid, ctype, images=(), preds=(), text=None):
        return ExtractedClaim(
            claim_id=cid,
            text=text or f"claim {cid}",
            claim_type=ctype,
            image_ids=list(images),
            prediction_ids=list(preds),
        )

    def run(self, claims, verifier, report_id=None, result_report_id=None):
        return verify_report(
            self.db,
            self.storage,
            verifier,
            report_id or self.report.id,
            ClaimExtractionResult(report_id=result_report_id, claims=claims),
            CONFIG,
        )

    def results(self, run):
        rows = self.db.scalars(
            select(VerificationResult).where(VerificationResult.verification_run_id == run.id)
        ).all()
        return {r.claim_id: r for r in rows}


@pytest.fixture
def world(db_session, storage):
    return World(db_session, storage)


@requires_db
def test_generator_to_verified_results_end_to_end(world):
    generator = StubStructuredReportGenerator(
        prediction_ids_by_image={world.image.id: [world.prediction.id]}
    )
    evidence = EvidenceBundle(
        images=[ImageRef(image_id=world.image.id, path=world.storage.resolve(world.image.storage_path))],
        detections=[Detection(image_id=world.image.id, damage_type="dent", confidence=0.9, bounding_box=BOX)],
    )
    structured = generator.generate_structured(evidence)
    verifier = StubClaimVerifier()  # every checkable claim: supported

    run = world.run(structured.claims.claims, verifier)

    rows = world.results(run)
    assert run.status is RunStatus.COMPLETED and run.completed_at is not None
    assert run.error_message is None
    assert run.verifier_model == "stub-verifier" and run.config["policy_version"] == POLICY_VERSION
    # one result per claim, none dropped
    assert set(rows) == {c.claim_id for c in structured.claims.claims}

    presence = next(c for c in structured.claims.claims if c.claim_type is ClaimType.VISIBLE_DAMAGE_PRESENCE)
    typed = next(c for c in structured.claims.claims if c.claim_type is ClaimType.DAMAGE_TYPE)
    assert rows[presence.claim_id].verdict is Verdict.SUPPORTED
    assert rows[presence.claim_id].needs_human_review is False
    assert rows[typed.claim_id].verdict is Verdict.SUPPORTED
    assert rows[typed.claim_id].verification_label is VerificationLabel.SUPPORTED
    record = EvidenceRecord.model_validate(rows[typed.claim_id].evidence_reference)
    assert record.prediction_ids == [world.prediction.id] and record.detection_match == "matches_claim"

    # claims the pixels cannot establish are stored, not dropped, and never reach the verifier
    for ctype in (ClaimType.DAMAGE_SEVERITY, ClaimType.HIDDEN_STRUCTURAL_DAMAGE, ClaimType.REPAIR_COST):
        row = next(r for r in rows.values() if r.claim_type is ctype)
        assert row.verdict is Verdict.INSUFFICIENT_EVIDENCE and row.needs_human_review is True
        assert row.verification_label is None and row.explanation.startswith("Verifier not called")
    called = {text for texts, _ in verifier.calls for text in texts}
    assert called == {presence.text, typed.text}
    # the verifier saw only the cited original image, never the report
    assert all(ids == (str(world.image.id),) for _, ids in verifier.calls)
    assert world.db.get(GeneratedReport, world.report.id).content == "original report text"


@requires_db
def test_raw_verifier_label_is_kept_apart_from_policy_verdict(world):
    # verifier says contradicted, but the claim cites a real detection: disagreement -> review
    claim = world.claim("c1", ClaimType.DAMAGE_TYPE, [world.image.id], [world.prediction.id], "dent on door")
    verifier = StubClaimVerifier(labels={"dent on door": "contradicted"})
    row = world.results(world.run([claim], verifier))["c1"]
    assert row.verification_label is VerificationLabel.CONTRADICTED  # raw
    assert row.verdict is Verdict.INSUFFICIENT_EVIDENCE and row.needs_human_review  # final
    assert row.policy_reason_codes == [ReasonCode.DETECTION_MATCHES_CONTRADICTED.value]


@requires_db
def test_visual_contradiction_without_conflicting_detection_passes_through(world):
    claim = world.claim("c1", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="dent")
    row = world.results(world.run([claim], StubClaimVerifier(labels={"dent": "wrong_location"})))["c1"]
    assert row.verification_label is VerificationLabel.WRONG_LOCATION
    assert row.verdict is Verdict.CONTRADICTED and row.needs_human_review is False


@requires_db
def test_label_without_a_factual_verdict_fails_closed(world):
    claim = world.claim("c1", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="dent")
    row = world.results(world.run([claim], StubClaimVerifier(labels={"dent": "unsupported_severity"})))["c1"]
    assert row.verification_label is VerificationLabel.UNSUPPORTED_SEVERITY  # raw kept
    assert row.verdict is Verdict.INSUFFICIENT_EVIDENCE and row.needs_human_review
    assert row.policy_reason_codes == [ReasonCode.VERIFIER_INVALID.value]


@requires_db
def test_unrecognised_label_is_not_stored(world):
    claim = world.claim("c1", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="dent")
    row = world.results(world.run([claim], StubClaimVerifier(labels={"dent": "totally-made-up"})))["c1"]
    assert row.verification_label is None and "totally-made-up" not in row.explanation
    assert row.verdict is Verdict.INSUFFICIENT_EVIDENCE and row.needs_human_review


@requires_db
def test_one_verifier_exception_fails_that_claim_only(world):
    ok = world.claim("c1", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="fine")
    bad = world.claim("c2", ClaimType.DAMAGE_LOCATION, [world.image.id], text="boom")
    run = world.run([ok, bad], StubClaimVerifier(raise_on={"boom"}))
    rows = world.results(run)
    assert run.status is RunStatus.COMPLETED
    assert rows["c1"].verdict is Verdict.SUPPORTED
    assert rows["c2"].verdict is Verdict.INSUFFICIENT_EVIDENCE and rows["c2"].needs_human_review
    assert rows["c2"].policy_reason_codes == [ReasonCode.VERIFIER_ERROR.value]
    assert "boom" not in rows["c2"].explanation  # no verifier internals leak into the record


@requires_db
def test_every_verifier_call_failing_marks_the_run_failed_but_keeps_all_claims(world):
    a = world.claim("c1", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="a")
    b = world.claim("c2", ClaimType.DAMAGE_TYPE, [world.image.id], text="b")
    run = world.run([a, b], StubClaimVerifier(raise_on={"a", "b"}))
    assert run.status is RunStatus.FAILED and run.error_message == ALL_FAILED_MESSAGE
    assert run.completed_at is not None
    assert set(world.results(run)) == {"c1", "c2"}


@pytest.mark.parametrize("mode", ["empty_for", "extra_for"])
@requires_db
def test_wrong_result_count_is_treated_as_a_verifier_error(world, mode):
    claim = world.claim("c1", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="dent")
    row = world.results(world.run([claim], StubClaimVerifier(**{mode: {"dent"}})))["c1"]
    assert row.verdict is Verdict.INSUFFICIENT_EVIDENCE and row.needs_human_review
    assert row.policy_reason_codes == [ReasonCode.VERIFIER_ERROR.value]


@requires_db
def test_missing_mismatched_and_corrupt_evidence_never_reach_the_verifier(world):
    foreign = add_image(world.db, world.storage, world.other)
    corrupt = add_image(world.db, world.storage, world.inspection, order=1)
    world.storage.resolve(corrupt.storage_path).write_bytes(b"not an image")  # now differs from its checksum
    world.db.commit()
    claims = [
        world.claim("none", ClaimType.VISIBLE_DAMAGE_PRESENCE, text="no evidence"),
        world.claim("unknown", ClaimType.VISIBLE_DAMAGE_PRESENCE, [uuid.uuid4()], text="unknown image"),
        world.claim("foreign", ClaimType.VISIBLE_DAMAGE_PRESENCE, [foreign.id], text="other inspection"),
        world.claim("corrupt", ClaimType.VISIBLE_DAMAGE_PRESENCE, [corrupt.id], text="bad file"),
    ]
    verifier = StubClaimVerifier()
    rows = world.results(world.run(claims, verifier))
    assert verifier.calls == []
    for cid in ("none", "unknown", "foreign", "corrupt"):
        assert rows[cid].verdict is Verdict.INSUFFICIENT_EVIDENCE and rows[cid].needs_human_review
        assert rows[cid].verification_label is None
    assert rows["none"].policy_reason_codes == [ReasonCode.NO_EVIDENCE_CITED.value]
    assert rows["unknown"].policy_reason_codes == [ReasonCode.EVIDENCE_UNRESOLVED.value]
    assert rows["foreign"].policy_reason_codes == [ReasonCode.EVIDENCE_UNRESOLVED.value]
    assert rows["corrupt"].policy_reason_codes[0].startswith("integrity_")
    assert EvidenceRecord.model_validate(rows["foreign"].evidence_reference).image_ids == []
    assert rows["unknown"].evidence_reference["problems"][0].startswith("image_not_in_inspection:")


@requires_db
def test_no_claims_is_a_completed_run_with_no_results(world):
    run = world.run([], StubClaimVerifier())
    assert run.status is RunStatus.COMPLETED and world.results(run) == {}


@requires_db
def test_unknown_report_and_mismatched_claim_report_are_rejected(world):
    with pytest.raises(AppError) as unknown:
        world.run([], StubClaimVerifier(), report_id=uuid.uuid4())
    assert unknown.value.status_code == 404
    with pytest.raises(AppError) as mismatch:
        world.run([], StubClaimVerifier(), result_report_id=uuid.uuid4())
    assert mismatch.value.status_code == 422


@requires_db
def test_reverification_creates_a_separate_run(world):
    claim = world.claim("c1", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="dent")
    first = world.run([claim], StubClaimVerifier())
    second = world.run([claim], StubClaimVerifier(default_label="contradicted"))
    assert first.id != second.id
    assert world.results(first)["c1"].verdict is Verdict.SUPPORTED
    assert world.results(second)["c1"].verdict is Verdict.CONTRADICTED


@requires_db
def test_no_transaction_is_open_while_the_verifier_runs(world):
    seen = []

    class Probe(StubClaimVerifier):
        def verify(self, claims, evidence):
            seen.append(world.db.in_transaction())
            return super().verify(claims, evidence)

    claim = world.claim("c1", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="dent")
    world.run([claim], Probe())
    assert seen == [False]


@requires_db
def test_interruption_marks_the_run_failed_and_propagates(world):
    class Interrupt:
        model = ModelInfo(name="stub-interrupt", version="0")

        def verify(self, claims, evidence):
            raise KeyboardInterrupt

    claim = world.claim("c1", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="dent")
    with pytest.raises(KeyboardInterrupt):
        world.run([claim], Interrupt())
    from app.models import VerificationRun

    run = world.db.scalars(select(VerificationRun).order_by(VerificationRun.created_at.desc())).first()
    assert run.status is RunStatus.FAILED and run.error_message == INTERRUPTED_MESSAGE
    assert run.completed_at is not None
    assert world.results(run) == {}

