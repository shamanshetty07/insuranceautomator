"""Verification orchestration with the development stubs (no models, no network)."""

import importlib.util
import typing
import uuid
from pathlib import Path

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
from app.services import verification_service as vs
from app.services.claim_policy import (
    POLICY_VERSION,
    DetectionMatch,
    EvidenceAssociation,
    EvidenceIntegrity,
    ReasonCode,
)
from app.services.evidence_association import ClaimEvidence
from app.services.dev_stubs import StubClaimVerifier, StubStructuredReportGenerator
from app.services.types import ClaimVerdict, Detection, EvidenceBundle, ImageRef, ModelInfo
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


# ---- omitted_damage regression and label handling (no database) ---------------------------

_IMG = uuid.UUID("00000000-0000-0000-0000-00000000000b")
_CK_OMITTED = "verification_label IS DISTINCT FROM 'omitted_damage' OR verdict IS NULL"


def _db_check_violations(row: VerificationResult) -> list[str]:
    """The verification_results CHECK constraints, evaluated in Python on a built row."""
    label = row.verification_label.value if row.verification_label is not None else None
    verdict = row.verdict.value if row.verdict is not None else None
    found = []
    if label == "omitted_damage" and verdict is not None:
        found.append("ck_verification_results_omitted_has_no_verdict")
    if verdict is not None and verdict not in {v.value for v in Verdict}:
        found.append("verdict")
    return found


def test_python_check_mirror_matches_the_model_constraint():
    ck = next(c for c in VerificationResult.__table__.constraints
              if c.name == "ck_verification_results_omitted_has_no_verdict")
    assert str(ck.sqltext) == _CK_OMITTED


class _OneLabel:
    model = ModelInfo(name="stub-one-label", version="0")

    def __init__(self, labels):
        self.labels = labels  # claim text -> label

    def verify(self, claims, evidence):
        return [ClaimVerdict(c.text, self.labels[c.text], "said so") for c in claims]


def _claim(cid, text):
    return ExtractedClaim(claim_id=cid, text=text, claim_type=ClaimType.VISIBLE_DAMAGE_PRESENCE,
                          image_ids=[_IMG])


def _evidence(cid, det=DetectionMatch.MATCHES_CLAIM):
    return ClaimEvidence(cid, EvidenceAssociation.ASSOCIATED, EvidenceIntegrity.OK, det,
                         ((_IMG, Path("unused.jpg")),), (), (), ())


def _row_for(label, det=DetectionMatch.MATCHES_CLAIM):
    claim = _claim("c1", "dent")
    ev = _evidence("c1", det)
    raw = vs._ask_verifier(_OneLabel({"dent": label}), claim, ev)
    decision = claim_policy.evaluate_claim(vs._policy_input(claim, ev, raw.outcome))
    return vs._build_row(uuid.uuid4(), claim, ev, raw, decision)


def test_regression_omitted_damage_label_builds_a_row_the_database_accepts():
    # Before the fix: verification_label=omitted_damage with verdict=insufficient_evidence,
    # which ck_verification_results_omitted_has_no_verdict rejects at commit.
    row = _row_for("omitted_damage")
    assert _db_check_violations(row) == []
    assert row.verification_label is None  # an omission finding is not this claim's raw label
    assert row.verdict is Verdict.INSUFFICIENT_EVIDENCE and row.needs_human_review
    assert row.policy_reason_codes == [ReasonCode.VERIFIER_INVALID.value]
    assert "omitted_damage" in row.explanation  # what the verifier said is still recorded


@pytest.mark.parametrize("det", list(DetectionMatch))
def test_regression_unverifiable_label_is_stored_for_review(det):
    # Before claim-policy-3 the stored row had needs_human_review=False, so a verifier's
    # "can't tell" left the review queue, even when detection conflicted with the claim.
    row = _row_for("unverifiable", det)
    assert _db_check_violations(row) == []
    assert row.verification_label.value == "unverifiable"
    assert row.verdict is Verdict.INSUFFICIENT_EVIDENCE and row.needs_human_review
    assert row.policy_reason_codes == [ReasonCode.VERIFIER_INSUFFICIENT.value]


# label -> (stored raw label, verdict, review, reason) with a corroborating detection.
LABEL_ROWS = {
    "supported": ("supported", Verdict.SUPPORTED, False, ReasonCode.VERIFIER_SUPPORTED),
    "contradicted": ("contradicted", Verdict.INSUFFICIENT_EVIDENCE, True,
                     ReasonCode.DETECTION_MATCHES_CONTRADICTED),
    "wrong_location": ("wrong_location", Verdict.INSUFFICIENT_EVIDENCE, True,
                       ReasonCode.DETECTION_MATCHES_CONTRADICTED),
    "wrong_damage_type": ("wrong_damage_type", Verdict.INSUFFICIENT_EVIDENCE, True,
                          ReasonCode.DETECTION_MATCHES_CONTRADICTED),
    "unverifiable": ("unverifiable", Verdict.INSUFFICIENT_EVIDENCE, True, ReasonCode.VERIFIER_INSUFFICIENT),
    "unsupported_severity": ("unsupported_severity", Verdict.INSUFFICIENT_EVIDENCE, True,
                             ReasonCode.VERIFIER_INVALID),
    "omitted_damage": (None, Verdict.INSUFFICIENT_EVIDENCE, True, ReasonCode.VERIFIER_INVALID),
    "made-up-label": (None, Verdict.INSUFFICIENT_EVIDENCE, True, ReasonCode.VERIFIER_INVALID),
}


def test_label_table_covers_every_verification_label():
    assert {m.value for m in VerificationLabel} <= set(LABEL_ROWS)


@pytest.mark.parametrize("label", list(LABEL_ROWS))
def test_every_verifier_label_is_stored_consistently(label):
    stored, verdict, review, reason = LABEL_ROWS[label]
    row = _row_for(label)
    assert _db_check_violations(row) == []
    assert (row.verification_label.value if row.verification_label else None) == stored
    assert (row.verdict, row.needs_human_review, row.policy_reason_codes) == (verdict, review, [reason.value])


class _CheckingSession:
    """Fake session whose commit enforces the verification_results CHECK constraints."""

    def __init__(self):
        self.pending: list[VerificationResult] = []
        self.saved: list[VerificationResult] = []
        self.run_updates: list[object] = []

    def add_all(self, rows):
        self.pending.extend(rows)

    def execute(self, stmt):
        self.run_updates.append(stmt)

    def commit(self):
        bad = [v for r in self.pending for v in _db_check_violations(r)]
        if bad:
            self.pending.clear()
            raise RuntimeError(f"IntegrityError: {bad}")  # the whole transaction is lost
        self.saved.extend(self.pending)
        self.pending.clear()

    def rollback(self):
        self.pending.clear()


def test_regression_one_omitted_label_no_longer_loses_the_whole_run(monkeypatch):
    claims = [_claim("ok", "fine"), _claim("odd", "omitted"), _claim("ok2", "also fine")]
    evidence = [_evidence(c.claim_id) for c in claims]
    monkeypatch.setattr(vs, "associate_claims", lambda *a, **k: evidence)
    monkeypatch.setattr(vs, "_load_run", lambda db, run_id: run_id)
    failed = []
    monkeypatch.setattr(vs, "_fail_if_running", lambda db, run_id, msg: failed.append(msg))
    db = _CheckingSession()
    verifier = _OneLabel({"fine": "supported", "omitted": "omitted_damage", "also fine": "supported"})

    vs._execute(db, None, verifier, uuid.uuid4(), uuid.uuid4(), claims)

    assert failed == []  # the run was not failed with SAVE_FAILED_MESSAGE
    assert [r.claim_id for r in db.saved] == ["ok", "odd", "ok2"]  # nothing dropped
    assert [r.verdict for r in db.saved] == [Verdict.SUPPORTED, Verdict.INSUFFICIENT_EVIDENCE, Verdict.SUPPORTED]
    run_update = str(db.run_updates[0].compile()).lower()
    assert "update verification_runs" in run_update and "verification_runs.status" in run_update


def test_a_save_failure_still_fails_the_run_accurately_and_keeps_no_partial_rows(monkeypatch):
    claims = [_claim("ok", "fine")]
    monkeypatch.setattr(vs, "associate_claims", lambda *a, **k: [_evidence("ok")])
    monkeypatch.setattr(vs, "_load_run", lambda db, run_id: run_id)
    failed = []
    monkeypatch.setattr(vs, "_fail_if_running", lambda db, run_id, msg: failed.append(msg))

    class Broken(_CheckingSession):
        def commit(self):
            self.pending.clear()
            raise RuntimeError("connection lost")

    db = Broken()
    vs._execute(db, None, _OneLabel({"fine": "supported"}), uuid.uuid4(), uuid.uuid4(), claims)
    assert failed == [vs.SAVE_FAILED_MESSAGE] and db.saved == []


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
        self.prediction.bounding_box = BOX  # a "dent" at BOX
        self.report = make_report(self.inspection)
        self.report.content = "original report text"
        db.add(self.report)
        db.commit()

    def claim(self, cid, ctype, images=(), preds=(), text=None, damage_type=None):
        return ExtractedClaim(
            claim_id=cid,
            text=text or f"claim {cid}",
            claim_type=ctype,
            damage_type=damage_type,
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
    # the presence claim cites the image but not its detection: not comparable -> review
    assert rows[presence.claim_id].verdict is Verdict.SUPPORTED
    assert rows[presence.claim_id].needs_human_review is True
    assert rows[presence.claim_id].policy_reason_codes == [ReasonCode.DETECTION_INCONCLUSIVE.value]
    assert rows[typed.claim_id].needs_human_review is False  # same class, same place
    assert rows[typed.claim_id].verdict is Verdict.SUPPORTED
    assert rows[typed.claim_id].verification_label is VerificationLabel.SUPPORTED
    record = EvidenceRecord.model_validate(rows[typed.claim_id].evidence_reference)
    assert record.prediction_ids == (world.prediction.id,) and record.detection_match == "matches_claim"

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
    # verifier says contradicted, but the cited detection is a dent: disagreement -> review
    claim = world.claim("c1", ClaimType.DAMAGE_TYPE, [world.image.id], [world.prediction.id], "dent on door",
                        damage_type="dent")
    verifier = StubClaimVerifier(labels={"dent on door": "contradicted"})
    row = world.results(world.run([claim], verifier))["c1"]
    assert row.verification_label is VerificationLabel.CONTRADICTED  # raw
    assert row.verdict is Verdict.INSUFFICIENT_EVIDENCE and row.needs_human_review  # final
    assert row.policy_reason_codes == [ReasonCode.DETECTION_MATCHES_CONTRADICTED.value]


@requires_db
def test_visual_contradiction_without_corroborating_detection_needs_review(world):
    claim = world.claim("c1", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="dent")
    row = world.results(world.run([claim], StubClaimVerifier(labels={"dent": "wrong_location"})))["c1"]
    assert row.verification_label is VerificationLabel.WRONG_LOCATION
    assert row.verdict is Verdict.CONTRADICTED and row.needs_human_review is True
    assert row.policy_reason_codes == [ReasonCode.DETECTION_INCONCLUSIVE.value]


@requires_db
def test_contradiction_with_a_conflicting_detection_still_needs_review(world):
    # The claim says scratch and cites a detection classed as dent at the same place.
    claim = world.claim("c1", ClaimType.DAMAGE_TYPE, [world.image.id], [world.prediction.id], "scratch",
                        damage_type="scratch")
    row = world.results(world.run([claim], StubClaimVerifier(labels={"scratch": "wrong_damage_type"})))["c1"]
    # claim-policy-4: a conflict is built from generator-written fields, so not final alone
    assert row.verdict is Verdict.CONTRADICTED and row.needs_human_review is True
    assert row.policy_reason_codes == [ReasonCode.DETECTION_CONFLICT_UNCONFIRMED.value]
    record = EvidenceRecord.model_validate(row.evidence_reference)
    assert record.detection_match == "conflicts_with_claim"
    assert f"damage_type_mismatch:{world.prediction.id}" in record.detection_details


@requires_db
def test_regression_omitted_damage_label_keeps_every_result_in_the_database(world):
    a = world.claim("c1", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="fine")
    b = world.claim("c2", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="odd")
    run = world.run([a, b], StubClaimVerifier(labels={"odd": "omitted_damage"}))
    rows = world.results(run)
    assert run.status is RunStatus.COMPLETED and run.error_message is None
    assert set(rows) == {"c1", "c2"}
    assert rows["c2"].verification_label is None and "omitted_damage" in rows["c2"].explanation
    assert rows["c2"].verdict is Verdict.INSUFFICIENT_EVIDENCE and rows["c2"].needs_human_review
    assert rows["c1"].verdict is Verdict.SUPPORTED


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
    assert EvidenceRecord.model_validate(rows["foreign"].evidence_reference).image_ids == ()
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

