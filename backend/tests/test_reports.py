"""Report generation/verification pipeline: storage mapping, provenance, safety, API."""

import importlib.util
import types
import uuid
from datetime import UTC, datetime

import pytest
from alembic.script import ScriptDirectory
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.api.deps import get_vlm_provider
from app.core.config import Settings
from app.core.errors import AppError
from app.main import app
from app.models import (
    DamagePrediction,
    GeneratedReport,
    ReportClaim,
    ReportClaimImage,
    ReportPrediction,
    VerificationResult,
)
from app.models.run_status import RunStatus
from app.schemas.claims import ClaimExtractionResult, ClaimType, ExtractedClaim, StructuredReport, Verdict
from app.schemas.report import VlmStatusRead, report_detail, report_summary
from app.schemas.runs import (
    GenerationConfig,
    SamplingParams,
    StoredGenerationConfig,
    StoredVerificationConfig,
    VerificationConfig,
)
from app.services import model_guard, report_service
from app.services.dev_stubs import StubClaimVerifier
from app.services.report_claims import claim_from_row, claim_row_values, claims_from_rows
from app.services.types import Detection, EvidenceBundle, ImageRef, ModelInfo
from app.services.verification_service import run_config
from app.services.vlm.adapters import QwenClaimVerifier, QwenStructuredReportGenerator, VlmLimits
from app.services.vlm.errors import (
    VlmBusy,
    VlmError,
    VlmInferenceError,
    VlmInputError,
    VlmOutputError,
    VlmTimeout,
    VlmUnavailable,
)
from app.services.vlm.provider import VlmStatus
from tests.conftest import BACKEND_DIR, requires_db
from tests.test_vlm import FakeRunner, report_json

NO_DB = "postgresql+psycopg://none:none@127.0.0.1:1/none"
IMG1, IMG2, P1, P2 = (uuid.UUID(int=n) for n in (1, 2, 11, 12))
BOX = {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6}
SAMPLING = SamplingParams(temperature=0, top_p=1, max_new_tokens=8)


def claim(cid="c1", **kw):
    base = dict(claim_id=cid, text="A dent on the door.", claim_type="damage_type", damage_type="dent",
                image_ids=[IMG1], prediction_ids=[P1],
                evidence=[{"image_id": str(IMG1), "prediction_id": str(P1), "crop": BOX, "note": "door"}])
    return ExtractedClaim(**{**base, **kw})


def bundle(**kw):
    return EvidenceBundle(
        images=[ImageRef(IMG1, kw.get("path1", "a")), ImageRef(IMG2, "b")],
        detections=[Detection(IMG1, "dent", 0.9, BOX, prediction_id=P1)],
    )


def row_for(c, position=0):
    return types.SimpleNamespace(id=uuid.uuid4(), **claim_row_values(c, position))


def test_claim_round_trip_is_exact_and_ordered():
    claims = [claim("c1"), claim("c2", text="Visible damage.", claim_type="visible_damage_presence",
                                 damage_type=None, image_ids=[IMG2, IMG1], prediction_ids=[], evidence=[])]
    rows = [row_for(c, n) for n, c in enumerate(claims)]
    assert rows[1].image_ids == [str(IMG2), str(IMG1)]
    assert rows[0].evidence[0]["crop"] == BOX and rows[0].evidence[0]["note"] == "door"
    back = claims_from_rows(uuid.UUID(int=99), rows)
    assert back.claims == tuple(claims) and back.report_id == uuid.UUID(int=99)


def test_a_corrupted_stored_claim_is_rejected_not_trusted():
    row = row_for(claim())
    row.image_ids = [str(IMG1), str(IMG1)]
    with pytest.raises(ValidationError):
        claim_from_row(row)


def structured(*claims):
    return StructuredReport(report_text="x", claims=ClaimExtractionResult(claims=list(claims)))


def test_citations_must_be_within_the_bundle():
    report_service.check_citations(structured(claim()), bundle())
    for bad in (
        claim(image_ids=[uuid.UUID(int=7)], prediction_ids=[], evidence=[]),
        claim(prediction_ids=[P2], evidence=[]),
    ):
        with pytest.raises(VlmOutputError) as exc:
            report_service.check_citations(structured(bad), bundle())
        assert exc.value.code == "uncited_evidence"


def test_supplied_predictions_are_linked_whether_cited_or_not():
    b = EvidenceBundle(images=[ImageRef(IMG1, "a")], detections=[
        Detection(IMG1, "dent", 0.9, None, prediction_id=P2),
        Detection(IMG1, "dent", 0.8, None, prediction_id=P1),
        Detection(IMG1, "dent", 0.7, None, prediction_id=P2),
    ])
    assert report_service.supplied_prediction_ids(b) == [P2, P1]


def test_generation_provenance_is_recorded_by_the_server():
    runner = FakeRunner()
    generator = QwenStructuredReportGenerator(runner, VlmLimits(max_claims=5))
    run_ids = (uuid.UUID(int=21),)
    config = report_service.stored_generation_config(generator, report_service.EvidenceSet(bundle(), run_ids))
    dumped = config.model_dump(mode="json")
    assert dumped["prompt_version"] == "qwen-report-1" and dumped["sampling"]["temperature"] == 0
    assert dumped["model"] == {"weights_sha256": "f" * 64, "packages": {"transformers": None, "torch": None}}
    assert dumped["limits"]["max_claims"] == 5
    assert dumped["evidence"] == {"image_ids": [str(IMG1), str(IMG2)], "detection_run_ids": [str(run_ids[0])]}
    assert "path" not in str(dumped)
    with pytest.raises(TypeError):
        report_service.stored_generation_config(types.SimpleNamespace(model=runner.info), report_service.EvidenceSet(bundle(), ()))


def test_stored_configs_accept_rows_written_before_provenance_existed():
    old = {"prompt_version": "p1", "sampling": SAMPLING.model_dump()}
    assert StoredGenerationConfig.model_validate(old).model is None
    assert StoredVerificationConfig.model_validate(old).verifier_independent is None
    with pytest.raises(ValidationError):
        StoredGenerationConfig.model_validate({**old, "model": {"weights_sha256": "not-a-digest"}})
    with pytest.raises(ValidationError):
        GenerationConfig.model_validate({**old, "model": {}})


def test_verification_run_config_comes_from_the_verifier():
    verifier = QwenClaimVerifier(FakeRunner(), VlmLimits())
    config = run_config(verifier, verifier_independent=False).model_dump(mode="json")
    assert config["prompt_version"] == "qwen-verify-1" and config["policy_version"] == "claim-policy-5"
    assert config["verifier_independent"] is False and config["model"]["weights_sha256"] == "f" * 64
    with pytest.raises(TypeError):
        run_config(types.SimpleNamespace(model=ModelInfo("m", "1")), True)


def test_stubs_are_refused_only_in_production(monkeypatch):
    stub = StubClaimVerifier()
    assert model_guard.is_stub(stub) and not model_guard.is_stub(QwenClaimVerifier(FakeRunner(), VlmLimits()))
    model_guard.refuse_stub_in_production(stub, "vlm_unavailable")
    monkeypatch.setattr(model_guard, "get_settings", lambda: Settings(database_url=NO_DB, app_env="production"))
    with pytest.raises(AppError) as exc:
        model_guard.refuse_stub_in_production(stub, "vlm_unavailable")
    assert (exc.value.status_code, exc.value.code) == (503, "vlm_unavailable")
    renamed = types.SimpleNamespace(model=ModelInfo("stub-anything", "0"))
    with pytest.raises(AppError):
        model_guard.refuse_stub_in_production(renamed, "vlm_unavailable")


@pytest.mark.parametrize("gen_sha,ver,expected", [
    ("a" * 64, ModelInfo("qwen", "1", "a" * 64), False),
    ("a" * 64, ModelInfo("other", "9", "a" * 64), False),
    ("a" * 64, ModelInfo("qwen", "1", "b" * 64), True),
    (None, ModelInfo("qwen", "1", "b" * 64), False),
    (None, ModelInfo("other", "1"), True),
])
def test_verifier_independence(gen_sha, ver, expected):
    config = {"model": {"weights_sha256": gen_sha}} if gen_sha else {"prompt_version": "p1"}
    assert model_guard.verifier_is_independent("qwen", "1", config, ver) is expected


def test_fail_fast_after_timeout_skips_remaining_claims():
    calls = []

    class Inner:
        model = ModelInfo("m", "1")
        verification_config = VerificationConfig(prompt_version="v", sampling=SAMPLING)

        def verify(self, claims, evidence):
            calls.append(claims)
            raise VlmTimeout("slow")

    wrapped = report_service.FailFastVerifier(Inner())
    assert wrapped.model.name == "m" and wrapped.verification_config.prompt_version == "v"
    with pytest.raises(VlmTimeout):
        wrapped.verify(["a"], None)
    with pytest.raises(VlmBusy) as exc:
        wrapped.verify(["b"], None)
    assert exc.value.code == "skipped" and calls == [["a"]]


def test_fail_fast_does_not_stop_on_ordinary_errors():
    class Inner:
        model = ModelInfo("m", "1")

        def __init__(self):
            self.n = 0

        def verify(self, claims, evidence):
            self.n += 1
            raise VlmOutputError("bad", code="schema")

    inner = Inner()
    wrapped = report_service.FailFastVerifier(inner)
    for _ in range(2):
        with pytest.raises(VlmOutputError):
            wrapped.verify(["x"], None)
    assert inner.n == 2


@pytest.mark.parametrize("exc,status,code", [
    (VlmUnavailable("x"), 503, "vlm_unavailable"),
    (VlmBusy("x"), 503, "vlm_busy"),
    (VlmTimeout("x"), 504, "vlm_timeout"),
    (VlmInputError("x", code="too_many_images"), 409, "too_many_images"),
    (VlmOutputError("x", code="unknown_alias"), 502, "vlm_output_rejected"),
    (VlmInferenceError("x"), 502, "vlm_inference_error"),
    (VlmError("x"), 502, "vlm_error"),
])
def test_model_errors_map_to_stable_http_errors(exc, status, code):
    err = report_service.vlm_error_to_app_error(exc)
    assert (err.status_code, err.code) == (status, code)
    if isinstance(exc, VlmOutputError):
        assert "unknown_alias" in err.message


def _report(claims):
    return types.SimpleNamespace(
        id=uuid.uuid4(), inspection_id=uuid.uuid4(), content="x", model_name="qwen", model_version="1",
        evidence_strategy="full_image", created_at=datetime.now(UTC), claims=claims,
        generation_config={"prompt_version": "qwen-report-1"},
    )


def _result(row_id, claim_id, verdict, review):
    return types.SimpleNamespace(
        report_claim_id=row_id, claim_id=claim_id, verdict=verdict, needs_human_review=review,
        policy_reason_codes=["verifier_not_independent"] if review else ["verifier_supported"],
        verification_label="supported", explanation="ok",
    )


def test_report_detail_attaches_only_the_latest_runs_results():
    rows = [row_for(claim("c1"), 0), row_for(claim("c2"), 1), row_for(claim("c3"), 2)]
    run = types.SimpleNamespace(
        id=uuid.uuid4(), generated_report_id=uuid.uuid4(), verifier_model="qwen", verifier_version="1",
        status=RunStatus.COMPLETED, error_message=None, created_at=datetime.now(UTC), completed_at=datetime.now(UTC),
        config={"policy_version": "claim-policy-5", "verifier_independent": False},
        results=[_result(rows[0].id, "c1", Verdict.SUPPORTED, True),
                 _result(None, "c2", Verdict.SUPPORTED, False)],
    )
    detail = report_detail(_report(rows), run)
    assert [c.claim_id for c in detail.claims] == ["c1", "c2", "c3"]
    assert detail.claims[0].result.needs_human_review and detail.claims[1].result is not None
    assert detail.claims[2].result is None
    assert detail.verification.review_count == 1 and detail.verification.verifier_independent is False
    assert detail.claim_count == 3 and detail.claims[0].evidence[0].crop.x_max == 0.5
    unverified = report_detail(_report(rows), None)
    assert unverified.verification is None and all(c.result is None for c in unverified.claims)
    assert report_summary(_report([])).claim_count == 0


def test_vlm_status_route_never_loads_a_model(plain_client):
    class Provider:
        def status(self):
            return VlmStatus(available=False, backend="none", reason="disabled")

    app.dependency_overrides[get_vlm_provider] = lambda: Provider()
    try:
        body = plain_client.get("/api/v1/vlm").json()
    finally:
        app.dependency_overrides.pop(get_vlm_provider, None)
    assert body == {"available": False, "backend": "none", "reason": "disabled", "model_name": None,
                    "model_version": None, "weights_sha256": None}
    assert VlmStatusRead.from_status(VlmStatus(True, "transformers", model_name="q")).model_name == "q"


def _migration(name):
    path = BACKEND_DIR / "alembic" / "versions" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_0006_is_the_single_head_and_its_frozen_lists_match():
    from alembic.config import Config

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    assert ScriptDirectory.from_config(cfg).get_heads() == ["0006"]
    module = _migration("0006_report_claims.py")
    assert (module.revision, module.down_revision) == ("0006", "0005")
    assert module.CLAIM_TYPES == tuple(m.value for m in ClaimType)
    from app.models import report_claim

    assert (module.CLAIM_ID_CHECK, module.JSON_ARRAYS_CHECK) == (report_claim.CLAIM_ID_CHECK,
                                                                 report_claim.JSON_ARRAYS_CHECK)


def test_report_claim_tables_metadata():
    claims, links = ReportClaim.__table__, ReportClaimImage.__table__
    names = {c.name for c in claims.constraints if c.name}
    assert {"uq_report_claims_report_claim", "uq_report_claims_report_position", "ck_report_claims_claim_id",
            "ck_report_claims_text", "ck_report_claims_position", "ck_report_claims_json_arrays",
            "report_claim_type"} <= names
    (report_fk,) = claims.c.generated_report_id.foreign_keys
    assert report_fk.ondelete == "CASCADE"
    image_fk = next(iter(links.c.inspection_image_id.foreign_keys))
    assert (image_fk.ondelete, image_fk.deferrable, image_fk.initially) == (None, True, "DEFERRED")
    (result_fk,) = VerificationResult.__table__.c.report_claim_id.foreign_keys
    assert result_fk.ondelete == "CASCADE" and VerificationResult.__table__.c.report_claim_id.nullable
    assert "uq_verification_results_run_report_claim" in {i.name for i in VerificationResult.__table__.indexes}
    assert "created_at" in claims.c and "updated_at" not in claims.c
    CreateTable(claims).compile(dialect=postgresql.dialect())


class FakeVlmProvider:
    """Real Qwen adapters (prompts, parsing, limits) over a fake runner; no model."""

    def __init__(self, generator_responses=(), verifier_responses=(), unavailable=False):
        self.gen_runner = FakeRunner(generator_responses)
        self.ver_runner = FakeRunner(verifier_responses)
        self.unavailable = unavailable

    def status(self):
        return VlmStatus(available=not self.unavailable, backend="fake")

    def get_generator(self):
        if self.unavailable:
            raise VlmUnavailable("The vision-language model is disabled.")
        return QwenStructuredReportGenerator(self.gen_runner, VlmLimits())

    def get_verifier(self):
        if self.unavailable:
            raise VlmUnavailable("The vision-language model is disabled.")
        return QwenClaimVerifier(self.ver_runner, VlmLimits())


@pytest.fixture
def use_vlm():
    def install(provider):
        app.dependency_overrides[get_vlm_provider] = lambda: provider
        return provider

    yield install
    app.dependency_overrides.pop(get_vlm_provider, None)


def _inspection_with_detection(client, db_session):
    from tests.image_utils import make_image_bytes
    from tests.test_evidence_association import add_run

    iid = client.post("/api/v1/inspections", json={"title": "r"}).json()["id"]
    files = {"file": ("a.jpg", make_image_bytes("JPEG", (64, 48)), "image/jpeg")}
    uploaded = client.post(f"/api/v1/inspections/{iid}/images", files=files)
    assert uploaded.status_code == 201, uploaded.text
    image_id = uploaded.json()["id"]
    from app.models import InspectionImage

    image = db_session.get(InspectionImage, uuid.UUID(image_id))
    _, (pred,) = add_run(db_session, image, preds=1)
    pred.bounding_box = BOX
    db_session.commit()
    return iid, image_id, pred


GEN_OK = report_json(
    {"text": "There is a dent on the panel.", "type": "damage_type", "damage_type": "dent",
     "images": ["img1"], "detections": ["det1"],
     "evidence": [{"image": "img1", "detection": "det1", "crop": [0.1, 0.2, 0.5, 0.6]}]},
    {"text": "Repair would cost 900.", "type": "repair_cost", "images": ["img1"]},
)


@requires_db
def test_generate_then_verify_end_to_end(client, db_session, use_vlm):
    iid, image_id, pred = _inspection_with_detection(client, db_session)
    provider = use_vlm(FakeVlmProvider([GEN_OK], ['{"label": "supported", "explanation": "A dent is visible."}']))

    r = client.post(f"/api/v1/inspections/{iid}/reports", json={})
    assert r.status_code == 201, r.text
    report = r.json()
    assert report["claim_count"] == 2 and report["verification"] is None
    assert report["content"] == "There is a dent on the panel.\nRepair would cost 900."
    c1, c2 = report["claims"]
    assert (c1["claim_id"], c1["image_ids"], c1["prediction_ids"]) == ("c1", [image_id], [str(pred.id)])
    assert c1["result"] is None and c2["claim_type"] == "repair_cost"
    cfg = report["generation_config"]
    assert cfg["model"]["weights_sha256"] == "f" * 64 and cfg["evidence"]["image_ids"] == [image_id]
    rid = uuid.UUID(report["id"])
    assert db_session.scalars(select(ReportPrediction.damage_prediction_id)).all() == [pred.id]
    assert len(db_session.scalars(select(ReportClaimImage)).all()) == 2
    assert client.delete(f"/api/v1/inspections/{iid}/images/{image_id}").status_code == 409

    v = client.post(f"/api/v1/reports/{rid}/verifications", json={})
    assert v.status_code == 201, v.text
    detail = v.json()
    assert detail["verification"]["status"] == "completed"
    assert detail["verification"]["verifier_independent"] is False
    r1, r2 = (c["result"] for c in detail["claims"])
    assert (r1["verdict"], r1["needs_human_review"], r1["policy_reason_codes"]) == (
        "supported", True, ["verifier_not_independent"])
    assert (r2["verdict"], r2["needs_human_review"]) == ("insufficient_evidence", True)
    assert len(provider.ver_runner.requests) == 1
    assert [c["text"] for c in detail["claims"]] == [c1["text"], c2["text"]]
    assert client.get(f"/api/v1/reports/{rid}").json() == detail
    listed = client.get(f"/api/v1/inspections/{iid}/reports").json()["items"]
    assert [i["id"] for i in listed] == [report["id"]]


@requires_db
def test_generation_failures_persist_nothing(client, db_session, use_vlm):
    iid, _, _ = _inspection_with_detection(client, db_session)
    use_vlm(FakeVlmProvider(["not json"]))
    r = client.post(f"/api/v1/inspections/{iid}/reports", json={})
    assert (r.status_code, r.json()["error"]["code"]) == (502, "vlm_output_rejected")
    use_vlm(FakeVlmProvider(unavailable=True))
    r = client.post(f"/api/v1/inspections/{iid}/reports")
    assert (r.status_code, r.json()["error"]["code"]) == (503, "vlm_unavailable")
    assert db_session.scalars(select(GeneratedReport)).all() == []
    assert db_session.scalars(select(ReportClaim)).all() == []


@requires_db
def test_report_route_errors(client, use_vlm):
    use_vlm(FakeVlmProvider())
    missing = uuid.uuid4()
    assert client.post(f"/api/v1/inspections/{missing}/reports").status_code == 404
    assert client.get(f"/api/v1/inspections/{missing}/reports").status_code == 404
    assert client.get(f"/api/v1/reports/{missing}").status_code == 404
    assert client.post(f"/api/v1/reports/{missing}/verifications").status_code == 404
    iid = client.post("/api/v1/inspections", json={"title": "empty"}).json()["id"]
    r = client.post(f"/api/v1/inspections/{iid}/reports")
    assert (r.status_code, r.json()["error"]["code"]) == (409, "no_images")
    assert client.post(f"/api/v1/inspections/{iid}/reports", json={"model": "x"}).status_code == 422


@requires_db
def test_a_report_without_claims_cannot_be_verified(client, db_session, use_vlm):
    iid, _, _ = _inspection_with_detection(client, db_session)
    use_vlm(FakeVlmProvider([report_json()]))
    report = client.post(f"/api/v1/inspections/{iid}/reports").json()
    assert report["claim_count"] == 0
    r = client.post(f"/api/v1/reports/{report['id']}/verifications")
    assert (r.status_code, r.json()["error"]["code"]) == (409, "no_claims")


@requires_db
def test_deleting_the_inspection_removes_reports_claims_and_links(client, db_session, storage, use_vlm):
    from app.services.image_service import delete_inspection

    iid, _, _ = _inspection_with_detection(client, db_session)
    use_vlm(FakeVlmProvider([GEN_OK]))
    assert client.post(f"/api/v1/inspections/{iid}/reports").status_code == 201
    delete_inspection(db_session, storage, uuid.UUID(iid))
    for model in (GeneratedReport, ReportClaim, ReportClaimImage, DamagePrediction):
        assert db_session.scalars(select(model)).all() == []
