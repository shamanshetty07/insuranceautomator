"""Local VLM adapters with a fake runner: no model, no weights, no torch, no database (except the one requires_db end-to-end test, skipped without TEST_DATABASE_URL)."""

import json
import sys
import threading
import time
import types
import uuid

import pytest
from PIL import Image
from pydantic import ValidationError

from app.core.config import Settings
from app.models.run_status import RunStatus
from app.schemas.claims import ClaimType, Verdict
from app.services import claim_policy
from app.services.evidence_association import PredictionFacts, detection_signal
from app.services.evidence_bundle import evidence_bundle
from app.services.types import Detection, EvidenceBundle, ImageRef, ModelInfo, ReportClaim
from app.services.vlm import prompts, provider as provider_module
from app.services.vlm import transformers_runner as tr
from app.services.vlm.adapters import QwenClaimVerifier, QwenStructuredReportGenerator, VlmLimits
from app.services.vlm.errors import (
    VlmBusy,
    VlmInferenceError,
    VlmInputError,
    VlmOutputError,
    VlmTimeout,
    VlmUnavailable,
)
from app.services.vlm.parsing import NO_CLAIMS_TEXT, load_json_object, parse_structured_report, parse_verdict
from app.services.vlm.runner import VlmRequest, call_runner
from tests.conftest import requires_db

NO_DB = "postgresql+psycopg://none:none@127.0.0.1:1/none"
IMG1, IMG2 = uuid.UUID(int=1), uuid.UUID(int=2)
P1, P2 = uuid.UUID(int=11), uuid.UUID(int=12)
BOX = {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6}


class FakeRunner:
    """Returns canned text (or calls a function); records every request."""

    def __init__(self, responses=(), *, delay=0.0, raises=None):
        self.responses = list(responses)
        self.delay = delay
        self.raises = raises
        self.requests: list[VlmRequest] = []
        self.info = ModelInfo("fake-vlm", "0", weights_sha256="f" * 64)
        self.runtime = {"transformers": None, "torch": None}

    def generate(self, request):
        self.requests.append(request)
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        item = self.responses.pop(0)
        return item(request) if callable(item) else item


@pytest.fixture
def images(tmp_path):
    paths = []
    for n, size in ((1, (64, 48)), (2, (2000, 1000))):
        path = tmp_path / f"{n}.png"
        Image.new("RGB", size, (n * 50, 10, 10)).save(path)
        paths.append(path)
    return paths


@pytest.fixture
def evidence(images):
    return EvidenceBundle(
        images=[ImageRef(IMG1, images[0]), ImageRef(IMG2, images[1])],
        detections=[
            Detection(IMG1, "dent", 0.82, BOX, prediction_id=P1),
            Detection(IMG2, "scratch", 0.40, None, prediction_id=P2),
        ],
    )


def report_json(*claims):
    return json.dumps({"claims": list(claims)})


DENT_CLAIM = {
    "text": "There is a dent on the front door.", "type": "damage_type", "damage_type": "dent",
    "images": ["img1"], "detections": ["det1"],
    "evidence": [{"image": "img1", "detection": "det1", "crop": [0.1, 0.2, 0.5, 0.6]}],
}
PRESENCE_CLAIM = {"text": "Visible damage is present.", "type": "visible_damage_presence", "images": ["img2"]}
LIMITS = VlmLimits(timeout_seconds=5, max_claims=3, max_images=2, max_image_pixels=200_000)


def generator(runner, limits=LIMITS):
    return QwenStructuredReportGenerator(runner, limits)


def test_valid_output_maps_aliases_to_real_ids(evidence):
    runner = FakeRunner([report_json(DENT_CLAIM, PRESENCE_CLAIM)])
    report = generator(runner).generate_structured(evidence)
    first, second = report.claims.claims
    assert (first.claim_id, first.claim_type, first.damage_type) == ("c1", ClaimType.DAMAGE_TYPE, "dent")
    assert first.image_ids == (IMG1,) and first.prediction_ids == (P1,)
    assert first.evidence[0].prediction_id == P1 and first.evidence[0].crop.x_max == 0.5
    assert second.image_ids == (IMG2,) and second.prediction_ids == ()
    assert report.report_text == "There is a dent on the front door.\nVisible damage is present."


def test_prompt_uses_aliases_only_and_marks_detector_output_unverified(evidence):
    runner = FakeRunner([report_json()])
    generator(runner).generate_structured(evidence)
    (request,) = runner.requests
    text = request.system + request.user_text
    assert str(IMG1) not in text and str(P1) not in text
    assert "det1 on img1: class=dent confidence=0.82 box=[0.100, 0.200, 0.500, 0.600]" in request.user_text
    assert "det2 on img2: class=scratch confidence=0.40 box=none" in request.user_text
    assert "unverified" in request.user_text and "Never state repair costs" in request.system
    assert len(request.images) == 2 and all(i.mode == "RGB" for i in request.images)
    assert request.images[1].size[0] * request.images[1].size[1] <= LIMITS.max_image_pixels
    assert request.images[0].size == (64, 48)


def test_fenced_json_is_accepted(evidence):
    raw = "```json\n" + report_json(PRESENCE_CLAIM) + "\n```"
    assert len(generator(FakeRunner([raw])).generate_structured(evidence).claims.claims) == 1


def test_zero_claims_is_a_valid_answer_attributed_to_the_model(evidence):
    report = generator(FakeRunner([report_json()])).generate_structured(evidence)
    assert report.claims.claims == () and report.report_text == NO_CLAIMS_TEXT


def test_out_of_scope_claim_types_are_kept_for_the_policy(evidence):
    cost = {"text": "Repair would cost 900.", "type": "repair_cost", "images": ["img1"]}
    hidden = {"text": "The frame may be bent.", "type": "hidden_structural_damage", "images": ["img1"]}
    report = generator(FakeRunner([report_json(cost, hidden)])).generate_structured(evidence)
    assert [c.claim_type for c in report.claims.claims] == [ClaimType.REPAIR_COST, ClaimType.HIDDEN_STRUCTURAL_DAMAGE]


@pytest.mark.parametrize("raw,code", [
    ("", "empty"),
    ("   ", "empty"),
    ("Sure! Here is the JSON: {}", "not_json"),
    ("{not json", "not_json"),
    ('{"claims": [], "claims": []}', "not_json"),
    ('{"claims": [{"text": "x", "type": "damage_type", "images": ["img1"], "confidence": NaN}]}', "not_json"),
    ("x" * 64_001, "too_long"),
    ('{"claims": "none"}', "schema"),
    ('{"claims": [], "summary": "extra key"}', "schema"),
    ('[]', "schema"),
    (report_json({"text": "x", "type": "damage_type", "images": ["img9"]}), "unknown_alias"),
    (report_json({"text": "x", "type": "damage_type", "images": ["img1"], "detections": ["det7"]}), "unknown_alias"),
    (report_json({"text": "x", "type": "damage_type", "images": ["img1"], "detections": ["det2"]}),
     "detection_image_mismatch"),
    (report_json({"text": "x", "type": "damage_type", "images": ["img1", "img2"], "detections": ["det2"],
                  "evidence": [{"image": "img1", "detection": "det2"}]}), "detection_image_mismatch"),
    (report_json({"text": "x", "type": "damage_type", "images": ["img1"],
                  "evidence": [{"image": "img1", "crop": [0.5, 0.2, 0.1, 0.6]}]}), "invalid_crop"),
    (report_json({"text": "x", "type": "made_up_type", "images": ["img1"]}), "invalid_claim"),
    (report_json({"text": "x", "type": "damage_type", "images": ["img1", "img1"]}), "invalid_claim"),
    (report_json({"text": "x", "type": "damage_type", "images": ["img1"],
                  "evidence": [{"image": "img1"}, {"image": "img1"}]}), "invalid_claim"),
    (report_json({"text": "x", "type": "damage_type", "images": [],
                  "evidence": [{"image": "img1"}]}), "invalid_claim"),
    (report_json(*[PRESENCE_CLAIM] * 4), "too_many_claims"),
])
def test_bad_output_is_rejected_whole(evidence, raw, code):
    with pytest.raises(VlmOutputError) as exc:
        generator(FakeRunner([raw])).generate_structured(evidence)
    assert exc.value.code == code
    assert raw[:40] not in str(exc.value) or raw == ""


def test_load_json_object_strips_exactly_one_fence():
    assert load_json_object("```\n{\"a\": 1}\n```") == {"a": 1}
    with pytest.raises(VlmOutputError):
        load_json_object("```json\n```json\n{}\n```\n```")


def test_detection_without_stored_id_is_refused(images):
    ev = EvidenceBundle(images=[ImageRef(IMG1, images[0])], detections=[Detection(IMG1, "dent", 0.9)])
    with pytest.raises(VlmInputError) as exc:
        generator(FakeRunner()).generate_structured(ev)
    assert exc.value.code == "uncitable_detection"


def test_input_limits_fail_instead_of_truncating(evidence, images):
    with pytest.raises(VlmInputError, match="Too many images"):
        generator(FakeRunner(), VlmLimits(max_images=1)).generate_structured(evidence)
    with pytest.raises(VlmInputError, match="Too many detections"):
        generator(FakeRunner(), VlmLimits(max_detections=1)).generate_structured(evidence)
    with pytest.raises(VlmInputError) as exc:
        generator(FakeRunner()).generate_structured(EvidenceBundle(images=[]))
    assert exc.value.code == "no_images"
    images[0].write_bytes(b"not an image")
    with pytest.raises(VlmInputError, match="could not be read"):
        generator(FakeRunner()).generate_structured(evidence)


def test_limits_are_validated():
    with pytest.raises(ValueError):
        VlmLimits(max_claims=101)
    with pytest.raises(ValueError):
        VlmLimits(timeout_seconds=0)


def _request():
    return VlmRequest("s", "u", (), 8, 1.0)


def test_runner_exception_becomes_inference_error():
    with pytest.raises(VlmInferenceError) as exc:
        call_runner(FakeRunner(raises=RuntimeError("CUDA out of memory /secret/path")), _request(), 2,
                    lock=threading.Lock())
    assert "secret" not in str(exc.value) and isinstance(exc.value.__cause__, RuntimeError)


def test_non_text_output_is_rejected():
    with pytest.raises(VlmOutputError):
        call_runner(FakeRunner([None]), _request(), 2, lock=threading.Lock())


def test_timeout_then_busy_until_the_stuck_inference_ends():
    lock = threading.Lock()
    slow = FakeRunner(["late"], delay=0.6)
    with pytest.raises(VlmTimeout):
        call_runner(slow, _request(), 0.1, lock=lock)
    with pytest.raises(VlmBusy):
        call_runner(FakeRunner(["x"]), _request(), 0.1, lock=lock)
    time.sleep(0.7)
    assert call_runner(FakeRunner(["ok"]), _request(), 1, lock=lock) == "ok"


def test_generator_passes_a_cooperative_time_budget(evidence):
    runner = FakeRunner([report_json()])
    generator(runner, VlmLimits(timeout_seconds=60, max_images=2)).generate_structured(evidence)
    assert runner.requests[0].max_time_seconds == 55 and runner.requests[0].max_new_tokens == 1024


def verifier(runner):
    return QwenClaimVerifier(runner, LIMITS)


def test_verifier_one_inference_per_claim_and_label_passthrough(images):
    runner = FakeRunner([
        '{"label": "supported", "explanation": "A dent is visible."}',
        '{"label": "omitted_damage", "explanation": "x"}',
    ])
    ev = EvidenceBundle(images=[ImageRef(IMG1, images[0])])
    out = verifier(runner).verify([ReportClaim("a dent"), ReportClaim("b")], ev)
    assert [(v.claim_text, v.label) for v in out] == [("a dent", "supported"), ("b", "omitted_damage")]
    assert out[0].verifier.name == "fake-vlm" and out[0].evidence_reference == {"image_ids": [str(IMG1)]}
    assert "<<<a dent>>>" in runner.requests[0].user_text
    assert "omitted_damage" not in runner.requests[0].system


def test_verifier_never_sees_detections(evidence):
    with pytest.raises(VlmInputError) as exc:
        verifier(FakeRunner()).verify([ReportClaim("x")], evidence)
    assert exc.value.code == "detections_not_allowed"


@pytest.mark.parametrize("raw", ['{"verdict": "supported"}', '{"label": ""}', "supported", '{"label": 1}'])
def test_malformed_verifier_output_raises(images, raw):
    with pytest.raises(VlmOutputError):
        verifier(FakeRunner([raw])).verify([ReportClaim("x")], EvidenceBundle(images=[ImageRef(IMG1, images[0])]))


def test_verifier_does_not_pretend_to_split_reports_or_find_omissions():
    v = verifier(FakeRunner())
    with pytest.raises(NotImplementedError):
        v.extract_claims("text")
    with pytest.raises(NotImplementedError):
        v.find_omissions("text", EvidenceBundle(images=[]))


def test_parse_verdict_strips_and_bounds():
    assert parse_verdict('{"label": " supported ", "explanation": " ok "}') == ("supported", "ok")
    with pytest.raises(VlmOutputError):
        parse_verdict(json.dumps({"label": "supported", "explanation": "x" * 2001}))


def test_verifier_label_vocabulary_excludes_omitted_damage():
    assert "omitted_damage" not in prompts.VERIFIER_SYSTEM
    assert '"unverifiable"' in prompts.VERIFIER_SYSTEM and '"supported"' in prompts.VERIFIER_SYSTEM


def test_configs_record_greedy_decoding_and_prompt_versions():
    runner = FakeRunner()
    g, v = generator(runner).generation_config, verifier(runner).verification_config
    assert (g.prompt_version, g.sampling.temperature, g.sampling.max_new_tokens) == ("qwen-report-1", 0, 1024)
    assert (v.prompt_version, v.sampling.temperature, v.sampling.max_new_tokens) == ("qwen-verify-1", 0, 256)


def test_generated_claims_flow_into_the_claim_policy(evidence):
    """Generator output -> detection_signal -> evaluate_claim, with a verifier verdict."""
    mismatched_text = {**DENT_CLAIM, "text": "There is a scratch on the front door."}
    raw = report_json(DENT_CLAIM, mismatched_text,
                      {"text": "Repair would cost 900.", "type": "repair_cost", "images": ["img1"]})
    report = generator(FakeRunner([raw])).generate_structured(evidence)
    run_id = uuid.uuid4()
    facts = {P1: PredictionFacts(IMG1, run_id, "dent", BOX)}
    decisions = []
    for claim in report.claims.claims:
        cited = {p: facts[p] for p in claim.prediction_ids}
        match, _ = detection_signal(claim, claim.image_ids, cited, {IMG1: run_id}, {IMG1: 1})
        decisions.append(claim_policy.evaluate_claim(claim_policy.ClaimPolicyInput(
            claim.claim_type, claim_policy.VerifierOutcome(Verdict.SUPPORTED), claim_policy.EvidenceIntegrity.OK,
            claim_policy.EvidenceAssociation.ASSOCIATED, match)))
    corroborated, mismatched, cost = decisions
    assert (corroborated.verdict, corroborated.needs_human_review) == (Verdict.SUPPORTED, False)
    assert mismatched.needs_human_review
    assert (cost.verdict, cost.needs_human_review) == (Verdict.INSUFFICIENT_EVIDENCE, True)


def _run(image_id, status, preds):
    return types.SimpleNamespace(inspection_image_id=image_id, status=status, predictions=preds)


def _pred(pid, conf, damage="dent", box=None):
    return types.SimpleNamespace(id=pid, damage_type=damage, confidence=conf, bounding_box=box)


def test_evidence_bundle_uses_completed_runs_only_with_prediction_ids(tmp_path):
    a, b, c = (tmp_path / n for n in "abc")
    img3 = uuid.UUID(int=3)
    runs = [
        _run(IMG1, RunStatus.COMPLETED, [_pred(P1, 0.3), _pred(P2, 0.9, "scratch", BOX)]),
        _run(IMG2, RunStatus.FAILED, [_pred(uuid.uuid4(), 0.99)]),
    ]
    bundle = evidence_bundle([(IMG1, a), (IMG2, b), (img3, c)], runs)
    assert [r.image_id for r in bundle.images] == [IMG1, IMG2, img3]
    assert [(d.prediction_id, d.damage_type) for d in bundle.detections] == [(P2, "scratch"), (P1, "dent")]
    assert bundle.detections[0].bounding_box == BOX
    prompts.build_alias_map(bundle)


def test_evidence_bundle_refuses_ambiguous_input(tmp_path):
    with pytest.raises(ValueError):
        evidence_bundle([(IMG1, tmp_path)], [_run(IMG2, RunStatus.COMPLETED, [])])
    with pytest.raises(ValueError):
        evidence_bundle([(IMG1, tmp_path)], [_run(IMG1, RunStatus.COMPLETED, [])] * 2)
    with pytest.raises(ValueError):
        evidence_bundle([(IMG1, tmp_path), (IMG1, tmp_path)], [])


def _model_dir(tmp_path, model_type="qwen2_vl", extra=()):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    (d / "model.safetensors").write_bytes(b"weights")
    for name in extra:
        (d / name).write_bytes(b"x")
    return d


def test_model_dir_checks(tmp_path):
    assert tr.inspect_model_dir(_model_dir(tmp_path)) == "qwen2_vl"
    for kwargs, match in (
        ({"model_type": "llama"}, "unsupported model_type"),
        ({"extra": ["pytorch_model.bin"]}, "pickle-format"),
    ):
        sub = tmp_path / match.replace(" ", "_")
        sub.mkdir()
        with pytest.raises(tr.RunnerRefused, match=match):
            tr.inspect_model_dir(_model_dir(sub, **kwargs))
    with pytest.raises(tr.RunnerRefused, match="not a directory"):
        tr.inspect_model_dir(tmp_path / "missing")


def test_model_dir_digest_covers_every_model_file_but_not_dotfiles(tmp_path):
    d = _model_dir(tmp_path)
    before = tr.model_dir_digest(d)
    (d / ".cache").mkdir()
    (d / ".cache" / "x").write_bytes(b"ignored")
    assert tr.model_dir_digest(d) == before
    (d / "tokenizer.json").write_text("{}")
    assert tr.model_dir_digest(d) != before


def test_pin_mismatch_refused_before_transformers_is_imported(tmp_path, monkeypatch):
    d = _model_dir(tmp_path)
    imported = []
    monkeypatch.setattr(tr.importlib, "import_module", lambda name: imported.append(name))
    with pytest.raises(tr.RunnerRefused, match="VLM_WEIGHTS_SHA256"):
        tr.TransformersQwenRunner(d, "m", "1", expected_sha256="0" * 64)
    assert imported == []


def test_importing_the_package_does_not_import_model_libraries():
    assert "transformers" not in sys.modules and "torch" not in sys.modules


def _settings(tmp_path, **kw):
    base = dict(database_url=NO_DB, vlm_backend="transformers", vlm_model_path=tmp_path,
                vlm_model_name="qwen2-vl-2b", vlm_model_version="1")
    return Settings(**{**base, **kw})


@pytest.fixture(autouse=True)
def _clear_vlm_cache():
    provider_module.clear_vlm_cache()
    yield
    provider_module.clear_vlm_cache()


def test_provider_disabled_by_default_and_never_returns_stubs():
    p = provider_module.SettingsVlmProvider(Settings(database_url=NO_DB))
    assert p.status().available is False and "VLM_BACKEND=none" in p.status().reason
    with pytest.raises(VlmUnavailable):
        p.get_generator()


def test_provider_reports_missing_packages(tmp_path, monkeypatch):
    monkeypatch.setattr(provider_module.importlib.util, "find_spec", lambda name: None)
    status = provider_module.SettingsVlmProvider(_settings(tmp_path)).status()
    assert not status.available and "'vlm' extra" in status.reason


def test_provider_caches_one_runner_and_status_never_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(provider_module.importlib.util, "find_spec", lambda name: object())
    built = []

    def factory(settings):
        built.append(1)
        return FakeRunner()

    p = provider_module.SettingsVlmProvider(_settings(tmp_path), runner_factory=factory)
    assert p.status().available and built == []
    g, v = p.get_generator(), p.get_verifier()
    assert built == [1] and g.model == v.model
    assert p.status().weights_sha256 == "f" * 64


def test_provider_maps_load_failures_without_caching_them(tmp_path, monkeypatch):
    monkeypatch.setattr(provider_module.importlib.util, "find_spec", lambda name: object())
    attempts = []

    def refused(settings):
        attempts.append(1)
        raise tr.RunnerRefused("pin mismatch at /secret")

    p = provider_module.SettingsVlmProvider(_settings(tmp_path), runner_factory=refused)
    for _ in range(2):
        with pytest.raises(VlmUnavailable) as exc:
            p.get_generator()
        assert "safeguard" in str(exc.value) and "secret" not in str(exc.value)
    assert attempts == [1, 1]

    def broken(settings):
        raise RuntimeError("boom")

    with pytest.raises(VlmUnavailable, match="failed to load"):
        provider_module.SettingsVlmProvider(_settings(tmp_path), runner_factory=broken).get_verifier()


def test_vlm_settings_validation(tmp_path):
    s = Settings(database_url=NO_DB)
    assert (s.vlm_backend, s.vlm_timeout_seconds, s.vlm_max_claims, s.vlm_dtype) == ("none", 180, 20, "bfloat16")
    assert _settings(tmp_path, vlm_weights_sha256="A" * 64).vlm_weights_sha256 == "a" * 64
    for bad in ({"vlm_max_claims": 101}, {"vlm_weights_sha256": "abc"}, {"vlm_dtype": "int4"},
                {"vlm_backend": "openai"}, {"vlm_timeout_seconds": 0}):
        with pytest.raises(ValidationError):
            _settings(tmp_path, **bad)


@requires_db
def test_verify_report_with_the_qwen_verifier(db_session, storage):
    from tests.test_verification_service import World

    world = World(db_session, storage)
    runner = FakeRunner([
        '{"label": "supported", "explanation": "A dent is visible."}',
        '{"label": "omitted_damage", "explanation": "x"}',
    ])
    a = world.claim("c1", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="dent")
    b = world.claim("c2", ClaimType.VISIBLE_DAMAGE_PRESENCE, [world.image.id], text="odd")
    run = world.run([a, b], QwenClaimVerifier(runner, LIMITS))
    rows = world.results(run)
    assert run.status is RunStatus.COMPLETED and run.verifier_model == "fake-vlm"
    assert rows["c1"].verdict is Verdict.SUPPORTED and rows["c1"].needs_human_review
    assert rows["c2"].verification_label is None and rows["c2"].needs_human_review
    assert all(not r.images or r.images[0].mode == "RGB" for r in runner.requests)
    assert all("<<<" in r.user_text for r in runner.requests)
