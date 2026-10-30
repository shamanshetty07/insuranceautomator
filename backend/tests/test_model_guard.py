"""verifier_is_independent: independence needs two known, different weight digests (no database)."""

import pytest

from app.services import model_guard
from app.services.dev_stubs import StubClaimVerifier, StubStructuredReportGenerator
from app.services.types import ModelInfo

A, B = "a" * 64, "b" * 64


def _config(sha):
    return {"prompt_version": "p1", "model": {"weights_sha256": sha, "packages": {}}}


@pytest.mark.parametrize("gen_name,gen_sha,verifier,expected", [
    ("qwen", A, ModelInfo("qwen", "1", A), False),
    ("qwen", A, ModelInfo("other", "9", A), False),
    ("qwen", A, ModelInfo("other", "9", B), True),
    ("qwen", None, ModelInfo("other", "9", B), False),
    ("qwen", A, ModelInfo("other", "9"), False),
    ("qwen", None, ModelInfo("other", "9"), False),
    ("qwen", None, ModelInfo("qwen", "1"), False),
    ("qwen", A, ModelInfo("qwen", "1", B), False),
], ids=[
    "equal-digests-same-name",
    "equal-digests-different-names",
    "different-digests-different-names",
    "missing-generator-digest-different-names",
    "missing-verifier-digest",
    "both-missing-different-names",
    "both-missing-same-names",
    "same-name-different-digests",
])
def test_independence_requires_known_different_digests(gen_name, gen_sha, verifier, expected):
    config = _config(gen_sha) if gen_sha else {"prompt_version": "p1"}
    assert model_guard.verifier_is_independent(gen_name, "1", config, verifier) is expected


def test_same_name_but_different_version_and_digest_is_independent():
    assert model_guard.verifier_is_independent("qwen", "1", _config(A), ModelInfo("qwen", "2", B)) is True


@pytest.mark.parametrize("config", [
    None,
    {},
    {"model": None},
    {"model": "a" * 64},
    {"model": {"weights_sha256": None}},
    {"model": {"weights_sha256": ""}},
    {"model": {"weights_sha256": "not-a-digest"}},
    {"model": {"weights_sha256": "a" * 63}},
    {"model": {"weights_sha256": 123}},
    ["model"],
    "model",
])
def test_unusable_generator_record_is_never_independent(config):
    assert model_guard.verifier_is_independent("qwen", "1", config, ModelInfo("other", "9", B)) is False


@pytest.mark.parametrize("sha", ["", "zz" * 32, "b" * 65, 0])
def test_unusable_verifier_digest_is_never_independent(sha):
    assert model_guard.verifier_is_independent("qwen", "1", _config(A), ModelInfo("other", "9", sha)) is False


def test_digest_case_and_surrounding_space_do_not_create_independence():
    padded = ModelInfo("other", "9", " " + A.upper())
    assert model_guard.verifier_is_independent("qwen", "1", _config(A), padded) is False
    assert model_guard.generator_digest(_config(A.upper())) == A


def test_development_stubs_are_not_shown_independent():
    """Neither stub carries a digest, so their different names prove nothing."""
    gen, ver = StubStructuredReportGenerator.model, StubClaimVerifier.model
    assert gen.weights_sha256 is None and ver.weights_sha256 is None and gen.name != ver.name
    assert model_guard.verifier_is_independent(gen.name, gen.version, _config(None), ver) is False


def test_models_independent_is_symmetric():
    pairs = [
        (ModelInfo("x", "1", A), ModelInfo("y", "1", B)),
        (ModelInfo("x", "1", A), ModelInfo("x", "1", B)),
        (ModelInfo("x", "1"), ModelInfo("y", "1", B)),
        (ModelInfo("x", "1", A), ModelInfo("y", "1", A)),
    ]
    for a, b in pairs:
        assert model_guard.models_independent(a, b) is model_guard.models_independent(b, a)
