"""Repeated verification of one report claim: disagreement forces review, agreement adds no trust."""

import dataclasses
import uuid

import pytest

from app.models.verification_result import VerificationLabel
from app.schemas.claims import Verdict
from app.services.types import ModelInfo
from app.services.verdict_consistency import ConsistencyCheck, PriorVerdict, check_consistency, raw_verdict

QWEN = ModelInfo("qwen", "1", "a" * 64)
OTHER = ModelInfo("other", "2", "b" * 64)
NO_DIGEST = ModelInfo("qwen", "1")


def prior(label, verifier=QWEN, run=None):
    return PriorVerdict(run_id=run or uuid.uuid4(), raw_label=label, verifier=verifier)


@pytest.mark.parametrize("current,previous,disagrees", [
    ("supported", "supported", False),
    ("supported", "contradicted", True),
    ("supported", "unverifiable", True),
    ("contradicted", "unverifiable", True),
    ("contradicted", "wrong_location", False),
    ("wrong_location", "wrong_damage_type", False),
    ("unverifiable", "supported", True),
    ("unverifiable", "unverifiable", False),
])
def test_any_change_of_raw_verdict_is_a_disagreement(current, previous, disagrees):
    check = check_consistency(current, QWEN, [prior(previous)])
    assert check.disagrees is disagrees
    assert check.compared == 1 and check.ignored == 0


def test_comparison_uses_the_raw_label_enum_as_stored():
    check = check_consistency(VerificationLabel.SUPPORTED, QWEN, [prior(VerificationLabel.CONTRADICTED)])
    assert check.disagrees and check.current is Verdict.SUPPORTED


@pytest.mark.parametrize("unusable", [None, "unsupported_severity", "omitted_damage", "maybe", "Supported", 1])
def test_unusable_prior_is_neither_agreement_nor_disagreement(unusable):
    check = check_consistency("supported", QWEN, [prior(unusable)])
    assert not check.disagrees and (check.compared, check.ignored) == (0, 1)


@pytest.mark.parametrize("unusable", [None, "unsupported_severity", "omitted_damage", "maybe"])
def test_unusable_current_verdict_compares_nothing(unusable):
    check = check_consistency(unusable, QWEN, [prior("contradicted"), prior("supported")])
    assert not check.disagrees and check.current is None and check.compared == 0 and check.ignored == 2


def test_one_disagreeing_run_among_many_agreeing_runs_still_disagrees():
    odd = uuid.uuid4()
    priors = [prior("supported") for _ in range(9)] + [prior("contradicted", run=odd)]
    check = check_consistency("supported", QWEN, priors)
    assert check.disagrees and check.disagreeing_runs == (str(odd),)
    assert "supported: 10" in check.summary and "contradicted: 1" in check.summary


def test_disagreement_from_a_different_model_also_forces_review():
    check = check_consistency("supported", QWEN, [prior("contradicted", verifier=OTHER)])
    assert check.disagrees and check.same_model == 0


@pytest.mark.parametrize("verifier,same", [
    (QWEN, 1),
    (NO_DIGEST, 1),
    (ModelInfo("qwen", "1", "b" * 64), 1),
    (ModelInfo("renamed", "9", "a" * 64), 1),
    (OTHER, 0),
])
def test_repeats_by_the_same_or_unknown_model_are_counted_as_same_model(verifier, same):
    check = check_consistency("supported", QWEN, [prior("supported", verifier=verifier)])
    assert not check.disagrees and check.same_model == same


def test_agreement_never_reads_as_confirmation():
    fields = {f.name.replace("disagree", "") for f in dataclasses.fields(ConsistencyCheck)}
    assert not any(word in name for name in fields for word in ("confirm", "trust", "agree", "independent"))
    check = check_consistency("supported", QWEN, [prior("supported") for _ in range(5)])
    assert not check.disagrees
    assert "adds no trust" in check.summary and "5 earlier run(s)" in check.summary


def test_no_priors():
    check = check_consistency("supported", QWEN, [])
    assert (check.disagrees, check.compared, check.ignored) == (False, 0, 0)


def test_priors_may_be_a_generator():
    check = check_consistency("supported", QWEN, (prior(x) for x in ("supported", "unverifiable")))
    assert check.disagrees and check.compared == 2


def test_raw_verdict_never_raises():
    assert raw_verdict("wrong_damage_type") is Verdict.CONTRADICTED
    assert raw_verdict(["supported"]) is None and raw_verdict(object()) is None
