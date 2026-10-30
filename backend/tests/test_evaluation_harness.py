"""SYNTHETIC: replays hand-written verifier answers; says nothing about Qwen-VL or detector accuracy."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.evaluation.harness import TrialSet, evaluate, load_trials, result_json, run_configurations
from app.evaluation.manifest import load_manifest
from app.evaluation.records import Configuration as C
from app.evaluation.records import Disposition
from app.services.claim_policy import POLICY_VERSION
from app.services.claim_policy import ReasonCode as R

FIXTURES = Path(__file__).parent / "fixtures" / "evaluation"
MANIFEST = FIXTURES / "synthetic_manifest_v0.json"
TRIALS = FIXTURES / "synthetic_trials_v0.json"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST)


@pytest.fixture(scope="module")
def trials():
    return load_trials(TRIALS)


def by_key(outcomes):
    return {o.claim_key: o for o in outcomes}


def test_configurations_are_paired_on_the_same_claims(manifest, trials):
    out = run_configurations(manifest, trials)
    keys = [c.claim_key for c in manifest.claims]
    for config in C:
        assert [o.claim_key for o in out[config]] == keys
        assert {o.configuration for o in out[config]} == {config}


def test_baseline_accepts_everything_unverified(manifest, trials):
    out = run_configurations(manifest, trials)[C.BASELINE]
    assert all(o.disposition is Disposition.ACCEPTED and not o.verifier_called for o in out)


def test_self_verification_uses_the_raw_answer_without_evidence_checks(manifest, trials):
    out = by_key(run_configurations(manifest, trials)[C.SELF_VERIFICATION])
    assert out["bravo-c04"].disposition is Disposition.ACCEPTED
    assert out["charlie-c03"].disposition is Disposition.ACCEPTED
    assert out["alpha-c02"].disposition is Disposition.REJECTED
    assert out["delta-c01"].verifier_output_invalid and out["delta-c01"].disposition is Disposition.HUMAN_REVIEW
    assert out["bravo-c02"].verifier_output_invalid
    assert out["delta-c03"].verifier_failed
    assert all(o.verifier_called for o in out.values())


def test_hybrid_with_the_same_model_verifying_accepts_nothing(manifest, trials):
    assert trials.verifier_independent is False
    out = run_configurations(manifest, trials)[C.HYBRID]
    assert {o.disposition for o in out} == {Disposition.HUMAN_REVIEW}


def test_hybrid_reuses_production_skip_and_integrity_rules(manifest, trials):
    out = by_key(run_configurations(manifest, trials)[C.HYBRID])
    assert out["bravo-c04"].reason_codes == (R.EVIDENCE_UNRESOLVED,) and not out["bravo-c04"].verifier_called
    assert out["charlie-c03"].reason_codes == (R.EVIDENCE_UNRESOLVED,) and out["charlie-c03"].integrity_failure
    assert out["alpha-c05"].reason_codes == (R.COST_NOT_VALIDATED,) and out["alpha-c05"].raw_verifier_verdict is None
    assert out["delta-c01"].reason_codes == (R.VERIFIER_INVALID,)


def test_limitation_counterfactual_independent_hybrid_accepts_a_falsely_supported_invented_claim(manifest, trials):
    out = by_key(run_configurations(manifest, trials, force_independent=True)[C.HYBRID])
    accepted = sorted(k for k, o in out.items() if o.disposition is Disposition.ACCEPTED)
    assert accepted == ["alpha-c01", "bravo-c01", "charlie-c01"]
    assert out["charlie-c01"].reason_codes == (R.VERIFIER_SUPPORTED,)


def test_evaluate_labels_results_synthetic_and_records_provenance(manifest, trials):
    result = evaluate(manifest, trials)
    assert result.synthetic and "SYNTHETIC" in result.result_label
    assert result.policy_version == POLICY_VERSION
    assert len(result.manifest_digest) == 64
    data = json.loads(result_json(result))
    assert set(data["reports"]) == {c.value for c in C}
    counterfactual = evaluate(manifest, trials, force_independent=True)
    assert "COUNTERFACTUAL" in counterfactual.result_label


def test_unlabelled_claims_are_excluded_not_guessed(manifest, trials):
    reports = evaluate(manifest, trials).reports
    for report in reports.values():
        assert (report.total, report.labelled, report.unlabelled) == (16, 15, 1)
    assert reports[C.BASELINE].unsupported_acceptance_rate.to_dict() == {"numerator": 12, "denominator": 12,
                                                                         "value": 1.0}
    assert reports[C.HYBRID].claim_precision.value is None


def test_real_manifest_with_synthetic_trials_is_still_labelled_synthetic(manifest, trials):
    real = manifest.model_copy(update={"dataset_kind": "real"})
    assert "SYNTHETIC verifier answers" in evaluate(real, trials).result_label


@pytest.mark.parametrize("change", ["version", "missing", "extra"])
def test_trials_must_match_the_manifest(manifest, trials, change):
    data = trials.model_dump(mode="json")
    if change == "version":
        data["manifest_dataset_version"] = "other"
    elif change == "missing":
        data["trials"].pop("alpha-c01")
    else:
        data["trials"]["zzz"] = data["trials"]["alpha-c01"]
    with pytest.raises(ValueError):
        run_configurations(manifest, TrialSet.model_validate(data))


def test_trial_rejects_unknown_fields_and_values():
    with pytest.raises(ValueError):
        TrialSet.model_validate({"manifest_dataset_version": "v", "source": "s", "synthetic": True,
                                 "verifier_independent": False,
                                 "trials": {"a": {"integrity": "fine", "detection_match": "matches_claim"}}})


def test_cli_prints_labelled_json():
    script = Path(__file__).parents[1] / "scripts" / "evaluate_claims.py"
    proc = subprocess.run([sys.executable, str(script), "--manifest", str(MANIFEST), "--trials", str(TRIALS)],
                          capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout)
    assert data["synthetic"] is True and data["reports"]["hybrid"]["counts"]["total"] == 16
    assert "SYNTHETIC" in proc.stderr
