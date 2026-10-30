import json

import pytest

from app.evaluation.metrics import (
    Ratio,
    cohens_kappa,
    compare,
    compute_metrics,
    verifier_agreement,
)
from app.evaluation.records import (
    ClaimOutcome,
    Configuration,
    Disposition,
    GoldLabel,
    LabelledOutcome,
    baseline_outcome,
    join_labels,
    outcome_from_decision,
)
from app.schemas.claims import ClaimType, Verdict
from app.services.claim_policy import (
    ClaimPolicyInput,
    DetectionMatch,
    EvidenceAssociation,
    EvidenceIntegrity,
    ReasonCode,
    VerifierOutcome,
    evaluate_claim,
)

S, C, U = GoldLabel.SUPPORTED, GoldLabel.CONTRADICTED, GoldLabel.UNVERIFIABLE
VS, VC, VI = Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.INSUFFICIENT_EVIDENCE
SELF = Configuration.SELF_VERIFICATION


def lo(key, verdict, review, gold, codes=(), raw=None, called=True, config=SELF):
    return LabelledOutcome(ClaimOutcome(key, config, verdict, review, tuple(codes), raw, called), gold)


def fixture():
    """Hand-built: 10 outcomes, 8 labelled. Expected values are worked out in the tests below."""
    return [
        lo("a", VS, False, S, raw=VS),
        lo("b", VS, False, C, raw=VS),
        lo("c", VC, False, S, raw=VC),
        lo("d", VC, False, C, raw=VC),
        lo("e", VI, True, U, raw=VI),
        lo("f", VI, True, S, codes=[ReasonCode.VERIFIER_INVALID]),
        lo("g", VI, True, U, codes=[ReasonCode.VERIFIER_ERROR]),
        lo("h", VI, True, None, codes=[ReasonCode.INTEGRITY_MISMATCHED], called=False),
        lo("i", VS, False, None, raw=VS),
        lo("j", VS, True, U, codes=[ReasonCode.VERIFIER_SUPPORTED_NO_DETECTION], raw=VS),
    ]


def test_ratio_value_and_str():
    assert Ratio(3, 4).value == 0.75
    assert str(Ratio(3, 4)) == "3/4 (0.750)"
    assert Ratio(0, 0).value is None
    assert str(Ratio(0, 0)) == "0/0 (undefined)"
    assert Ratio(0, 0).to_dict() == {"numerator": 0, "denominator": 0, "value": None}
    with pytest.raises(ValueError):
        Ratio(2, 1)
    with pytest.raises(ValueError):
        Ratio(-1, 1)


def test_wilson_interval():
    assert Ratio(0, 0).wilson_95() is None
    lo_, hi = Ratio(5, 10).wilson_95()
    assert lo_ == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    lo0, hi0 = Ratio(0, 10).wilson_95()
    assert lo0 == 0.0 and hi0 == pytest.approx(0.2775, abs=1e-4)
    lo1, hi1 = Ratio(10, 10).wilson_95()
    assert hi1 == 1.0 and lo1 == pytest.approx(0.7225, abs=1e-4)


def test_counts():
    r = compute_metrics(fixture())
    assert r.configuration is SELF
    assert (r.total, r.labelled, r.unlabelled, r.verifier_called) == (10, 8, 2, 9)
    assert r.by_disposition == {Disposition.ACCEPTED: 3, Disposition.REJECTED: 2, Disposition.HUMAN_REVIEW: 5}
    assert r.by_gold == {S: 3, C: 2, U: 3}


def test_claim_metrics_use_labelled_only():
    r = compute_metrics(fixture())
    assert r.claim_precision == Ratio(1, 2)
    assert r.claim_recall == Ratio(1, 3)
    assert r.unsupported_acceptance_rate == Ratio(1, 5)
    assert r.false_rejection_rate == Ratio(1, 3)


def test_operational_rates():
    r = compute_metrics(fixture())
    assert r.human_review_rate == Ratio(5, 10)
    assert r.invalid_output_rate == Ratio(1, 9)
    assert r.verifier_failure_rate == Ratio(1, 9)
    assert r.invalid_output_rejection_rate == Ratio(1, 1)
    assert r.evidence_integrity_failure_rate == Ratio(1, 10)


def test_unlabelled_accepted_claim_does_not_move_precision():
    items = fixture()
    extra = items + [lo("k", VS, False, None, raw=VS), lo("l", VS, False, None, raw=VS)]
    a, b = compute_metrics(items), compute_metrics(extra)
    assert b.unlabelled == 4
    assert b.claim_precision == a.claim_precision
    assert b.unsupported_acceptance_rate == a.unsupported_acceptance_rate
    assert b.verifier_agreement == a.verifier_agreement
    assert b.human_review_rate == Ratio(5, 12)


def test_verifier_agreement_hand_calculation():
    # rows gold (S, C, I), cols verifier: a S/S, c S/C, b C/S, d C/C, e I/I, j I/S.
    # po = 3/6; row totals 2,2,2; col totals 3,2,1; pe = (6+4+2)/36 = 1/3; kappa = (1/2-1/3)/(2/3) = 1/4.
    ag = compute_metrics(fixture()).verifier_agreement
    assert ag.n == 6
    assert ag.confusion == ((1, 1, 0), (1, 1, 0), (1, 0, 1))
    assert ag.agreement == Ratio(3, 6)
    assert ag.kappa == pytest.approx(0.25)


def test_kappa_known_values():
    assert cohens_kappa([[20, 5], [10, 15]]) == pytest.approx(0.4)
    assert cohens_kappa([[5, 0], [0, 5]]) == pytest.approx(1.0)
    assert cohens_kappa([[0, 5], [5, 0]]) == pytest.approx(-1.0)
    assert cohens_kappa([[5, 0], [0, 0]]) is None
    assert cohens_kappa([[0, 0], [0, 0]]) is None


def test_agreement_ignores_policy_and_missing_raw():
    # Policy sent this to review, but the raw verifier verdict still agrees with the human.
    items = [lo("x", VS, True, S, raw=VS), lo("y", VI, True, S), lo("z", VS, False, None, raw=VC)]
    ag = verifier_agreement(items)
    assert ag.n == 1 and ag.agreement == Ratio(1, 1) and ag.kappa is None


def test_all_zero_denominators():
    r = compute_metrics([])
    assert r.configuration is None and r.total == 0
    for name in (
        "claim_precision",
        "claim_recall",
        "unsupported_acceptance_rate",
        "false_rejection_rate",
        "human_review_rate",
        "invalid_output_rate",
        "verifier_failure_rate",
        "invalid_output_rejection_rate",
        "evidence_integrity_failure_rate",
    ):
        assert getattr(r, name) == Ratio(0, 0), name
    assert r.verifier_agreement.n == 0 and r.verifier_agreement.kappa is None


def test_zero_denominators_with_only_unlabelled_claims():
    r = compute_metrics([lo("a", VS, False, None, raw=VS), lo("b", VC, False, None, raw=VC)])
    assert r.claim_precision.value is None
    assert r.claim_recall.value is None
    assert r.unsupported_acceptance_rate.value is None
    assert r.false_rejection_rate.value is None
    assert r.invalid_output_rejection_rate.value is None
    assert r.human_review_rate == Ratio(0, 2)


def test_baseline_accepts_everything():
    gold = {"s1": S, "s2": S, "c1": C, "u1": U, "n1": None}
    items = join_labels([baseline_outcome(k) for k in gold], gold)
    r = compute_metrics(items)
    assert r.configuration is Configuration.BASELINE
    assert r.by_disposition[Disposition.ACCEPTED] == 5
    assert r.unsupported_acceptance_rate == Ratio(2, 2)
    assert r.unsupported_acceptance_rate.value == 1.0
    assert r.claim_precision == Ratio(2, 4)
    assert r.claim_recall == Ratio(2, 2)
    assert r.false_rejection_rate == Ratio(0, 2)
    assert r.human_review_rate == Ratio(0, 5)
    assert r.verifier_called == 0
    assert r.invalid_output_rate.value is None
    assert r.verifier_agreement.n == 0


def test_policy_never_accepts_invalid_verifier_output():
    items = []
    for n, verdict in enumerate(["maybe", None, "SUPPORTED"]):
        decision = evaluate_claim(
            ClaimPolicyInput(
                claim_type=ClaimType.DAMAGE_TYPE,
                verifier=VerifierOutcome(verdict=verdict),
                integrity=EvidenceIntegrity.OK,
                association=EvidenceAssociation.ASSOCIATED,
                detection_match=DetectionMatch.MATCHES_CLAIM,
            )
        )
        items.append(LabelledOutcome(outcome_from_decision(f"k{n}", Configuration.HYBRID, decision), S))
    r = compute_metrics(items)
    assert r.invalid_output_rate == Ratio(3, 3)
    assert r.invalid_output_rejection_rate == Ratio(3, 3)
    assert r.by_disposition[Disposition.ACCEPTED] == 0


def test_duplicate_claim_keys_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        compute_metrics([lo("a", VS, False, S), lo("a", VC, False, S)])


def test_mixed_configurations_rejected():
    with pytest.raises(ValueError, match="mixed"):
        compute_metrics([lo("a", VS, False, S), lo("b", VS, False, S, config=Configuration.HYBRID)])


def test_compare_same_claims():
    gold = {"a": S, "b": C}
    base = join_labels([baseline_outcome(k) for k in gold], gold)
    hyb = [lo("a", VS, False, S, config=Configuration.HYBRID), lo("b", VC, False, C, config=Configuration.HYBRID)]
    out = compare({Configuration.BASELINE: base, Configuration.HYBRID: hyb})
    assert out[Configuration.BASELINE].unsupported_acceptance_rate == Ratio(1, 1)
    assert out[Configuration.HYBRID].unsupported_acceptance_rate == Ratio(0, 1)


def test_compare_rejects_different_claim_sets():
    base = join_labels([baseline_outcome("a"), baseline_outcome("b")], {"a": S, "b": C})
    hyb = [lo("a", VS, False, S, config=Configuration.HYBRID)]
    with pytest.raises(ValueError, match="same claims"):
        compare({Configuration.BASELINE: base, Configuration.HYBRID: hyb})


def test_compare_rejects_inconsistent_gold_and_misfiled_outcomes():
    base = join_labels([baseline_outcome("a")], {"a": S})
    hyb = [lo("a", VS, False, C, config=Configuration.HYBRID)]
    with pytest.raises(ValueError, match="gold"):
        compare({Configuration.BASELINE: base, Configuration.HYBRID: hyb})
    with pytest.raises(ValueError, match="belong"):
        compare({Configuration.HYBRID: base})


def test_to_dict_is_json_serialisable():
    d = compute_metrics(fixture()).to_dict()
    json.dumps(d)
    assert d["configuration"] == "self_verification"
    assert d["counts"]["by_disposition"] == {"accepted": 3, "rejected": 2, "human_review": 5}
    assert d["claim_precision"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert d["verifier_agreement"]["confusion_rows_gold_cols_verifier"] == [[1, 1, 0], [1, 1, 0], [1, 0, 1]]
    assert compute_metrics([]).to_dict()["claim_recall"]["value"] is None
