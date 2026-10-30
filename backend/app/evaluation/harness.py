"""Replay recorded per-claim verifier answers through the three configurations and score them."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.manifest import Manifest, ManifestClaim, gold_labels, manifest_digest
from app.evaluation.metrics import MetricsReport, compare
from app.evaluation.records import (
    ClaimOutcome,
    Configuration,
    baseline_outcome,
    join_labels,
    outcome_from_decision,
)
from app.schemas.claims import ExtractedClaim, Verdict
from app.services import verification_service as vs
from app.services.claim_policy import (
    POLICY_VERSION,
    ClaimPolicyDecision,
    DetectionMatch,
    EvidenceAssociation,
    EvidenceIntegrity,
    ReasonCode,
    VerifierOutcome,
    evaluate_claim,
)
from app.services.evidence_association import ClaimEvidence
from app.services.types import ClaimVerdict, ModelInfo


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Trial(_Strict):
    """What the pipeline observed for one claim; raw_label is the verifier's answer as returned, or null."""

    integrity: EvidenceIntegrity
    detection_match: DetectionMatch
    raw_label: str | None = Field(default=None, max_length=64)
    verifier_errored: bool = False


class TrialSet(_Strict):
    """Recorded verifier answers for a manifest; `source` says whether they came from a real model."""

    manifest_dataset_version: str
    source: str = Field(min_length=1, max_length=200)
    synthetic: bool
    verifier_independent: bool
    trials: dict[str, Trial]


def load_trials(path: str | Path) -> TrialSet:
    return TrialSet.model_validate_json(Path(path).read_text(encoding="utf-8"), strict=False)


def association_for(manifest: Manifest, claim: ManifestClaim) -> EvidenceAssociation:
    """Same rule as production: no citation, or any citation outside the claim's inspection, fails association."""
    if not claim.cited_fixture_ids:
        return EvidenceAssociation.NONE_CITED
    owner = manifest.image_inspections()
    if any(owner.get(f) != claim.inspection_key for f in claim.cited_fixture_ids):
        return EvidenceAssociation.UNRESOLVED
    return EvidenceAssociation.ASSOCIATED


class _Replay:
    """A verifier that returns the recorded answer, so production's `_ask_verifier` mapping is reused."""

    model = ModelInfo("recorded", "0")

    def __init__(self, trial: Trial) -> None:
        self._trial = trial

    def verify(self, claims, evidence):  # noqa: ANN001, ANN201
        if self._trial.verifier_errored:
            raise RuntimeError("recorded verifier error")
        return [ClaimVerdict(claim_text=claims[0].text, label=self._trial.raw_label, explanation="")]


def _extracted(claim: ManifestClaim) -> ExtractedClaim:
    return ExtractedClaim(claim_id=claim.claim_key, text=claim.text, claim_type=claim.claim_type,
                          damage_type=claim.damage_type)


def _evidence(claim: ManifestClaim, association: EvidenceAssociation, trial: Trial) -> ClaimEvidence:
    return ClaimEvidence(claim.claim_key, association, trial.integrity, trial.detection_match, (), (), (), ())


_SELF_REASON = {
    Verdict.SUPPORTED: ReasonCode.VERIFIER_SUPPORTED,
    Verdict.CONTRADICTED: ReasonCode.VERIFIER_CONTRADICTED,
    Verdict.INSUFFICIENT_EVIDENCE: ReasonCode.VERIFIER_INSUFFICIENT,
}


def _self_verification(key: str, raw: "vs._Raw") -> ClaimOutcome:
    """Raw verdict only, no evidence checks or policy gate; the verifier is asked about every claim."""
    verdict = raw.outcome.verdict if isinstance(raw.outcome.verdict, Verdict) else None
    if raw.errored:
        reason = ReasonCode.VERIFIER_ERROR
    elif verdict is None:
        reason = ReasonCode.VERIFIER_INVALID
    else:
        reason = _SELF_REASON[verdict]
    final = verdict if verdict in (Verdict.SUPPORTED, Verdict.CONTRADICTED) else Verdict.INSUFFICIENT_EVIDENCE
    decision = ClaimPolicyDecision(final, final is Verdict.INSUFFICIENT_EVIDENCE, (reason,), "")
    return outcome_from_decision(key, Configuration.SELF_VERIFICATION, decision, verdict)


def run_configurations(
    manifest: Manifest, trials: TrialSet, *, force_independent: bool = False
) -> dict[Configuration, list[ClaimOutcome]]:
    """Paired outcomes: every configuration sees the same claims and the same recorded verifier answers."""
    if trials.manifest_dataset_version != manifest.dataset_version:
        raise ValueError("the trials were recorded for a different manifest version")
    keys = [c.claim_key for c in manifest.claims]
    if set(trials.trials) != set(keys):
        differing = sorted(set(trials.trials) ^ set(keys))
        raise ValueError(f"trials must cover exactly the manifest claims; differing: {differing}")
    independent = trials.verifier_independent or force_independent
    out: dict[Configuration, list[ClaimOutcome]] = {c: [] for c in Configuration}
    for claim in manifest.claims:
        trial = trials.trials[claim.claim_key]
        extracted = _extracted(claim)
        ev = _evidence(claim, association_for(manifest, claim), trial)
        asked = vs._ask_verifier(_Replay(trial), extracted, ev)
        out[Configuration.BASELINE].append(baseline_outcome(claim.claim_key))
        out[Configuration.SELF_VERIFICATION].append(_self_verification(claim.claim_key, asked))
        skipped = vs._decided_without_verifier(extracted, ev)
        outcome = VerifierOutcome(verdict=None) if skipped else asked.outcome
        decision = evaluate_claim(vs._policy_input(extracted, ev, outcome, independent))
        raw_verdict = None if skipped or not isinstance(asked.outcome.verdict, Verdict) else asked.outcome.verdict
        out[Configuration.HYBRID].append(
            outcome_from_decision(claim.claim_key, Configuration.HYBRID, decision, raw_verdict, not skipped)
        )
    return out


@dataclass(frozen=True)
class EvaluationResult:
    result_label: str
    synthetic: bool
    manifest_digest: str
    policy_version: str
    trials_source: str
    verifier_independent: bool
    forced_independent: bool
    reports: Mapping[Configuration, MetricsReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_label": self.result_label,
            "synthetic": self.synthetic,
            "manifest_digest": self.manifest_digest,
            "policy_version": self.policy_version,
            "trials_source": self.trials_source,
            "verifier_independent": self.verifier_independent,
            "forced_independent": self.forced_independent,
            "reports": {c.value: r.to_dict() for c, r in self.reports.items()},
        }


def evaluate(
    manifest: Manifest,
    trials: TrialSet,
    *,
    require_status: tuple[str, ...] = ("adjudicated",),
    force_independent: bool = False,
) -> EvaluationResult:
    """Score all three configurations; results are SYNTHETIC unless both manifest and trials are real."""
    gold = gold_labels(manifest, require_status=require_status)
    outcomes = run_configurations(manifest, trials, force_independent=force_independent)
    synthetic = manifest.is_synthetic or trials.synthetic
    label = manifest.result_label if manifest.is_synthetic else (
        f"SYNTHETIC verifier answers ({trials.source})" if trials.synthetic else manifest.result_label
    )
    if force_independent:
        label += "; COUNTERFACTUAL: verifier independence forced"
    return EvaluationResult(
        result_label=label,
        synthetic=synthetic,
        manifest_digest=manifest_digest(manifest),
        policy_version=POLICY_VERSION,
        trials_source=trials.source,
        verifier_independent=trials.verifier_independent,
        forced_independent=force_independent,
        reports=compare({c: join_labels(o, gold) for c, o in outcomes.items()}),
    )


def result_json(result: EvaluationResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True)


__all__ = ["Trial", "TrialSet", "evaluate", "load_trials", "result_json", "run_configurations"]
