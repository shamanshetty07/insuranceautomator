"""Claim-level evaluation metrics over labelled outcomes; pure, stdlib only, never imputes labels."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from app.evaluation.records import Configuration, Disposition, GoldLabel, LabelledOutcome
from app.schemas.claims import Verdict

GOLD_TO_VERDICT: Mapping[GoldLabel, Verdict] = {
    GoldLabel.SUPPORTED: Verdict.SUPPORTED,
    GoldLabel.CONTRADICTED: Verdict.CONTRADICTED,
    GoldLabel.UNVERIFIABLE: Verdict.INSUFFICIENT_EVIDENCE,
}
VERDICT_ORDER: tuple[Verdict, ...] = (
    Verdict.SUPPORTED,
    Verdict.CONTRADICTED,
    Verdict.INSUFFICIENT_EVIDENCE,
)
UNSUPPORTED_GOLD = frozenset({GoldLabel.CONTRADICTED, GoldLabel.UNVERIFIABLE})
WILSON_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Ratio:
    """A count ratio; value is None when the denominator is zero (undefined, never imputed)."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator < 0 or self.denominator < 0 or self.numerator > self.denominator:
            raise ValueError(f"invalid ratio {self.numerator}/{self.denominator}")

    @property
    def value(self) -> float | None:
        return None if self.denominator == 0 else self.numerator / self.denominator

    def wilson_95(self) -> tuple[float, float] | None:
        """Wilson score 95% interval for the proportion; None when the denominator is zero."""
        n = self.denominator
        if n == 0:
            return None
        z = WILSON_Z_95
        p = self.numerator / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        low = 0.0 if self.numerator == 0 else max(0.0, centre - half)
        high = 1.0 if self.numerator == n else min(1.0, centre + half)
        return (low, high)

    def to_dict(self) -> dict[str, int | float | None]:
        return {"numerator": self.numerator, "denominator": self.denominator, "value": self.value}

    def __str__(self) -> str:
        v = self.value
        return f"{self.numerator}/{self.denominator} ({'undefined' if v is None else f'{v:.3f}'})"


def _ratio(
    items: Sequence[LabelledOutcome],
    num: Callable[[LabelledOutcome], bool],
    den: Callable[[LabelledOutcome], bool],
) -> Ratio:
    pool = [i for i in items if den(i)]
    return Ratio(sum(1 for i in pool if num(i)), len(pool))


@dataclass(frozen=True)
class VerifierAgreement:
    """Raw verifier verdict vs human gold (mapped to Verdict); rows = gold, columns = verifier."""

    n: int
    agreement: Ratio
    confusion: tuple[tuple[int, ...], ...]
    kappa: float | None

    def to_dict(self) -> dict[str, object]:
        labels = [v.value for v in VERDICT_ORDER]
        return {
            "n": self.n,
            "agreement": self.agreement.to_dict(),
            "labels": labels,
            "confusion_rows_gold_cols_verifier": [list(row) for row in self.confusion],
            "kappa": self.kappa,
        }


def cohens_kappa(confusion: Sequence[Sequence[int]]) -> float | None:
    """Cohen's kappa for a square confusion matrix; None when empty or expected agreement is 1."""
    n = sum(sum(row) for row in confusion)
    if n == 0:
        return None
    k = len(confusion)
    observed = sum(confusion[i][i] for i in range(k)) / n
    rows = [sum(confusion[i]) for i in range(k)]
    cols = [sum(confusion[r][j] for r in range(k)) for j in range(k)]
    expected = sum(rows[i] * cols[i] for i in range(k)) / (n * n)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def verifier_agreement(items: Sequence[LabelledOutcome]) -> VerifierAgreement:
    """Agreement over labelled outcomes that have a raw verifier verdict; independent of policy."""
    index = {v: i for i, v in enumerate(VERDICT_ORDER)}
    matrix = [[0] * len(VERDICT_ORDER) for _ in VERDICT_ORDER]
    for item in items:
        raw = item.outcome.raw_verifier_verdict
        if item.gold is None or raw is None:
            continue
        matrix[index[GOLD_TO_VERDICT[item.gold]]][index[raw]] += 1
    n = sum(sum(row) for row in matrix)
    agree = sum(matrix[i][i] for i in range(len(matrix)))
    return VerifierAgreement(
        n=n,
        agreement=Ratio(agree, n),
        confusion=tuple(tuple(row) for row in matrix),
        kappa=cohens_kappa(matrix),
    )


@dataclass(frozen=True)
class MetricsReport:
    """Metrics for one configuration; claim metrics use labelled claims, rates over all outcomes say so."""

    configuration: Configuration | None
    total: int
    labelled: int
    unlabelled: int
    by_disposition: Mapping[Disposition, int]
    by_gold: Mapping[GoldLabel, int]
    verifier_called: int
    claim_precision: Ratio
    claim_recall: Ratio
    unsupported_acceptance_rate: Ratio
    false_rejection_rate: Ratio
    human_review_rate: Ratio
    invalid_output_rate: Ratio
    verifier_failure_rate: Ratio
    invalid_output_rejection_rate: Ratio
    evidence_integrity_failure_rate: Ratio
    verifier_agreement: VerifierAgreement

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration": None if self.configuration is None else self.configuration.value,
            "counts": {
                "total": self.total,
                "labelled": self.labelled,
                "unlabelled": self.unlabelled,
                "verifier_called": self.verifier_called,
                "by_disposition": {d.value: self.by_disposition.get(d, 0) for d in Disposition},
                "by_gold": {g.value: self.by_gold.get(g, 0) for g in GoldLabel},
            },
            "claim_precision": self.claim_precision.to_dict(),
            "claim_recall": self.claim_recall.to_dict(),
            "unsupported_acceptance_rate": self.unsupported_acceptance_rate.to_dict(),
            "false_rejection_rate": self.false_rejection_rate.to_dict(),
            "human_review_rate": self.human_review_rate.to_dict(),
            "invalid_output_rate": self.invalid_output_rate.to_dict(),
            "verifier_failure_rate": self.verifier_failure_rate.to_dict(),
            "invalid_output_rejection_rate": self.invalid_output_rejection_rate.to_dict(),
            "evidence_integrity_failure_rate": self.evidence_integrity_failure_rate.to_dict(),
            "verifier_agreement": self.verifier_agreement.to_dict(),
        }


def _accepted(i: LabelledOutcome) -> bool:
    return i.outcome.disposition is Disposition.ACCEPTED


def _gold_supported(i: LabelledOutcome) -> bool:
    return i.gold is GoldLabel.SUPPORTED


def _gold_unsupported(i: LabelledOutcome) -> bool:
    return i.gold in UNSUPPORTED_GOLD


def _check(items: Sequence[LabelledOutcome]) -> Configuration | None:
    keys = [i.outcome.claim_key for i in items]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"duplicate claim_key values: {dupes}")
    configs = {i.outcome.configuration for i in items}
    if len(configs) > 1:
        raise ValueError(f"mixed configurations in one call: {sorted(c.value for c in configs)}")
    return next(iter(configs), None)


def compute_metrics(items: Sequence[LabelledOutcome]) -> MetricsReport:
    """Compute every metric for one configuration; rejects duplicate claim keys and mixed configurations."""
    configuration = _check(items)
    labelled = [i for i in items if i.gold is not None]
    called = [i for i in items if i.outcome.verifier_called]
    invalid = [i for i in items if i.outcome.verifier_output_invalid]
    return MetricsReport(
        configuration=configuration,
        total=len(items),
        labelled=len(labelled),
        unlabelled=len(items) - len(labelled),
        by_disposition={d: sum(1 for i in items if i.outcome.disposition is d) for d in Disposition},
        by_gold={g: sum(1 for i in labelled if i.gold is g) for g in GoldLabel},
        verifier_called=len(called),
        claim_precision=_ratio(labelled, _gold_supported, _accepted),
        claim_recall=_ratio(labelled, _accepted, _gold_supported),
        unsupported_acceptance_rate=_ratio(labelled, _accepted, _gold_unsupported),
        false_rejection_rate=_ratio(
            labelled, lambda i: i.outcome.disposition is Disposition.REJECTED, _gold_supported
        ),
        human_review_rate=_ratio(
            items, lambda i: i.outcome.disposition is Disposition.HUMAN_REVIEW, lambda i: True
        ),
        invalid_output_rate=_ratio(called, lambda i: i.outcome.verifier_output_invalid, lambda i: True),
        verifier_failure_rate=_ratio(called, lambda i: i.outcome.verifier_failed, lambda i: True),
        invalid_output_rejection_rate=_ratio(invalid, lambda i: not _accepted(i), lambda i: True),
        evidence_integrity_failure_rate=_ratio(items, lambda i: i.outcome.integrity_failure, lambda i: True),
        verifier_agreement=verifier_agreement(labelled),
    )


def compare(by_config: Mapping[Configuration, Sequence[LabelledOutcome]]) -> dict[Configuration, MetricsReport]:
    """Per-configuration reports; requires identical claim keys and identical gold labels across configurations."""
    reference: dict[str, GoldLabel | None] | None = None
    for config, items in by_config.items():
        wrong = {i.outcome.configuration for i in items} - {config}
        if wrong:
            raise ValueError(f"outcomes under {config.value} belong to {sorted(c.value for c in wrong)}")
        _check(items)
        labels = {i.outcome.claim_key: i.gold for i in items}
        if reference is None:
            reference = labels
            continue
        if labels.keys() != reference.keys():
            diff = sorted(labels.keys() ^ reference.keys())
            raise ValueError(f"{config.value} is not evaluated on the same claims; differing keys: {diff}")
        changed = sorted(k for k in labels if labels[k] != reference[k])
        if changed:
            raise ValueError(f"{config.value} has different gold labels for claims: {changed}")
    return {config: compute_metrics(items) for config, items in by_config.items()}
