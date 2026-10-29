"""DEVELOPMENT/TEST-ONLY deterministic test doubles. These are NOT AI.

Nothing here inspects pixels or uses a model; outputs are fixed functions of the inputs.
These stubs must never be wired into production code paths. Every ModelInfo name starts
with "stub-" and every class sets IS_STUB = True so they are easy to detect and refuse.
"""

import re
import uuid
from collections.abc import Collection, Mapping, Sequence

from pydantic import ValidationError

from app.schemas.claims import (
    ClaimExtractionResult,
    ClaimType,
    EvidenceReference,
    ExtractedClaim,
    StructuredReport,
    CropBox,
)
from app.services.types import ClaimVerdict, EvidenceBundle, ModelInfo, ReportClaim


class StubVerifierError(RuntimeError):
    """Raised by StubClaimVerifier.verify for claims configured in raise_on."""


class StubStructuredReportGenerator:
    """Deterministic test double (not AI); never wire into production.

    Implements StructuredReportGenerationService and ReportGenerationService.
    """

    IS_STUB = True
    model = ModelInfo("stub-generator", "0")

    def __init__(
        self,
        prediction_ids_by_image: Mapping[uuid.UUID, Sequence[uuid.UUID]] | None = None,
        include_unverifiable_claims: bool = True,
    ) -> None:
        self.prediction_ids_by_image = dict(prediction_ids_by_image or {})
        self.include_unverifiable_claims = include_unverifiable_claims

    def generate_structured(self, evidence: EvidenceBundle) -> StructuredReport:
        if not evidence.images:
            return StructuredReport(
                report_text="No images were provided.", claims=ClaimExtractionResult(claims=[])
            )
        image_ids = [i.image_id for i in evidence.images]
        index = {iid: n + 1 for n, iid in enumerate(image_ids)}
        claims: list[ExtractedClaim] = []

        def add(text, ctype, ids, evidence_refs=(), preds=()):
            claims.append(
                ExtractedClaim(
                    claim_id=f"c{len(claims) + 1}",
                    text=text,
                    claim_type=ctype,
                    image_ids=list(ids),
                    prediction_ids=list(preds),
                    evidence=list(evidence_refs),
                )
            )

        for iid in dict.fromkeys(image_ids):
            add(
                f"Visible damage is present in image {index[iid]}.",
                ClaimType.VISIBLE_DAMAGE_PRESENCE,
                [iid],
                [EvidenceReference(image_id=iid)],
            )
        seen: dict[uuid.UUID, int] = {}
        for det in evidence.detections:
            if det.image_id not in index:
                continue  # never cite ids outside the evidence
            k = seen.get(det.image_id, 0)
            seen[det.image_id] = k + 1
            crop = None
            if det.bounding_box:
                try:
                    crop = CropBox(**det.bounding_box)
                except (ValidationError, TypeError):
                    crop = None
            pids = self.prediction_ids_by_image.get(det.image_id, ())
            pid = pids[k] if k < len(pids) else None
            add(
                f"Image {index[det.image_id]} shows a {det.damage_type}.",
                ClaimType.DAMAGE_TYPE,
                [det.image_id],
                [EvidenceReference(image_id=det.image_id, prediction_id=pid, crop=crop)],
                [pid] if pid else [],
            )
        if self.include_unverifiable_claims:
            first = image_ids[0]
            add("The damage is severe.", ClaimType.DAMAGE_SEVERITY, [first],
                [EvidenceReference(image_id=first)])
            add("The frame behind the damaged panel may also be damaged.",
                ClaimType.HIDDEN_STRUCTURAL_DAMAGE, [first], [EvidenceReference(image_id=first)])
            add("Repair would cost about 1200.", ClaimType.REPAIR_COST, [first],
                [EvidenceReference(image_id=first)])
        text = "\n".join(c.text for c in claims)
        return StructuredReport(report_text=text, claims=ClaimExtractionResult(claims=claims))

    def generate(self, evidence: EvidenceBundle) -> str:
        return self.generate_structured(evidence).report_text


class StubClaimVerifier:
    """Deterministic test double (not AI); never wire into production.

    Implements ClaimVerificationService. Records every call in `calls`.
    """

    IS_STUB = True
    model = ModelInfo("stub-verifier", "0")

    def __init__(
        self,
        default_label: str = "supported",
        labels: Mapping[str, str] | None = None,
        raise_on: Collection[str] = (),
        empty_for: Collection[str] = (),
        explanations: Mapping[str, str] | None = None,
        extra_for: Collection[str] = (),
    ) -> None:
        self.default_label = default_label
        self.labels = dict(labels or {})
        self.raise_on = set(raise_on)
        self.empty_for = set(empty_for)
        self.explanations = dict(explanations or {})
        self.extra_for = set(extra_for)
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def extract_claims(self, report_text: str) -> list[ReportClaim]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", report_text)
        return [ReportClaim(p.strip()) for p in parts if p.strip()]

    def verify(self, claims: list[ReportClaim], evidence: EvidenceBundle) -> list[ClaimVerdict]:
        ids = tuple(str(i.image_id) for i in evidence.images)
        self.calls.append((tuple(c.text for c in claims), tuple(str(i.image_id) for i in evidence.images)))
        out: list[ClaimVerdict] = []
        for c in claims:
            if c.text in self.raise_on:
                raise StubVerifierError(f"configured failure for claim: {c.text}")
            if c.text in self.empty_for:
                continue
            v = ClaimVerdict(
                claim_text=c.text,
                label=self.labels.get(c.text, self.default_label),
                explanation=self.explanations.get(c.text, "Stub verdict (not AI)."),
                evidence_reference={"image_ids": list(ids)},
                verifier=self.model,
            )
            out.append(v)
            if c.text in self.extra_for:
                out.append(v)
        return out

    def find_omissions(self, report_text: str, evidence: EvidenceBundle) -> list[ClaimVerdict]:
        return []
