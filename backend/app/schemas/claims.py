"""Typed contracts for report claims and their evidence (data only; no behavior).

Enums here are shared by the claim policy (app.services.claim_policy). A Verdict is a
semantic judgement about a claim and never encodes operational routing such as
"needs human review", which is a separate flag.

Every model is frozen and every collection is a tuple, so a validated object cannot be
changed afterwards (e.g. by appending a duplicate id). JSON input and output still use
arrays. `model_copy(update=...)` and `model_construct` skip validation, as in all
Pydantic models; never use them on these contracts.
"""

import enum
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.damage import BoundingBox


class ClaimType(str, enum.Enum):
    """What kind of statement a report claim makes; decides what evidence could support it."""

    VISIBLE_DAMAGE_PRESENCE = "visible_damage_presence"  # damage exists at all in the photos
    DAMAGE_TYPE = "damage_type"  # dent, scratch, crack, ...
    DAMAGE_LOCATION = "damage_location"  # where on the vehicle or in the image
    DAMAGE_SEVERITY = "damage_severity"  # minor/moderate/severe or extent
    HIDDEN_STRUCTURAL_DAMAGE = "hidden_structural_damage"  # frame, mounts, anything not visible
    REPAIR_RECOMMENDATION = "repair_recommendation"  # repair vs replace, procedure
    REPAIR_COST = "repair_cost"  # any monetary estimate
    DAMAGE_CAUSE = "damage_cause"  # how or when the damage happened
    OTHER = "other"  # anything else; treated conservatively


class Verdict(str, enum.Enum):
    """Semantic outcome of checking one claim against the available evidence."""

    SUPPORTED = "supported"  # the evidence sufficiently supports the claim
    CONTRADICTED = "contradicted"  # the evidence clearly conflicts with the claim
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # the evidence cannot establish the claim


MAX_CLAIM_TEXT_LENGTH = 1000
MAX_CLAIMS_PER_REPORT = 100
MAX_REFERENCES_PER_CLAIM = 50

ClaimId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
# Same length limit as damage_predictions.damage_type (a detector class name).
DamageTypeName = Annotated[str, Field(min_length=1, max_length=100)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CropBox(BoundingBox):
    """BoundingBox that also rejects unknown keys (damage.BoundingBox ignores them)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceReference(_Strict):
    """One piece of evidence a claim points at: an image, optionally a prediction and a crop."""

    image_id: uuid.UUID  # InspectionImage id
    prediction_id: uuid.UUID | None = None  # DamagePrediction id
    crop: CropBox | None = None  # normalized to the stored image
    note: str | None = Field(default=None, max_length=500)

    @field_validator("note", mode="before")
    @classmethod
    def strip_note(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v

    @field_validator("note")
    @classmethod
    def note_not_empty(cls, v: str | None) -> str | None:
        if v is not None and not v:
            raise ValueError("note must not be empty when given")
        return v

    def identity(self) -> tuple[uuid.UUID, uuid.UUID | None, tuple[float, ...] | None]:
        """What the reference points at: image, prediction and crop. The note is not part of it."""
        crop = self.crop
        return (
            self.image_id,
            self.prediction_id,
            (crop.x_min, crop.y_min, crop.x_max, crop.y_max) if crop else None,
        )


class ExtractedClaim(_Strict):
    """One atomic statement from a report, with the evidence it cites.

    image_ids and prediction_ids must not contain duplicates (rejected, not silently
    deduplicated). Every evidence reference must be covered by them: its image_id must
    be in image_ids and its prediction_id (if any) in prediction_ids. Two evidence
    references with the same identity (image, prediction, crop) are duplicates, even if
    their notes differ. A claim that cites nothing has empty lists; that means "no
    evidence cited", not "no problem".

    damage_type is the claimed damage class (e.g. "dent") in the detector's vocabulary,
    used to compare a DAMAGE_TYPE claim with the detections it cites. Optional: without
    it a damage-type comparison is inconclusive, never a match.
    """

    claim_id: ClaimId  # report-local, e.g. "c1"
    text: str = Field(min_length=1, max_length=MAX_CLAIM_TEXT_LENGTH)
    claim_type: ClaimType
    damage_type: DamageTypeName | None = None
    image_ids: tuple[uuid.UUID, ...] = Field(default=(), max_length=MAX_REFERENCES_PER_CLAIM)
    prediction_ids: tuple[uuid.UUID, ...] = Field(default=(), max_length=MAX_REFERENCES_PER_CLAIM)
    evidence: tuple[EvidenceReference, ...] = Field(default=(), max_length=MAX_REFERENCES_PER_CLAIM)

    @field_validator("text", "damage_type", mode="before")
    @classmethod
    def strip_text(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def check_references(self) -> "ExtractedClaim":
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("image_ids must not contain duplicates")
        if len(set(self.prediction_ids)) != len(self.prediction_ids):
            raise ValueError("prediction_ids must not contain duplicates")
        identities = [ref.identity() for ref in self.evidence]
        if len(set(identities)) != len(identities):
            raise ValueError("evidence must not contain duplicate references (same image, prediction and crop)")
        images, predictions = set(self.image_ids), set(self.prediction_ids)
        for ref in self.evidence:
            if ref.image_id not in images:
                raise ValueError("evidence image_id must be listed in image_ids")
            if ref.prediction_id is not None and ref.prediction_id not in predictions:
                raise ValueError("evidence prediction_id must be listed in prediction_ids")
        return self


class ClaimExtractionResult(_Strict):
    """Structured claims extracted from ONE report; the report text stays in GeneratedReport.

    report_id is the GeneratedReport id, absent while the report row does not exist yet.
    An empty claims list means nothing was extracted; it must never be read as "the
    report found no problems" or "all claims are supported".
    """

    report_id: uuid.UUID | None = None
    claims: tuple[ExtractedClaim, ...] = Field(default=(), max_length=MAX_CLAIMS_PER_REPORT)

    @model_validator(mode="after")
    def check_unique_claim_ids(self) -> "ClaimExtractionResult":
        ids = [c.claim_id for c in self.claims]
        if len(set(ids)) != len(ids):
            raise ValueError("claim_id values must be unique within a result")
        return self


class EvidenceRecord(_Strict):
    """What evidence association found for one claim; stored in verification_results.evidence_reference.

    Values mirror claim_policy.EvidenceAssociation / EvidenceIntegrity / DetectionMatch
    (a test keeps them in sync). image_ids and prediction_ids list only references that
    resolved inside the report's own inspection; anything that did not resolve is named
    in `problems` (stable codes), never dropped silently. `detection_details` names the
    facts behind `detection_match` (stable codes). This is provenance, not a verdict: it
    says what was checked, not whether the claim is true.
    """

    association: Literal["associated", "none_cited", "unresolved"]
    integrity: Literal["unchecked", "ok", "missing", "mismatched", "corrupt"]
    detection_match: Literal[
        "matches_claim",
        "no_detection",
        "conflicts_with_claim",
        "inconclusive",
        "unavailable",
        "not_applicable",
    ]
    image_ids: tuple[uuid.UUID, ...] = Field(default=(), max_length=MAX_REFERENCES_PER_CLAIM)
    prediction_ids: tuple[uuid.UUID, ...] = Field(default=(), max_length=MAX_REFERENCES_PER_CLAIM)
    crops: tuple[EvidenceReference, ...] = Field(default=(), max_length=MAX_REFERENCES_PER_CLAIM)
    problems: tuple[str, ...] = Field(default=(), max_length=MAX_REFERENCES_PER_CLAIM)
    detection_details: tuple[str, ...] = Field(default=(), max_length=MAX_REFERENCES_PER_CLAIM)


class StructuredReport(_Strict):
    """A generator's output: the report text plus its atomic claims, produced together."""

    report_text: str = Field(min_length=1)
    claims: ClaimExtractionResult
