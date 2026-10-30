"""Annotation manifest: images, claims and human gold labels for offline evaluation (data + pure helpers).

The manifest is the only source of gold labels. A label counts only when its annotation status
is accepted by the caller (adjudicated by default); everything else is excluded, never guessed.
Labels describe what the images show, never the physical state of the vehicle beyond them.

Fields added after the first release (provenance, privacy_redacted, visibility, annotations,
adjudication_method) are optional and omitted from serialization at their defaults, so manifests
that do not use them validate and digest exactly as before.
"""

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

from app.evaluation.records import GoldLabel
from app.schemas.claims import MAX_CLAIM_TEXT_LENGTH, MAX_REFERENCES_PER_CLAIM, ClaimId, ClaimType, CropBox
from app.services.evidence_association import normalize_damage_type

SCHEMA_VERSION = "1"
MAX_IMAGES = 10_000
MAX_CLAIMS = 100_000
MAX_ANNOTATORS_PER_CLAIM = 10

Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ImageQuality = Literal["ok", "blurry", "low_resolution", "occluded", "unknown"]
AnnotationStatus = Literal["unannotated", "draft", "adjudicated", "disputed"]
Visibility = Literal["clearly_visible", "partially_visible", "occluded", "out_of_frame", "not_assessable"]
DamageVisibility = Literal["clearly_visible", "partially_visible"]
SEEN_VISIBILITY: frozenset[str] = frozenset({"clearly_visible", "partially_visible"})
AdjudicationMethod = Literal["unanimous", "discussion", "third_annotator"]
SourceKind = Literal["own_photo", "contributed_with_consent", "licensed", "public_domain", "synthetic"]
RecordRef = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")]
LicenseId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,63}$")]
LABEL_BEARING_STATUSES: frozenset[str] = frozenset({"draft", "adjudicated"})
ALL_STATUSES: frozenset[str] = frozenset({"unannotated", "draft", "adjudicated", "disputed"})
AdversarialTag = Literal[
    "invented_damage",
    "wrong_location",
    "wrong_damage_type",
    "wrong_severity",
    "area_not_visible",
    "low_quality_evidence",
    "ambiguous",
    "not_visually_decidable",
    "fabricated_reference",
    "cross_inspection_reference",
]

SYNTHETIC_SOURCE_PREFIX = "synthetic:"
_SYNTHETIC_SOURCE = re.compile(r"^synthetic:[a-z0-9][a-z0-9_-]{0,63}$")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    _OMIT_AT_DEFAULT: ClassVar[tuple[str, ...]] = ()

    @model_serializer(mode="wrap")
    def _omit_new_defaults(self, handler: Any) -> Any:
        data = handler(self)
        for name in self._OMIT_AT_DEFAULT:
            if getattr(self, name) == type(self).model_fields[name].default:
                data.pop(name, None)
        return data


def _strip(v: object) -> object:
    return v.strip() if isinstance(v, str) else v


class VisibleDamage(_Strict):
    """One damage instance a human can see in the image. Location is free text (see docs vocabulary)."""

    _OMIT_AT_DEFAULT = ("visibility",)

    damage_type: str = Field(min_length=1, max_length=100)
    location: str = Field(min_length=1, max_length=200)
    bbox: CropBox | None = None
    visibility: DamageVisibility | None = None

    @field_validator("damage_type", "location", mode="before")
    @classmethod
    def strip_text(cls, v: object) -> object:
        return _strip(v)

    @field_validator("damage_type")
    @classmethod
    def normalize_type(cls, v: str) -> str:
        return normalize_damage_type(v)


class Provenance(_Strict):
    """Where an image came from and the record (kept outside the repository) that permits its use."""

    source_kind: SourceKind
    consent_or_license_ref: RecordRef | None = None
    license_id: LicenseId | None = None
    collected_at: date | None = None
    contributor: Slug | None = None

    @model_validator(mode="after")
    def check_kind(self) -> "Provenance":
        if self.source_kind == "licensed" and self.license_id is None:
            raise ValueError("a licensed image must name its license_id")
        if self.source_kind == "contributed_with_consent" and self.contributor is None:
            raise ValueError("a contributed image must name its pseudonymous contributor")
        return self


class ManifestImage(_Strict):
    """One image: either produced by a named synthetic generator or a file relative to the dataset root."""

    _OMIT_AT_DEFAULT = ("provenance", "privacy_redacted")

    fixture_id: Slug
    inspection_key: Slug
    source: str = Field(min_length=1, max_length=500)
    sha256: Sha256Hex | None = None
    visible_damage: tuple[VisibleDamage, ...] = Field(default=(), max_length=100)
    image_quality: ImageQuality = "unknown"
    provenance: Provenance | None = None
    privacy_redacted: bool | None = None

    @property
    def is_synthetic(self) -> bool:
        return self.source.startswith(SYNTHETIC_SOURCE_PREFIX)

    @field_validator("source")
    @classmethod
    def check_source(cls, v: str) -> str:
        if v.startswith(SYNTHETIC_SOURCE_PREFIX):
            if not _SYNTHETIC_SOURCE.fullmatch(v):
                raise ValueError("synthetic source must be 'synthetic:<generator-slug>'")
            return v
        if v != v.strip() or "\\" in v or "://" in v or "\x00" in v:
            raise ValueError("source must be a plain relative POSIX path")
        if v.startswith("/") or v.startswith("~") or _WINDOWS_DRIVE.match(v):
            raise ValueError("source must be relative to the dataset root, not absolute")
        parts = v.split("/")
        if any(p in ("", ".", "..") for p in parts):
            raise ValueError("source must not contain '..', '.' or empty path segments")
        return v

    @model_validator(mode="after")
    def check_digest(self) -> "ManifestImage":
        if not self.is_synthetic and self.sha256 is None:
            raise ValueError("sha256 is required for file sources (optional only for synthetic generators)")
        if self.provenance is not None and (self.provenance.source_kind == "synthetic") != self.is_synthetic:
            raise ValueError("provenance source_kind 'synthetic' is required for, and only for, synthetic: sources")
        return self


def _check_visibility(label: GoldLabel | None, visibility: str | None, notes: str | None, where: str) -> None:
    """A label may only say what the images show: supported/contradicted need the relevant region seen."""
    if label is None or visibility is None or label is GoldLabel.UNVERIFIABLE:
        return
    if visibility not in SEEN_VISIBILITY:
        raise ValueError(f"{where}: label '{label.value}' requires a visible region, not '{visibility}'")
    if label is GoldLabel.CONTRADICTED and visibility == "partially_visible" and notes is None:
        raise ValueError(f"{where}: 'contradicted' on a partially visible region needs notes explaining what is seen")


class AnnotatorLabel(_Strict):
    """One annotator's independent judgement of a claim, made blind to other labels and system outputs."""

    annotator: Slug
    label: GoldLabel | None
    visibility: Visibility | None = None
    annotated_at: datetime | None = None
    notes: str | None = Field(default=None, min_length=1, max_length=2000)

    @field_validator("notes", mode="before")
    @classmethod
    def strip_notes(cls, v: object) -> object:
        return _strip(v)

    @model_validator(mode="after")
    def check_visibility(self) -> "AnnotatorLabel":
        _check_visibility(self.label, self.visibility, self.notes, f"annotator {self.annotator}")
        return self


class Annotation(_Strict):
    """Who labelled the claim and how final it is. Annotator ids are pseudonymous slugs, never emails or names.

    For adjudicated claims `annotator` is the adjudicator.
    """

    _OMIT_AT_DEFAULT = ("adjudication_method",)

    annotator: Slug | None = None
    status: AnnotationStatus
    annotated_at: datetime | None = None
    notes: str | None = Field(default=None, max_length=2000)
    adjudication_method: AdjudicationMethod | None = None

    @field_validator("notes", mode="before")
    @classmethod
    def strip_notes(cls, v: object) -> object:
        return _strip(v)

    @model_validator(mode="after")
    def check_annotator(self) -> "Annotation":
        if self.status != "unannotated" and self.annotator is None:
            raise ValueError("annotator is required unless status is 'unannotated'")
        if self.notes == "":
            raise ValueError("notes must not be empty when given")
        if self.adjudication_method is not None and self.status != "adjudicated":
            raise ValueError("adjudication_method is only allowed when status is 'adjudicated'")
        return self


class ManifestClaim(_Strict):
    """One report claim to evaluate, the images it cites, and its (optional) gold label."""

    _OMIT_AT_DEFAULT = ("visibility", "annotations")

    claim_key: ClaimId
    inspection_key: Slug
    text: str = Field(min_length=1, max_length=MAX_CLAIM_TEXT_LENGTH)
    claim_type: ClaimType
    damage_type: str | None = Field(default=None, min_length=1, max_length=100)
    cited_fixture_ids: tuple[ClaimId, ...] = Field(default=(), max_length=MAX_REFERENCES_PER_CLAIM)
    adversarial_tag: AdversarialTag | None = None
    label: GoldLabel | None = None
    annotation: Annotation
    visibility: Visibility | None = None
    annotations: tuple[AnnotatorLabel, ...] = Field(default=(), max_length=MAX_ANNOTATORS_PER_CLAIM)

    @field_validator("text", "damage_type", mode="before")
    @classmethod
    def strip_text(cls, v: object) -> object:
        return _strip(v)

    @field_validator("damage_type")
    @classmethod
    def normalize_type(cls, v: str | None) -> str | None:
        return normalize_damage_type(v) if v is not None else None

    @model_validator(mode="after")
    def check_label(self) -> "ManifestClaim":
        if len(set(self.cited_fixture_ids)) != len(self.cited_fixture_ids):
            raise ValueError("cited_fixture_ids must not contain duplicates")
        status = self.annotation.status
        if self.label is not None and status not in LABEL_BEARING_STATUSES:
            raise ValueError(f"label must be null when annotation status is '{status}'")
        if status == "adjudicated" and self.label is None:
            raise ValueError("an adjudicated claim must carry a label")
        _check_visibility(self.label, self.visibility, self.annotation.notes, f"claim {self.claim_key}")
        self._check_annotations()
        return self

    def _check_annotations(self) -> None:
        ids = [a.annotator for a in self.annotations]
        if len(set(ids)) != len(ids):
            raise ValueError("each annotator may label a claim only once")
        if self.annotations and self.annotation.status == "unannotated":
            raise ValueError("a claim with annotator labels cannot have status 'unannotated'")
        method = self.annotation.adjudication_method
        if self.annotation.status == "adjudicated" and len(ids) >= 2 and method is None:
            raise ValueError("an adjudicated claim with two or more annotator labels must state adjudication_method")
        if method == "unanimous":
            labels = {a.label for a in self.annotations}
            if len(ids) < 2 or labels != {self.label}:
                raise ValueError("'unanimous' needs two or more annotator labels all equal to the adjudicated label")
        if method == "third_annotator" and self.annotation.annotator in ids:
            raise ValueError("a 'third_annotator' adjudicator must not be one of the claim's annotators")


class Manifest(_Strict):
    """A versioned evaluation dataset. dataset_kind must be stated; synthetic results must be labelled as such."""

    schema_version: Literal["1"]
    dataset_version: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    dataset_kind: Literal["synthetic", "real"]
    description: str = Field(min_length=1, max_length=5000)
    created: date
    images: tuple[ManifestImage, ...] = Field(max_length=MAX_IMAGES)
    claims: tuple[ManifestClaim, ...] = Field(max_length=MAX_CLAIMS)

    @property
    def is_synthetic(self) -> bool:
        return self.dataset_kind == "synthetic"

    @property
    def result_label(self) -> str:
        """Tag for reports: results on synthetic manifests are not evidence about real-world accuracy."""
        if self.is_synthetic:
            return f"SYNTHETIC dataset {self.dataset_version} (labels are not human ground truth)"
        return f"real dataset {self.dataset_version}"

    def image_inspections(self) -> dict[str, str]:
        return {img.fixture_id: img.inspection_key for img in self.images}

    @model_validator(mode="after")
    def check_references(self) -> "Manifest":
        fixture_ids = [img.fixture_id for img in self.images]
        if len(set(fixture_ids)) != len(fixture_ids):
            raise ValueError("fixture_id values must be unique")
        keys = [c.claim_key for c in self.claims]
        if len(set(keys)) != len(keys):
            raise ValueError("claim_key values must be unique")
        inspection_of_digest: dict[str, str] = {}
        for img in self.images:
            if img.sha256 is not None:
                other = inspection_of_digest.setdefault(img.sha256, img.inspection_key)
                if other != img.inspection_key:
                    raise ValueError(f"image {img.fixture_id}: the same file appears in inspections {other} and "
                                     f"{img.inspection_key}, which would leak across splits")
            if not self.is_synthetic:
                _check_real_image(img)
            if self.is_synthetic and not img.is_synthetic:
                raise ValueError(f"synthetic manifest image {img.fixture_id} must use a synthetic: source")
            if not self.is_synthetic and img.is_synthetic:
                raise ValueError(f"real manifest image {img.fixture_id} must not use a synthetic: source")
        owner = self.image_inspections()
        inspections = set(owner.values())
        for claim in self.claims:
            if claim.inspection_key not in inspections:
                raise ValueError(f"claim {claim.claim_key} references unknown inspection_key {claim.inspection_key}")
            unknown = [f for f in claim.cited_fixture_ids if f not in owner]
            foreign = [f for f in claim.cited_fixture_ids if f in owner and owner[f] != claim.inspection_key]
            if unknown and claim.adversarial_tag != "fabricated_reference":
                raise ValueError(
                    f"claim {claim.claim_key} cites unknown fixture ids {unknown}; "
                    "only allowed with adversarial_tag 'fabricated_reference'"
                )
            if foreign and claim.adversarial_tag != "cross_inspection_reference":
                raise ValueError(
                    f"claim {claim.claim_key} cites images of another inspection {foreign}; "
                    "only allowed with adversarial_tag 'cross_inspection_reference'"
                )
        return self


def _check_real_image(img: ManifestImage) -> None:
    prov = img.provenance
    if prov is None or prov.source_kind == "synthetic" or prov.consent_or_license_ref is None:
        raise ValueError(
            f"real manifest image {img.fixture_id} needs provenance with a non-synthetic source_kind "
            "and a consent_or_license_ref"
        )
    if img.privacy_redacted is not True:
        raise ValueError(f"real manifest image {img.fixture_id} must be privacy_redacted (plates, faces, people)")


def gold_labels(
    manifest: Manifest, *, require_status: tuple[str, ...] = ("adjudicated",)
) -> dict[str, GoldLabel | None]:
    """claim_key -> label for the metrics join; claims whose status is not accepted map to None (excluded)."""
    accepted = frozenset(require_status)
    if not accepted:
        raise ValueError("require_status must name at least one status")
    if not accepted <= ALL_STATUSES:
        raise ValueError(f"unknown annotation status in {sorted(accepted - ALL_STATUSES)}")
    return {c.claim_key: (c.label if c.annotation.status in accepted else None) for c in manifest.claims}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant not allowed: {name}")


def parse_manifest(text: str) -> Manifest:
    """Strict parse: duplicate keys, NaN/Infinity, unknown fields and type coercion are all rejected."""
    json.loads(text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    return Manifest.model_validate_json(text, strict=True)


def load_manifest(path: str | Path) -> Manifest:
    return parse_manifest(Path(path).read_text(encoding="utf-8"))


def canonical_json(manifest: Manifest) -> str:
    return json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def manifest_digest(manifest: Manifest) -> str:
    """SHA-256 of the canonical (key-sorted) JSON; record it with every evaluation run."""
    return hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ManifestSplit:
    """Disjoint dev/test partitions at inspection granularity (all claims of one inspection share a split)."""

    dev_inspections: tuple[str, ...]
    test_inspections: tuple[str, ...]
    dev_claims: tuple[str, ...]
    test_claims: tuple[str, ...]


def split_of(inspection_key: str, *, test_fraction: float, salt: str) -> Literal["dev", "test"]:
    """Deterministic, order-independent assignment by hashing the inspection key."""
    h = hashlib.sha256(f"{salt}\x00{inspection_key}".encode()).digest()
    bucket = int.from_bytes(h[:8], "big") / 2**64
    return "test" if bucket < test_fraction else "dev"


def split_by_inspection(manifest: Manifest, *, test_fraction: float = 0.3, salt: str = "") -> ManifestSplit:
    """Split by inspection to prevent leakage: images and claims of one vehicle never straddle dev and test."""
    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError("test_fraction must be within [0, 1]")
    inspections = sorted({img.inspection_key for img in manifest.images})
    assign = {k: split_of(k, test_fraction=test_fraction, salt=salt) for k in inspections}
    return ManifestSplit(
        dev_inspections=tuple(k for k in inspections if assign[k] == "dev"),
        test_inspections=tuple(k for k in inspections if assign[k] == "test"),
        dev_claims=tuple(c.claim_key for c in manifest.claims if assign[c.inspection_key] == "dev"),
        test_claims=tuple(c.claim_key for c in manifest.claims if assign[c.inspection_key] == "test"),
    )


@dataclass(frozen=True)
class AgreementInputs:
    """Paired independent labels of two annotators; feed `confusion` to metrics.cohens_kappa.

    Rows are annotator_a, columns annotator_b, both in GoldLabel declaration order. Claims where
    either annotator gave no label (absent or abstained) are listed in `excluded`, not counted.
    """

    annotator_a: str
    annotator_b: str
    pairs: tuple[tuple[str, GoldLabel, GoldLabel], ...]
    excluded: tuple[str, ...]

    @property
    def confusion(self) -> tuple[tuple[int, ...], ...]:
        order = list(GoldLabel)
        counts = [[0] * len(order) for _ in order]
        for _, a, b in self.pairs:
            counts[order.index(a)][order.index(b)] += 1
        return tuple(tuple(row) for row in counts)

    @property
    def observed_agreement(self) -> float | None:
        return sum(a is b for _, a, b in self.pairs) / len(self.pairs) if self.pairs else None


def annotator_ids(manifest: Manifest) -> tuple[str, ...]:
    return tuple(sorted({a.annotator for c in manifest.claims for a in c.annotations}))


def agreement_inputs(
    manifest: Manifest, annotator_a: str, annotator_b: str, *, claim_keys: Iterable[str] | None = None
) -> AgreementInputs:
    """Pre-adjudication label pairs for inter-annotator agreement, optionally restricted to some claims."""
    if annotator_a == annotator_b:
        raise ValueError("agreement needs two different annotators")
    wanted = None if claim_keys is None else frozenset(claim_keys)
    pairs: list[tuple[str, GoldLabel, GoldLabel]] = []
    excluded: list[str] = []
    for claim in manifest.claims:
        if wanted is not None and claim.claim_key not in wanted:
            continue
        by = {a.annotator: a.label for a in claim.annotations}
        a, b = by.get(annotator_a), by.get(annotator_b)
        if a is None or b is None:
            excluded.append(claim.claim_key)
        else:
            pairs.append((claim.claim_key, a, b))
    return AgreementInputs(annotator_a, annotator_b, tuple(pairs), tuple(excluded))
