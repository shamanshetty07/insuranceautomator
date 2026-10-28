"""Shared value types for AI service interfaces (data only, no behavior)."""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageRef:
    image_id: uuid.UUID
    path: Path


@dataclass(frozen=True)
class Detection:
    """One detected damage instance; coordinates follow app.schemas.damage.BoundingBox."""

    image_id: uuid.UUID
    damage_type: str
    confidence: float
    bounding_box: dict[str, float] | None = None
    segmentation_data: list[Any] | None = None


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    # Lowercase hex SHA-256 of the weights file, when the model loads one.
    weights_sha256: str | None = None


@dataclass(frozen=True)
class ReportClaim:
    text: str


@dataclass(frozen=True)
class ClaimVerdict:
    claim_text: str
    label: str  # a VerificationLabel value
    explanation: str
    evidence_reference: dict[str, Any] | None = None
    verifier: ModelInfo | None = None


@dataclass(frozen=True)
class EvidenceBundle:
    """What a report generator or verifier is allowed to see."""

    images: list[ImageRef]
    detections: list[Detection] = field(default_factory=list)
