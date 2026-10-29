"""Extension points for AI services. Implementations must run locally.

DamageDetectionService has a local Ultralytics implementation (app.services.detection);
the others arrive in later phases.
"""

from typing import Protocol

from app.schemas.claims import StructuredReport
from app.schemas.runs import DetectionConfig
from app.services.types import (
    ClaimVerdict,
    Detection,
    EvidenceBundle,
    ImageRef,
    ModelInfo,
    ReportClaim,
)


class DamageDetectionService(Protocol):
    model: ModelInfo
    # The inference parameters actually applied; recorded on every DetectionRun.
    config: DetectionConfig

    def detect(self, images: list[ImageRef]) -> list[Detection]:
        """Return detections across the given images (possibly empty).

        Boxes follow app.schemas.damage.BoundingBox, relative to the stored image.
        """
        ...


class VisionAnalysisService(Protocol):
    model: ModelInfo

    def describe(self, images: list[ImageRef], prompt: str) -> str:
        """Return free-text visual analysis of the images for the prompt."""
        ...


class ReportGenerationService(Protocol):
    model: ModelInfo

    def generate(self, evidence: EvidenceBundle) -> str:
        """Return an insurance damage report grounded in the evidence."""
        ...


class ClaimVerificationService(Protocol):
    model: ModelInfo

    def extract_claims(self, report_text: str) -> list[ReportClaim]:
        """Split a report into individually verifiable claims."""
        ...

    def verify(self, claims: list[ReportClaim], evidence: EvidenceBundle) -> list[ClaimVerdict]:
        """Label each claim against the evidence."""
        ...

    def find_omissions(self, report_text: str, evidence: EvidenceBundle) -> list[ClaimVerdict]:
        """Compare images to the report to find damage the report omits (label omitted_damage)."""
        ...


class StructuredReportGenerationService(Protocol):
    """Generates a report and its atomic claims in one call (no separate extraction model)."""

    model: ModelInfo

    def generate_structured(self, evidence: EvidenceBundle) -> StructuredReport:
        """Return the report text and the structured claims, citing only ids in the evidence."""
        ...
