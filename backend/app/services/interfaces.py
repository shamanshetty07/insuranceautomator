"""Extension points for AI services. Phase 1 defines contracts only; no implementations.

Implementations (local YOLO / Qwen) arrive in later phases and must run locally.
"""

from typing import Protocol

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

    def detect(self, images: list[ImageRef]) -> list[Detection]:
        """Return detections across the given images (possibly empty)."""
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
