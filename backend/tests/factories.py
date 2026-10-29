import uuid

"""Small helpers for building valid model graphs in tests."""

from app.models import (
    DamagePrediction,
    DetectionRun,
    EvidenceStrategy,
    GeneratedReport,
    Inspection,
    InspectionImage,
    VerificationLabel,
    VerificationResult,
    VerificationRun,
)

SHA = "a" * 64


def make_image(inspection: Inspection, order: int = 0) -> InspectionImage:
    return InspectionImage(
        inspection=inspection,
        storage_path=f"{uuid.uuid4()}/{order}.jpg",
        original_filename="a.jpg",
        media_type="image/jpeg",
        file_size_bytes=10,
        image_order=order,
    )


def make_run(image: InspectionImage, version: str = "1", **kw) -> DetectionRun:
    return DetectionRun(inspection_image=image, model_name="yolo", model_version=version, **kw)


def make_prediction(run: DetectionRun, confidence: float = 0.9, damage_type: str = "dent") -> DamagePrediction:
    return DamagePrediction(detection_run=run, damage_type=damage_type, confidence=confidence)


def make_report(
    inspection: Inspection, strategy: EvidenceStrategy = EvidenceStrategy.FULL_IMAGE
) -> GeneratedReport:
    return GeneratedReport(
        inspection=inspection,
        content="c",
        model_name="qwen",
        model_version="1",
        evidence_strategy=strategy,
        generation_config={"prompt_version": "p1"},
    )


def make_vrun(report: GeneratedReport, version: str = "1") -> VerificationRun:
    return VerificationRun(generated_report=report, verifier_model="qwen", verifier_version=version)


def make_result(
    run: VerificationRun,
    claim: str = "claim",
    label: VerificationLabel | None = VerificationLabel.SUPPORTED,
    **kw,
) -> VerificationResult:
    """Extra keyword args (claim_id, claim_type, verdict, ...) pass through to the model."""
    return VerificationResult(
        verification_run=run, claim_text=claim, verification_label=label, explanation="e", **kw
    )
