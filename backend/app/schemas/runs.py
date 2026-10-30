"""Validated shapes for the JSONB `config` / `generation_config` columns."""

import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


IMAGE_SIZE_STRIDE = 32


def check_image_size_multiple(value: int, stride: int = IMAGE_SIZE_STRIDE) -> int:
    """Return `value` unchanged if it is a positive multiple of `stride`; never rounds."""
    if value <= 0 or value % stride:
        raise ValueError(f"image_size must be a positive multiple of {stride}, got {value}")
    return value


class _DetectionParams(_Strict):
    confidence_threshold: float = Field(ge=0, le=1)
    iou_threshold: float = Field(ge=0, le=1)
    image_size: int = Field(gt=0)
    max_detections: int = Field(gt=0)
    class_filter: list[str] | None = None
    device: str | None = None


class DetectionConfig(_DetectionParams):
    """Inference parameters for a new YOLO run (image_size: positive multiple of 32)."""

    @field_validator("image_size")
    @classmethod
    def check_image_size(cls, v: int) -> int:
        return check_image_size_multiple(v)


class RuntimeVersions(_Strict):
    """Installed package versions recorded on a run (name -> version; None if unknown)."""

    ultralytics: str | None = None
    torch: str | None = None


class StoredDetectionConfig(_DetectionParams):
    """What a DetectionRun stores: the inference parameters plus server-recorded runtime."""

    runtime: RuntimeVersions | None = None


class SamplingParams(_Strict):
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    max_new_tokens: int = Field(gt=0)
    seed: int | None = None


class EvidenceSettings(_Strict):
    """Per-strategy settings; keys that do not apply to the strategy stay unset."""

    crop_padding: float | None = Field(default=None, ge=0, le=1)
    mask_opacity: float | None = Field(default=None, ge=0, le=1)


class GenerationConfig(_Strict):
    prompt_version: str = Field(min_length=1)
    sampling: SamplingParams
    evidence_settings: EvidenceSettings = EvidenceSettings()


class VerificationConfig(_Strict):
    prompt_version: str = Field(min_length=1)
    sampling: SamplingParams
    policy_version: str | None = None


class ModelRecord(_Strict):
    """What exactly ran: weights digest and installed package versions."""

    weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    packages: dict[str, str | None] = Field(default_factory=dict)


class VlmLimitsRecord(_Strict):
    """Limits that shape what the model saw or could answer (VLM_* settings)."""

    timeout_seconds: float | None = None
    max_new_tokens: int | None = None
    max_images: int | None = None
    max_image_pixels: int | None = None
    max_detections: int | None = None
    max_claims: int | None = None


class EvidenceInputs(_Strict):
    """The evidence handed to the generator, in prompt order."""

    image_ids: tuple[uuid.UUID, ...] = ()
    detection_run_ids: tuple[uuid.UUID, ...] = ()


class StoredGenerationConfig(GenerationConfig):
    """generated_reports.generation_config as written by the report service."""

    model: ModelRecord | None = None
    limits: VlmLimitsRecord | None = None
    evidence: EvidenceInputs | None = None


class StoredVerificationConfig(VerificationConfig):
    """verification_runs.config as written by the verification service."""

    model: ModelRecord | None = None
    limits: VlmLimitsRecord | None = None
    verifier_independent: bool | None = None
