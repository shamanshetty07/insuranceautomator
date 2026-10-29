"""Validated shapes for the JSONB `config` / `generation_config` columns.

Validate with these before writing; the database stores JSON as given. Model name
and version are NOT repeated here (they have dedicated columns), and neither are the
prediction IDs used by a report (see the report_predictions table).
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Standard YOLO detection models downsample by 32; Ultralytics silently rounds any
# other inference size up (and the recorded size would then be wrong), so it is
# rejected instead. A model with a larger stride (e.g. a P6 model, 64) is checked
# against its own stride when it loads (ultralytics_detector).
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
    """Installed package versions recorded on a run (name -> version; None if unknown).

    Written only by the server (detection_service); never part of DetectionConfig, so a
    client-supplied DetectionConfig cannot carry or forge it.
    """

    ultralytics: str | None = None
    torch: str | None = None


class StoredDetectionConfig(_DetectionParams):
    """What a DetectionRun stores: the inference parameters plus server-recorded runtime.

    `runtime` is optional so rows written before it existed still validate. Likewise
    image_size only has to be positive here: a historical run recorded with a size
    that is not a multiple of 32 stays readable. New runs get DetectionConfig's check.
    """

    runtime: RuntimeVersions | None = None


class SamplingParams(_Strict):
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    max_new_tokens: int = Field(gt=0)
    seed: int | None = None


class EvidenceSettings(_Strict):
    """Per-strategy settings; keys that do not apply to the strategy stay unset."""

    crop_padding: float | None = Field(default=None, ge=0, le=1)  # cropped_region: fraction of box size
    mask_opacity: float | None = Field(default=None, ge=0, le=1)  # image_with_segmentation_mask


class GenerationConfig(_Strict):
    prompt_version: str = Field(min_length=1)
    sampling: SamplingParams
    evidence_settings: EvidenceSettings = EvidenceSettings()


class VerificationConfig(_Strict):
    prompt_version: str = Field(min_length=1)
    sampling: SamplingParams
    policy_version: str | None = None  # claim_policy.POLICY_VERSION; set by the verification service
