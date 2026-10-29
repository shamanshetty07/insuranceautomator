"""Validated shapes for the JSONB `config` / `generation_config` columns.

Validate with these before writing; the database stores JSON as given. Model name
and version are NOT repeated here (they have dedicated columns), and neither are the
prediction IDs used by a report (see the report_predictions table).
"""

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectionConfig(_Strict):
    """Inference parameters for a YOLO run."""

    confidence_threshold: float = Field(ge=0, le=1)
    iou_threshold: float = Field(ge=0, le=1)
    image_size: int = Field(gt=0)
    max_detections: int = Field(gt=0)
    class_filter: list[str] | None = None
    device: str | None = None


class RuntimeVersions(_Strict):
    """Installed package versions recorded on a run (name -> version; None if unknown).

    Written only by the server (detection_service); never part of DetectionConfig, so a
    client-supplied DetectionConfig cannot carry or forge it.
    """

    ultralytics: str | None = None
    torch: str | None = None


class StoredDetectionConfig(DetectionConfig):
    """What a DetectionRun stores: the inference parameters plus server-recorded runtime.

    `runtime` is optional so rows written before it existed still validate.
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
