import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.models.run_status import RunStatus
from app.schemas.damage import BoundingBox
from app.schemas.runs import DetectionConfig, StoredDetectionConfig


class DetectorStatusRead(BaseModel):
    """Whether detection can run. Never includes filesystem paths."""

    available: bool
    backend: Literal["none", "ultralytics"]
    model_name: str | None = None
    model_version: str | None = None
    # weights_sha256 and classes are known only once the model has been loaded.
    weights_sha256: str | None = None
    classes: list[str] | None = None
    config: DetectionConfig | None = None
    reason: str | None = None


class DetectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # None/omitted = every image of the inspection.
    image_ids: list[uuid.UUID] | None = Field(default=None, min_length=1, max_length=100)


class PredictionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    damage_type: str
    confidence: float
    # Normalized to the stored image: 0-1, origin top-left, y down.
    bounding_box: BoundingBox | None


class DetectionRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    inspection_image_id: uuid.UUID
    model_name: str
    model_version: str
    weights_sha256: str | None
    config: StoredDetectionConfig
    status: RunStatus
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None
    predictions: list[PredictionRead]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_ms(self) -> int | None:
        if self.completed_at is None:
            return None
        return max(0, round((self.completed_at - self.created_at).total_seconds() * 1000))

    @classmethod
    def from_run(cls, run: object) -> "DetectionRunRead":
        read = cls.model_validate(run)
        read.predictions.sort(key=lambda p: p.confidence, reverse=True)
        return read


class DetectionRunList(BaseModel):
    items: list[DetectionRunRead]
