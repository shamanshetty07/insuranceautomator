import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.inspection import InspectionStatus
from app.schemas.detection import DetectionRunRead
from app.schemas.report import ReportDetailRead


class InspectionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=5000)
    vehicle_make: str | None = Field(default=None, max_length=100)
    vehicle_model: str | None = Field(default=None, max_length=100)
    vehicle_year: int | None = Field(default=None, ge=1886, le=2100)
    vehicle_registration: str | None = Field(default=None, max_length=32)
    vehicle_color: str | None = Field(default=None, max_length=50)

    @field_validator("title", "notes", "vehicle_make", "vehicle_model", "vehicle_registration", "vehicle_color")
    @classmethod
    def blank_to_none(cls, value: str | None) -> str | None:
        """Trim whitespace; treat blank strings as not provided."""
        if value is None:
            return None
        value = value.strip()
        return value or None


class InspectionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    notes: str | None
    vehicle_make: str | None = None
    vehicle_model: str | None = None
    vehicle_year: int | None = None
    vehicle_registration: str | None = None
    vehicle_color: str | None = None
    status: InspectionStatus
    created_at: datetime
    updated_at: datetime


class InspectionList(BaseModel):
    items: list[InspectionRead]
    total: int
    limit: int
    offset: int


class InspectionWorkflowRead(BaseModel):
    """Result of a synchronous complete inspection run."""

    inspection: InspectionRead
    detections: list[DetectionRunRead]
    report: ReportDetailRead
