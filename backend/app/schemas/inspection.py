import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.inspection import InspectionStatus


class InspectionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=5000)

    @field_validator("title", "notes")
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
    status: InspectionStatus
    created_at: datetime
    updated_at: datetime


class InspectionList(BaseModel):
    items: list[InspectionRead]
    total: int
    limit: int
    offset: int
