import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, computed_field


class ImageRead(BaseModel):
    """Public image metadata. The storage path is deliberately not exposed."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    inspection_id: uuid.UUID
    original_filename: str
    media_type: str
    file_size_bytes: int
    sha256: str | None
    width_px: int | None
    height_px: int | None
    exif_orientation: int | None
    image_order: int
    created_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_url(self) -> str:
        return f"/api/v1/inspections/{self.inspection_id}/images/{self.id}/content"


class ImageList(BaseModel):
    items: list[ImageRead]
    total: int
