from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    """Axis-aligned box normalized to image size; origin top-left, x right, y down."""

    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check_order(self) -> "BoundingBox":
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("bounding box must have x_min < x_max and y_min < y_max")
        return self
