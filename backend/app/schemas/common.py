from pydantic import BaseModel


class FieldError(BaseModel):
    field: str
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[FieldError] | None = None


class ErrorResponse(BaseModel):
    """Every non-2xx response from the API has this shape."""

    error: ErrorBody


class StatusResponse(BaseModel):
    status: str


class DbHealthResponse(BaseModel):
    status: str
    database: str
