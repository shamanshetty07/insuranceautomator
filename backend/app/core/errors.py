import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.common import ErrorBody, ErrorResponse, FieldError

logger = logging.getLogger(__name__)


def _response(status_code: int, code: str, message: str, details: list[FieldError] | None = None):
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=body.model_dump(exclude_none=True))


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        details = [
            FieldError(field=".".join(str(p) for p in e["loc"]), message=e["msg"])
            for e in exc.errors()
        ]
        return _response(422, "validation_error", "Request validation failed.", details)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return _response(500, "internal_error", "An unexpected error occurred.")

