import logging

from fastapi import APIRouter, Response, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import DbSession
from app.schemas.common import DbHealthResponse, StatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=StatusResponse)
def api_health() -> StatusResponse:
    return StatusResponse(status="ok")


@router.get("/db", response_model=DbHealthResponse)
def db_health(db: DbSession, response: Response) -> DbHealthResponse:
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        logger.exception("Database health check failed")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return DbHealthResponse(status="unavailable", database="unreachable")
    return DbHealthResponse(status="ok", database="reachable")
