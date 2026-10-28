import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.body_limit import BodySizeLimitMiddleware
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.db.session import get_engine
from app.schemas.common import StatusResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # The engine connects lazily; startup does not require the database to be up.
    logger.info("Starting application")
    yield
    get_engine().dispose()
    logger.info("Application stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    # Added first so CORS (added next) wraps it: its 413/411 then carry CORS headers.
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    register_exception_handlers(app)
    app.include_router(api_router)

    @app.get("/health", response_model=StatusResponse, tags=["health"])
    def health() -> StatusResponse:
        """Process liveness only. Use /api/v1/health/db for database connectivity."""
        return StatusResponse(status="ok")

    return app


app = create_app()
