from fastapi import APIRouter

from app.api.routes import health, images, inspections

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(inspections.router)
api_router.include_router(images.router)
