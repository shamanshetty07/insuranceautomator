from fastapi import APIRouter

from app.api.routes import detections, health, images, inspections, reports

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(inspections.router)
api_router.include_router(images.router)
api_router.include_router(detections.router)
api_router.include_router(reports.router)
