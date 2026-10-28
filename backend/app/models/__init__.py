"""Import all models so Base.metadata (used by Alembic) sees every table."""

from app.models.damage_prediction import DamagePrediction
from app.models.generated_report import GeneratedReport
from app.models.inspection import Inspection, InspectionStatus
from app.models.inspection_image import InspectionImage
from app.models.verification_result import VerificationLabel, VerificationResult

__all__ = [
    "DamagePrediction",
    "GeneratedReport",
    "Inspection",
    "InspectionImage",
    "InspectionStatus",
    "VerificationLabel",
    "VerificationResult",
]
