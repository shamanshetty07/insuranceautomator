"""Import all models so Base.metadata (used by Alembic) sees every table."""

from app.models.damage_prediction import DamagePrediction
from app.models.detection_run import DetectionRun
from app.models.generated_report import EvidenceStrategy, GeneratedReport
from app.models.inspection import Inspection, InspectionStatus
from app.models.inspection_image import InspectionImage
from app.models.report_claim import ReportClaim, ReportClaimImage
from app.models.report_prediction import ReportPrediction
from app.models.run_status import RunStatus
from app.models.verification_result import VerificationLabel, VerificationResult
from app.models.verification_run import VerificationRun

__all__ = [
    "DamagePrediction",
    "DetectionRun",
    "EvidenceStrategy",
    "GeneratedReport",
    "Inspection",
    "InspectionImage",
    "InspectionStatus",
    "ReportClaim",
    "ReportClaimImage",
    "ReportPrediction",
    "RunStatus",
    "VerificationLabel",
    "VerificationResult",
    "VerificationRun",
]
