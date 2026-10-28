"""Local damage detection: the Ultralytics adapter and the provider that builds it."""

from app.services.detection.provider import (
    DetectorProvider,
    DetectorUnavailable,
    SettingsDetectorProvider,
    clear_detector_cache,
    detection_config_from_settings,
)
from app.services.detection.ultralytics_detector import UltralyticsDetector, detections_from_result

__all__ = [
    "DetectorProvider",
    "DetectorUnavailable",
    "SettingsDetectorProvider",
    "UltralyticsDetector",
    "clear_detector_cache",
    "detection_config_from_settings",
    "detections_from_result",
]
