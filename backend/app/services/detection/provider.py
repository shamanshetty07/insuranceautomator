"""Builds the configured detector from Settings and caches it per process."""

import importlib.util
import logging
import threading
from typing import Protocol

from app.core.config import Settings
from app.schemas.detection import DetectorStatusRead
from app.schemas.runs import DetectionConfig
from app.services.detection.ultralytics_detector import (
    UltralyticsDetector,
    UnsafeLoadRefused,
    is_weights_file,
)
from app.services.interfaces import DamageDetectionService

logger = logging.getLogger(__name__)


class DetectorUnavailable(Exception):
    """No detector can run. The message is client-safe (no paths or internals)."""


class DetectorProvider(Protocol):
    def status(self) -> DetectorStatusRead:
        """Report availability without loading a model that is not loaded yet."""
        ...

    def get(self) -> DamageDetectionService:
        """Return a ready detector, loading it if needed; raise DetectorUnavailable."""
        ...


def detection_config_from_settings(settings: Settings) -> DetectionConfig:
    return DetectionConfig(
        confidence_threshold=settings.detector_confidence_threshold,
        iou_threshold=settings.detector_iou_threshold,
        image_size=settings.detector_image_size,
        max_detections=settings.detector_max_detections,
        device=settings.detector_device,
    )


_load_lock = threading.Lock()
_loaded: dict[tuple[object, ...], UltralyticsDetector] = {}


def clear_detector_cache() -> None:
    with _load_lock:
        _loaded.clear()


class SettingsDetectorProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._config = detection_config_from_settings(settings)

    def _key(self) -> tuple[object, ...]:
        s = self._settings
        path = s.detector_weights_path.resolve() if s.detector_weights_path else None
        return (
            path,
            s.detector_model_name,
            s.detector_model_version,
            self._config.model_dump_json(),
            s.detector_safe_load,
            s.detector_weights_sha256,
        )

    def _unavailable_reason(self) -> str | None:
        s = self._settings
        if s.detector_backend == "none":
            return "Damage detection is disabled (DETECTOR_BACKEND=none)."
        if not s.detector_model_name or not s.detector_model_version:
            return "DETECTOR_MODEL_NAME and DETECTOR_MODEL_VERSION must be set."
        if s.detector_weights_path is None or not is_weights_file(s.detector_weights_path):
            return "The configured weights file is missing, is not a regular file, or is not a .pt file."
        if importlib.util.find_spec("ultralytics") is None:
            return "The ultralytics package is not installed (install the 'detection' extra)."
        return None

    def status(self) -> DetectorStatusRead:
        s = self._settings
        reason = self._unavailable_reason()
        if reason is not None:
            return DetectorStatusRead(available=False, backend=s.detector_backend, reason=reason)
        loaded = _loaded.get(self._key())
        return DetectorStatusRead(
            available=True,
            backend=s.detector_backend,
            model_name=s.detector_model_name,
            model_version=s.detector_model_version,
            weights_sha256=loaded.model.weights_sha256 if loaded else None,
            classes=loaded.classes if loaded else None,
            config=self._config,
        )

    def get(self) -> DamageDetectionService:
        reason = self._unavailable_reason()
        if reason is not None:
            raise DetectorUnavailable(reason)
        s = self._settings
        assert s.detector_weights_path and s.detector_model_name and s.detector_model_version
        key = self._key()
        with _load_lock:
            detector = _loaded.get(key)
            if detector is None:
                try:
                    detector = UltralyticsDetector(
                        s.detector_weights_path, s.detector_model_name, s.detector_model_version, self._config,
                        safe_load=s.detector_safe_load, expected_sha256=s.detector_weights_sha256,
                    )
                except UnsafeLoadRefused:
                    logger.exception("Refused to load the detection model")
                    raise DetectorUnavailable(
                        "The detection model was refused by a configured safeguard (see the server log)."
                    ) from None
                except Exception:
                    logger.exception("Failed to load the detection model")
                    raise DetectorUnavailable("The detection model failed to load.") from None
                _loaded[key] = detector
        return detector
