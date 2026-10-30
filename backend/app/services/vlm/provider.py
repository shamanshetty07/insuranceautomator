"""Builds the configured VLM adapters from Settings; one loaded model per process."""

import importlib.util
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import Settings
from app.services.vlm.adapters import QwenClaimVerifier, QwenStructuredReportGenerator, VlmLimits
from app.services.vlm.errors import VlmUnavailable
from app.services.vlm.runner import VlmRunner
from app.services.vlm.transformers_runner import RunnerRefused, TransformersQwenRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VlmStatus:
    available: bool
    backend: str
    reason: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    weights_sha256: str | None = None


RunnerFactory = Callable[[Settings], VlmRunner]


def _transformers_runner(s: Settings) -> VlmRunner:
    assert s.vlm_model_path and s.vlm_model_name and s.vlm_model_version
    return TransformersQwenRunner(
        s.vlm_model_path, s.vlm_model_name, s.vlm_model_version,
        device=s.vlm_device, dtype=s.vlm_dtype, expected_sha256=s.vlm_weights_sha256,
    )


_load_lock = threading.Lock()
_loaded: dict[tuple[object, ...], VlmRunner] = {}


def clear_vlm_cache() -> None:
    with _load_lock:
        _loaded.clear()


def limits_from_settings(s: Settings) -> VlmLimits:
    return VlmLimits(
        timeout_seconds=s.vlm_timeout_seconds,
        max_new_tokens=s.vlm_max_new_tokens,
        max_images=s.vlm_max_images,
        max_image_pixels=s.vlm_max_image_pixels,
        max_detections=s.vlm_max_detections,
        max_claims=s.vlm_max_claims,
    )


class SettingsVlmProvider:
    def __init__(self, settings: Settings, runner_factory: RunnerFactory = _transformers_runner) -> None:
        self._settings = settings
        self._factory = runner_factory
        self._limits = limits_from_settings(settings)

    def _key(self) -> tuple[object, ...]:
        s = self._settings
        path = s.vlm_model_path.resolve() if s.vlm_model_path else None
        return (path, s.vlm_model_name, s.vlm_model_version, s.vlm_weights_sha256, s.vlm_device, s.vlm_dtype)

    def _unavailable_reason(self) -> str | None:
        s = self._settings
        if s.vlm_backend == "none":
            return "The vision-language model is disabled (VLM_BACKEND=none)."
        if not s.vlm_model_name or not s.vlm_model_version:
            return "VLM_MODEL_NAME and VLM_MODEL_VERSION must be set."
        if s.vlm_model_path is None or not s.vlm_model_path.is_dir():
            return "The configured model directory is missing."
        for package in ("transformers", "torch"):
            if importlib.util.find_spec(package) is None:
                return f"The {package} package is not installed (install the 'vlm' extra)."
        return None

    def status(self) -> VlmStatus:
        s = self._settings
        reason = self._unavailable_reason()
        if reason is not None:
            return VlmStatus(available=False, backend=s.vlm_backend, reason=reason)
        loaded = _loaded.get(self._key())
        return VlmStatus(
            available=True,
            backend=s.vlm_backend,
            model_name=s.vlm_model_name,
            model_version=s.vlm_model_version,
            weights_sha256=loaded.info.weights_sha256 if loaded else None,
        )

    def _runner(self) -> VlmRunner:
        reason = self._unavailable_reason()
        if reason is not None:
            raise VlmUnavailable(reason)
        key = self._key()
        with _load_lock:
            runner = _loaded.get(key)
            if runner is None:
                try:
                    runner = self._factory(self._settings)
                except RunnerRefused:
                    logger.exception("Refused to load the vision-language model")
                    raise VlmUnavailable(
                        "The vision-language model was refused by a configured safeguard (see the server log)."
                    ) from None
                except Exception:
                    logger.exception("Failed to load the vision-language model")
                    raise VlmUnavailable("The vision-language model failed to load.") from None
                _loaded[key] = runner
        return runner

    def get_generator(self) -> QwenStructuredReportGenerator:
        return QwenStructuredReportGenerator(self._runner(), self._limits)

    def get_verifier(self) -> QwenClaimVerifier:
        return QwenClaimVerifier(self._runner(), self._limits)
