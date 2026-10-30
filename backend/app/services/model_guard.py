"""Checks on model services before they touch stored data."""

from collections.abc import Mapping
from typing import Any

from app.core.config import get_settings
from app.core.errors import AppError
from app.services.types import ModelInfo


def is_stub(service: object) -> bool:
    model = getattr(service, "model", None)
    name = getattr(model, "name", "") or ""
    return bool(getattr(service, "IS_STUB", False)) or name.startswith("stub-")


def refuse_stub_in_production(service: object, unavailable_code: str) -> None:
    if is_stub(service) and get_settings().app_env == "production":
        raise AppError(503, unavailable_code, "No real model is configured (development stubs are refused).")


def verifier_is_independent(
    generator_name: str,
    generator_version: str,
    generation_config: Mapping[str, Any] | None,
    verifier: ModelInfo,
) -> bool:
    """False when the verifier is, or may be, the generator's model."""
    model = (generation_config or {}).get("model") or {}
    generator_sha = model.get("weights_sha256") if isinstance(model, Mapping) else None
    if generator_sha and verifier.weights_sha256:
        return generator_sha != verifier.weights_sha256
    return (generator_name, generator_version) != (verifier.name, verifier.version)
