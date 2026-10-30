"""Checks on model services before they touch stored data."""

import re
from collections.abc import Mapping
from typing import Any

from app.core.config import get_settings
from app.core.errors import AppError
from app.services.types import ModelInfo

_SHA256 = re.compile(r"[0-9a-f]{64}")


def is_stub(service: object) -> bool:
    model = getattr(service, "model", None)
    name = getattr(model, "name", "") or ""
    return bool(getattr(service, "IS_STUB", False)) or name.startswith("stub-")


def refuse_stub_in_production(service: object, unavailable_code: str) -> None:
    if is_stub(service) and get_settings().app_env == "production":
        raise AppError(503, unavailable_code, "No real model is configured (development stubs are refused).")


def _digest(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if _SHA256.fullmatch(value) else None


def generator_digest(generation_config: Mapping[str, Any] | None) -> str | None:
    """The generator's weights digest as stored in generated_reports.generation_config, if valid."""
    model = generation_config.get("model") if isinstance(generation_config, Mapping) else None
    return _digest(model.get("weights_sha256")) if isinstance(model, Mapping) else None


def verifier_is_independent(
    generator_name: str,
    generator_version: str,
    generation_config: Mapping[str, Any] | None,
    verifier: ModelInfo,
) -> bool:
    """True only when the verifier is shown to run different weights from the generator.

    Requires both weight digests (well-formed SHA-256) and that they differ, and also that the
    configured (name, version) differ. Anything unknown is "not shown independent", so the policy
    sends the claim to review. Names alone never establish independence: they are free-text
    settings. Equal (name, version) with different digests is treated as the same model, because
    the digest covers every file in the model directory (tokenizer, config, a re-download), so it
    can change while the weights do not.

    Limitation: different digests and names do not prove independent judgement. A fine-tune or
    quantisation of the same base model, or two models trained on the same data, share errors.
    """
    return models_independent(
        ModelInfo(generator_name, generator_version, generator_digest(generation_config)), verifier
    )


def models_independent(a: ModelInfo, b: ModelInfo) -> bool:
    """The rule of `verifier_is_independent` for two model identities (symmetric)."""
    a_sha = _digest(getattr(a, "weights_sha256", None))
    b_sha = _digest(getattr(b, "weights_sha256", None))
    if a_sha is None or b_sha is None or a_sha == b_sha:
        return False
    return (a.name, a.version) != (b.name, b.version)
