"""Local vision-language model (Qwen-VL) adapters. See docs/architecture.md.

Importing this package never imports transformers or torch.
"""

from app.services.vlm.adapters import QwenClaimVerifier, QwenStructuredReportGenerator, VlmLimits
from app.services.vlm.errors import (
    VlmBusy,
    VlmError,
    VlmInferenceError,
    VlmInputError,
    VlmOutputError,
    VlmTimeout,
    VlmUnavailable,
)
from app.services.vlm.provider import SettingsVlmProvider, VlmStatus

__all__ = [
    "QwenClaimVerifier",
    "QwenStructuredReportGenerator",
    "SettingsVlmProvider",
    "VlmBusy",
    "VlmError",
    "VlmInferenceError",
    "VlmInputError",
    "VlmLimits",
    "VlmOutputError",
    "VlmStatus",
    "VlmTimeout",
    "VlmUnavailable",
]
