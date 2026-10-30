"""The seam between the adapters and an actual model: a runner turns one request into text."""

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from app.services.types import ModelInfo
from app.services.vlm.errors import VlmBusy, VlmInferenceError, VlmOutputError, VlmTimeout

logger = logging.getLogger(__name__)

_STOP_MARGIN_SECONDS = 5.0


@dataclass(frozen=True)
class VlmRequest:
    system: str
    user_text: str
    images: tuple[Image.Image, ...]
    max_new_tokens: int
    max_time_seconds: float


class VlmRunner(Protocol):
    info: ModelInfo
    runtime: Mapping[str, str | None]

    def generate(self, request: VlmRequest) -> str:
        """Return the model's raw text output (greedy decoding; deterministic for fixed inputs)."""
        ...


# Models are not safe for concurrent use: one inference per process.
_INFERENCE_LOCK = threading.Lock()


def call_runner(
    runner: VlmRunner,
    request: VlmRequest,
    timeout_seconds: float,
    lock: threading.Lock = _INFERENCE_LOCK,
) -> str:
    """Run one inference with a lock and a hard wait limit; return its raw text."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not lock.acquire(timeout=timeout_seconds):
        raise VlmBusy("The vision-language model is busy; try again later.")

    outcome: dict[str, object] = {}

    def work() -> None:
        try:
            outcome["text"] = runner.generate(request)
        except BaseException as exc:  # noqa: BLE001 - re-raised as VlmInferenceError below
            outcome["error"] = exc
        finally:
            lock.release()

    thread = threading.Thread(target=work, name="vlm-inference", daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        logger.error("VLM inference exceeded %.1f s; still running in the background", timeout_seconds)
        raise VlmTimeout("The vision-language model did not finish in time.")
    if "error" in outcome:
        logger.error("VLM inference failed", exc_info=outcome["error"])  # type: ignore[arg-type]
        raise VlmInferenceError("The vision-language model failed.") from outcome["error"]  # type: ignore[misc]
    text = outcome.get("text")
    if not isinstance(text, str):
        raise VlmOutputError("The vision-language model returned no text.", code="empty")
    return text


def max_time_for(timeout_seconds: float) -> float:
    """Generation time budget passed to the runner, leaving a margin for decoding/cleanup."""
    return max(1.0, timeout_seconds - _STOP_MARGIN_SECONDS)
