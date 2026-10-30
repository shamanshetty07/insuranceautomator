"""The seam between the adapters and an actual model: a runner turns one request into text.

Adapters (prompting, parsing, validation) never import a model library; a runner does.
Tests inject a fake runner. `call_runner` adds the bounded wait every caller needs:
one inference at a time per process, and a timeout.
"""

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from app.services.types import ModelInfo
from app.services.vlm.errors import VlmBusy, VlmInferenceError, VlmOutputError, VlmTimeout

logger = logging.getLogger(__name__)

# Seconds kept back from the timeout for the runner's own cooperative stop (max_time).
_STOP_MARGIN_SECONDS = 5.0


@dataclass(frozen=True)
class VlmRequest:
    system: str
    user_text: str
    images: tuple[Image.Image, ...]  # RGB, already size-limited, in prompt order
    max_new_tokens: int
    max_time_seconds: float  # the runner should stop generating after this long


class VlmRunner(Protocol):
    info: ModelInfo
    # Installed package versions (name -> version or None), recorded with results.
    runtime: Mapping[str, str | None]

    def generate(self, request: VlmRequest) -> str:
        """Return the model's raw text output (greedy decoding; deterministic for fixed inputs)."""
        ...


# Models are large and not safe for concurrent use; one inference per process.
_INFERENCE_LOCK = threading.Lock()


def call_runner(
    runner: VlmRunner,
    request: VlmRequest,
    timeout_seconds: float,
    lock: threading.Lock = _INFERENCE_LOCK,
) -> str:
    """Run one inference with a lock and a hard wait limit; return its raw text.

    A Python thread cannot be killed: on timeout the inference keeps running in a daemon
    thread (holding the lock until it ends; real runners also get max_time so generation
    stops on its own) and VlmTimeout is raised. Meanwhile other calls wait up to
    `timeout_seconds` for the lock and then raise VlmBusy instead of piling up.
    """
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
