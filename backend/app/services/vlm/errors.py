"""Errors raised by the local vision-language model (VLM) adapters."""


class VlmError(RuntimeError):
    code = "vlm_error"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class VlmUnavailable(VlmError):
    """No model can run (disabled, misconfigured, package missing, failed or refused load)."""

    code = "vlm_unavailable"


class VlmBusy(VlmError):
    """Another inference still holds the model (e.g. one that timed out is still running)."""

    code = "vlm_busy"


class VlmTimeout(VlmError):
    """Inference did not finish within VLM_TIMEOUT_SECONDS."""

    code = "vlm_timeout"


class VlmInferenceError(VlmError):
    """The model runner raised; the original exception is chained and logged."""

    code = "vlm_inference_error"


class VlmInputError(VlmError):
    """The evidence cannot be given to the model as-is (too many images, uncitable detection, ...)."""

    code = "vlm_input_error"


class VlmOutputError(VlmError):
    """The model's output was rejected; `code` says why (not_json, schema, unknown_alias, ...)."""

    code = "vlm_output_error"
