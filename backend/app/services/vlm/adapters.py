"""Qwen-VL adapters for the existing service protocols (app.services.interfaces).

- QwenStructuredReportGenerator: StructuredReportGenerationService (+ ReportGenerationService).
  Sees the original images and the detector's output; returns claims citing them.
- QwenClaimVerifier: ClaimVerificationService.verify. Sees ONE claim's text and that
  claim's original images only, never the report or the detections, and refuses
  evidence that carries detections (independence from the generator's inputs).

Both share one runner (one model in memory). Neither knows about the database; the
verification service and a future report service own persistence. Model output is an
untrusted prediction: the claim policy decides what it means.
"""

import uuid
from collections.abc import Sequence

from PIL import Image

from app.schemas.claims import MAX_CLAIMS_PER_REPORT, StructuredReport
from app.schemas.runs import GenerationConfig, SamplingParams, VerificationConfig
from app.services.detection.ultralytics_detector import to_rgb
from app.services.types import ClaimVerdict, EvidenceBundle, ImageRef, ModelInfo, ReportClaim
from app.services.vlm import prompts
from app.services.vlm.errors import VlmInputError
from app.services.vlm.parsing import parse_structured_report, parse_verdict
from app.services.vlm.runner import VlmRequest, VlmRunner, call_runner, max_time_for

VERIFIER_MAX_NEW_TOKENS = 256


class VlmLimits:
    """Per-call limits, normally from Settings (VLM_*)."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 180,
        max_new_tokens: int = 1024,
        max_images: int = 6,
        max_image_pixels: int = 802_816,
        max_detections: int = 30,
        max_claims: int = 20,
    ) -> None:
        if not 0 < max_claims <= MAX_CLAIMS_PER_REPORT:
            raise ValueError(f"max_claims must be between 1 and {MAX_CLAIMS_PER_REPORT}")
        if min(timeout_seconds, max_new_tokens, max_images, max_image_pixels, max_detections) <= 0:
            raise ValueError("limits must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_new_tokens = max_new_tokens
        self.max_images = max_images
        self.max_image_pixels = max_image_pixels
        self.max_detections = max_detections
        self.max_claims = max_claims


def load_image(ref: ImageRef, max_pixels: int) -> Image.Image:
    """Stored original -> RGB, downscaled (never upscaled) to at most max_pixels."""
    with Image.open(ref.path) as img:
        rgb = to_rgb(img)
    rgb.load()
    w, h = rgb.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return rgb


def _images(refs: Sequence[ImageRef], limits: VlmLimits) -> tuple[Image.Image, ...]:
    if not refs:
        raise VlmInputError("No images were provided.", code="no_images")
    if len(refs) > limits.max_images:
        raise VlmInputError("Too many images for the vision-language model.", code="too_many_images")
    try:
        return tuple(load_image(ref, limits.max_image_pixels) for ref in refs)
    except OSError:
        raise VlmInputError("An image could not be read.", code="image_unreadable") from None


class QwenStructuredReportGenerator:
    """Report text + structured claims in one call; the report text IS the claims."""

    def __init__(self, runner: VlmRunner, limits: VlmLimits) -> None:
        self._runner = runner
        self._limits = limits
        self.model: ModelInfo = runner.info

    @property
    def generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            prompt_version=prompts.GENERATOR_PROMPT_VERSION,
            sampling=SamplingParams(temperature=0, top_p=1, max_new_tokens=self._limits.max_new_tokens),
        )

    def generate_structured(self, evidence: EvidenceBundle) -> StructuredReport:
        if len(evidence.detections) > self._limits.max_detections:
            raise VlmInputError("Too many detections for the vision-language model.", code="too_many_detections")
        try:
            aliases = prompts.build_alias_map(evidence)
        except ValueError:
            raise VlmInputError("The detections cannot be cited (missing stored ids).",
                                code="uncitable_detection") from None
        request = VlmRequest(
            system=prompts.GENERATOR_SYSTEM,
            user_text=prompts.generator_user_text(aliases, self._limits.max_claims),
            images=_images(evidence.images, self._limits),
            max_new_tokens=self._limits.max_new_tokens,
            max_time_seconds=max_time_for(self._limits.timeout_seconds),
        )
        raw = call_runner(self._runner, request, self._limits.timeout_seconds)
        return parse_structured_report(raw, aliases, self._limits.max_claims)

    def generate(self, evidence: EvidenceBundle) -> str:
        return self.generate_structured(evidence).report_text


class QwenClaimVerifier:
    """Labels each claim from its own images; one inference per claim."""

    def __init__(self, runner: VlmRunner, limits: VlmLimits) -> None:
        self._runner = runner
        self._limits = limits
        self.model: ModelInfo = runner.info

    @property
    def verification_config(self) -> VerificationConfig:
        max_tokens = min(VERIFIER_MAX_NEW_TOKENS, self._limits.max_new_tokens)
        return VerificationConfig(
            prompt_version=prompts.VERIFIER_PROMPT_VERSION,
            sampling=SamplingParams(temperature=0, top_p=1, max_new_tokens=max_tokens),
        )

    def verify(self, claims: list[ReportClaim], evidence: EvidenceBundle) -> list[ClaimVerdict]:
        if evidence.detections:
            # The verifier must judge pixels, not the detector or generator inputs.
            raise VlmInputError("The verifier must not be given detections.", code="detections_not_allowed")
        images = _images(evidence.images, self._limits)
        image_ids: list[uuid.UUID] = [ref.image_id for ref in evidence.images]
        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            request = VlmRequest(
                system=prompts.VERIFIER_SYSTEM,
                user_text=prompts.verifier_user_text(claim.text, len(images)),
                images=images,
                max_new_tokens=self.verification_config.sampling.max_new_tokens,
                max_time_seconds=max_time_for(self._limits.timeout_seconds),
            )
            label, explanation = parse_verdict(call_runner(self._runner, request, self._limits.timeout_seconds))
            verdicts.append(
                ClaimVerdict(
                    claim_text=claim.text,
                    label=label,
                    explanation=explanation,
                    evidence_reference={"image_ids": [str(i) for i in image_ids]},
                    verifier=self.model,
                )
            )
        return verdicts

    def extract_claims(self, report_text: str) -> list[ReportClaim]:
        # Claims come structured from the generator; splitting prose is not supported.
        raise NotImplementedError("claims are produced by the structured report generator")

    def find_omissions(self, report_text: str, evidence: EvidenceBundle) -> list[ClaimVerdict]:
        # Not returning []: that would read as "nothing omitted".
        raise NotImplementedError("omission detection is not implemented")
