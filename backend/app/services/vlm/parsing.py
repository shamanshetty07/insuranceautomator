"""Parse and validate raw VLM output. Fails closed: any problem rejects the whole output.

Model output is an untrusted prediction. The generator's claims are produced together,
so an invented alias, a duplicate reference or too many claims means the output as a
whole cannot be trusted: nothing is dropped, truncated or repaired, and no retry is
made (decoding is greedy, so a retry would repeat it). Valid claims become the existing
ExtractedClaim / StructuredReport contracts, reusing their validators.
"""

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.schemas.claims import (
    MAX_CLAIM_TEXT_LENGTH,
    ClaimExtractionResult,
    CropBox,
    EvidenceReference,
    ExtractedClaim,
    StructuredReport,
)
from app.services.vlm.errors import VlmOutputError
from app.services.vlm.prompts import AliasMap

MAX_OUTPUT_CHARS = 64_000
MAX_EXPLANATION_CHARS = 2000
# Attributed to the model on purpose: it is not a finding that the vehicle is undamaged.
NO_CLAIMS_TEXT = "The model described no visible damage in the provided images."
_FENCE = re.compile(r"\A```(?:json)?\s*\n(.*)\n?```\Z", re.DOTALL | re.IGNORECASE)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _RawEvidence(_Strict):
    image: str
    detection: str | None = None
    crop: tuple[float, float, float, float] | None = None


class _RawClaim(_Strict):
    text: str = Field(min_length=1, max_length=MAX_CLAIM_TEXT_LENGTH)
    type: str
    damage_type: str | None = None
    images: list[str] = []
    detections: list[str] = []
    evidence: list[_RawEvidence] = []


class _RawReport(_Strict):
    claims: list[_RawClaim]


class _RawVerdict(_Strict):
    label: str = Field(min_length=1, max_length=64)
    explanation: str = Field(default="", max_length=MAX_EXPLANATION_CHARS)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


def _reject_constant(name: str) -> None:
    raise ValueError(f"non-finite number {name}")


def load_json_object(raw: str) -> Any:
    """Strict JSON from model text: at most one surrounding ``` fence, nothing else around it."""
    if len(raw) > MAX_OUTPUT_CHARS:
        raise VlmOutputError("The model output was too long.", code="too_long")
    text = raw.strip()
    if not text:
        raise VlmOutputError("The model returned no output.", code="empty")
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        raise VlmOutputError("The model output was not valid JSON.", code="not_json") from None


def _resolve(aliases: AliasMap, raw: _RawClaim, claim_id: str) -> ExtractedClaim:
    def image(alias: str):
        if alias not in aliases.images:
            raise VlmOutputError("The model cited an image it was not given.", code="unknown_alias")
        return aliases.images[alias]

    def detection(alias: str):
        if alias not in aliases.detections:
            raise VlmOutputError("The model cited a detection it was not given.", code="unknown_alias")
        return aliases.detections[alias]

    image_ids = [image(a) for a in raw.images]
    dets = [detection(a) for a in raw.detections]
    for det in dets:
        if det.image_id not in image_ids:
            raise VlmOutputError("A cited detection is on an image the claim does not cite.",
                                 code="detection_image_mismatch")
    evidence = []
    for ev in raw.evidence:
        image_id = image(ev.image)
        det = detection(ev.detection) if ev.detection is not None else None
        if det is not None and det.image_id != image_id:
            raise VlmOutputError("An evidence item pairs a detection with the wrong image.",
                                 code="detection_image_mismatch")
        try:
            crop = CropBox(x_min=ev.crop[0], y_min=ev.crop[1], x_max=ev.crop[2], y_max=ev.crop[3]) if ev.crop else None
        except ValidationError:
            raise VlmOutputError("The model gave an invalid crop.", code="invalid_crop") from None
        evidence.append(EvidenceReference(image_id=image_id, prediction_id=det.prediction_id if det else None,
                                          crop=crop))
    try:
        return ExtractedClaim(
            claim_id=claim_id,
            text=raw.text,
            claim_type=raw.type,
            damage_type=raw.damage_type,
            image_ids=image_ids,
            prediction_ids=[d.prediction_id for d in dets],
            evidence=evidence,
        )
    except ValidationError:
        # Unknown claim type, duplicate ids/evidence, uncovered evidence, bad lengths, ...
        raise VlmOutputError("A claim in the model output is invalid.", code="invalid_claim") from None


def parse_structured_report(raw: str, aliases: AliasMap, max_claims: int) -> StructuredReport:
    """Model text -> StructuredReport whose report text is exactly its claims, in order.

    Building the report from the claims means every sentence in it is a structured,
    verifiable claim: there is no unverified free prose. Zero claims is a valid answer
    ("no damage seen"); it is NOT evidence that the vehicle is undamaged.
    """
    data = load_json_object(raw)
    try:
        report = _RawReport.model_validate(data)
    except ValidationError:
        raise VlmOutputError("The model output does not match the expected shape.", code="schema") from None
    if len(report.claims) > max_claims:
        raise VlmOutputError("The model produced more claims than allowed.", code="too_many_claims")
    claims = [_resolve(aliases, c, f"c{n}") for n, c in enumerate(report.claims, start=1)]
    text = "\n".join(c.text for c in claims) if claims else NO_CLAIMS_TEXT
    return StructuredReport(report_text=text, claims=ClaimExtractionResult(claims=claims))


def parse_verdict(raw: str) -> tuple[str, str]:
    """Model text -> (label, explanation). The label is passed through unvalidated on purpose:
    verification_service decides what an unknown or non-claim label (omitted_damage) means."""
    data = load_json_object(raw)
    try:
        verdict = _RawVerdict.model_validate(data)
    except ValidationError:
        raise VlmOutputError("The verifier output does not match the expected shape.", code="schema") from None
    return verdict.label.strip(), verdict.explanation.strip()
