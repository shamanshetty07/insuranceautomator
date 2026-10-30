"""Image-quality and detection-confidence safeguards for claim verification; pure (no DB, no files).

Only facts the system already stores are judged, against values it already configures:
stored width/height vs. ``min_image_side_px`` (the upload policy), the VLM pixel budget
(``vlm_max_image_pixels``), and each detection's confidence vs. its own run's stored
``confidence_threshold``. Nothing here can make a claim more acceptable: every finding is
either descriptive (INFO), a reason for human review (REVIEW), or a reason the cited image
cannot serve as evidence (INSUFFICIENT). Detector confidence is never treated as proof.

Blur, lighting and occlusion are NOT assessed. ``describe_pixels`` reports uncalibrated
pixel statistics for logging/research only; they produce no findings and no effect.
"""

import enum
import math
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageFilter, ImageStat
from pydantic import ValidationError

from app.schemas.damage import BoundingBox
from app.services.evidence_association import (
    LOCATION_MATCH_IOU,
    box_iou,
    normalize_damage_type,
)

# Float32 scores (Ultralytics) stored as float64 can differ from the threshold by one float32
# ulp (~6e-8 near 1.0); a score this far below the threshold cannot be rounding.
THRESHOLD_TOLERANCE = 1e-6


class QualityEffect(enum.IntEnum):
    """Ordered: a larger value is a stronger move away from an automatic decision."""

    INFO = 0
    REVIEW = 1
    INSUFFICIENT = 2


@dataclass(frozen=True)
class QualityFinding:
    code: str
    subject: str
    effect: QualityEffect

    def __str__(self) -> str:
        return f"{self.code}:{self.subject}"


@dataclass(frozen=True)
class QualityAssessment:
    findings: tuple[QualityFinding, ...] = ()

    @property
    def effect(self) -> QualityEffect:
        return max((f.effect for f in self.findings), default=QualityEffect.INFO)

    @property
    def evidence_clearly_insufficient(self) -> bool:
        return self.effect is QualityEffect.INSUFFICIENT

    @property
    def needs_review(self) -> bool:
        return self.effect >= QualityEffect.REVIEW

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(f) for f in self.findings))


def combine(*assessments: QualityAssessment) -> QualityAssessment:
    found: list[QualityFinding] = []
    for a in assessments:
        found.extend(f for f in a.findings if f not in found)
    return QualityAssessment(tuple(found))


@dataclass(frozen=True)
class QualityLimits:
    """Existing configured values only: Settings.min_image_side_px / vlm_max_image_pixels."""

    min_side_px: int
    vlm_max_image_pixels: int

    @classmethod
    def from_settings(cls, settings: Any) -> "QualityLimits":  # noqa: ANN401
        return cls(settings.min_image_side_px, settings.vlm_max_image_pixels)


@dataclass(frozen=True)
class ImageQualityFacts:
    image_id: uuid.UUID
    width_px: object
    height_px: object

    @classmethod
    def from_model(cls, image: Any) -> "ImageQualityFacts":  # noqa: ANN401 - InspectionImage
        return cls(image.id, image.width_px, image.height_px)


@dataclass(frozen=True)
class DetectionQualityFacts:
    """One cited prediction plus its run's stored config (DetectionRun.config)."""

    prediction_id: uuid.UUID
    image_id: uuid.UUID
    run_id: uuid.UUID
    damage_type: str
    confidence: object
    bounding_box: Mapping[str, Any] | None
    run_config: Mapping[str, Any] | None


def _box(raw: object) -> BoundingBox | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        return BoundingBox.model_validate(raw)
    except ValidationError:
        return None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _unit_float(value: object) -> float | None:
    """A finite float in [0, 1], else None (bool and non-numbers are rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) and 0.0 <= f <= 1.0 else None


def vlm_input_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """The size the VLM adapter hands to the model (mirrors vlm.adapters.load_image): never upscaled."""
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        return max(1, int(width * scale)), max(1, int(height * scale))
    return width, height


def assess_image(facts: ImageQualityFacts, limits: QualityLimits) -> QualityAssessment:
    iid = str(facts.image_id)
    w, h = _positive_int(facts.width_px), _positive_int(facts.height_px)
    if facts.width_px is None or facts.height_px is None:
        return QualityAssessment((QualityFinding("image_dimensions_unknown", iid, QualityEffect.REVIEW),))
    if w is None or h is None:
        return QualityAssessment((QualityFinding("image_dimensions_invalid", iid, QualityEffect.REVIEW),))
    found: list[QualityFinding] = []
    if min(w, h) < limits.min_side_px:
        found.append(QualityFinding("image_below_min_side", iid, QualityEffect.INSUFFICIENT))
    vw, vh = vlm_input_size(w, h, limits.vlm_max_image_pixels)
    if (vw, vh) != (w, h):
        found.append(QualityFinding("image_downscaled_for_vlm", f"{iid}:{w}x{h}->{vw}x{vh}", QualityEffect.INFO))
    return QualityAssessment(tuple(found))


def region_pixel_size(box: Mapping[str, Any] | None, width: object, height: object) -> tuple[float, float] | None:
    """Width/height in stored pixels of a normalized box; None if box or dimensions are unusable."""
    b, w, h = _box(box), _positive_int(width), _positive_int(height)
    if b is None or w is None or h is None:
        return None
    return (b.x_max - b.x_min) * w, (b.y_max - b.y_min) * h


def assess_region(
    image: ImageQualityFacts, box: Mapping[str, Any] | None, subject: str, limits: QualityLimits
) -> QualityAssessment:
    """INFO only: the upload minimum was set for whole images, not for damage regions."""
    size = region_pixel_size(box, image.width_px, image.height_px)
    if size is None or min(size) >= limits.min_side_px:
        return QualityAssessment()
    return QualityAssessment((QualityFinding("region_below_min_side", subject, QualityEffect.INFO),))


def run_confidence_threshold(run_config: Mapping[str, Any] | None) -> float | None:
    if not isinstance(run_config, Mapping):
        return None
    return _unit_float(run_config.get("confidence_threshold"))


def assess_detections(detections: Sequence[DetectionQualityFacts]) -> QualityAssessment:
    """Data-integrity and agreement checks on the cited detections of one claim.

    Never rewards confidence: a high score adds nothing; only impossible or inconsistent
    values and disagreeing detections produce findings.
    """
    found: list[QualityFinding] = []
    unknown_runs: list[str] = []
    for d in detections:
        pid = str(d.prediction_id)
        conf = _unit_float(d.confidence)
        if conf is None:
            found.append(QualityFinding("detection_confidence_invalid", pid, QualityEffect.REVIEW))
            continue
        threshold = run_confidence_threshold(d.run_config)
        if threshold is None:
            unknown_runs.append(str(d.run_id))
        elif conf < threshold - THRESHOLD_TOLERANCE:
            found.append(QualityFinding("detection_below_run_threshold", pid, QualityEffect.REVIEW))
    found.extend(
        QualityFinding("run_threshold_unknown", r, QualityEffect.INFO) for r in dict.fromkeys(unknown_runs)
    )
    found.extend(_disagreements(detections))
    return QualityAssessment(tuple(found))


def _disagreements(detections: Sequence[DetectionQualityFacts]) -> Iterable[QualityFinding]:
    """Cited detections on the same region of the same image with different classes."""
    unique = list({d.prediction_id: d for d in detections}.values())
    for i, a in enumerate(unique):
        box_a = _box(a.bounding_box)
        if box_a is None:
            continue
        for b in unique[i + 1 :]:
            box_b = _box(b.bounding_box)
            if (
                box_b is not None
                and a.image_id == b.image_id
                and normalize_damage_type(a.damage_type) != normalize_damage_type(b.damage_type)
                and box_iou(box_a, box_b) >= LOCATION_MATCH_IOU
            ):
                ids = ",".join(sorted((str(a.prediction_id), str(b.prediction_id))))
                yield QualityFinding("detections_disagree", ids, QualityEffect.REVIEW)


@dataclass(frozen=True)
class PixelStats:
    """UNCALIBRATED descriptive statistics. No threshold exists; never use these to decide."""

    width: int
    height: int
    edge_variance: float
    mean_luminance: float
    luminance_stddev: float
    dark_fraction: float
    bright_fraction: float
    calibrated: bool = False


def describe_pixels(image: Image.Image) -> PixelStats:
    """Edge variance (Laplacian-like FIND_EDGES on luminance), brightness and clipping fractions."""
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    # FIND_EDGES leaves a 1px border artefact; measure the interior when there is one.
    if gray.width > 2 and gray.height > 2:
        edges = edges.crop((1, 1, gray.width - 1, gray.height - 1))
    edge_stat, gray_stat = ImageStat.Stat(edges), ImageStat.Stat(gray)
    hist = gray.histogram()
    total = gray.width * gray.height
    return PixelStats(
        width=image.width,
        height=image.height,
        edge_variance=float(edge_stat.var[0]),
        mean_luminance=float(gray_stat.mean[0]),
        luminance_stddev=float(gray_stat.stddev[0]),
        dark_fraction=sum(hist[:5]) / total,
        bright_fraction=sum(hist[251:]) / total,
    )
