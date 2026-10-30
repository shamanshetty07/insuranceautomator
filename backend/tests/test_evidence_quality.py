"""Pure unit tests for evidence_quality: synthetic values and PIL images only, no DB."""

import math
import uuid
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFilter

from app.core.config import Settings
from app.schemas.claims import ClaimType, Verdict
from app.services.claim_policy import (
    ClaimPolicyInput,
    DetectionMatch,
    EvidenceAssociation,
    EvidenceIntegrity,
    EvidenceQuality,
    ReasonCode,
    VerifierOutcome,
    evaluate_claim,
)
from app.services.evidence_quality import (
    THRESHOLD_TOLERANCE,
    DetectionQualityFacts,
    ImageQualityFacts,
    PixelStats,
    QualityAssessment,
    QualityEffect,
    QualityFinding,
    QualityLimits,
    assess_detections,
    assess_image,
    assess_region,
    combine,
    describe_pixels,
    region_pixel_size,
    run_confidence_threshold,
    vlm_input_size,
)
from app.services.types import ImageRef
from app.services.vlm.adapters import load_image

LIMITS = QualityLimits(min_side_px=32, vlm_max_image_pixels=802_816)
IMG = uuid.UUID(int=1)
RUN = uuid.UUID(int=2)
BOX = {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}
CFG = {"confidence_threshold": 0.25, "iou_threshold": 0.45, "image_size": 640, "max_detections": 100}



def image(w: object, h: object) -> ImageQualityFacts:
    return ImageQualityFacts(IMG, w, h)


def det(n: int, conf: object = 0.9, dtype: str = "dent", box: object = BOX, cfg: object = CFG,
        image_id: uuid.UUID = IMG) -> DetectionQualityFacts:
    return DetectionQualityFacts(uuid.UUID(int=100 + n), image_id, RUN, dtype, conf, box, cfg)


def codes(a: QualityAssessment) -> list[str]:
    return [f.code for f in a.findings]


class TestLimits:
    def test_from_settings_uses_existing_configured_values(self):
        s = Settings(database_url="postgresql://unused/none", _env_file=None)
        assert QualityLimits.from_settings(s) == QualityLimits(s.min_image_side_px, s.vlm_max_image_pixels)
        assert LIMITS == QualityLimits(32, 802_816)  # the shipped defaults


class TestImage:
    def test_ok_image_has_no_findings(self):
        a = assess_image(image(640, 480), LIMITS)
        assert a.findings == () and a.effect is QualityEffect.INFO and not a.needs_review

    @pytest.mark.parametrize("w,h", [(12, 9), (31, 1000), (1000, 31)])
    def test_below_configured_upload_min_side_is_insufficient(self, w, h):
        a = assess_image(image(w, h), LIMITS)
        assert a.codes == (f"image_below_min_side:{IMG}",)
        assert a.evidence_clearly_insufficient and a.needs_review

    def test_exactly_min_side_passes(self):
        assert assess_image(image(32, 32), LIMITS).findings == ()

    def test_min_side_follows_the_configured_value(self):
        assert assess_image(image(64, 64), QualityLimits(100, 802_816)).evidence_clearly_insufficient

    @pytest.mark.parametrize("w,h", [(None, 10), (10, None), (None, None)])
    def test_unknown_dimensions_need_review_but_are_not_insufficient(self, w, h):
        a = assess_image(image(w, h), LIMITS)
        assert codes(a) == ["image_dimensions_unknown"]
        assert a.needs_review and not a.evidence_clearly_insufficient

    @pytest.mark.parametrize("w,h", [(0, 10), (-5, 10), (10.5, 10), (True, 10), ("640", 480)])
    def test_invalid_dimensions_need_review(self, w, h):
        assert codes(assess_image(image(w, h), LIMITS)) == ["image_dimensions_invalid"]

    def test_downscale_is_info_only(self):
        a = assess_image(image(4000, 3000), LIMITS)
        assert codes(a) == ["image_downscaled_for_vlm"]
        assert a.effect is QualityEffect.INFO and not a.needs_review
        assert "4000x3000->" in a.codes[0]

    def test_from_model(self):
        row = SimpleNamespace(id=IMG, width_px=50, height_px=40, sha256=None)
        assert ImageQualityFacts.from_model(row) == ImageQualityFacts(IMG, 50, 40)


class TestVlmInputSize:
    @pytest.mark.parametrize("size", [(12, 9), (64, 48), (896, 896), (2000, 1000), (5000, 37)])
    def test_matches_adapter_load_image(self, tmp_path, size):
        path = tmp_path / "x.png"
        Image.new("RGB", size, (90, 90, 90)).save(path)
        loaded = load_image(ImageRef(image_id=IMG, path=path), LIMITS.vlm_max_image_pixels)
        assert vlm_input_size(*size, LIMITS.vlm_max_image_pixels) == loaded.size

    def test_never_upscales(self):
        assert vlm_input_size(12, 9, 802_816) == (12, 9)


class TestRegion:
    def test_pixel_size(self):
        assert region_pixel_size(BOX, 100, 50) == pytest.approx((40, 20))

    @pytest.mark.parametrize("box,w,h", [(None, 100, 100), ({"x_min": 0.5}, 100, 100), (BOX, None, 100)])
    def test_unusable_inputs(self, box, w, h):
        assert region_pixel_size(box, w, h) is None

    def test_small_region_is_info_only(self):
        a = assess_region(image(50, 50), BOX, "p1", LIMITS)
        assert a.codes == ("region_below_min_side:p1",) and not a.needs_review

    def test_large_region_has_no_finding(self):
        assert assess_region(image(1000, 1000), BOX, "p1", LIMITS).findings == ()


class TestRunThreshold:
    @pytest.mark.parametrize("cfg,expected", [
        (CFG, 0.25), ({"confidence_threshold": 0}, 0.0), ({}, None), (None, None), ([], None),
        ({"confidence_threshold": "0.25"}, None), ({"confidence_threshold": float("nan")}, None),
        ({"confidence_threshold": 1.5}, None), ({"confidence_threshold": True}, None),
    ])
    def test_read_from_stored_config(self, cfg, expected):
        assert run_confidence_threshold(cfg) == expected


class TestDetections:
    @pytest.mark.parametrize("conf", [0.25, 0.9, 1.0, 1])
    def test_valid_confidence_produces_nothing_and_high_confidence_is_not_rewarded(self, conf):
        a = assess_detections([det(1, conf)])
        assert a.findings == () and a.effect is QualityEffect.INFO

    @pytest.mark.parametrize("conf", [None, float("nan"), math.inf, -math.inf, -0.01, 1.01, "0.9", True])
    def test_invalid_confidence_needs_review(self, conf):
        a = assess_detections([det(1, conf)])
        assert a.codes == (f"detection_confidence_invalid:{uuid.UUID(int=101)}",)
        assert a.needs_review and not a.evidence_clearly_insufficient

    def test_below_own_run_threshold_is_inconsistent(self):
        a = assess_detections([det(1, 0.1)])
        assert codes(a) == ["detection_below_run_threshold"] and a.needs_review

    def test_float32_rounding_at_threshold_is_tolerated(self):
        assert assess_detections([det(1, 0.25 - THRESHOLD_TOLERANCE / 10)]).findings == ()

    def test_unknown_run_threshold_is_info_and_reported_once(self):
        a = assess_detections([det(1, 0.01, cfg={}), det(2, 0.02, cfg=None, box=None)])
        assert a.codes == (f"run_threshold_unknown:{RUN}",) and not a.needs_review

    def test_overlapping_different_classes_disagree(self):
        a = assess_detections([det(1, dtype="dent"), det(2, dtype="Scratch")])
        assert a.codes == (f"detections_disagree:{uuid.UUID(int=101)},{uuid.UUID(int=102)}",)
        assert a.needs_review and not a.evidence_clearly_insufficient

    def test_same_class_in_different_spelling_agrees(self):
        assert assess_detections([det(1, dtype="paint-scratch"), det(2, dtype="Paint Scratch")]).findings == ()

    @pytest.mark.parametrize("other", [
        {"box": {"x_min": 0.6, "y_min": 0.6, "x_max": 0.9, "y_max": 0.9}},  # disjoint region
        {"box": None},
        {"box": {"x_min": 0.9, "y_min": 0.1, "x_max": 0.2, "y_max": 0.5}},  # invalid box
        {"image_id": uuid.UUID(int=9)},  # other image
    ])
    def test_no_disagreement_without_a_shared_region(self, other):
        assert assess_detections([det(1, dtype="dent"), det(2, dtype="scratch", **other)]).findings == ()

    def test_duplicate_citations_do_not_disagree_with_themselves(self):
        assert assess_detections([det(1), det(1)]).findings == ()

    def test_empty(self):
        assert assess_detections([]).findings == ()


class TestAssessment:
    def test_effect_is_the_strongest_and_combine_dedupes(self):
        i = QualityFinding("a", "1", QualityEffect.INFO)
        r = QualityFinding("b", "2", QualityEffect.REVIEW)
        x = QualityFinding("c", "3", QualityEffect.INSUFFICIENT)
        merged = combine(QualityAssessment((i, r)), QualityAssessment((r, x)))
        assert merged.findings == (i, r, x) and merged.evidence_clearly_insufficient
        assert merged.codes == ("a:1", "b:2", "c:3")
        assert QualityAssessment().effect is QualityEffect.INFO


def test_tiny_image_scenario_is_insufficient_under_the_policy():
    assert assess_image(image(12, 9), LIMITS).evidence_clearly_insufficient
    d = evaluate_claim(ClaimPolicyInput(
        ClaimType.VISIBLE_DAMAGE_PRESENCE, VerifierOutcome(Verdict.SUPPORTED), EvidenceIntegrity.OK,
        EvidenceAssociation.ASSOCIATED, DetectionMatch.MATCHES_CLAIM, evidence_quality=EvidenceQuality.INSUFFICIENT,
    ))
    assert (d.verdict, d.needs_human_review, d.reason_codes) == (
        Verdict.INSUFFICIENT_EVIDENCE, True, (ReasonCode.EVIDENCE_QUALITY_INSUFFICIENT,)
    )


def _checkerboard(size: int = 64, cell: int = 4) -> Image.Image:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=255)
    return img.convert("RGB")


class TestDescribePixels:
    def test_is_marked_uncalibrated(self):
        stats = describe_pixels(_checkerboard())
        assert isinstance(stats, PixelStats) and stats.calibrated is False
        assert (stats.width, stats.height) == (64, 64)

    def test_blur_lowers_edge_variance(self):
        sharp = _checkerboard()
        blurred = sharp.filter(ImageFilter.GaussianBlur(3))
        assert describe_pixels(blurred).edge_variance < describe_pixels(sharp).edge_variance

    def test_flat_image_has_zero_edge_variance(self):
        assert describe_pixels(Image.new("RGB", (40, 40), (128, 128, 128))).edge_variance == 0

    def test_dark_and_bright_fractions(self):
        dark = describe_pixels(Image.new("RGB", (10, 10), (0, 0, 0)))
        bright = describe_pixels(Image.new("RGB", (10, 10), (255, 255, 255)))
        assert (dark.dark_fraction, dark.bright_fraction, dark.mean_luminance) == (1.0, 0.0, 0.0)
        assert (bright.dark_fraction, bright.bright_fraction) == (0.0, 1.0)

    @pytest.mark.parametrize("size", [(1, 1), (2, 2), (3, 1)])
    def test_tiny_images_do_not_crash(self, size):
        describe_pixels(Image.new("RGB", size, (10, 20, 30)))
