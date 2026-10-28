"""Optional end-to-end check with a real Ultralytics install and real weights.

Skipped unless `ultralytics` is importable and DETECTOR_TEST_WEIGHTS points at a
trusted local weights file (.pt files are pickles: loading one executes code).
"""

import importlib.util
import os
import uuid
from pathlib import Path

import pytest

from app.schemas.runs import DetectionConfig
from app.services.types import ImageRef
from tests.image_utils import make_image_bytes

WEIGHTS = os.environ.get("DETECTOR_TEST_WEIGHTS")

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None or not WEIGHTS,
    reason="needs the 'detection' extra and DETECTOR_TEST_WEIGHTS",
)


def test_real_model_returns_normalized_boxes(tmp_path: Path):
    from app.services.detection import UltralyticsDetector

    image = tmp_path / "car.jpg"
    image.write_bytes(make_image_bytes(size=(320, 240)))
    config = DetectionConfig(
        confidence_threshold=0.0, iou_threshold=0.45, image_size=320, max_detections=10, device="cpu"
    )
    detector = UltralyticsDetector(Path(WEIGHTS), "test", "test", config)

    detections = detector.detect([ImageRef(uuid.uuid4(), image)])

    assert detector.classes
    assert len(detector.model.weights_sha256) == 64
    for d in detections:
        box = d.bounding_box
        assert 0 <= box["x_min"] < box["x_max"] <= 1 and 0 <= box["y_min"] < box["y_max"] <= 1
        assert 0 <= d.confidence <= 1 and d.damage_type in detector.classes
