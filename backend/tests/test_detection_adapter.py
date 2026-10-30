"""Ultralytics adapter, provider and settings, with fake Ultralytics objects (no package needed)."""

import hashlib
import json
import math
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.api.deps import get_detector_provider
from app.core.config import Settings, get_settings
from app.main import app
from app.schemas.detection import DetectionRunRead
from app.schemas.runs import DetectionConfig
from app.services.detection import (
    DetectorUnavailable,
    SettingsDetectorProvider,
    clear_detector_cache,
    detections_from_result,
)
from app.services.detection import provider as provider_module
from app.services.detection import ultralytics_detector
from app.services.types import ImageRef
from tests.image_utils import make_image_bytes

NO_DB = "postgresql+psycopg://none:none@127.0.0.1:1/none"
IMAGE_ID = uuid.uuid4()
NAMES = {0: "dent", 1: "Glass Shatter"}


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def tolist(self):
        return self.data


class FakeBoxes:
    def __init__(self, xyxyn, conf, cls):
        self.xyxyn = FakeTensor(xyxyn)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def result(*rows):
    return FakeResult(FakeBoxes([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]))


def test_conversion_keeps_normalized_boxes_and_verbatim_class_names():
    out = detections_from_result(IMAGE_ID, result(([0.1, 0.2, 0.5, 0.6], 0.9, 1.0)), NAMES)
    assert len(out) == 1
    d = out[0]
    assert d.image_id == IMAGE_ID
    assert d.damage_type == "Glass Shatter"
    assert d.confidence == pytest.approx(0.9)
    assert d.bounding_box == pytest.approx({"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6})


def test_conversion_clamps_to_unit_square():
    out = detections_from_result(IMAGE_ID, result(([-0.05, -1.0, 1.2, 1.0001], 1.0000001, 0)), NAMES)
    assert out[0].bounding_box == {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
    assert out[0].confidence == 1.0


@pytest.mark.parametrize(
    "box",
    [
        [0.5, 0.2, 0.5, 0.6],
        [0.1, 0.6, 0.5, 0.6],
        [1.1, 0.2, 1.5, 0.6],
        [0.6, 0.2, 0.5, 0.6],
        [math.nan, 0.2, 0.5, 0.6],
        [0.1, 0.2, math.inf, 0.6],
    ],
)
def test_conversion_drops_degenerate_or_non_finite_boxes(box):
    assert detections_from_result(IMAGE_ID, result((box, 0.9, 0)), NAMES) == []


def test_conversion_drops_non_finite_confidence():
    assert detections_from_result(IMAGE_ID, result(([0.1, 0.2, 0.5, 0.6], math.nan, 0)), NAMES) == []


def test_conversion_handles_no_boxes():
    assert detections_from_result(IMAGE_ID, FakeResult(None), NAMES) == []
    assert detections_from_result(IMAGE_ID, result(), NAMES) == []


class FakeYOLO:
    instances: list["FakeYOLO"] = []
    names = {0: "dent", 1: "scratch"}

    def __init__(self, path: str, task: str) -> None:
        self.path, self.task = path, task
        self.predict_calls: list[dict] = []
        FakeYOLO.instances.append(self)

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [result(([0.1, 0.2, 0.5, 0.6], 0.8, 1))]


@pytest.fixture
def fake_yolo(monkeypatch):
    FakeYOLO.instances = []
    FakeYOLO.names = {0: "dent", 1: "scratch"}
    monkeypatch.setattr(ultralytics_detector, "_import_yolo", lambda: FakeYOLO)
    return FakeYOLO


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    path.write_bytes(b"fake weights")
    return path


CONFIG = DetectionConfig(
    confidence_threshold=0.3, iou_threshold=0.5, image_size=320, max_detections=7, device="cpu"
)


def test_adapter_loads_hashes_and_passes_parameters(fake_yolo, weights, tmp_path):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(make_image_bytes(size=(80, 40)))
    detector = ultralytics_detector.UltralyticsDetector(weights, "damage-yolo", "2024.1", CONFIG)

    out = detector.detect([ImageRef(IMAGE_ID, image_path)])

    assert detector.model.weights_sha256 == hashlib.sha256(b"fake weights").hexdigest()
    assert detector.classes == ["dent", "scratch"]
    assert detector.config == CONFIG
    model = fake_yolo.instances[0]
    assert model.task == "detect"
    call = model.predict_calls[0]
    assert (call["conf"], call["iou"], call["imgsz"], call["max_det"], call["device"]) == (0.3, 0.5, 320, 7, "cpu")
    assert call["classes"] is None and call["save"] is False
    assert call["source"].size == (80, 40) and call["source"].mode == "RGB"
    assert [d.damage_type for d in out] == ["scratch"]


def test_adapter_maps_class_filter_and_rejects_unknown_names(fake_yolo, weights):
    config = CONFIG.model_copy(update={"class_filter": ["scratch"]})
    detector = ultralytics_detector.UltralyticsDetector(weights, "m", "v", config)
    assert detector._class_ids == [1]
    with pytest.raises(ValueError, match="class_filter"):
        ultralytics_detector.UltralyticsDetector(
            weights, "m", "v", CONFIG.model_copy(update={"class_filter": ["rust"]})
        )


def test_adapter_rejects_overlong_class_names(fake_yolo, weights):
    fake_yolo.names = {0: "x" * 101}
    with pytest.raises(ValueError, match="class names"):
        ultralytics_detector.UltralyticsDetector(weights, "m", "v", CONFIG)


def test_adapter_refuses_missing_weights_before_importing(monkeypatch, tmp_path):
    def boom():
        raise AssertionError("ultralytics must not be imported (it would try to download)")

    monkeypatch.setattr(ultralytics_detector, "_import_yolo", boom)
    with pytest.raises(FileNotFoundError):
        ultralytics_detector.UltralyticsDetector(tmp_path / "missing.pt", "m", "v", CONFIG)
    with pytest.raises(FileNotFoundError):
        ultralytics_detector.UltralyticsDetector(tmp_path, "m", "v", CONFIG)


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_detector_cache()
    yield
    clear_detector_cache()


def settings(**kw) -> Settings:
    base = {
        "database_url": NO_DB,
        "detector_backend": "ultralytics",
        "detector_model_name": "damage-yolo",
        "detector_model_version": "2024.1",
    }
    return Settings(**(base | kw))


@pytest.fixture
def ultralytics_installed(monkeypatch):
    real = provider_module.importlib.util.find_spec
    monkeypatch.setattr(
        provider_module.importlib.util,
        "find_spec",
        lambda name, *a: object() if name == "ultralytics" else real(name, *a),
    )


def test_provider_disabled_by_default():
    provider = SettingsDetectorProvider(Settings(database_url=NO_DB))
    status = provider.status()
    assert status.available is False and status.backend == "none"
    assert "disabled" in status.reason
    with pytest.raises(DetectorUnavailable):
        provider.get()


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"detector_model_name": None}, "DETECTOR_MODEL_NAME"),
        ({"detector_weights_path": None}, "weights file"),
        ({"detector_weights_path": Path("/nonexistent/secret-dir/w.pt")}, "weights file"),
    ],
)
def test_provider_reports_configuration_problems_without_paths(overrides, fragment, weights, ultralytics_installed):
    status = SettingsDetectorProvider(settings(**({"detector_weights_path": weights} | overrides))).status()
    assert status.available is False
    assert fragment in status.reason
    assert "secret-dir" not in status.model_dump_json() and str(weights.parent) not in status.model_dump_json()


def test_provider_reports_missing_package(weights, monkeypatch):
    real = provider_module.importlib.util.find_spec
    monkeypatch.setattr(
        provider_module.importlib.util,
        "find_spec",
        lambda name, *a: None if name == "ultralytics" else real(name, *a),
    )
    status = SettingsDetectorProvider(settings(detector_weights_path=weights)).status()
    assert status.available is False and "ultralytics" in status.reason


def test_provider_status_does_not_load_and_get_caches(fake_yolo, weights, ultralytics_installed):
    provider = SettingsDetectorProvider(settings(detector_weights_path=weights))

    status = provider.status()
    assert status.available is True
    assert status.classes is None and status.weights_sha256 is None
    assert status.config.confidence_threshold == 0.25
    assert fake_yolo.instances == []

    first = provider.get()
    second = SettingsDetectorProvider(settings(detector_weights_path=weights)).get()
    assert first is second and len(fake_yolo.instances) == 1

    loaded = provider.status()
    assert loaded.classes == ["dent", "scratch"]
    assert loaded.weights_sha256 == first.model.weights_sha256
    assert str(weights) not in loaded.model_dump_json()


def test_provider_load_failure_is_unavailable_and_not_cached(monkeypatch, weights, ultralytics_installed):
    def broken():
        raise RuntimeError(f"corrupt file at {weights}")

    monkeypatch.setattr(ultralytics_detector, "_import_yolo", broken)
    provider = SettingsDetectorProvider(settings(detector_weights_path=weights))
    with pytest.raises(DetectorUnavailable, match="failed to load") as excinfo:
        provider.get()
    assert str(weights) not in str(excinfo.value)
    assert provider_module._loaded == {}


@pytest.mark.parametrize(
    "overrides",
    [
        {"detector_confidence_threshold": 1.5},
        {"detector_iou_threshold": -0.1},
        {"detector_image_size": 0},
        {"detector_max_detections": 0},
        {"detector_backend": "openai"},
        {"detector_model_name": "x" * 101},
    ],
)
def test_invalid_detector_settings_rejected(overrides):
    with pytest.raises(ValidationError):
        Settings(database_url=NO_DB, **overrides)


def test_detection_config_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        DetectionConfig.model_validate(CONFIG.model_dump() | {"secret": "x"})


def test_duration_is_null_until_completed():
    now = datetime.now(UTC)
    base = {
        "id": uuid.uuid4(),
        "inspection_image_id": uuid.uuid4(),
        "model_name": "m",
        "model_version": "v",
        "weights_sha256": None,
        "config": CONFIG.model_dump(),
        "status": "running",
        "error_message": None,
        "created_at": now,
        "completed_at": None,
        "predictions": [],
    }
    assert DetectionRunRead.model_validate(base).duration_ms is None
    done = base | {"status": "completed", "completed_at": now + timedelta(milliseconds=1500)}
    assert DetectionRunRead.model_validate(done).duration_ms == 1500


def test_status_endpoint_disabled(plain_client):
    app.dependency_overrides[get_settings] = lambda: Settings(database_url=NO_DB)
    try:
        body = plain_client.get("/api/v1/detector").json()
    finally:
        app.dependency_overrides.clear()
    assert body["available"] is False and body["backend"] == "none"
    assert body["classes"] is None and body["config"] is None


def test_status_endpoint_available_without_leaking_paths(plain_client, fake_yolo, weights, ultralytics_installed):
    app.dependency_overrides[get_settings] = lambda: settings(detector_weights_path=weights)
    try:
        response = plain_client.get("/api/v1/detector")
    finally:
        app.dependency_overrides.clear()
    body = response.json()
    assert response.status_code == 200
    assert body["available"] is True and body["model_name"] == "damage-yolo"
    assert set(body["config"]) == set(json.loads(CONFIG.model_dump_json()))
    assert str(weights.parent) not in response.text
    assert fake_yolo.instances == []


def test_default_dependency_uses_settings_provider():
    provider = get_detector_provider(Settings(database_url=NO_DB))
    assert isinstance(provider, SettingsDetectorProvider)
