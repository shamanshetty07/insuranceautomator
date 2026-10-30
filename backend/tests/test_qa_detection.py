"""QA pass 2: the detection boundary (threat model T-1..T-8) and the request body cap."""

import io
import sys
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy.orm import sessionmaker
from starlette.middleware.cors import CORSMiddleware

from app.api.deps import get_detector_provider
from app.core.config import Settings
from app.main import app
from app.models import DamagePrediction, DetectionRun
from app.models.run_status import RunStatus
from app.schemas.detection import DetectorStatusRead
from app.schemas.runs import DetectionConfig
from app.services import detection_service
from app.services.detection import (
    DetectorUnavailable,
    SettingsDetectorProvider,
    UltralyticsDetector,
    clear_detector_cache,
)
from app.services.detection import provider as provider_module
from app.services.detection import ultralytics_detector
from app.services.image_processing import ImageLimits, process_image
from app.services.types import Detection, ImageRef, ModelInfo
from tests.conftest import requires_db
from tests.image_utils import make_image_bytes

NO_DB = "postgresql+psycopg://none:none@127.0.0.1:1/none"
CONFIG = DetectionConfig(confidence_threshold=0.25, iou_threshold=0.45, image_size=640, max_detections=100)
BOX = {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6}


class CapturingYOLO:
    """Stands in for ultralytics.YOLO; records what predict() receives."""

    names = {0: "dent"}
    instances: list["CapturingYOLO"] = []
    init_delay = 0.0
    predict_delay = 0.0
    active = 0
    max_active = 0
    _guard = threading.Lock()

    def __init__(self, path: str, task: str) -> None:
        time.sleep(CapturingYOLO.init_delay)
        self.path = path
        self.sources: list[Image.Image] = []
        CapturingYOLO.instances.append(self)

    def predict(self, **kwargs):
        cls = CapturingYOLO
        with cls._guard:
            cls.active += 1
            cls.max_active = max(cls.max_active, cls.active)
        time.sleep(cls.predict_delay)
        self.sources.append(kwargs["source"])
        with cls._guard:
            cls.active -= 1
        return []


@pytest.fixture
def capturing_yolo(monkeypatch):
    CapturingYOLO.instances = []
    CapturingYOLO.init_delay = CapturingYOLO.predict_delay = 0.0
    CapturingYOLO.active = CapturingYOLO.max_active = 0
    monkeypatch.setattr(ultralytics_detector, "_import_yolo", lambda: CapturingYOLO)
    monkeypatch.setattr(
        provider_module.importlib.util,
        "find_spec",
        lambda name, *a: object() if name == "ultralytics" else None,
    )
    clear_detector_cache()
    yield CapturingYOLO
    clear_detector_cache()


def _weights(tmp_path: Path, name: str = "model.pt") -> Path:
    path = tmp_path / name
    path.write_bytes(b"fake weights")
    return path


def _detector(tmp_path: Path) -> UltralyticsDetector:
    return UltralyticsDetector(_weights(tmp_path), "qa-model", "1", CONFIG)


def _provider_settings(weights: Path) -> Settings:
    return Settings(
        database_url=NO_DB,
        detector_backend="ultralytics",
        detector_weights_path=weights,
        detector_model_name="qa-model",
        detector_model_version="1",
    )


def _stored_file(tmp_path: Path, raw: bytes) -> Path:
    processed = process_image(raw, ImageLimits(min_side_px=16, max_pixels=1_000_000))
    path = tmp_path / f"stored.{processed.extension}"
    path.write_bytes(processed.data)
    return path


def _encode(image: Image.Image, fmt: str = "PNG") -> bytes:
    out = io.BytesIO()
    image.save(out, format=fmt)
    return out.getvalue()


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P", "1", "L", "RGB"])
def test_every_stored_mode_reaches_the_model_as_rgb_at_stored_size(tmp_path, capturing_yolo, mode):
    path = _stored_file(tmp_path, _encode(Image.new(mode, (64, 48))))
    detector = _detector(tmp_path)
    detector.detect([ImageRef(image_id=uuid.uuid4(), path=path)])
    source = capturing_yolo.instances[0].sources[0]
    assert source.mode == "RGB" and source.size == (64, 48)


def test_exif_rotated_upload_reaches_the_model_upright(tmp_path, capturing_yolo):
    path = _stored_file(tmp_path, make_image_bytes("JPEG", (64, 32), orientation=6))
    _detector(tmp_path).detect([ImageRef(image_id=uuid.uuid4(), path=path)])
    assert capturing_yolo.instances[0].sources[0].size == (32, 64)


def test_sixteen_bit_png_is_not_saturated_before_inference(tmp_path, capturing_yolo):
    image = Image.new("I;16", (64, 48))
    image.putdata([32768] * (64 * 48))
    path = _stored_file(tmp_path, _encode(image))
    assert Image.open(path).mode == "I;16"
    _detector(tmp_path).detect([ImageRef(image_id=uuid.uuid4(), path=path)])
    red, _, _ = capturing_yolo.instances[0].sources[0].getpixel((10, 10))
    assert 100 <= red <= 160


@pytest.mark.parametrize("name", ["model.yaml", "MODEL.YML"])
def test_model_definition_files_are_refused_as_weights(tmp_path, capturing_yolo, name):
    provider = SettingsDetectorProvider(_provider_settings(_weights(tmp_path, name)))
    assert provider.status().available is False
    with pytest.raises(DetectorUnavailable):
        provider.get()


def test_import_forces_ultralytics_offline_and_no_autoinstall(monkeypatch):
    monkeypatch.delenv("YOLO_OFFLINE", raising=False)
    monkeypatch.delenv("YOLO_AUTOINSTALL", raising=False)
    fake = types.ModuleType("ultralytics")
    fake.YOLO = object
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    assert ultralytics_detector._import_yolo() is object
    assert ultralytics_detector.os.environ["YOLO_OFFLINE"] == "1"
    assert ultralytics_detector.os.environ["YOLO_AUTOINSTALL"] == "0"


def test_status_never_loads_or_hashes_weights(tmp_path, capturing_yolo, monkeypatch):
    def no_hash(path):
        raise AssertionError("status must not hash the weights")

    monkeypatch.setattr(ultralytics_detector, "file_sha256", no_hash)
    status = SettingsDetectorProvider(_provider_settings(_weights(tmp_path))).status()
    assert status.available is True and status.weights_sha256 is None
    assert capturing_yolo.instances == []
    assert str(tmp_path) not in status.model_dump_json()


def test_concurrent_detect_calls_never_overlap_in_the_model(tmp_path, capturing_yolo):
    capturing_yolo.predict_delay = 0.03
    path = _stored_file(tmp_path, make_image_bytes())
    detectors = [_detector(tmp_path / str(i)) for i in range(2) if (tmp_path / str(i)).mkdir() is None]

    def run(i: int) -> None:
        detectors[i % 2].detect([ImageRef(image_id=uuid.uuid4(), path=path)])

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(run, range(6)))
    assert capturing_yolo.max_active == 1


def test_concurrent_first_requests_load_the_model_once(tmp_path, capturing_yolo):
    capturing_yolo.init_delay = 0.05
    provider = SettingsDetectorProvider(_provider_settings(_weights(tmp_path)))
    with ThreadPoolExecutor(max_workers=5) as pool:
        detectors = list(pool.map(lambda _: provider.get(), range(5)))
    assert len(capturing_yolo.instances) == 1
    assert all(d is detectors[0] for d in detectors)


class FakeDetector:
    def __init__(self, behavior) -> None:
        self.model = ModelInfo(name="qa-fake", version="1", weights_sha256="c" * 64)
        self.config = CONFIG
        self.behavior = behavior

    def detect(self, images: list[ImageRef]) -> list[Detection]:
        return self.behavior(images[0])


class FakeProvider:
    def __init__(self, detector: FakeDetector) -> None:
        self.detector = detector

    def status(self) -> DetectorStatusRead:
        return DetectorStatusRead(available=True, backend="ultralytics", config=CONFIG)

    def get(self) -> FakeDetector:
        return self.detector


def _install(behavior) -> FakeProvider:
    provider = FakeProvider(FakeDetector(behavior))
    app.dependency_overrides[get_detector_provider] = lambda: provider
    return provider


def _inspection_with_images(client, colors) -> tuple[str, list[str]]:
    iid = client.post("/api/v1/inspections", json={"title": "qa"}).json()["id"]
    ids = []
    for color in colors:
        r = client.post(
            f"/api/v1/inspections/{iid}/images",
            files={"file": ("car.jpg", make_image_bytes(color=color), "image/jpeg")},
        )
        assert r.status_code == 201, r.text
        ids.append(r.json()["id"])
    return iid, ids


@requires_db
def test_failure_details_never_reach_the_client(client, db_session):
    def explode(ref):
        raise RuntimeError(f"CUDA OOM at /very/secret/weights.pt reading {ref.path}")

    _install(explode)
    iid, _ = _inspection_with_images(client, [(1, 2, 3)])
    r = client.post(f"/api/v1/inspections/{iid}/detections")
    assert r.status_code == 200
    run = r.json()["items"][0]
    assert (run["status"], run["error_message"]) == ("failed", "Detection failed for this image.")
    assert "secret" not in r.text and "CUDA" not in r.text and "storage" not in r.text
    listed = client.get(f"/api/v1/inspections/{iid}/detections").text
    assert "secret" not in listed


@requires_db
def test_unstorable_class_name_fails_only_that_run(client, db_session):
    def by_color(ref):
        bad = Image.open(ref.path).convert("RGB").getpixel((40, 20))[0] > 100
        name = "x" * 101 if bad else "dent"
        return [Detection(image_id=ref.image_id, damage_type=name, confidence=0.9, bounding_box=BOX)]

    _install(by_color)
    iid, (bad_id, good_id) = _inspection_with_images(client, [(250, 10, 10), (10, 10, 10)])
    r = client.post(f"/api/v1/inspections/{iid}/detections")
    assert r.status_code == 200
    runs = {run["inspection_image_id"]: run for run in r.json()["items"]}
    assert runs[bad_id]["status"] == "failed" and runs[bad_id]["predictions"] == []
    assert runs[bad_id]["error_message"] == "The detection results could not be saved."
    assert runs[good_id]["status"] == "completed" and len(runs[good_id]["predictions"]) == 1
    assert db_session.query(DamagePrediction).count() == 1


@requires_db
def test_duplicate_image_ids_create_one_run_per_image(client):
    _install(lambda ref: [])
    iid, (image_id,) = _inspection_with_images(client, [(5, 5, 5)])
    r = client.post(f"/api/v1/inspections/{iid}/detections", json={"image_ids": [image_id, image_id]})
    assert r.status_code == 200 and len(r.json()["items"]) == 1


@requires_db
def test_stale_running_run_is_reported_truthfully(client, db_session):
    _install(lambda ref: [])
    iid, (image_id,) = _inspection_with_images(client, [(7, 7, 7)])
    assert client.post(f"/api/v1/inspections/{iid}/detections").status_code == 200
    crashed = DetectionRun(
        inspection_image_id=uuid.UUID(image_id), model_name="qa-fake", model_version="1",
        config=CONFIG.model_dump(mode="json"), status=RunStatus.RUNNING,
        created_at=datetime.now(UTC) + timedelta(seconds=5),
    )
    db_session.add(crashed)
    db_session.commit()
    latest = client.get(f"/api/v1/inspections/{iid}/detections").json()["items"]
    assert len(latest) == 1
    assert (latest[0]["id"], latest[0]["status"]) == (str(crashed.id), "running")
    assert latest[0]["completed_at"] is None and latest[0]["duration_ms"] is None
    everything = client.get(f"/api/v1/inspections/{iid}/detections?latest=false").json()["items"]
    assert [run["status"] for run in everything] == ["running", "completed"]


@requires_db
def test_concurrent_detection_requests_are_independent_and_latest_is_deterministic(
    client, db_engine, storage
):
    provider = _install(lambda ref: [Detection(ref.image_id, "dent", 0.5, BOX)])
    iid, image_ids = _inspection_with_images(client, [(20, 20, 20), (90, 90, 90)])
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)

    def request(_):
        with factory() as session:
            return [r.id for r in detection_service.run_detection(
                session, storage, provider, uuid.UUID(iid), None)]

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(request, range(3)))
    assert all(len(ids) == 2 for ids in results)
    assert len({run_id for ids in results for run_id in ids}) == 6
    first = client.get(f"/api/v1/inspections/{iid}/detections").json()["items"]
    second = client.get(f"/api/v1/inspections/{iid}/detections").json()["items"]
    assert [r["inspection_image_id"] for r in first] == image_ids
    assert first == second and all(r["status"] == "completed" for r in first)


def _allowed_origin() -> str:
    return next(m for m in app.user_middleware if m.cls is CORSMiddleware).kwargs["allow_origins"][0]


@requires_db
def test_body_cap_counts_framing_allowance_only_once(client, test_settings):
    iid = client.post("/api/v1/inspections", json={"title": "qa"}).json()["id"]
    r = client.post(
        f"/api/v1/inspections/{iid}/images",
        content=b"x" * (test_settings.max_upload_bytes + 64 * 1024 + 1),
        headers={"Content-Type": "multipart/form-data; boundary=zz"},
    )
    assert (r.status_code, r.json()["error"]["code"]) == (413, "file_too_large")


@requires_db
def test_body_cap_rejection_is_readable_by_the_browser_frontend(client, test_settings):
    iid = client.post("/api/v1/inspections", json={"title": "qa"}).json()["id"]
    origin = _allowed_origin()
    r = client.post(
        f"/api/v1/inspections/{iid}/images",
        content=b"x" * (test_settings.max_upload_bytes + 64 * 1024 + 1),
        headers={"Content-Type": "multipart/form-data; boundary=zz", "Origin": origin},
    )
    assert r.status_code == 413
    assert r.headers.get("access-control-allow-origin") == origin
