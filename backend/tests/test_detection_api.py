"""Detection endpoints and persistence, with a fake detector (no ultralytics needed)."""

import uuid
from collections.abc import Callable

import pytest
from sqlalchemy import select, text

from app.api.deps import get_detector_provider
from app.main import app
from app.models import DamagePrediction, DetectionRun, InspectionImage
from app.models.run_status import RunStatus
from app.schemas.detection import DetectorStatusRead
from app.schemas.runs import DetectionConfig
from app.services.detection import DetectorUnavailable
from app.services.types import Detection, ImageRef, ModelInfo
from tests.image_utils import make_image_bytes

CONFIG = DetectionConfig(confidence_threshold=0.25, iou_threshold=0.45, image_size=640, max_detections=100)
BOX = {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6}

Behavior = Callable[[ImageRef], list[Detection]]


class FakeDetector:
    def __init__(self, behavior: Behavior) -> None:
        self.model = ModelInfo(name="fake-yolo", version="1.0", weights_sha256="b" * 64)
        self.config = CONFIG
        self.behavior = behavior
        self.calls: list[ImageRef] = []

    def detect(self, images: list[ImageRef]) -> list[Detection]:
        assert len(images) == 1
        self.calls.append(images[0])
        return self.behavior(images[0])


class FakeProvider:
    def __init__(self, detector: FakeDetector | None) -> None:
        self.detector = detector
        self.get_calls = 0

    def status(self) -> DetectorStatusRead:
        if self.detector is None:
            return DetectorStatusRead(available=False, backend="none", reason="disabled")
        return DetectorStatusRead(available=True, backend="ultralytics", config=CONFIG)

    def get(self) -> FakeDetector:
        self.get_calls += 1
        if self.detector is None:
            raise DetectorUnavailable("Damage detection is disabled (DETECTOR_BACKEND=none).")
        return self.detector


def one_dent(ref: ImageRef) -> list[Detection]:
    return [
        Detection(image_id=ref.image_id, damage_type="scratch", confidence=0.4, bounding_box=BOX),
        Detection(image_id=ref.image_id, damage_type="dent", confidence=0.9, bounding_box=BOX),
    ]


@pytest.fixture
def use_detector(client) -> Callable[[Behavior | None], FakeProvider]:
    def install(behavior: Behavior | None) -> FakeProvider:
        provider = FakeProvider(FakeDetector(behavior) if behavior else None)
        app.dependency_overrides[get_detector_provider] = lambda: provider
        return provider

    return install


def create_inspection(client) -> str:
    response = client.post("/api/v1/inspections", json={"title": "t"})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def upload(client, inspection_id: str, color: tuple[int, int, int]) -> str:
    response = client.post(
        f"/api/v1/inspections/{inspection_id}/images",
        files={"file": ("car.jpg", make_image_bytes(color=color), "image/jpeg")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def detections_url(inspection_id: str) -> str:
    return f"/api/v1/inspections/{inspection_id}/detections"


def test_happy_path_persists_runs_and_predictions(client, db_session, use_detector):
    use_detector(lambda ref: one_dent(ref) if ref.image_id == first else [])
    inspection_id = create_inspection(client)
    first = uuid.UUID(upload(client, inspection_id, (200, 0, 0)))
    second = uuid.UUID(upload(client, inspection_id, (0, 200, 0)))

    response = client.post(detections_url(inspection_id))

    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert [i["inspection_image_id"] for i in items] == [str(first), str(second)]
    run = items[0]
    assert run["status"] == "completed"
    assert run["error_message"] is None
    assert run["model_name"] == "fake-yolo" and run["model_version"] == "1.0"
    assert run["weights_sha256"] == "b" * 64
    assert run["config"]["confidence_threshold"] == 0.25
    assert run["completed_at"] is not None and run["duration_ms"] >= 0
    assert [p["damage_type"] for p in run["predictions"]] == ["dent", "scratch"]
    assert run["predictions"][0]["bounding_box"] == BOX
    assert items[1]["status"] == "completed" and items[1]["predictions"] == []

    db_session.expire_all()
    stored = db_session.scalars(select(DetectionRun).order_by(DetectionRun.created_at)).all()
    assert [r.status for r in stored] == [RunStatus.COMPLETED, RunStatus.COMPLETED]
    stored_config = dict(stored[0].config)
    assert set(stored_config.pop("runtime")) == {"ultralytics", "torch"}
    assert stored_config == CONFIG.model_dump(mode="json")
    preds = db_session.scalars(select(DamagePrediction)).all()
    assert {p.damage_type for p in preds} == {"dent", "scratch"}
    assert all(p.detection_run_id == stored[0].id for p in preds)


def test_explicit_image_ids_runs_only_those(client, use_detector):
    provider = use_detector(lambda ref: [])
    inspection_id = create_inspection(client)
    upload(client, inspection_id, (200, 0, 0))
    second = upload(client, inspection_id, (0, 200, 0))

    response = client.post(detections_url(inspection_id), json={"image_ids": [second, second]})

    assert response.status_code == 200
    assert [i["inspection_image_id"] for i in response.json()["items"]] == [second]
    assert len(provider.detector.calls) == 1


def test_one_image_failing_does_not_abort_others(client, db_session, use_detector):
    def behavior(ref: ImageRef) -> list[Detection]:
        if ref.image_id == first:
            raise RuntimeError(f"CUDA exploded reading {ref.path}")
        return one_dent(ref)

    use_detector(behavior)
    inspection_id = create_inspection(client)
    first = uuid.UUID(upload(client, inspection_id, (200, 0, 0)))
    upload(client, inspection_id, (0, 200, 0))

    response = client.post(detections_url(inspection_id))

    assert response.status_code == 200
    failed, completed = response.json()["items"]
    assert failed["status"] == "failed"
    assert failed["error_message"] == "Detection failed for this image."
    assert failed["predictions"] == [] and failed["completed_at"] is not None
    assert "CUDA" not in response.text and "/" not in failed["error_message"]
    assert completed["status"] == "completed" and len(completed["predictions"]) == 2


def test_invalid_detector_output_fails_the_run_without_partial_predictions(client, db_session, use_detector):
    def behavior(ref: ImageRef) -> list[Detection]:
        bad = {"x_min": 0.6, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6}
        return [*one_dent(ref), Detection(image_id=ref.image_id, damage_type="dent", confidence=0.5, bounding_box=bad)]

    use_detector(behavior)
    inspection_id = create_inspection(client)
    upload(client, inspection_id, (200, 0, 0))

    run = client.post(detections_url(inspection_id)).json()["items"][0]

    assert run["status"] == "failed"
    assert run["error_message"] == "The detector returned an invalid bounding box."
    db_session.expire_all()
    assert db_session.scalars(select(DamagePrediction)).all() == []


@pytest.mark.parametrize(
    "detection",
    [
        Detection(image_id=uuid.uuid4(), damage_type="dent", confidence=0.5),
        None,
    ],
)
def test_detector_contract_violations_fail_the_run(client, use_detector, detection):
    def behavior(ref: ImageRef) -> list[Detection]:
        return [detection or Detection(image_id=ref.image_id, damage_type="dent", confidence=1.5)]

    use_detector(behavior)
    inspection_id = create_inspection(client)
    upload(client, inspection_id, (200, 0, 0))

    run = client.post(detections_url(inspection_id)).json()["items"][0]

    assert run["status"] == "failed"
    assert run["error_message"].startswith("The detector returned")


def test_missing_file_fails_that_run_without_calling_detector(client, db_session, storage, use_detector):
    provider = use_detector(one_dent)
    inspection_id = create_inspection(client)
    image_id = upload(client, inspection_id, (200, 0, 0))
    storage.delete(db_session.get(InspectionImage, uuid.UUID(image_id)).storage_path)

    run = client.post(detections_url(inspection_id)).json()["items"][0]

    assert run["status"] == "failed"
    assert run["error_message"] == "The image file is not available."
    assert provider.detector.calls == []


def test_image_deleted_during_inference_is_dropped_from_response(client, db_engine, use_detector):
    def behavior(ref: ImageRef) -> list[Detection]:
        with db_engine.begin() as conn:
            conn.execute(text("DELETE FROM inspection_images WHERE id = :id"), {"id": ref.image_id})
        return one_dent(ref)

    use_detector(behavior)
    inspection_id = create_inspection(client)
    upload(client, inspection_id, (200, 0, 0))

    response = client.post(detections_url(inspection_id))

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_run_is_committed_as_running_before_inference(client, db_engine, use_detector):
    seen: list[tuple[str, object]] = []

    def behavior(ref: ImageRef) -> list[Detection]:
        with db_engine.connect() as conn:
            seen.extend(conn.execute(text("SELECT status, completed_at FROM detection_runs")).all())
        return []

    use_detector(behavior)
    inspection_id = create_inspection(client)
    upload(client, inspection_id, (200, 0, 0))

    client.post(detections_url(inspection_id))

    assert seen == [("running", None)]


def test_unavailable_detector_returns_503_and_creates_no_runs(client, db_session, use_detector):
    use_detector(None)
    inspection_id = create_inspection(client)
    upload(client, inspection_id, (200, 0, 0))

    response = client.post(detections_url(inspection_id))

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "detector_unavailable"
    assert db_session.scalars(select(DetectionRun)).all() == []


def test_missing_inspection_is_404_before_detector_is_loaded(client, use_detector):
    provider = use_detector(one_dent)
    response = client.post(detections_url(str(uuid.uuid4())))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert provider.get_calls == 0


def test_image_from_another_inspection_is_404(client, use_detector):
    use_detector(one_dent)
    inspection_id = create_inspection(client)
    upload(client, inspection_id, (200, 0, 0))
    other = create_inspection(client)
    foreign = upload(client, other, (0, 0, 200))

    response = client.post(detections_url(inspection_id), json={"image_ids": [foreign]})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_inspection_without_images_is_409(client, use_detector):
    use_detector(one_dent)
    response = client.post(detections_url(create_inspection(client)))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "no_images"


@pytest.mark.parametrize("body", [{"image_ids": []}, {"image_ids": ["not-a-uuid"]}, {"unexpected": 1}])
def test_invalid_body_is_422(client, use_detector, body):
    use_detector(one_dent)
    inspection_id = create_inspection(client)
    response = client.post(detections_url(inspection_id), json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_listing_latest_and_all_runs(client, use_detector):
    use_detector(one_dent)
    inspection_id = create_inspection(client)
    first = upload(client, inspection_id, (200, 0, 0))
    second = upload(client, inspection_id, (0, 200, 0))
    initial = {i["id"] for i in client.post(detections_url(inspection_id)).json()["items"]}
    rerun = client.post(detections_url(inspection_id)).json()["items"]

    latest = client.get(detections_url(inspection_id)).json()["items"]
    everything = client.get(detections_url(inspection_id), params={"latest": "false"}).json()["items"]

    assert [i["inspection_image_id"] for i in latest] == [first, second]
    assert [i["id"] for i in latest] == [r["id"] for r in rerun]
    assert len(everything) == 4
    assert {i["id"] for i in everything[2:]} == initial
    created = [i["created_at"] for i in everything]
    assert created == sorted(created, reverse=True)


def test_listing_empty_and_missing_inspection(client):
    inspection_id = create_inspection(client)
    assert client.get(detections_url(inspection_id)).json() == {"items": []}
    response = client.get(detections_url(str(uuid.uuid4())))
    assert response.status_code == 404


def test_runs_cascade_when_image_deleted(client, db_session, use_detector):
    use_detector(one_dent)
    inspection_id = create_inspection(client)
    image_id = upload(client, inspection_id, (200, 0, 0))
    client.post(detections_url(inspection_id))

    assert client.delete(f"/api/v1/inspections/{inspection_id}/images/{image_id}").status_code == 204

    db_session.expire_all()
    assert db_session.scalars(select(DetectionRun)).all() == []
    assert db_session.scalars(select(DamagePrediction)).all() == []
