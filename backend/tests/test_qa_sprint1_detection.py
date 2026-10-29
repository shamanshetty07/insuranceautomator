"""Sprint 1 (Agent C): stuck-run handling, runtime versions on runs, opt-in safe load.

Mostly DB-less (fake session/detector). Tests marked requires_db exercise the real
conditional UPDATE and are skipped without TEST_DATABASE_URL.
"""

import contextlib
import importlib.metadata
import os
import sys
import types
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.models import DetectionRun
from app.models.run_status import RunStatus
from app.schemas.detection import DetectionRunRead
from app.schemas.runs import DetectionConfig, StoredDetectionConfig
from app.services import detection_service as svc
from app.services.detection import runtime as runtime_module
from app.services.detection import ultralytics_detector
from app.services.types import Detection, ImageRef, ModelInfo
from tests.conftest import requires_db
from tests.test_detection_hardening import install_fake_ultralytics as _install_fake_ultralytics
from tests.factories import make_image, make_run
from app.models import Inspection

NO_DB = "postgresql+psycopg://none:none@127.0.0.1:1/none"
CONFIG = DetectionConfig(confidence_threshold=0.25, iou_threshold=0.45, image_size=640, max_detections=100)


# --- fakes ---------------------------------------------------------------------


class FakeSession:
    def __init__(self, fail_commits_after: int | None = None):
        self.added: list[object] = []
        self.commits = 0
        self.rollbacks = 0
        self.executed: list[object] = []
        self.fail_commits_after = fail_commits_after  # commit number (1-based) from which commits fail
        self.state_at_commit: list[tuple[str, object]] = []
        self.refreshed: list[tuple[object, bool]] = []
        self.on_refresh = None  # simulate another transaction changing the row before the lock

    def add(self, obj):
        self.added.append(obj)

    def add_all(self, objs):
        self.added.extend(objs)

    def commit(self):
        self.commits += 1
        run = self.added[0]
        self.state_at_commit.append((run.status.value, run.completed_at))
        if self.fail_commits_after is not None and self.commits >= self.fail_commits_after:
            raise RuntimeError("commit failed")

    def rollback(self):
        self.rollbacks += 1

    def execute(self, stmt):
        self.executed.append(stmt)

    def refresh(self, obj, with_for_update=False):
        self.refreshed.append((obj, with_for_update))
        if self.on_refresh:
            self.on_refresh(obj)


class FakeStorage:
    def __init__(self, path: Path):
        self._path = path

    def resolve(self, _):
        return self._path


class FakeDetector:
    model = ModelInfo(name="m", version="1", weights_sha256="a" * 64)
    config = CONFIG

    def __init__(self, behaviour):
        self.behaviour = behaviour

    def detect(self, images: list[ImageRef]) -> list[Detection]:
        return self.behaviour(images)


@pytest.fixture
def target(tmp_path):
    f = tmp_path / "i.jpg"
    f.write_bytes(b"x")
    return svc._Target(image_id=uuid.uuid4(), storage_path="a/b.jpg"), FakeStorage(f)


def _no_heartbeat(run_id):
    return contextlib.nullcontext()


def _run_one(db, storage, tgt, behaviour, heartbeat=_no_heartbeat):
    return svc._run_one(db, storage, FakeDetector(behaviour), tgt, heartbeat)


# --- Task A --------------------------------------------------------------------


def test_ordinary_exception_leaves_run_failed_with_safe_message(target):
    tgt, storage = target
    db = FakeSession()

    def boom(_):
        raise RuntimeError("/secret/path CUDA")

    assert _run_one(db, storage, tgt, boom) is not None
    run = db.added[0]
    assert run.status is RunStatus.FAILED and run.completed_at is not None
    assert run.error_message == "Detection failed for this image."


def test_failed_finish_commit_returns_none_and_does_not_raise(target):
    tgt, storage = target
    db = FakeSession(fail_commits_after=2)  # create succeeds, every later commit fails

    def boom(_):
        raise RuntimeError("x")

    assert _run_one(db, storage, tgt, boom) is None
    assert db.rollbacks >= 1  # session left usable


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_base_exception_marks_run_failed_atomically_and_reraises(target, exc):
    tgt, storage = target
    db = FakeSession()

    def interrupted(_):
        raise exc()

    with pytest.raises(exc):
        _run_one(db, storage, tgt, interrupted)
    assert db.rollbacks == 1 and db.commits == 2  # create + interrupted-marker
    sql = str(db.executed[0].compile()).lower()
    assert sql.startswith("update detection_runs")
    assert "detection_runs.status" in sql and "where" in sql  # conditional on still running


def test_base_exception_marker_failure_does_not_mask_original(target):
    tgt, storage = target
    db = FakeSession()

    def execute(_):
        raise RuntimeError("db down")

    db.execute = execute

    def interrupted(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run_one(db, storage, tgt, interrupted)


def test_stale_cutoff_arithmetic():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert svc.stale_cutoff(now, timedelta(minutes=30)) == datetime(2026, 1, 1, 11, 30, tzinfo=UTC)
    with pytest.raises(ValueError):
        svc.stale_cutoff(now, timedelta(0))
    with pytest.raises(ValueError):
        svc.stale_cutoff(now, timedelta(minutes=-1))


def test_stale_setting_default_and_validation():
    assert Settings(database_url=NO_DB).detector_stale_run_minutes == 30
    for bad in (0, -5):
        with pytest.raises(ValidationError):
            Settings(database_url=NO_DB, detector_stale_run_minutes=bad)


def test_fail_stale_runs_issues_one_conditional_update():
    db = FakeSession()
    db.added.append(types.SimpleNamespace(status=RunStatus.RUNNING, completed_at=None))
    db.execute = lambda stmt: (db.executed.append(stmt), types.SimpleNamespace(rowcount=3))[1]
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert svc.fail_stale_runs(db, timedelta(minutes=30), now=now) == 3
    sql = str(db.executed[0].compile()).lower()
    assert "status" in sql and "completed_at" in sql
    assert "coalesce(detection_runs.heartbeat_at, detection_runs.created_at) <" in sql
    assert db.commits == 1


def _db_inspection_image(db_session):
    inspection = Inspection(title="v")
    image = make_image(inspection)
    db_session.add_all([inspection, image])
    db_session.commit()
    return image


@requires_db
def test_fail_stale_runs_only_touches_old_running_runs(db_session):
    image = _db_inspection_image(db_session)
    now = datetime.now(UTC)
    old = make_run(image, "old", status=RunStatus.RUNNING, created_at=now - timedelta(hours=2))
    young = make_run(image, "young", status=RunStatus.RUNNING, created_at=now - timedelta(minutes=5))
    done = make_run(image, "done", status=RunStatus.COMPLETED, created_at=now - timedelta(hours=3),
                    completed_at=now - timedelta(hours=2))
    db_session.add_all([old, young, done])
    db_session.commit()

    assert svc.fail_stale_runs(db_session, timedelta(minutes=30)) == 1
    db_session.expire_all()
    assert (old.status, old.error_message) == (RunStatus.FAILED, svc.INTERRUPTED_MESSAGE)
    assert old.completed_at is not None
    assert young.status is RunStatus.RUNNING and young.completed_at is None
    assert done.status is RunStatus.COMPLETED
    assert svc.fail_stale_runs(db_session, timedelta(minutes=30)) == 0  # idempotent


@requires_db
def test_interrupted_marker_does_not_clobber_a_finished_run(db_session):
    image = _db_inspection_image(db_session)
    finished = make_run(image, status=RunStatus.COMPLETED, completed_at=datetime.now(UTC))
    db_session.add(finished)
    db_session.commit()
    svc._fail_if_running(db_session, finished.id)
    db_session.expire_all()
    assert finished.status is RunStatus.COMPLETED and finished.error_message is None


# --- Task B --------------------------------------------------------------------


def _patch_versions(monkeypatch, mapping):
    def version(name):
        value = mapping[name]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(importlib.metadata, "version", version)


def test_runtime_versions_both_available(monkeypatch):
    _patch_versions(monkeypatch, {"ultralytics": "8.4.191", "torch": "2.15.0"})
    assert runtime_module.runtime_versions().model_dump() == {"ultralytics": "8.4.191", "torch": "2.15.0"}


def test_runtime_versions_missing_or_broken(monkeypatch):
    missing = importlib.metadata.PackageNotFoundError("x")
    _patch_versions(monkeypatch, {"ultralytics": missing, "torch": "2.15.0"})
    assert runtime_module.runtime_versions().model_dump() == {"ultralytics": None, "torch": "2.15.0"}
    _patch_versions(monkeypatch, {"ultralytics": missing, "torch": ValueError("corrupt")})
    assert runtime_module.runtime_versions().model_dump() == {"ultralytics": None, "torch": None}


def test_created_run_records_versions_and_keeps_existing_config(monkeypatch, target):
    _patch_versions(monkeypatch, {"ultralytics": "8.4.191", "torch": "2.15.0"})
    tgt, storage = target
    db = FakeSession()
    _run_one(db, storage, tgt, lambda _: [])
    run = db.added[0]
    assert run.config["runtime"] == {"ultralytics": "8.4.191", "torch": "2.15.0"}
    assert {k: run.config[k] for k in CONFIG.model_dump()} == CONFIG.model_dump(mode="json")
    StoredDetectionConfig.model_validate(run.config)


def test_detector_config_cannot_inject_runtime(monkeypatch):
    _patch_versions(monkeypatch, {"ultralytics": "1", "torch": "2"})
    with pytest.raises(ValidationError):
        DetectionConfig(**CONFIG.model_dump(), runtime={"ultralytics": "forged"})
    detector = FakeDetector(lambda _: [])
    detector.config = types.SimpleNamespace(
        model_dump=lambda mode: {**CONFIG.model_dump(mode=mode), "runtime": {"torch": "forged"}}
    )
    assert svc._run_config(detector)["runtime"] == {"ultralytics": "1", "torch": "2"}


def _read(config: dict) -> DetectionRunRead:
    now = datetime.now(UTC)
    run = types.SimpleNamespace(
        id=uuid.uuid4(), inspection_image_id=uuid.uuid4(), model_name="m", model_version="1",
        weights_sha256=None, config=config, status=RunStatus.RUNNING, error_message=None,
        created_at=now, completed_at=None, predictions=[],
    )
    return DetectionRunRead.from_run(run)


def test_old_runs_without_runtime_still_serialise_and_new_ones_expose_it():
    old = _read(CONFIG.model_dump(mode="json"))
    assert old.model_dump(mode="json")["config"]["runtime"] is None
    new = _read({**CONFIG.model_dump(mode="json"), "runtime": {"ultralytics": "8", "torch": None}})
    assert new.config.runtime.ultralytics == "8" and new.config.runtime.torch is None
    with pytest.raises(ValidationError):
        _read({**CONFIG.model_dump(mode="json"), "runtime": {"python": "3.14"}})  # only package names


# --- Task C --------------------------------------------------------------------


def test_safe_load_is_off_by_default_and_leaves_env_alone(monkeypatch):
    monkeypatch.delenv("ULTRALYTICS_SAFE_LOAD", raising=False)
    ultralytics_detector.apply_safe_load(False)
    assert "ULTRALYTICS_SAFE_LOAD" not in os.environ
    assert Settings(database_url=NO_DB).detector_safe_load is False


def test_safe_load_env_is_set_before_the_lazy_import(monkeypatch, tmp_path):
    monkeypatch.delenv("ULTRALYTICS_SAFE_LOAD", raising=False)
    _install_fake_ultralytics(monkeypatch, safe_load_flag=True)
    fake = sys.modules["ultralytics"]

    class YOLO:
        names = {0: "dent"}

        def __init__(self, path, task):
            pass

    fake.YOLO = YOLO
    seen_at_import = {}
    weights = tmp_path / "m.pt"
    weights.write_bytes(b"w")

    original = ultralytics_detector._import_yolo

    def spying_import():
        seen_at_import["value"] = os.environ.get("ULTRALYTICS_SAFE_LOAD")
        return original()

    monkeypatch.setattr(ultralytics_detector, "_import_yolo", spying_import)
    try:
        ultralytics_detector.UltralyticsDetector(weights, "m", "1", CONFIG, safe_load=True)
    finally:
        os.environ.pop("ULTRALYTICS_SAFE_LOAD", None)
    assert seen_at_import["value"] == "1"
