"""Hardening: run liveness / recovery races, image-size validation, checkpoint safeguards."""

import contextlib
import importlib.machinery
import logging
import os
import sys
import threading
import time
import types
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.models import DamagePrediction, DetectionRun, Inspection
from app.models.run_status import RunStatus
from app.schemas.detection import DetectionRunRead
from app.schemas.runs import DetectionConfig, StoredDetectionConfig, check_image_size_multiple
from app.services import detection_service as svc
from app.services.detection import provider as provider_module
from app.services.detection import ultralytics_detector as ud
from app.services.types import Detection, ImageRef, ModelInfo
from tests.conftest import requires_db
from tests.factories import make_image, make_run

NO_DB = "postgresql+psycopg://none:none@127.0.0.1:1/none"
CONFIG = DetectionConfig(confidence_threshold=0.25, iou_threshold=0.45, image_size=640, max_detections=100)
SHA_OF_W = "50e721e49c013f00c62cf59f2163542a9d8df02464efeb615d31051b0fddc326"


class FakeSession:
    """Just enough Session for _run_one / _finish; `on_refresh` simulates a concurrent writer."""

    def __init__(self, on_refresh=None):
        self.added: list[object] = []
        self.commits = 0
        self.rollbacks = 0
        self.refreshed: list[bool] = []
        self.on_refresh = on_refresh

    def add(self, obj):
        self.added.append(obj)

    def add_all(self, objs):
        self.added.extend(objs)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1
        self.added = self.added[:1]

    def refresh(self, obj, with_for_update=False):
        self.refreshed.append(with_for_update)
        if self.on_refresh:
            self.on_refresh(obj)

    def execute(self, stmt):
        return types.SimpleNamespace(rowcount=0)


class FakeStorage:
    def __init__(self, path):
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


class RecordingHeartbeat:
    def __init__(self):
        self.events: list[tuple[str, uuid.UUID]] = []

    def __call__(self, run_id):
        @contextlib.contextmanager
        def cm():
            self.events.append(("enter", run_id))
            try:
                yield
            finally:
                self.events.append(("exit", run_id))

        return cm()


class FakeEngine:
    """engine.begin() context; records statements; optionally fails."""

    def __init__(self, fail=False):
        self.statements: list[object] = []
        self.fail = fail
        self.lock = threading.Lock()

    @contextlib.contextmanager
    def begin(self):
        if self.fail:
            raise RuntimeError("database unreachable")
        conn = types.SimpleNamespace(execute=lambda stmt: self._record(stmt))
        yield conn

    def _record(self, stmt):
        with self.lock:
            self.statements.append(stmt)


@pytest.fixture
def target(tmp_path):
    f = tmp_path / "i.jpg"
    f.write_bytes(b"x")
    return svc._Target(image_id=uuid.uuid4(), storage_path="a/b.jpg"), FakeStorage(f)


def one_dent(images):
    return [Detection(images[0].image_id, "dent", 0.9, {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5})]


def _fail_like_recovery(obj):
    obj.status = RunStatus.FAILED
    obj.error_message = svc.INTERRUPTED_MESSAGE
    obj.completed_at = datetime.now(UTC)


def test_new_run_starts_with_a_heartbeat(target):
    tgt, storage = target
    db = FakeSession()
    svc._run_one(db, storage, FakeDetector(lambda _: []), tgt, RecordingHeartbeat())
    run = db.added[0]
    assert run.heartbeat_at is not None and run.heartbeat_at == run.created_at


def test_worker_locks_the_row_and_rechecks_before_its_final_write(target):
    tgt, storage = target
    db = FakeSession()
    svc._run_one(db, storage, FakeDetector(one_dent), tgt, RecordingHeartbeat())
    assert db.refreshed == [True]
    run = db.added[0]
    assert run.status is RunStatus.COMPLETED and run.error_message is None
    assert len(db.added) == 2


def test_worker_keeps_a_failure_written_by_recovery_and_discards_predictions(target):
    tgt, storage = target
    db = FakeSession(on_refresh=_fail_like_recovery)
    run_id = svc._run_one(db, storage, FakeDetector(one_dent), tgt, RecordingHeartbeat())
    run = db.added[0]
    assert run_id == run.id
    assert (run.status, run.error_message) == (RunStatus.FAILED, svc.INTERRUPTED_MESSAGE)
    assert db.added == [run]
    assert db.commits == 1


def test_worker_does_not_overwrite_recovery_failure_on_its_own_failure_path(target):
    tgt, storage = target
    db = FakeSession(on_refresh=_fail_like_recovery)

    def boom(_):
        raise RuntimeError("x")

    svc._run_one(db, storage, FakeDetector(boom), tgt, RecordingHeartbeat())
    run = db.added[0]
    assert run.error_message == svc.INTERRUPTED_MESSAGE
    assert db.commits == 1


def test_finish_on_a_vanished_row_returns_false(target):
    tgt, storage = target

    def vanished(_):
        raise RuntimeError("row is gone")

    db = FakeSession(on_refresh=vanished)
    assert svc._run_one(db, storage, FakeDetector(lambda _: []), tgt, RecordingHeartbeat()) is None


def test_heartbeat_wraps_inference_and_stops_even_on_interruption(target):
    tgt, storage = target
    hb = RecordingHeartbeat()
    run_id = svc._run_one(FakeSession(), storage, FakeDetector(lambda _: []), tgt, hb)
    assert hb.events == [("enter", run_id), ("exit", run_id)]

    hb = RecordingHeartbeat()

    def interrupted(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        svc._run_one(FakeSession(), storage, FakeDetector(interrupted), tgt, hb)
    assert [e for e, _ in hb.events] == ["enter", "exit"]


def test_heartbeat_thread_beats_while_open_and_stops_on_exit():
    engine = FakeEngine()
    run_id = uuid.uuid4()
    with svc.RunHeartbeat(engine, run_id, 0.01):
        deadline = time.monotonic() + 2
        while len(engine.statements) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
    beats = len(engine.statements)
    assert beats >= 2
    time.sleep(0.05)
    assert len(engine.statements) == beats
    sql = str(engine.statements[0].compile()).lower()
    assert sql.startswith("update detection_runs set heartbeat_at")
    assert "detection_runs.id" in sql and "detection_runs.status" in sql


def test_heartbeat_failure_is_logged_never_raised(caplog):
    hb = svc.RunHeartbeat(FakeEngine(fail=True), uuid.uuid4(), 1)
    with caplog.at_level(logging.WARNING):
        hb.beat()
    assert "Could not record heartbeat" in caplog.text


def test_heartbeat_interval_must_be_positive():
    with pytest.raises(ValueError):
        svc.RunHeartbeat(FakeEngine(), uuid.uuid4(), 0)


def test_settings_stale_threshold_must_cover_several_heartbeats():
    s = Settings(database_url=NO_DB)
    assert (s.detector_heartbeat_seconds, s.detector_stale_run_minutes) == (30, 30)
    Settings(database_url=NO_DB, detector_stale_run_minutes=2, detector_heartbeat_seconds=30)
    with pytest.raises(ValidationError):
        Settings(database_url=NO_DB, detector_stale_run_minutes=1, detector_heartbeat_seconds=30)
    with pytest.raises(ValidationError):
        Settings(database_url=NO_DB, detector_heartbeat_seconds=0)


def _image(db_session):
    inspection = Inspection(title="v")
    image = make_image(inspection)
    db_session.add_all([inspection, image])
    db_session.commit()
    return image


def _stale_running_run(image_id):
    return DetectionRun(
        inspection_image_id=image_id, model_name="yolo", model_version="1",
        status=RunStatus.RUNNING, created_at=datetime.now(UTC) - timedelta(hours=2),
    )


@requires_db
def test_recovery_spares_an_old_run_whose_worker_is_still_beating(db_session):
    image = _image(db_session)
    now = datetime.now(UTC)
    live = make_run(image, "live", status=RunStatus.RUNNING, created_at=now - timedelta(hours=5),
                    heartbeat_at=now - timedelta(seconds=20))
    dead = make_run(image, "dead", status=RunStatus.RUNNING, created_at=now - timedelta(hours=5),
                    heartbeat_at=now - timedelta(hours=1))
    legacy = make_run(image, "legacy", status=RunStatus.RUNNING, created_at=now - timedelta(hours=2))
    young_legacy = make_run(image, "young", status=RunStatus.RUNNING, created_at=now - timedelta(minutes=5))
    db_session.add_all([live, dead, legacy, young_legacy])
    db_session.commit()

    assert svc.fail_stale_runs(db_session, timedelta(minutes=30)) == 2
    db_session.expire_all()
    assert live.status is RunStatus.RUNNING and young_legacy.status is RunStatus.RUNNING
    assert dead.status is RunStatus.FAILED and legacy.status is RunStatus.FAILED
    assert dead.error_message == svc.INTERRUPTED_MESSAGE and dead.completed_at is not None
    assert svc.fail_stale_runs(db_session, timedelta(minutes=30)) == 0


@requires_db
def test_competing_recovery_processes_fail_each_run_exactly_once(db_engine, db_session):
    image = _image(db_session)
    old = datetime.now(UTC) - timedelta(hours=2)
    db_session.add_all([make_run(image, str(i), status=RunStatus.RUNNING, created_at=old) for i in range(5)])
    db_session.commit()
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    barrier = threading.Barrier(4)
    counts: list[int] = []

    def recover():
        with factory() as s:
            barrier.wait()
            counts.append(svc.fail_stale_runs(s, timedelta(minutes=30)))

    threads = [threading.Thread(target=recover) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sum(counts) == 5
    statuses = db_session.scalars(select(DetectionRun.status)).all()
    assert statuses == [RunStatus.FAILED] * 5


@requires_db
def test_worker_finishing_after_recovery_keeps_the_failure(db_engine, db_session, tmp_path):
    image = _image(db_session)
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    with factory() as worker_db:
        run = _stale_running_run(image.id)
        worker_db.add(run)
        worker_db.commit()
        with factory() as recovery_db:
            assert svc.fail_stale_runs(recovery_db, timedelta(minutes=30)) == 1
        prediction = DamagePrediction(detection_run_id=run.id, damage_type="dent", confidence=0.9)
        assert svc._finish(worker_db, run, run.id, RunStatus.COMPLETED, None, [prediction]) is True
    db_session.expire_all()
    stored = db_session.get(DetectionRun, run.id)
    assert (stored.status, stored.error_message) == (RunStatus.FAILED, svc.INTERRUPTED_MESSAGE)
    assert db_session.scalars(select(DamagePrediction)).all() == []


@requires_db
def test_recovery_waits_for_a_finishing_worker_and_then_changes_nothing(db_engine, db_session):
    image = _image(db_session)
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    with factory() as worker_db:
        run = _stale_running_run(image.id)
        worker_db.add(run)
        worker_db.commit()
        worker_db.refresh(run, with_for_update=True)
        counts: list[int] = []

        def recover():
            with factory() as recovery_db:
                counts.append(svc.fail_stale_runs(recovery_db, timedelta(minutes=30)))

        t = threading.Thread(target=recover)
        t.start()
        time.sleep(0.3)
        assert t.is_alive()
        run.status, run.completed_at = RunStatus.COMPLETED, datetime.now(UTC)
        worker_db.commit()
        t.join(10)
    assert counts == [0]
    db_session.expire_all()
    stored = db_session.get(DetectionRun, run.id)
    assert (stored.status, stored.error_message) == (RunStatus.COMPLETED, None)


@requires_db
def test_heartbeat_refreshes_the_stored_row_only_while_running(db_engine, db_session):
    image = _image(db_session)
    before = datetime.now(UTC) - timedelta(hours=1)
    running = make_run(image, "r", status=RunStatus.RUNNING, heartbeat_at=before)
    done = make_run(image, "d", status=RunStatus.COMPLETED, completed_at=before, heartbeat_at=before)
    db_session.add_all([running, done])
    db_session.commit()
    svc.RunHeartbeat(db_engine, running.id, 60).beat()
    svc.RunHeartbeat(db_engine, done.id, 60).beat()
    db_session.expire_all()
    assert running.heartbeat_at > before and done.heartbeat_at == before


@pytest.mark.parametrize("bad", [0, -32, -1, 1, 31, 33, 100, 641, 650])
def test_image_size_rejects_non_multiples_of_32_and_non_positive(bad):
    with pytest.raises(ValidationError):
        DetectionConfig(**{**CONFIG.model_dump(), "image_size": bad})
    with pytest.raises(ValidationError):
        Settings(database_url=NO_DB, detector_image_size=bad)
    with pytest.raises(ValueError):
        check_image_size_multiple(bad)


@pytest.mark.parametrize("good", [32, 320, 640, 1280])
def test_image_size_accepts_multiples_unchanged(good):
    assert DetectionConfig(**{**CONFIG.model_dump(), "image_size": good}).image_size == good
    assert Settings(database_url=NO_DB, detector_image_size=good).detector_image_size == good


def test_historical_run_with_unrounded_image_size_stays_readable():
    stored = {**CONFIG.model_dump(mode="json"), "image_size": 650}
    assert StoredDetectionConfig.model_validate(stored).image_size == 650
    run = types.SimpleNamespace(
        id=uuid.uuid4(), inspection_image_id=uuid.uuid4(), model_name="m", model_version="1",
        weights_sha256=None, config=stored, status=RunStatus.COMPLETED, error_message=None,
        created_at=datetime.now(UTC), completed_at=datetime.now(UTC), predictions=[],
    )
    assert DetectionRunRead.from_run(run).config.image_size == 650
    with pytest.raises(ValidationError):
        StoredDetectionConfig.model_validate({**stored, "image_size": 0})


def install_fake_ultralytics(monkeypatch, *, safe_load_flag=True, supported=True, stride=None, layout=True):
    """Fake `ultralytics` with the 8.4.191 attributes confirm_safe_load_active reads."""
    pkg = types.ModuleType("ultralytics")
    pkg.__spec__ = importlib.machinery.ModuleSpec("ultralytics", None)
    nn = types.ModuleType("ultralytics.nn")
    tasks = types.ModuleType("ultralytics.nn.tasks")
    if layout:
        tasks.SAFE_LOAD = safe_load_flag
        tasks._SafeLoad = types.SimpleNamespace(SUPPORTED=supported)
    constructed: list[str] = []

    class YOLO:
        names = {0: "dent"}

        def __init__(self, path, task):
            constructed.append(path)
            if stride is not None:
                self.model = types.SimpleNamespace(stride=stride)

    pkg.YOLO = YOLO
    for name, module in (("ultralytics", pkg), ("ultralytics.nn", nn), ("ultralytics.nn.tasks", tasks)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("ULTRALYTICS_SAFE_LOAD", "0")
    monkeypatch.delenv("ULTRALYTICS_SAFE_LOAD")
    return constructed


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / "m.pt"
    path.write_bytes(b"w")
    return path


def _load(weights, **kw):
    return ud.UltralyticsDetector(weights, "m", "1", CONFIG, **kw)


def test_safe_load_succeeds_only_when_ultralytics_reports_it_active(monkeypatch, weights):
    constructed = install_fake_ultralytics(monkeypatch, safe_load_flag=True, supported=True)
    _load(weights, safe_load=True)
    assert constructed == [str(weights)] and os.environ["ULTRALYTICS_SAFE_LOAD"] == "1"


def test_safe_load_refused_when_ultralytics_was_imported_without_it(monkeypatch, weights):
    constructed = install_fake_ultralytics(monkeypatch, safe_load_flag=False)
    with pytest.raises(ud.UnsafeLoadRefused, match="imported before"):
        _load(weights, safe_load=True)
    assert constructed == []
    assert os.environ.get("ULTRALYTICS_SAFE_LOAD") == "1"


def test_safe_load_refused_when_torch_cannot_restrict(monkeypatch, weights):
    constructed = install_fake_ultralytics(monkeypatch, supported=False)
    with pytest.raises(ud.UnsafeLoadRefused, match="torch >= 2.6"):
        _load(weights, safe_load=True)
    assert constructed == []


def test_safe_load_refused_when_loader_internals_are_unknown(monkeypatch, weights):
    constructed = install_fake_ultralytics(monkeypatch, layout=False)
    with pytest.raises(ud.UnsafeLoadRefused):
        _load(weights, safe_load=True)
    assert constructed == []


def test_safe_load_off_skips_the_check_and_leaves_env_alone(monkeypatch, weights):
    constructed = install_fake_ultralytics(monkeypatch, safe_load_flag=False, supported=False)
    _load(weights)
    assert constructed == [str(weights)] and "ULTRALYTICS_SAFE_LOAD" not in os.environ


def test_pinned_digest_match_loads(monkeypatch, weights):
    constructed = install_fake_ultralytics(monkeypatch)
    detector = _load(weights, expected_sha256=SHA_OF_W)
    assert constructed == [str(weights)] and detector.model.weights_sha256 == SHA_OF_W


def test_pinned_digest_mismatch_refused_before_ultralytics_is_imported(monkeypatch, weights):
    constructed = install_fake_ultralytics(monkeypatch)
    imported = []
    monkeypatch.setattr(ud, "_import_yolo", lambda: imported.append(1))
    with pytest.raises(ud.UnsafeLoadRefused, match="DETECTOR_WEIGHTS_SHA256"):
        _load(weights, expected_sha256="0" * 64, safe_load=True)
    assert imported == [] and constructed == []


def test_missing_pin_loads_with_a_warning(monkeypatch, weights, caplog):
    install_fake_ultralytics(monkeypatch)
    with caplog.at_level(logging.WARNING):
        _load(weights)
    assert "No DETECTOR_WEIGHTS_SHA256 pin configured" in caplog.text


def test_settings_pin_is_validated_and_normalised():
    assert Settings(database_url=NO_DB).detector_weights_sha256 is None
    assert Settings(database_url=NO_DB, detector_weights_sha256=SHA_OF_W.upper()).detector_weights_sha256 == SHA_OF_W
    for bad in ("abc", "g" * 64, "a" * 63):
        with pytest.raises(ValidationError):
            Settings(database_url=NO_DB, detector_weights_sha256=bad)


def test_provider_reports_a_refused_safeguard_without_internals(monkeypatch, weights):
    install_fake_ultralytics(monkeypatch)
    provider_module.clear_detector_cache()
    settings = Settings(
        database_url=NO_DB, detector_backend="ultralytics", detector_weights_path=weights,
        detector_model_name="m", detector_model_version="1", detector_weights_sha256="0" * 64,
    )
    with pytest.raises(provider_module.DetectorUnavailable) as exc:
        provider_module.SettingsDetectorProvider(settings).get()
    assert "safeguard" in str(exc.value) and str(weights) not in str(exc.value)
    provider_module.clear_detector_cache()


def test_image_size_checked_against_the_loaded_models_own_stride(monkeypatch, weights):
    install_fake_ultralytics(monkeypatch, stride=[8.0, 16.0, 32.0, 64.0])
    cfg = DetectionConfig(**{**CONFIG.model_dump(), "image_size": 352})
    with pytest.raises(ValueError, match="stride 64"):
        ud.UltralyticsDetector(weights, "m", "1", cfg)
    ud.UltralyticsDetector(weights, "m", "1", CONFIG)


def test_unknown_model_stride_falls_back_to_the_32_rule(monkeypatch, weights, caplog):
    install_fake_ultralytics(monkeypatch)
    with caplog.at_level(logging.WARNING):
        _load(weights)
    assert "Could not read the model stride" in caplog.text
    assert ud.model_stride(types.SimpleNamespace(model=types.SimpleNamespace(stride=32))) == 32


def test_heartbeat_column_matches_migration_0005():
    import importlib.util

    from tests.conftest import BACKEND_DIR

    path = BACKEND_DIR / "alembic" / "versions" / "0005_detection_run_heartbeat.py"
    spec = importlib.util.spec_from_file_location("mig0005", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (module.revision, module.down_revision) == ("0005", "0004")
    column = DetectionRun.__table__.c.heartbeat_at
    assert column.nullable and column.type.timezone and column.server_default is None
