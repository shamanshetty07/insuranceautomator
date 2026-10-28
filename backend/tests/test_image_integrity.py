"""Partial-failure behaviour between PostgreSQL and the filesystem."""

import hashlib
import io
import logging
import uuid

import pytest
from PIL import Image
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import Inspection, InspectionImage
from app.services import image_service
from app.services.image_storage import ImageStorage
from tests.conftest import requires_db
from tests.image_utils import make_image_bytes

pytestmark = requires_db


@pytest.fixture
def inspection(db_session: Session) -> Inspection:
    insp = Inspection(title="x")
    db_session.add(insp)
    db_session.commit()
    return insp


def _files(storage: ImageStorage) -> list:
    return [p for p in storage.root.rglob("*") if p.is_file()] if storage.root.exists() else []


def _upload(db, storage, settings, insp, data=None, name="a.jpg"):
    return image_service.upload_image(
        db, storage, settings, insp.id, data or make_image_bytes(), name
    )


def test_checksum_and_dimensions_describe_the_stored_normalized_bytes(
    db_session, storage, test_settings, inspection
):
    image = _upload(db_session, storage, test_settings, inspection, make_image_bytes("JPEG", (64, 32), orientation=8))
    on_disk = storage.read(image.storage_path)
    assert hashlib.sha256(on_disk).hexdigest() == image.sha256
    assert len(on_disk) == image.file_size_bytes
    assert Image.open(io.BytesIO(on_disk)).size == (image.width_px, image.height_px) == (32, 64)
    assert image.exif_orientation == 8
    assert image.storage_path == f"{inspection.id}/{image.id}.jpg"


def test_db_failure_on_commit_removes_the_new_file(db_session, storage, test_settings, inspection, monkeypatch):
    def boom():
        raise OperationalError("COMMIT", {}, Exception("db down"))

    monkeypatch.setattr(db_session, "commit", boom)
    with pytest.raises(OperationalError):
        _upload(db_session, storage, test_settings, inspection)
    assert _files(storage) == []
    monkeypatch.undo()
    assert db_session.query(InspectionImage).count() == 0


def test_cleanup_failure_is_logged_with_the_orphan_path(
    db_session, storage, test_settings, inspection, monkeypatch, caplog
):
    monkeypatch.setattr(db_session, "commit", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    monkeypatch.setattr(storage, "delete", lambda p: (_ for _ in ()).throw(OSError("disk error")))
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="db down"):
        _upload(db_session, storage, test_settings, inspection)
    orphan = _files(storage)
    assert len(orphan) == 1  # still there, and the log says where
    assert any(orphan[0].name in r.getMessage() and "reconciliation" in r.getMessage() for r in caplog.records)


def test_failed_delete_transaction_keeps_the_valid_file(
    db_session, storage, test_settings, inspection, monkeypatch
):
    image = _upload(db_session, storage, test_settings, inspection)
    path = image.storage_path
    monkeypatch.setattr(db_session, "commit", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    with pytest.raises(RuntimeError):
        image_service.delete_image(db_session, storage, inspection.id, image.id)
    monkeypatch.undo()
    assert storage.exists(path)
    assert db_session.get(InspectionImage, image.id) is not None


def test_file_delete_failure_after_commit_is_logged_and_recoverable(
    db_session, storage, test_settings, inspection, monkeypatch, caplog
):
    from app.services.reconciliation import reconcile

    image = _upload(db_session, storage, test_settings, inspection)
    path = image.storage_path
    monkeypatch.setattr(storage, "delete", lambda p: (_ for _ in ()).throw(OSError("busy")))
    with caplog.at_level(logging.ERROR):
        image_service.delete_image(db_session, storage, inspection.id, image.id)
    assert db_session.get(InspectionImage, image.id) is None  # row is gone
    assert any(path in r.getMessage() for r in caplog.records)
    monkeypatch.undo()
    assert reconcile(db_session, storage).orphaned_files == [path]  # recoverable via the report


def test_delete_inspection_removes_rows_then_files(db_session, storage, test_settings, inspection):
    paths = [
        _upload(db_session, storage, test_settings, inspection, make_image_bytes(color=(i * 100, 50, 0))).storage_path
        for i in range(2)
    ]
    image_service.delete_inspection(db_session, storage, inspection.id)
    assert db_session.get(Inspection, inspection.id) is None
    assert not any(storage.exists(p) for p in paths)


def test_delete_inspection_rollback_keeps_files(db_session, storage, test_settings, inspection, monkeypatch):
    path = _upload(db_session, storage, test_settings, inspection).storage_path
    monkeypatch.setattr(db_session, "commit", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    with pytest.raises(RuntimeError):
        image_service.delete_inspection(db_session, storage, inspection.id)
    assert storage.exists(path)


def test_concurrent_style_uploads_get_distinct_orders(db_session, storage, test_settings, inspection):
    orders = [
        _upload(db_session, storage, test_settings, inspection, make_image_bytes(color=(i * 30, 0, 0))).image_order
        for i in range(3)
    ]
    assert orders == [0, 1, 2]


def test_unknown_inspection_writes_nothing(db_session, storage, test_settings):
    with pytest.raises(Exception) as info:
        image_service.upload_image(db_session, storage, test_settings, uuid.uuid4(), make_image_bytes(), "a.jpg")
    assert getattr(info.value, "status_code", None) == 404
    assert _files(storage) == []


def test_truly_concurrent_uploads_serialize(db_engine, storage, test_settings, inspection):
    """Separate sessions/threads: the row lock must prevent order collisions and over-limit inserts."""
    from concurrent.futures import ThreadPoolExecutor

    from sqlalchemy.orm import sessionmaker

    from app.core.errors import AppError

    factory = sessionmaker(bind=db_engine, expire_on_commit=False)

    def attempt(i: int):
        with factory() as session:
            try:
                return image_service.upload_image(
                    session, storage, test_settings, inspection.id,
                    make_image_bytes(color=(i * 40, 90, 10)), f"{i}.jpg",
                ).image_order
            except AppError as exc:
                return exc.code

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(attempt, range(5)))

    orders = sorted(r for r in results if isinstance(r, int))
    assert orders == [0, 1, 2]  # limit is 3: unique orders, none over the limit
    assert results.count("image_limit_reached") == 2
    assert len(_files(storage)) == 3  # rejected uploads left no files
