import hashlib
import json
import uuid

import pytest
from sqlalchemy.orm import Session

from app.models import Inspection, InspectionImage
from app.services import image_service
from app.services.image_storage import ImageStorage, make_relative_path
from app.services.reconciliation import main, reconcile
from tests.conftest import requires_db
from tests.image_utils import make_image_bytes

pytestmark = requires_db


@pytest.fixture
def stored(db_session: Session, storage: ImageStorage, test_settings):
    insp = Inspection()
    db_session.add(insp)
    db_session.commit()
    return image_service.upload_image(db_session, storage, test_settings, insp.id, make_image_bytes(), "a.jpg")


def _snapshot(storage: ImageStorage) -> dict[str, bytes]:
    return {p.relative_to(storage.root).as_posix(): p.read_bytes() for p in storage.root.rglob("*") if p.is_file()}


def test_clean_state_reports_nothing(db_session, storage, stored):
    report = reconcile(db_session, storage)
    assert not report.has_problems
    assert (report.checked_records, report.checked_files) == (1, 1)


def test_detects_missing_file(db_session, storage, stored):
    (storage.root / stored.storage_path).unlink()
    report = reconcile(db_session, storage)
    assert report.missing_files == [stored.storage_path] and report.has_problems


def test_detects_orphaned_file_without_deleting_it(db_session, storage, stored):
    orphan = make_relative_path(uuid.uuid4(), uuid.uuid4(), "jpg")
    storage.write(orphan, b"stray")
    before = _snapshot(storage)
    report = reconcile(db_session, storage)
    assert report.orphaned_files == [orphan]
    assert _snapshot(storage) == before
    assert storage.exists(orphan)


def test_detects_checksum_mismatch(db_session, storage, stored):
    (storage.root / stored.storage_path).write_bytes(b"tampered")
    report = reconcile(db_session, storage)
    assert report.checksum_mismatches == [stored.storage_path]


def test_record_without_checksum_is_listed_as_unverified(db_session, storage, stored):
    stored.sha256 = None
    db_session.commit()
    report = reconcile(db_session, storage)
    assert report.unchecked_no_checksum == [stored.storage_path] and not report.has_problems


def test_unsafe_database_path_is_flagged_and_never_touched(db_session, storage, stored, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    stored.storage_path = "../../victim.txt"
    db_session.commit()
    report = reconcile(db_session, storage)
    assert report.invalid_paths == ["../../victim.txt"]
    assert victim.read_text() == "keep"


def test_symlinks_are_skipped_not_followed(db_session, storage, stored, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jpg").write_bytes(b"x")
    (storage.root / "link").symlink_to(outside, target_is_directory=True)
    report = reconcile(db_session, storage)
    assert report.skipped_symlinks == ["link"]
    assert report.orphaned_files == []


def test_cli_is_report_only_and_sets_exit_code(db_session, storage, stored, monkeypatch, capsys):
    import app.core.config as config
    import app.db.session as session

    class _Settings:
        storage_root = storage.root

    monkeypatch.setattr(config, "get_settings", lambda: _Settings())
    monkeypatch.setattr(session, "get_session_factory", lambda: (lambda: db_session))
    storage.write(make_relative_path(uuid.uuid4(), uuid.uuid4(), "png"), b"stray")
    before = _snapshot(storage)

    assert main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["orphaned_files"]) == 1
    assert _snapshot(storage) == before
    assert hashlib.sha256(before[stored.storage_path]).hexdigest() == stored.sha256
