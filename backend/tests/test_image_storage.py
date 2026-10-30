import uuid

import pytest

from app.services.image_storage import ImageStorage, StorageError, make_relative_path


def _path(ext="jpg"):
    return make_relative_path(uuid.uuid4(), uuid.uuid4(), ext)


def test_write_read_delete_and_directory_cleanup(tmp_path):
    storage = ImageStorage(tmp_path / "root")
    rel = _path()
    storage.write(rel, b"data")
    assert storage.read(rel) == b"data" and storage.exists(rel)
    storage.delete(rel)
    assert not storage.exists(rel)
    assert not (storage.root / rel.split("/")[0]).exists()
    storage.delete(rel)


def test_write_never_overwrites(tmp_path):
    storage = ImageStorage(tmp_path)
    rel = _path()
    storage.write(rel, b"one")
    with pytest.raises(FileExistsError):
        storage.write(rel, b"two")
    assert storage.read(rel) == b"one"


@pytest.mark.parametrize(
    "bad",
    ["../evil.jpg", "../../etc/passwd", "/etc/passwd", "a/b.jpg", "x" * 36 + "/" + "y" * 36 + ".jpg",
     f"{uuid.uuid4()}/{uuid.uuid4()}.exe", f"{uuid.uuid4()}/../{uuid.uuid4()}.jpg", ""],
)
def test_unsafe_relative_paths_are_rejected(tmp_path, bad):
    storage = ImageStorage(tmp_path)
    with pytest.raises(StorageError):
        storage.resolve(bad)
    assert not storage.exists(bad)


def test_symlinked_directory_cannot_escape_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    storage = ImageStorage(tmp_path / "root")
    storage.root.mkdir()
    inspection = uuid.uuid4()
    (storage.root / str(inspection)).symlink_to(outside, target_is_directory=True)
    rel = make_relative_path(inspection, uuid.uuid4(), "jpg")
    with pytest.raises(StorageError):
        storage.write(rel, b"x")
    assert list(outside.iterdir()) == []


def test_symlinked_file_is_refused(tmp_path):
    storage = ImageStorage(tmp_path / "root")
    rel = _path()
    storage.write(rel, b"real")
    target = tmp_path / "secret.txt"
    target.write_text("secret")
    path = storage.root / rel
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(StorageError):
        storage.read(rel)
    assert target.read_text() == "secret"
