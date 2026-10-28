"""Filesystem storage for normalized images, confined to a configured root.

Paths stored in the database are relative ("<inspection_id>/<image_id>.<ext>") and
are always re-validated here: they must match the generated pattern, must not pass
through symlinks, and must resolve inside the storage root.
"""

import logging
import os
import re
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_REL_PATH = re.compile(
    r"^(?P<dir>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/"
    r"(?P<name>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.(jpg|png|webp)$"
)


class StorageError(Exception):
    """Unsafe path or filesystem failure."""


def make_relative_path(inspection_id: uuid.UUID, image_id: uuid.UUID, extension: str) -> str:
    return f"{inspection_id}/{image_id}.{extension}"


class ImageStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def is_valid_relative_path(self, relative_path: str) -> bool:
        return _REL_PATH.fullmatch(relative_path) is not None

    def resolve(self, relative_path: str) -> Path:
        """Absolute path for a stored image, or StorageError if it is not safely inside the root."""
        if not self.is_valid_relative_path(relative_path):
            raise StorageError("storage path does not match the managed pattern")
        directory = self.root / relative_path.split("/")[0]
        path = self.root / relative_path
        # Reject symlinked directories/files so nothing can point outside the root.
        if directory.is_symlink() or path.is_symlink():
            raise StorageError("symlinks are not allowed in managed storage")
        if not path.resolve().is_relative_to(self.root):
            raise StorageError("path escapes the storage root")
        return path

    def write(self, relative_path: str, data: bytes) -> None:
        """Create a new file (never overwrites) and flush it to disk."""
        path = self.resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink() or not path.parent.resolve().is_relative_to(self.root):
            raise StorageError("storage directory is not safely inside the root")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o640)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)  # do not leave a partial file behind
            raise

    def read(self, relative_path: str) -> bytes:
        return self.resolve(relative_path).read_bytes()

    def exists(self, relative_path: str) -> bool:
        try:
            return self.resolve(relative_path).is_file()
        except StorageError:
            return False

    def delete(self, relative_path: str) -> None:
        """Delete one managed file (missing is fine) and its directory if now empty."""
        path = self.resolve(relative_path)
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()  # only succeeds when empty
        except OSError:
            pass
