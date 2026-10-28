"""Report-only comparison of the database and the managed storage directory.

Detects (and never changes anything):
- missing:           database rows whose file is absent or whose path is unsafe
- orphaned:          managed files with no database row
- checksum_mismatch: files whose SHA-256 differs from the stored checksum

Run:  python -m app.services.reconciliation [--json]
There is deliberately no delete option. Review the report, then act manually.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.inspection_image import InspectionImage
from app.services.image_storage import ImageStorage

logger = logging.getLogger(__name__)

_IGNORED_NAMES = {".gitkeep", ".DS_Store"}


@dataclass
class ReconciliationReport:
    checked_records: int = 0
    checked_files: int = 0
    missing_files: list[str] = field(default_factory=list)
    invalid_paths: list[str] = field(default_factory=list)
    orphaned_files: list[str] = field(default_factory=list)
    skipped_symlinks: list[str] = field(default_factory=list)
    checksum_mismatches: list[str] = field(default_factory=list)
    unchecked_no_checksum: list[str] = field(default_factory=list)

    @property
    def has_problems(self) -> bool:
        return bool(
            self.missing_files or self.invalid_paths or self.orphaned_files or self.checksum_mismatches
        )


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_files(storage: ImageStorage, report: ReconciliationReport) -> set[str]:
    """Relative paths of regular files under the root. Never follows symlinks."""
    found: set[str] = set()
    if not storage.root.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(storage.root, followlinks=False):
        current = Path(dirpath)
        for name in list(dirnames):  # symlinked directories are listed, not entered
            if (current / name).is_symlink():
                report.skipped_symlinks.append((current / name).relative_to(storage.root).as_posix())
                dirnames.remove(name)
        for name in filenames:
            if name in _IGNORED_NAMES:
                continue
            path = current / name
            relative = path.relative_to(storage.root).as_posix()
            if path.is_symlink():
                report.skipped_symlinks.append(relative)
            else:
                found.add(relative)
    return found


def reconcile(db: Session, storage: ImageStorage) -> ReconciliationReport:
    report = ReconciliationReport()
    files_on_disk = _scan_files(storage, report)
    report.checked_files = len(files_on_disk)

    known: set[str] = set()
    for image in db.scalars(select(InspectionImage)):
        report.checked_records += 1
        path = image.storage_path
        if not storage.is_valid_relative_path(path):
            report.invalid_paths.append(path)  # never touched on disk
            continue
        known.add(path)
        if not storage.exists(path):
            report.missing_files.append(path)
            continue
        if image.sha256 is None:
            report.unchecked_no_checksum.append(path)
        elif _sha256_of(storage.resolve(path)) != image.sha256:
            report.checksum_mismatches.append(path)

    report.orphaned_files = sorted(files_on_disk - known)
    for items in (report.missing_files, report.invalid_paths, report.checksum_mismatches):
        items.sort()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report-only storage/database reconciliation.")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    from app.core.config import get_settings
    from app.db.session import get_session_factory

    storage = ImageStorage(get_settings().storage_root)
    with get_session_factory()() as db:
        report = reconcile(db, storage)

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(f"Storage root: {storage.root}")
        print(f"Records checked: {report.checked_records}   Files found: {report.checked_files}")
        for title, items in (
            ("Missing files (DB row, no file)", report.missing_files),
            ("Invalid/unsafe database paths", report.invalid_paths),
            ("Orphaned files (file, no DB row)", report.orphaned_files),
            ("Checksum mismatches", report.checksum_mismatches),
            ("Symlinks skipped", report.skipped_symlinks),
            ("Records without a checksum (not verified)", report.unchecked_no_checksum),
        ):
            print(f"{title}: {len(items)}")
            for item in items:
                print(f"  - {item}")
        print("Report only: no files were changed.")
    return 1 if report.has_problems else 0


if __name__ == "__main__":
    sys.exit(main())
