"""Installed package versions recorded on every DetectionRun (reproducibility).

Read from package metadata only: ultralytics and torch are never imported here, so this
is cheap and works without the optional `detection` extra. Only package-name -> version
strings are recorded; nothing about the environment, paths or machine.
"""

import importlib.metadata
import logging

from app.schemas.runs import RuntimeVersions

logger = logging.getLogger(__name__)


def _version(package: str) -> str | None:
    try:
        return str(importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # a broken install must never fail a detection run
        logger.warning("Could not read the installed version of %s", package, exc_info=True)
        return None


def runtime_versions() -> RuntimeVersions:
    return RuntimeVersions(ultralytics=_version("ultralytics"), torch=_version("torch"))
