"""Installed package versions recorded on every DetectionRun (reproducibility)."""

import importlib.metadata
import logging

from app.schemas.runs import RuntimeVersions

logger = logging.getLogger(__name__)


def _version(package: str) -> str | None:
    try:
        return str(importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        logger.warning("Could not read the installed version of %s", package, exc_info=True)
        return None


def runtime_versions() -> RuntimeVersions:
    return RuntimeVersions(ultralytics=_version("ultralytics"), torch=_version("torch"))
