import logging
import sys

_FORMAT = "%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s"


def configure_logging(level: str) -> None:
    """Configure key=value structured logging to stdout."""
    root = logging.getLogger()
    root.setLevel(level)
    if not any(getattr(h, "_vda", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler._vda = True  # type: ignore[attr-defined]
        root.addHandler(handler)
