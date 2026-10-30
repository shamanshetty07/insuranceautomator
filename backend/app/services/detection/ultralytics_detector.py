"""Local YOLO inference through the optional `ultralytics` package."""

import hashlib
import hmac
import importlib
import logging
import math
import os
import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from app.schemas.runs import DetectionConfig
from app.services.types import Detection, ImageRef, ModelInfo

logger = logging.getLogger(__name__)

# Ultralytics models are not thread-safe; FastAPI runs sync routes in a thread pool.
_INFERENCE_LOCK = threading.Lock()

_MAX_CLASS_NAME_LENGTH = 100

WEIGHTS_SUFFIXES = frozenset({".pt"})

_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16B", "I;16L", "I"})


def is_weights_file(path: Path) -> bool:
    return path.suffix.lower() in WEIGHTS_SUFFIXES and path.is_file()


def to_rgb(image: Image.Image) -> Image.Image:
    """RGB for the model."""
    if image.mode in _SIXTEEN_BIT_MODES:
        image = image.convert("I").point(lambda v: v / 256).convert("L")
    return image.convert("RGB")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _import_yolo() -> Any:
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "0"
    from ultralytics import YOLO  # noqa: PLC0415 - optional dependency, imported on demand

    return YOLO


class UnsafeLoadRefused(RuntimeError):
    """A configured checkpoint safeguard could not be guaranteed, so nothing was loaded."""


def verify_weights_digest(actual: str, expected: str | None) -> None:
    """Refuse weights whose SHA-256 differs from the configured pin (checked before loading)."""
    if expected is None:
        logger.warning("No DETECTOR_WEIGHTS_SHA256 pin configured; loading unpinned weights sha256=%s", actual)
        return
    if not hmac.compare_digest(actual, expected):
        raise UnsafeLoadRefused("the weights file does not match DETECTOR_WEIGHTS_SHA256")


def apply_safe_load(enabled: bool) -> None:
    """Request Ultralytics' restricted checkpoint loading (weights_only + class allow-list)."""
    if enabled:
        os.environ["ULTRALYTICS_SAFE_LOAD"] = "1"


def confirm_safe_load_active() -> None:
    """Raise UnsafeLoadRefused unless the next checkpoint load really is restricted."""
    try:
        tasks = importlib.import_module("ultralytics.nn.tasks")
    except Exception as exc:
        raise UnsafeLoadRefused("cannot inspect Ultralytics' checkpoint loader") from exc
    if getattr(tasks, "SAFE_LOAD", None) is not True:
        raise UnsafeLoadRefused(
            "restricted loading is not active: Ultralytics was imported before it was "
            "requested; restart the process with DETECTOR_SAFE_LOAD=true"
        )
    if getattr(getattr(tasks, "_SafeLoad", None), "SUPPORTED", None) is not True:
        raise UnsafeLoadRefused("restricted loading is not supported here (Ultralytics needs torch >= 2.6)")


def model_stride(model: Any) -> int | None:
    """Largest stride of a loaded YOLO model (model.model.stride), or None if unknown."""
    try:
        raw = model.model.stride
        values = raw if isinstance(raw, int | float) else _as_list(raw)
    except Exception:
        return None
    flat = values if isinstance(values, list) else [values]
    try:
        strides = [int(v) for v in flat]
    except (TypeError, ValueError):
        return None
    return max(strides) if strides else None


def _as_list(value: Any) -> list[Any]:
    """Tensor / ndarray / plain sequence -> nested Python lists."""
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def detections_from_result(
    image_id: uuid.UUID, result: Any, names: Mapping[int, str]
) -> list[Detection]:
    """Convert one Ultralytics `Results` object into Detection values."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    coords = _as_list(boxes.xyxyn)
    confidences = _as_list(boxes.conf)
    classes = _as_list(boxes.cls)
    detections = []
    for box, confidence, cls in zip(coords, confidences, classes, strict=True):
        values = [float(v) for v in box]
        if len(values) != 4 or not all(math.isfinite(v) for v in [*values, float(confidence)]):
            continue
        x_min, y_min, x_max, y_max = (_clamp01(v) for v in values)
        if x_min >= x_max or y_min >= y_max:
            continue
        detections.append(
            Detection(
                image_id=image_id,
                damage_type=names[int(cls)],
                confidence=_clamp01(float(confidence)),
                bounding_box={"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
            )
        )
    return detections


def _class_ids(names: Mapping[int, str], class_filter: Sequence[str] | None) -> list[int] | None:
    if class_filter is None:
        return None
    by_name = {name: index for index, name in names.items()}
    unknown = [name for name in class_filter if name not in by_name]
    if unknown:
        raise ValueError(f"class_filter names not in the model: {unknown}")
    return [by_name[name] for name in class_filter]


class UltralyticsDetector:
    """DamageDetectionService backed by a local YOLO detection weights file."""

    def __init__(
        self, weights_path: Path, model_name: str, model_version: str, config: DetectionConfig,
        *, safe_load: bool = False, expected_sha256: str | None = None,
    ) -> None:
        if not is_weights_file(weights_path):
            # Checked first: given a missing path, Ultralytics would try to download it.
            raise FileNotFoundError("weights file is missing, not a regular file, or not a .pt file")
        weights_sha256 = file_sha256(weights_path)
        verify_weights_digest(weights_sha256, expected_sha256)
        apply_safe_load(safe_load)
        yolo = _import_yolo()
        if safe_load:
            confirm_safe_load_active()  # before the checkpoint is unpickled, never after
        model = yolo(str(weights_path), task="detect")
        stride = model_stride(model)
        if stride is None:
            logger.warning("Could not read the model stride; relying on the multiple-of-32 check")
        elif config.image_size % stride:
            raise ValueError(f"image_size {config.image_size} is not a multiple of the model stride {stride}")
        names: dict[int, str] = {int(k): str(v) for k, v in model.names.items()}
        too_long = [n for n in names.values() if len(n) > _MAX_CLASS_NAME_LENGTH]
        if too_long:
            raise ValueError(f"class names longer than {_MAX_CLASS_NAME_LENGTH} characters")

        self._model = model
        self._names = names
        self._class_ids = _class_ids(names, config.class_filter)
        self.model = ModelInfo(name=model_name, version=model_version, weights_sha256=weights_sha256)
        self.config = config

    @property
    def classes(self) -> list[str]:
        return [self._names[i] for i in sorted(self._names)]

    def detect(self, images: list[ImageRef]) -> list[Detection]:
        detections: list[Detection] = []
        for ref in images:
            with Image.open(ref.path) as img:
                rgb = to_rgb(img)
            with _INFERENCE_LOCK:
                results = self._model.predict(
                    source=rgb,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=self.config.image_size,
                    max_det=self.config.max_detections,
                    device=self.config.device,
                    classes=self._class_ids,
                    verbose=False,
                    save=False,
                )
            for result in results:
                detections.extend(detections_from_result(ref.image_id, result, self._names))
        return detections
