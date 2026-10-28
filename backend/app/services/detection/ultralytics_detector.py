"""Local YOLO inference through the optional `ultralytics` package.

The package is imported lazily, so the app (and every test that does not exercise a
real model) works without it installed: `pip install -e ".[detection]"`.
Ultralytics is licensed AGPL-3.0; see docs/architecture.md before distributing.

Images never leave the process: the package is forced offline before import, which
disables its usage telemetry, weight auto-downloads and pip auto-installs.
"""

import hashlib
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

# Ultralytics models keep mutable predictor state and are not safe to call from
# several threads at once; FastAPI runs sync routes in a thread pool.
_INFERENCE_LOCK = threading.Lock()

_MAX_CLASS_NAME_LENGTH = 100  # damage_predictions.damage_type is VARCHAR(100)

# Trained weights only: Ultralytics turns a model definition (.yaml) into a randomly
# initialised network, which would produce meaningless detections.
WEIGHTS_SUFFIXES = frozenset({".pt"})

_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16B", "I;16L", "I"})


def is_weights_file(path: Path) -> bool:
    return path.suffix.lower() in WEIGHTS_SUFFIXES and path.is_file()


def to_rgb(image: Image.Image) -> Image.Image:
    """RGB for the model. 16-bit greyscale is scaled down first: a plain convert("RGB")
    clips every value above 255 to white."""
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
    """Convert one Ultralytics `Results` object into Detection values.

    Uses `boxes.xyxyn` (normalized to the input image, origin top-left, y down),
    clamps to [0, 1] and drops degenerate or non-finite boxes.
    """
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
        self, weights_path: Path, model_name: str, model_version: str, config: DetectionConfig
    ) -> None:
        if not is_weights_file(weights_path):
            # Checked before loading: given a missing path, Ultralytics would try to download it.
            raise FileNotFoundError("weights file is missing, not a regular file, or not a .pt file")
        weights_sha256 = file_sha256(weights_path)
        model = _import_yolo()(str(weights_path), task="detect")
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
            # Decode with Pillow (as at upload) so boxes are relative to exactly the
            # stored, orientation-normalized pixels.
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
