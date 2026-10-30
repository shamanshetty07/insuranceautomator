"""Build the EvidenceBundle a report generator sees from stored images and detection runs."""

import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from app.models.run_status import RunStatus
from app.services.types import Detection, EvidenceBundle, ImageRef


class _Prediction(Protocol):
    id: uuid.UUID
    damage_type: str
    confidence: float
    bounding_box: Mapping[str, float] | None


class _Run(Protocol):
    inspection_image_id: uuid.UUID
    status: RunStatus
    predictions: Iterable[Any]


def evidence_bundle(images: Sequence[tuple[uuid.UUID, Path]], runs: Iterable[_Run]) -> EvidenceBundle:
    """images: (image_id, resolved stored path) in report order; runs: at most one per image."""
    order = [image_id for image_id, _ in images]
    if len(set(order)) != len(order):
        raise ValueError("duplicate image in evidence")
    by_image: dict[uuid.UUID, _Run] = {}
    for run in runs:
        if run.inspection_image_id not in set(order):
            raise ValueError("detection run for an image outside the evidence")
        if run.inspection_image_id in by_image:
            raise ValueError("more than one detection run for an image")
        by_image[run.inspection_image_id] = run

    detections: list[Detection] = []
    for image_id in order:
        run = by_image.get(image_id)
        if run is None or run.status is not RunStatus.COMPLETED:
            continue
        for p in sorted(run.predictions, key=lambda p: (-float(p.confidence), str(p.id))):
            detections.append(
                Detection(
                    image_id=image_id,
                    damage_type=p.damage_type,
                    confidence=float(p.confidence),
                    bounding_box=dict(p.bounding_box) if p.bounding_box is not None else None,
                    prediction_id=p.id,
                )
            )
    return EvidenceBundle(
        images=[ImageRef(image_id=image_id, path=path) for image_id, path in images],
        detections=detections,
    )
