"""Evidence association: which stored evidence does each claim really cite, and is it intact?"""

import hashlib
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.damage_prediction import DamagePrediction
from app.models.detection_run import DetectionRun
from app.models.inspection import Inspection
from app.models.inspection_image import InspectionImage
from app.models.run_status import RunStatus
from app.schemas.claims import (
    MAX_REFERENCES_PER_CLAIM,
    ClaimType,
    EvidenceRecord,
    EvidenceReference,
    ExtractedClaim,
)
from app.schemas.damage import BoundingBox
from app.services.claim_policy import DetectionMatch, EvidenceAssociation, EvidenceIntegrity
from app.services.claim_semantics import assertion_of
from app.services.image_storage import ImageStorage, StorageError

_VISUAL_TYPES = frozenset(
    {ClaimType.VISIBLE_DAMAGE_PRESENCE, ClaimType.DAMAGE_TYPE, ClaimType.DAMAGE_LOCATION}
)

# Uncalibrated: separates same place / disjoint / unclear; never makes a claim true.
LOCATION_MATCH_IOU = 0.3

_SEVERITY = {
    EvidenceIntegrity.OK: 0,
    EvidenceIntegrity.UNCHECKED: 1,
    EvidenceIntegrity.MISMATCHED: 2,
    EvidenceIntegrity.MISSING: 3,
    EvidenceIntegrity.CORRUPT: 4,
}


@dataclass(frozen=True)
class ClaimEvidence:
    claim_id: str
    association: EvidenceAssociation
    integrity: EvidenceIntegrity
    detection_match: DetectionMatch
    image_paths: tuple[tuple[uuid.UUID, Path], ...]
    prediction_ids: tuple[uuid.UUID, ...]
    crops: tuple[EvidenceReference, ...]
    problems: tuple[str, ...]
    detection_details: tuple[str, ...] = ()

    def to_record(self) -> EvidenceRecord:
        return EvidenceRecord(
            association=self.association.value,
            integrity=self.integrity.value,
            detection_match=self.detection_match.value,
            image_ids=tuple(i for i, _ in self.image_paths),
            prediction_ids=self.prediction_ids,
            crops=self.crops,
            problems=_capped(self.problems, "problems_truncated"),
            detection_details=_capped(self.detection_details, "details_truncated"),
        )


def _capped(codes: Sequence[str], marker: str) -> tuple[str, ...]:
    if len(codes) > MAX_REFERENCES_PER_CLAIM:
        return (*codes[: MAX_REFERENCES_PER_CLAIM - 1], marker)
    return tuple(codes)


@dataclass(frozen=True)
class PredictionFacts:
    """What association knows about one cited prediction that resolved in the inspection."""

    image_id: uuid.UUID
    run_id: uuid.UUID
    damage_type: str
    bounding_box: Mapping[str, Any] | None


def normalize_damage_type(name: str) -> str:
    """Case, whitespace and hyphen-insensitive class name ("Paint-Scratch" == "paint_scratch")."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def _box(raw: Mapping[str, Any] | None) -> BoundingBox | None:
    if raw is None:
        return None
    try:
        return BoundingBox.model_validate(raw)
    except ValidationError:
        return None


def box_iou(a: BoundingBox, b: BoundingBox) -> float:
    w = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
    h = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    return inter / (area_a + area_b - inter)


_MATCH, _CONFLICT, _UNKNOWN = "match", "conflict", "unknown"


def _location(claim: ExtractedClaim, pid: uuid.UUID, facts: PredictionFacts) -> tuple[str | None, str]:
    """(None | match | conflict | unknown, detail) for crops the claim pairs with this detection."""
    crops = [r.crop for r in claim.evidence if r.prediction_id == pid and r.crop is not None]
    if not crops:
        return None, f"location_not_given:{pid}"
    box = _box(facts.bounding_box)
    if box is None:
        return _UNKNOWN, f"detection_box_absent:{pid}"
    best = max(box_iou(c, box) for c in crops)
    if best >= LOCATION_MATCH_IOU:
        return _MATCH, f"location_overlap:{pid}"
    if best == 0:
        return _CONFLICT, f"location_no_overlap:{pid}"
    return _UNKNOWN, f"location_unclear:{pid}"


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^0-9a-z]+", text.lower()) if w]


def text_mentions_damage_type(text: str, damage_type: str) -> bool:
    """True if damage_type's words appear as a contiguous word sequence in text."""
    words, phrase = _words(text), _words(damage_type)
    n = len(phrase)
    return n > 0 and any(words[i : i + n] == phrase for i in range(len(words) - n + 1))


def _compare(claim: ExtractedClaim, pid: uuid.UUID, facts: PredictionFacts) -> tuple[str, list[str]]:
    location, loc_detail = _location(claim, pid, facts)
    if claim.claim_type is ClaimType.DAMAGE_LOCATION:
        return (location if location is not None else _UNKNOWN), [loc_detail]

    if claim.claim_type is ClaimType.VISIBLE_DAMAGE_PRESENCE:
        if claim.damage_type is not None and (
            normalize_damage_type(claim.damage_type) != normalize_damage_type(facts.damage_type)
        ):
            return _UNKNOWN, [f"damage_type_mismatch:{pid}", loc_detail]
        details = [f"detection_present:{pid}", loc_detail]
        return (_MATCH if location in (None, _MATCH) else _UNKNOWN), details

    if claim.damage_type is None:
        return _UNKNOWN, ["claim_damage_type_absent", loc_detail]
    if not text_mentions_damage_type(claim.text, claim.damage_type):
        return _UNKNOWN, ["claim_text_damage_type_mismatch", loc_detail]
    if normalize_damage_type(claim.damage_type) == normalize_damage_type(facts.damage_type):
        return (_MATCH if location in (None, _MATCH) else _UNKNOWN), [f"damage_type_match:{pid}", loc_detail]
    return (_CONFLICT if location in (None, _MATCH) else _UNKNOWN), [f"damage_type_mismatch:{pid}", loc_detail]


def _location_doubts(claim: ExtractedClaim, cited: Mapping[uuid.UUID, PredictionFacts]) -> list[str]:
    """Reasons the claim's crops point somewhere other than its cited detections; only ever demotes a match.

    Crops are the claim's only machine-readable location. They are compared with boxes, never with
    the location words, and a crop the generator copied from a detector box proves nothing about them.
    """
    doubts: list[str] = []
    boxes = {pid: _box(f.bounding_box) for pid, f in cited.items()}

    def overlaps(crop: BoundingBox, pid: uuid.UUID) -> bool:
        box = boxes[pid]
        return box is not None and box_iou(crop, box) >= LOCATION_MATCH_IOU

    located = [r for r in claim.evidence if r.crop is not None]
    for ref in located:
        pid = ref.prediction_id
        if pid is not None and pid in cited:
            if cited[pid].image_id != ref.image_id:
                doubts.append(f"location_crop_image_mismatch:{pid}")
            elif not overlaps(ref.crop, pid):
                doubts.append(f"location_crops_disagree:{pid}")
        elif not any(f.image_id == ref.image_id and overlaps(ref.crop, q) for q, f in cited.items()):
            doubts.append(f"location_crop_uncovered:{ref.image_id}")

    paired = {r.prediction_id for r in located if r.prediction_id is not None}
    for pid, f in cited.items():
        if pid in paired:
            continue
        crops = [r.crop for r in located if r.image_id == f.image_id]
        if not crops:
            continue
        if boxes[pid] is None:
            doubts.append(f"detection_box_absent:{pid}")
        elif not any(overlaps(c, pid) for c in crops):
            doubts.append(f"location_elsewhere:{pid}")
    return doubts


def detection_signal(
    claim: ExtractedClaim,
    cited_images: Sequence[uuid.UUID],
    cited_predictions: Mapping[uuid.UUID, PredictionFacts],
    latest_run: Mapping[uuid.UUID, uuid.UUID],
    latest_count: Mapping[uuid.UUID, int],
) -> tuple[DetectionMatch, tuple[str, ...]]:
    """Compare one ASSOCIATED claim with detection results; pure (no DB, no files)."""
    if claim.claim_type not in _VISUAL_TYPES:
        return DetectionMatch.NOT_APPLICABLE, ()
    assertion = assertion_of(claim.text)
    if not assertion.is_positive:
        cues = (f"claim_cue:{c}" for c in assertion.cues)
        return DetectionMatch.INCONCLUSIVE, (f"claim_not_positive:{assertion.assertion.value}", *cues)
    if not cited_images:
        return DetectionMatch.INCONCLUSIVE, ()
    not_run = [i for i in cited_images if i not in latest_run]
    if not_run:
        return DetectionMatch.UNAVAILABLE, tuple(f"detection_not_run:{i}" for i in not_run)
    if not cited_predictions:
        found = [i for i in cited_images if latest_count.get(i, 0) > 0]
        if not found:
            return DetectionMatch.NO_DETECTION, tuple(f"no_detections:{i}" for i in cited_images)
        return DetectionMatch.INCONCLUSIVE, tuple(f"uncited_detections:{i}" for i in found)

    outcomes: list[str] = []
    details: list[str] = []
    for pid, facts in cited_predictions.items():
        if latest_run.get(facts.image_id) != facts.run_id:
            outcomes.append(_UNKNOWN)
            details.append(f"prediction_not_latest:{pid}")
            continue
        outcome, more = _compare(claim, pid, facts)
        outcomes.append(outcome)
        details.extend(more)
    details_t = tuple(_dedupe(details))
    if all(o == _MATCH for o in outcomes):
        doubts = _location_doubts(claim, cited_predictions)
        if claim.claim_type is ClaimType.DAMAGE_LOCATION:
            doubts.append("location_text_unchecked")
        if doubts:
            return DetectionMatch.INCONCLUSIVE, tuple(_dedupe([*details, *doubts]))
        return DetectionMatch.MATCHES_CLAIM, details_t
    if all(o == _CONFLICT for o in outcomes):
        return DetectionMatch.CONFLICTS_WITH_CLAIM, details_t
    return DetectionMatch.INCONCLUSIVE, details_t


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worst(statuses: Sequence[EvidenceIntegrity]) -> EvidenceIntegrity:
    if not statuses:
        return EvidenceIntegrity.UNCHECKED
    return max(statuses, key=lambda s: _SEVERITY[s])


@dataclass(frozen=True)
class _ImageCheck:
    path: Path | None
    integrity: EvidenceIntegrity
    problems: tuple[str, ...]


def _check_image(storage: ImageStorage, image: InspectionImage, check_integrity: bool) -> _ImageCheck:
    iid = image.id
    try:
        path = storage.resolve(image.storage_path)
    except StorageError:
        return _ImageCheck(None, EvidenceIntegrity.MISSING, (f"image_file_missing:{iid}",))
    if not check_integrity:
        return _ImageCheck(path, EvidenceIntegrity.UNCHECKED, ())
    try:
        if not path.is_file():
            return _ImageCheck(path, EvidenceIntegrity.MISSING, (f"image_file_missing:{iid}",))
        actual_sha = _sha256_of(path)
    except OSError:
        return _ImageCheck(path, EvidenceIntegrity.MISSING, (f"image_file_missing:{iid}",))

    size: tuple[int, int] | None = None
    try:
        with Image.open(path) as pil:
            size = pil.size
            pil.verify()
    except Exception:  # noqa: BLE001 - any decoder failure means "unreadable"
        return _ImageCheck(path, EvidenceIntegrity.CORRUPT, (f"image_unreadable:{iid}",))

    found: list[tuple[EvidenceIntegrity, str]] = []
    if image.sha256 is None:
        found.append((EvidenceIntegrity.UNCHECKED, f"image_checksum_absent:{iid}"))
    elif image.sha256 != actual_sha:
        found.append((EvidenceIntegrity.MISMATCHED, f"image_checksum_mismatch:{iid}"))
    if image.width_px is not None and image.height_px is not None and size is not None:
        if (image.width_px, image.height_px) != size:
            found.append((EvidenceIntegrity.MISMATCHED, f"image_dimensions_mismatch:{iid}"))
    if not found:
        return _ImageCheck(path, EvidenceIntegrity.OK, ())
    return _ImageCheck(
        path, _worst([s for s, _ in found]), tuple(p for _, p in found)
    )


def _dedupe(items: Sequence) -> list:  # noqa: ANN401
    return list(dict.fromkeys(items))


def associate_claims(
    db: Session,
    storage: ImageStorage,
    inspection_id: uuid.UUID,
    claims: Sequence[ExtractedClaim],
    *,
    check_integrity: bool = True,
) -> list[ClaimEvidence]:
    """Associate every claim with evidence of ``inspection_id``; one result per claim, in order."""
    if db.scalar(select(Inspection.id).where(Inspection.id == inspection_id)) is None:
        raise ValueError("unknown inspection")

    cited_images: set[uuid.UUID] = set()
    cited_preds: set[uuid.UUID] = set()
    for c in claims:
        cited_images.update(c.image_ids)
        cited_images.update(r.image_id for r in c.evidence)
        cited_preds.update(c.prediction_ids)
        cited_preds.update(r.prediction_id for r in c.evidence if r.prediction_id is not None)

    images: dict[uuid.UUID, InspectionImage] = {}
    if cited_images:
        images = {
            i.id: i
            for i in db.scalars(
                select(InspectionImage).where(
                    InspectionImage.inspection_id == inspection_id,
                    InspectionImage.id.in_(cited_images),
                )
            )
        }

    pred_facts: dict[uuid.UUID, PredictionFacts] = {}
    if cited_preds:
        for pid, run_id, img_id, damage_type, bounding_box in db.execute(
            select(
                DamagePrediction.id,
                DamagePrediction.detection_run_id,
                DetectionRun.inspection_image_id,
                DamagePrediction.damage_type,
                DamagePrediction.bounding_box,
            )
            .join(DetectionRun, DamagePrediction.detection_run_id == DetectionRun.id)
            .join(InspectionImage, DetectionRun.inspection_image_id == InspectionImage.id)
            .where(
                DamagePrediction.id.in_(cited_preds),
                InspectionImage.inspection_id == inspection_id,
            )
        ):
            pred_facts[pid] = PredictionFacts(img_id, run_id, damage_type, bounding_box)
    pred_image = {pid: facts.image_id for pid, facts in pred_facts.items()}

    latest_run: dict[uuid.UUID, uuid.UUID] = {}
    latest_count: dict[uuid.UUID, int] = {}
    if images:
        for run_id, img_id in db.execute(
            select(DetectionRun.id, DetectionRun.inspection_image_id)
            .where(
                DetectionRun.inspection_image_id.in_(images.keys()),
                DetectionRun.status == RunStatus.COMPLETED,
            )
            .order_by(DetectionRun.created_at.desc(), DetectionRun.id.desc())
        ):
            latest_run.setdefault(img_id, run_id)
        counts: dict[uuid.UUID, int] = {}
        if latest_run:
            counts = dict(
                db.execute(
                    select(DamagePrediction.detection_run_id, func.count())
                    .where(DamagePrediction.detection_run_id.in_(latest_run.values()))
                    .group_by(DamagePrediction.detection_run_id)
                ).all()
            )
        latest_count = {img: counts.get(run, 0) for img, run in latest_run.items()}

    checks: dict[uuid.UUID, _ImageCheck] = {}

    def check(image_id: uuid.UUID) -> _ImageCheck:
        if image_id not in checks:
            checks[image_id] = _check_image(storage, images[image_id], check_integrity)
        return checks[image_id]

    results: list[ClaimEvidence] = []
    for claim in claims:
        c_images = _dedupe([*claim.image_ids, *(r.image_id for r in claim.evidence)])
        c_preds = _dedupe(
            [*claim.prediction_ids, *(r.prediction_id for r in claim.evidence if r.prediction_id)]
        )
        if not c_images and not c_preds and not claim.evidence:
            results.append(
                ClaimEvidence(
                    claim.claim_id,
                    EvidenceAssociation.NONE_CITED,
                    EvidenceIntegrity.UNCHECKED,
                    DetectionMatch.NOT_APPLICABLE,
                    (),
                    (),
                    (),
                    ("no_evidence_cited",),
                )
            )
            continue

        problems: list[str] = []
        resolved_images = [i for i in c_images if i in images]
        problems += [f"image_not_in_inspection:{i}" for i in c_images if i not in images]

        image_set = set(c_images)
        good_preds: list[uuid.UUID] = []
        if c_preds and not c_images:
            problems.append("prediction_without_image")
        for pid in c_preds:
            if pid not in pred_image:
                problems.append(f"prediction_not_in_inspection:{pid}")
            elif not c_images:
                continue
            elif pred_image[pid] not in image_set:
                problems.append(f"prediction_image_mismatch:{pid}")
            else:
                good_preds.append(pid)

        crops: list[EvidenceReference] = []
        for ref in claim.evidence:
            if ref.image_id not in images:
                continue
            if ref.prediction_id is not None and ref.prediction_id in pred_image:
                if pred_image[ref.prediction_id] != ref.image_id:
                    problems.append(f"prediction_image_mismatch:{ref.prediction_id}")
                    continue
            elif ref.prediction_id is not None:
                continue
            crops.append(ref)

        association = EvidenceAssociation.UNRESOLVED if problems else EvidenceAssociation.ASSOCIATED

        statuses: list[EvidenceIntegrity] = []
        image_paths: list[tuple[uuid.UUID, Path]] = []
        for iid in resolved_images:
            res = check(iid)
            statuses.append(res.integrity)
            problems.extend(res.problems)
            if res.path is not None:
                image_paths.append((iid, res.path))
        integrity = _worst(statuses)

        detection, detection_details = DetectionMatch.NOT_APPLICABLE, ()
        if association is EvidenceAssociation.ASSOCIATED:
            detection, detection_details = detection_signal(
                claim, c_images, {p: pred_facts[p] for p in good_preds}, latest_run, latest_count
            )

        results.append(
            ClaimEvidence(
                claim.claim_id,
                association,
                integrity,
                detection,
                tuple(image_paths),
                tuple(good_preds),
                tuple(crops),
                tuple(_dedupe(problems)),
                detection_details,
            )
        )
    return results
