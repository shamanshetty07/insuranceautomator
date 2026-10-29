"""Evidence association: which stored evidence does each claim really cite, and is it intact?

Read-only: this module never adds, flushes or commits anything. It returns exactly one
ClaimEvidence per input claim, in input order. A bad reference never raises and never
drops a claim; it becomes UNRESOLVED with stable problem codes. Only programmer errors
(unknown inspection id) raise ValueError.

Problem codes (stable; they never contain filesystem paths):
  no_evidence_cited
  image_not_in_inspection:<uuid>          (unknown and foreign images look identical)
  prediction_not_in_inspection:<uuid>     (unknown and foreign predictions look identical)
  prediction_image_mismatch:<uuid>
  prediction_without_image
  image_file_missing:<uuid>
  image_checksum_absent:<uuid>
  image_checksum_mismatch:<uuid>
  image_dimensions_mismatch:<uuid>
  image_unreadable:<uuid>
  problems_truncated                      (only when to_record had to cap the list)

Detection detail codes (EvidenceRecord.detection_details; why detection_match was chosen):
  detection_not_run:<image uuid>          no completed detection run (never run, failed, running)
  no_detections:<image uuid>              latest completed run found nothing
  uncited_detections:<image uuid>         detections exist that the claim does not cite
  prediction_not_latest:<uuid>            cited prediction is not from the image's latest completed run
  detection_present:<uuid>                a cited detection exists (presence claims)
  claim_damage_type_absent                the claim states no structured damage_type
  damage_type_match:<uuid> / damage_type_mismatch:<uuid>
  location_overlap:<uuid> / location_no_overlap:<uuid> / location_unclear:<uuid>
  location_not_given:<uuid>               no crop pairs the claim with this detection
  detection_box_absent:<uuid>             the detection has no usable bounding box
  details_truncated
"""

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
from app.services.image_storage import ImageStorage, StorageError

_VISUAL_TYPES = frozenset(
    {ClaimType.VISIBLE_DAMAGE_PRESENCE, ClaimType.DAMAGE_TYPE, ClaimType.DAMAGE_LOCATION}
)

# IoU at or above which a cited crop and a cited detection box count as the same region.
# Not calibrated on data: it only separates "same place" / "disjoint" / "unclear" and
# never makes a claim true. Disjoint (no overlap at all) is the only location conflict.
LOCATION_MATCH_IOU = 0.3

# Severity used to combine per-image results into one claim-level integrity.
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


# Per-prediction comparison outcome.
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


def _compare(claim: ExtractedClaim, pid: uuid.UUID, facts: PredictionFacts) -> tuple[str, list[str]]:
    location, loc_detail = _location(claim, pid, facts)
    if claim.claim_type is ClaimType.DAMAGE_LOCATION:
        return (location if location is not None else _UNKNOWN), [loc_detail]

    if claim.claim_type is ClaimType.VISIBLE_DAMAGE_PRESENCE:
        # A detection is consistent with "damage is present"; a crop elsewhere only
        # means this detection does not speak to the cited region (never a conflict).
        details = [f"detection_present:{pid}", loc_detail]
        return (_MATCH if location in (None, _MATCH) else _UNKNOWN), details

    # DAMAGE_TYPE
    if claim.damage_type is None:
        return _UNKNOWN, ["claim_damage_type_absent", loc_detail]
    if normalize_damage_type(claim.damage_type) == normalize_damage_type(facts.damage_type):
        return (_MATCH if location in (None, _MATCH) else _UNKNOWN), [f"damage_type_match:{pid}", loc_detail]
    # Different class: a conflict only where the detection is at the claimed place (or
    # no place is given). A different class somewhere else says nothing about the claim.
    return (_CONFLICT if location in (None, _MATCH) else _UNKNOWN), [f"damage_type_mismatch:{pid}", loc_detail]


def detection_signal(
    claim: ExtractedClaim,
    cited_images: Sequence[uuid.UUID],
    cited_predictions: Mapping[uuid.UUID, PredictionFacts],
    latest_run: Mapping[uuid.UUID, uuid.UUID],
    latest_count: Mapping[uuid.UUID, int],
) -> tuple[DetectionMatch, tuple[str, ...]]:
    """Compare one ASSOCIATED claim with detection results; pure (no DB, no files).

    cited_images: the claim's images (all resolved in the inspection).
    cited_predictions: the claim's resolved predictions, all on cited images.
    latest_run / latest_count: latest COMPLETED run per image and its prediction count.

    MATCHES_CLAIM / CONFLICTS_WITH_CLAIM need every cited detection to agree / disagree
    on the attributes the claim states; anything mixed or not comparable is
    INCONCLUSIVE. A cited image without a completed run makes the signal UNAVAILABLE;
    NO_DETECTION needs every cited image analysed with nothing found. None of these
    is a verdict: the policy decides (and never reads NO_DETECTION as absence).
    """
    if claim.claim_type not in _VISUAL_TYPES:
        return DetectionMatch.NOT_APPLICABLE, ()
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

    # Structural validity first: a corrupt file outranks every other finding.
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
    """Associate every claim with evidence of ``inspection_id``; one result per claim, in order.

    Reference validity (association: are the ids real and in this inspection?) is
    decided first and separately. Only for an ASSOCIATED visual claim is detection then
    compared semantically by `detection_signal`: the claim's structured damage_type
    against each cited detection's class, and its cited crops against the detection
    boxes. detection_match is only a SIGNAL for the policy, never proof: a matching
    class or an overlapping box does not make the claim true. The policy must never
    read NO_DETECTION as absence of damage.

    Predictions that are cited but unresolved (foreign, unknown, on an uncited image, or
    without any cited image) are not reported in prediction_ids/crops; resolved ones are.
    Images whose stored path is unsafe have no path and so are absent from image_paths
    (the problem code still names them).
    """
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

    # Facts for cited predictions, only for predictions inside this inspection.
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

    # Latest completed run per resolved image, with its prediction count.
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

    checks: dict[uuid.UUID, _ImageCheck] = {}  # each image hashed/decoded at most once

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
                continue  # already reported as prediction_without_image
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
                continue  # unknown prediction, already reported
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
