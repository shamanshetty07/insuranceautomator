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
"""

import hashlib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
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
from app.services.claim_policy import DetectionMatch, EvidenceAssociation, EvidenceIntegrity
from app.services.image_storage import ImageStorage, StorageError

_VISUAL_TYPES = frozenset(
    {ClaimType.VISIBLE_DAMAGE_PRESENCE, ClaimType.DAMAGE_TYPE, ClaimType.DAMAGE_LOCATION}
)

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

    def to_record(self) -> EvidenceRecord:
        problems = list(self.problems)
        if len(problems) > MAX_REFERENCES_PER_CLAIM:
            problems = problems[: MAX_REFERENCES_PER_CLAIM - 1] + ["problems_truncated"]
        return EvidenceRecord(
            association=self.association.value,
            integrity=self.integrity.value,
            detection_match=self.detection_match.value,
            image_ids=[i for i, _ in self.image_paths],
            prediction_ids=list(self.prediction_ids),
            crops=list(self.crops),
            problems=problems,
        )


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

    detection_match is only a SIGNAL for the policy. MATCHES_CLAIM means CITATION
    consistency: the claim cites a real prediction on a cited image of this inspection.
    It does not compare damage class or region semantically (ExtractedClaim has no
    structured attribute to compare), so CONFLICTS_WITH_CLAIM is never produced in this
    sprint. NO_DETECTION is returned only when every cited image has a completed
    detection run with zero predictions; the policy must never read it as absence of damage.

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

    # prediction id -> image id, only for predictions inside this inspection.
    pred_image: dict[uuid.UUID, uuid.UUID] = {}
    if cited_preds:
        for pid, img_id in db.execute(
            select(DamagePrediction.id, DetectionRun.inspection_image_id)
            .join(DetectionRun, DamagePrediction.detection_run_id == DetectionRun.id)
            .join(InspectionImage, DetectionRun.inspection_image_id == InspectionImage.id)
            .where(
                DamagePrediction.id.in_(cited_preds),
                InspectionImage.inspection_id == inspection_id,
            )
        ):
            pred_image[pid] = img_id

    # Latest completed run per resolved image, with its prediction count.
    latest_count: dict[uuid.UUID, int] = {}
    if images:
        latest_run: dict[uuid.UUID, uuid.UUID] = {}
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

        detection = DetectionMatch.NOT_APPLICABLE
        if association is EvidenceAssociation.ASSOCIATED and claim.claim_type in _VISUAL_TYPES:
            if good_preds:
                detection = DetectionMatch.MATCHES_CLAIM
            elif c_images and all(latest_count.get(i) == 0 for i in c_images):
                detection = DetectionMatch.NO_DETECTION

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
            )
        )
    return results
