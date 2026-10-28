"""Write path for report evidence provenance (service-layer invariants).

ReportPrediction means: "this detection prediction was SUPPLIED AS EVIDENCE to the
report-generation model". It does NOT mean the generated text mentions it. If three
predictions were supplied and the report discusses two, all three are linked; which
ones the text covers is evaluated later (coverage / omitted-damage research phase).

Invariants enforced here because plain foreign keys cannot express them:
- every linked prediction belongs to the same inspection as the report;
- a prediction's run is the run that produced it (guaranteed by the NOT NULL FK).
"""

import uuid
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.damage_prediction import DamagePrediction
from app.models.detection_run import DetectionRun
from app.models.generated_report import GeneratedReport
from app.models.inspection_image import InspectionImage
from app.models.report_prediction import ReportPrediction


class ProvenanceError(ValueError):
    """A report was linked to predictions from another inspection, or ids were unknown."""


def attach_predictions_to_report(
    db: Session, report: GeneratedReport, prediction_ids: Iterable[uuid.UUID]
) -> list[ReportPrediction]:
    ids = list(dict.fromkeys(prediction_ids))  # de-duplicate, keep order
    if not ids:
        return []
    rows = db.execute(
        select(DamagePrediction.id, InspectionImage.inspection_id)
        .join(DetectionRun, DamagePrediction.detection_run_id == DetectionRun.id)
        .join(InspectionImage, DetectionRun.inspection_image_id == InspectionImage.id)
        .where(DamagePrediction.id.in_(ids))
    ).all()
    owner = {prediction_id: inspection_id for prediction_id, inspection_id in rows}

    unknown = [i for i in ids if i not in owner]
    if unknown:
        raise ProvenanceError(f"unknown prediction ids: {', '.join(map(str, unknown))}")
    foreign = [i for i in ids if owner[i] != report.inspection_id]
    if foreign:
        raise ProvenanceError(
            f"predictions belong to a different inspection than report {report.id}: "
            f"{', '.join(map(str, foreign))}"
        )

    existing = set(
        db.scalars(
            select(ReportPrediction.damage_prediction_id).where(
                ReportPrediction.generated_report_id == report.id
            )
        )
    )
    links = [
        ReportPrediction(generated_report_id=report.id, damage_prediction_id=i)
        for i in ids
        if i not in existing
    ]
    db.add_all(links)
    db.flush()
    return links
