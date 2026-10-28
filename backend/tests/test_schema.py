"""Constraints and relationships of the Phase 2+ tables (no API exists for them yet)."""

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    DamagePrediction,
    GeneratedReport,
    Inspection,
    InspectionImage,
    VerificationLabel,
    VerificationResult,
)
from app.schemas.damage import BoundingBox
from tests.conftest import requires_db


def _image(inspection: Inspection, order: int = 0) -> InspectionImage:
    return InspectionImage(
        inspection=inspection,
        storage_path=f"{inspection.id}/{order}.jpg",
        original_filename="a.jpg",
        media_type="image/jpeg",
        file_size_bytes=10,
        image_order=order,
    )


@requires_db
def test_delete_inspection_cascades(db_session: Session):
    insp = Inspection()
    img = _image(insp)
    img.damage_predictions.append(
        DamagePrediction(
            damage_type="dent", confidence=0.9, model_name="m", model_version="1"
        )
    )
    report = GeneratedReport(inspection=insp, content="c", model_name="m", model_version="1")
    report.verification_results.append(
        VerificationResult(
            claim_text="c",
            verification_label=VerificationLabel.SUPPORTED,
            explanation="e",
            verifier_model="v",
        )
    )
    db_session.add_all([insp, img, report])
    db_session.commit()

    db_session.delete(insp)
    db_session.commit()
    for model in (InspectionImage, DamagePrediction, GeneratedReport, VerificationResult):
        assert db_session.query(model).count() == 0


@requires_db
def test_confidence_out_of_range_rejected(db_session: Session):
    insp = Inspection()
    img = _image(insp)
    img.damage_predictions.append(
        DamagePrediction(damage_type="dent", confidence=1.5, model_name="m", model_version="1")
    )
    db_session.add_all([insp, img])
    with pytest.raises(IntegrityError):
        db_session.commit()


@requires_db
def test_duplicate_image_order_rejected(db_session: Session):
    insp = Inspection()
    db_session.add_all([insp, _image(insp, 0)])
    db_session.flush()
    db_session.add(
        InspectionImage(
            inspection_id=insp.id,
            storage_path="other.jpg",
            original_filename="b.jpg",
            media_type="image/jpeg",
            file_size_bytes=1,
            image_order=0,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_bounding_box_validation():
    BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
    with pytest.raises(ValidationError):
        BoundingBox(x_min=0.5, y_min=0.1, x_max=0.4, y_max=0.5)
    with pytest.raises(ValidationError):
        BoundingBox(x_min=0, y_min=0, x_max=1.2, y_max=1)
