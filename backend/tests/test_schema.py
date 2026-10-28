"""Constraints and cascades of the tables that have no API yet."""

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    DamagePrediction,
    DetectionRun,
    GeneratedReport,
    Inspection,
    InspectionImage,
    ReportPrediction,
    VerificationResult,
    VerificationRun,
)
from app.schemas.damage import BoundingBox
from tests.conftest import requires_db
from tests.factories import SHA, make_image, make_prediction, make_report, make_result, make_run, make_vrun


@requires_db
def test_delete_inspection_cascades(db_session: Session):
    insp = Inspection()
    img = make_image(insp)
    run = make_run(img)
    pred = make_prediction(run)
    report = make_report(insp)
    vrun = make_vrun(report)
    db_session.add_all([insp, img, run, pred, report, vrun, make_result(vrun)])
    db_session.flush()
    db_session.add(ReportPrediction(generated_report_id=report.id, damage_prediction_id=pred.id))
    db_session.commit()

    db_session.delete(insp)
    db_session.commit()
    for model in (
        InspectionImage, DetectionRun, DamagePrediction, GeneratedReport,
        ReportPrediction, VerificationRun, VerificationResult,
    ):
        assert db_session.query(model).count() == 0, model.__name__


@requires_db
def test_confidence_out_of_range_rejected(db_session: Session):
    insp = Inspection()
    img = make_image(insp)
    db_session.add_all([insp, img, make_prediction(make_run(img), confidence=1.5)])
    with pytest.raises(IntegrityError):
        db_session.commit()


@requires_db
def test_duplicate_image_order_rejected(db_session: Session):
    insp = Inspection()
    db_session.add_all([insp, make_image(insp, 0)])
    db_session.flush()
    db_session.add(
        InspectionImage(
            inspection_id=insp.id, storage_path="other.jpg", original_filename="b.jpg",
            media_type="image/jpeg", file_size_bytes=1, image_order=0,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()


@requires_db
@pytest.mark.parametrize("bad", ["abc", "A" * 64, "g" * 64, "a" * 63])
def test_image_sha256_format_enforced(db_session: Session, bad: str):
    insp = Inspection()
    img = make_image(insp)
    img.sha256 = bad
    db_session.add_all([insp, img])
    with pytest.raises(IntegrityError):
        db_session.commit()


@requires_db
def test_image_sha256_optional_and_valid_accepted(db_session: Session):
    insp = Inspection()
    a, b = make_image(insp, 0), make_image(insp, 1)
    b.sha256 = SHA
    db_session.add_all([insp, a, b])
    db_session.commit()
    assert a.sha256 is None and b.sha256 == SHA


def test_bounding_box_validation():
    BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
    with pytest.raises(ValidationError):
        BoundingBox(x_min=0.5, y_min=0.1, x_max=0.4, y_max=0.5)
    with pytest.raises(ValidationError):
        BoundingBox(x_min=0, y_min=0, x_max=1.2, y_max=1)


@requires_db
def test_invalid_evidence_strategy_rejected_by_database(db_session: Session):
    insp = Inspection()
    db_session.add(insp)
    db_session.flush()
    with pytest.raises(IntegrityError):
        db_session.execute(
            text(
                "INSERT INTO generated_reports (id, inspection_id, content, model_name, model_version, "
                "evidence_strategy) VALUES (gen_random_uuid(), :i, 'c', 'm', '1', 'telepathy')"
            ),
            {"i": insp.id},
        )
