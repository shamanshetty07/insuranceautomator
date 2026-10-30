"""Run relationships, result isolation and report provenance."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    DamagePrediction,
    DetectionRun,
    EvidenceStrategy,
    Inspection,
    ReportPrediction,
    RunStatus,
    VerificationLabel,
    VerificationResult,
)
from app.schemas.runs import DetectionConfig, GenerationConfig, VerificationConfig
from tests.conftest import requires_db
from tests.factories import SHA, make_image, make_prediction, make_report, make_result, make_run, make_vrun

pytestmark = requires_db


def test_image_can_have_multiple_runs_with_own_predictions(db_session: Session):
    insp = Inspection()
    img = make_image(insp)
    run_a = make_run(img, "1", config={"confidence_threshold": 0.25})
    run_b = make_run(img, "2", config={"confidence_threshold": 0.5}, weights_sha256=SHA)
    empty = make_run(img, "3")
    make_prediction(run_a), make_prediction(run_a, damage_type="scratch"), make_prediction(run_b)
    db_session.add_all([insp, img, run_a, run_b, empty])
    db_session.commit()

    assert len(img.detection_runs) == 3
    assert {r.model_version: len(r.predictions) for r in img.detection_runs} == {"1": 2, "2": 1, "3": 0}
    assert run_b.config == {"confidence_threshold": 0.5}
    assert empty.config == {} and empty.status is RunStatus.PENDING and empty.completed_at is None


def test_prediction_requires_a_run(db_session: Session):
    db_session.add(DamagePrediction(damage_type="dent", confidence=0.5))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_deleting_a_run_removes_only_its_predictions(db_session: Session):
    insp = Inspection()
    img = make_image(insp)
    keep, drop = make_run(img, "1"), make_run(img, "2")
    keep_pred, drop_pred = make_prediction(keep), make_prediction(drop)
    db_session.add_all([insp, img, keep, drop, keep_pred, drop_pred])
    db_session.commit()
    db_session.delete(drop)
    db_session.commit()
    assert db_session.query(DamagePrediction).all() == [keep_pred]


@pytest.mark.parametrize(
    "status, completed_at, ok",
    [
        (RunStatus.PENDING, None, True),
        (RunStatus.RUNNING, None, True),
        (RunStatus.COMPLETED, datetime.now(timezone.utc), True),
        (RunStatus.FAILED, datetime.now(timezone.utc), True),
        (RunStatus.COMPLETED, None, False),
        (RunStatus.PENDING, datetime.now(timezone.utc), False),
    ],
)
def test_completed_at_matches_status(db_session: Session, status, completed_at, ok):
    insp = Inspection()
    img = make_image(insp)
    db_session.add_all([insp, img, make_run(img, status=status, completed_at=completed_at)])
    if ok:
        db_session.commit()
    else:
        with pytest.raises(IntegrityError):
            db_session.commit()


def test_weights_checksum_format_enforced(db_session: Session):
    insp = Inspection()
    img = make_image(insp)
    db_session.add_all([insp, img, make_run(img, weights_sha256="not-a-hash")])
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_verification_runs_do_not_mix_results(db_session: Session):
    insp = Inspection()
    report = make_report(insp)
    run_1, run_2 = make_vrun(report, "1"), make_vrun(report, "2")
    r1 = make_result(run_1, "Dent on left door", VerificationLabel.SUPPORTED)
    r2 = make_result(run_2, "Dent on left door", VerificationLabel.WRONG_LOCATION)
    r3 = make_result(run_2, "Missed scratch on hood", VerificationLabel.OMITTED_DAMAGE)
    db_session.add_all([insp, report, run_1, run_2, r1, r2, r3])
    db_session.commit()

    assert len(report.verification_runs) == 2
    assert [r.verification_label for r in run_1.results] == [VerificationLabel.SUPPORTED]
    assert {r.verification_label for r in run_2.results} == {
        VerificationLabel.WRONG_LOCATION, VerificationLabel.OMITTED_DAMAGE,
    }
    db_session.delete(run_2)
    db_session.commit()
    assert db_session.query(VerificationResult).all() == [r1]


def test_result_requires_a_run(db_session: Session):
    db_session.add(
        VerificationResult(claim_text="c", verification_label=VerificationLabel.SUPPORTED, explanation="e")
    )
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_report_records_strategy_and_exact_predictions(db_session: Session):
    insp = Inspection()
    img = make_image(insp)
    run = make_run(img)
    used, unused = make_prediction(run), make_prediction(run, damage_type="scratch")
    report = make_report(insp, EvidenceStrategy.CROPPED_REGION)
    db_session.add_all([insp, img, run, used, unused, report])
    db_session.flush()
    report.prediction_links.append(
        ReportPrediction(generated_report_id=report.id, damage_prediction_id=used.id)
    )
    db_session.commit()
    db_session.expire_all()

    assert report.evidence_strategy is EvidenceStrategy.CROPPED_REGION
    assert [l.damage_prediction_id for l in report.prediction_links] == [used.id]


def test_prediction_used_by_a_report_cannot_be_deleted_alone(db_session: Session):
    insp = Inspection()
    img = make_image(insp)
    run = make_run(img)
    pred = make_prediction(run)
    report = make_report(insp)
    db_session.add_all([insp, img, run, pred, report])
    db_session.flush()
    db_session.add(ReportPrediction(generated_report_id=report.id, damage_prediction_id=pred.id))
    db_session.commit()

    db_session.delete(pred)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    db_session.delete(run)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    db_session.delete(insp)
    db_session.commit()
    assert db_session.query(DetectionRun).count() == 0


def test_run_config_schemas():
    DetectionConfig(confidence_threshold=0.25, iou_threshold=0.7, image_size=640, max_detections=100)
    with pytest.raises(ValidationError):
        DetectionConfig(confidence_threshold=1.5, iou_threshold=0.7, image_size=640, max_detections=100)
    with pytest.raises(ValidationError):
        DetectionConfig(
            confidence_threshold=0.25, iou_threshold=0.7, image_size=640, max_detections=100, typo=1
        )
    sampling = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512, "seed": 7}
    cfg = GenerationConfig(
        prompt_version="v1", sampling=sampling, evidence_settings={"crop_padding": 0.1}
    )
    assert cfg.model_dump(exclude_none=True)["evidence_settings"] == {"crop_padding": 0.1}
    VerificationConfig(prompt_version="v1", sampling=sampling)
    with pytest.raises(ValidationError):
        GenerationConfig(prompt_version="", sampling=sampling)
