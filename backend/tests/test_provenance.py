import uuid

import pytest
from sqlalchemy.orm import Session

from app.models import Inspection
from app.services.provenance import ProvenanceError, attach_predictions_to_report
from tests.conftest import requires_db
from tests.factories import make_image, make_prediction, make_report, make_run

pytestmark = requires_db


def _inspection_with_prediction(db: Session):
    insp = Inspection()
    img = make_image(insp)
    pred = make_prediction(make_run(img))
    report = make_report(insp)
    db.add_all([insp, img, pred, report])
    db.commit()
    return insp, report, pred


def test_same_inspection_predictions_are_linked_once(db_session: Session):
    _, report, pred = _inspection_with_prediction(db_session)
    assert len(attach_predictions_to_report(db_session, report, [pred.id, pred.id])) == 1
    assert attach_predictions_to_report(db_session, report, [pred.id]) == []  # idempotent
    db_session.commit()
    assert [l.damage_prediction_id for l in report.prediction_links] == [pred.id]


def test_report_cannot_reference_a_prediction_from_another_inspection(db_session: Session):
    _, report_a, _ = _inspection_with_prediction(db_session)
    _, _, pred_b = _inspection_with_prediction(db_session)
    with pytest.raises(ProvenanceError, match="different inspection"):
        attach_predictions_to_report(db_session, report_a, [pred_b.id])
    db_session.rollback()
    assert report_a.prediction_links == []


def test_mixed_batch_is_rejected_atomically(db_session: Session):
    _, report_a, pred_a = _inspection_with_prediction(db_session)
    _, _, pred_b = _inspection_with_prediction(db_session)
    with pytest.raises(ProvenanceError):
        attach_predictions_to_report(db_session, report_a, [pred_a.id, pred_b.id])
    db_session.rollback()
    db_session.refresh(report_a)
    assert report_a.prediction_links == []  # the valid one was not linked either


def test_unknown_prediction_id_rejected(db_session: Session):
    _, report, _ = _inspection_with_prediction(db_session)
    with pytest.raises(ProvenanceError, match="unknown"):
        attach_predictions_to_report(db_session, report, [uuid.uuid4()])


def test_linking_a_subset_still_means_supplied_as_evidence(db_session: Session):
    """Three supplied predictions are all provenance, whatever the report text mentions."""
    insp = Inspection()
    img = make_image(insp)
    run = make_run(img)
    preds = [make_prediction(run, damage_type=t) for t in ("dent", "scratch", "crack")]
    report = make_report(insp)
    db_session.add_all([insp, img, run, report, *preds])
    db_session.commit()
    attach_predictions_to_report(db_session, report, [p.id for p in preds])
    db_session.commit()
    assert len(report.prediction_links) == 3
