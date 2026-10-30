import hashlib
import uuid
from datetime import datetime, timezone

import pytest

from app.models import DamagePrediction, DetectionRun, Inspection, InspectionImage
from app.models.run_status import RunStatus
from app.schemas.claims import ClaimType, EvidenceRecord, EvidenceReference, ExtractedClaim
from app.services import evidence_association as ea
from app.services.claim_policy import DetectionMatch, EvidenceAssociation, EvidenceIntegrity
from app.services.evidence_association import associate_claims
from app.services.image_storage import make_relative_path
from tests.conftest import requires_db
from tests.image_utils import make_image_bytes

pytestmark = requires_db

A, I, D = EvidenceAssociation, EvidenceIntegrity, DetectionMatch


def add_image(db, storage, insp, order=0, data=None, write=True, **over):
    iid = uuid.uuid4()
    data = data if data is not None else make_image_bytes("JPEG", (64, 32), color=(order * 40 % 255, 9, 9))
    rel = make_relative_path(insp.id, iid, "jpg")
    if write:
        storage.write(rel, data)
    kw = dict(
        id=iid, inspection_id=insp.id, storage_path=rel, original_filename="a.jpg",
        media_type="image/jpeg", file_size_bytes=len(data), image_order=order,
        sha256=hashlib.sha256(data).hexdigest(), width_px=64, height_px=32,
    )
    kw.update(over)
    img = InspectionImage(**kw)
    db.add(img)
    db.flush()
    return img


def add_run(db, img, status=RunStatus.COMPLETED, preds=0):
    run = DetectionRun(inspection_image_id=img.id, model_name="y", model_version="1", status=status)
    if status in (RunStatus.COMPLETED, RunStatus.FAILED):
        run.completed_at = datetime.now(timezone.utc)
    db.add(run)
    db.flush()
    out = []
    for _ in range(preds):
        p = DamagePrediction(detection_run_id=run.id, damage_type="dent", confidence=0.9)
        db.add(p)
        out.append(p)
    db.flush()
    return run, out


def claim(cid="c1", ctype=ClaimType.VISIBLE_DAMAGE_PRESENCE, images=(), preds=(), evidence=()):
    return ExtractedClaim(
        claim_id=cid, text="t", claim_type=ctype, image_ids=list(images),
        prediction_ids=list(preds), evidence=list(evidence),
    )


@pytest.fixture
def world(db_session, storage):
    insp = Inspection(title="a")
    other = Inspection(title="b")
    db_session.add_all([insp, other])
    db_session.flush()
    return db_session, storage, insp, other


def one(world, c, **kw):
    db, storage, insp, _ = world
    res = associate_claims(db, storage, insp.id, [c], **kw)
    assert len(res) == 1
    return res[0]


def test_valid_refs_ok_and_matches(world):
    db, storage, insp, _ = world
    img = add_image(db, storage, insp)
    _, (p,) = add_run(db, img, preds=1)
    ref = EvidenceReference(image_id=img.id, prediction_id=p.id)
    r = one(world, claim(images=[img.id], preds=[p.id], evidence=[ref]))
    assert (r.association, r.integrity, r.detection_match) == (A.ASSOCIATED, I.OK, D.MATCHES_CLAIM)
    assert r.problems == () and r.prediction_ids == (p.id,) and r.crops == (ref,)
    assert r.image_paths[0][0] == img.id and r.image_paths[0][1].is_file()
    rec = r.to_record()
    assert EvidenceRecord.model_validate(rec.model_dump(mode="json")) == rec
    assert rec.image_ids == (img.id,)


def test_no_refs(world):
    r = one(world, claim())
    assert (r.association, r.integrity, r.detection_match) == (A.NONE_CITED, I.UNCHECKED, D.NOT_APPLICABLE)
    assert r.problems == ("no_evidence_cited",)


def test_unknown_image(world):
    u = uuid.uuid4()
    r = one(world, claim(images=[u]))
    assert r.association is A.UNRESOLVED and r.problems == (f"image_not_in_inspection:{u}",)
    assert r.image_paths == ()


def test_image_from_other_inspection_same_as_unknown(world):
    db, storage, insp, other = world
    foreign = add_image(db, storage, other)
    own = add_image(db, storage, insp, order=1)
    r = one(world, claim(images=[own.id, foreign.id]))
    assert r.association is A.UNRESOLVED
    assert r.problems == (f"image_not_in_inspection:{foreign.id}",)
    assert [i for i, _ in r.image_paths] == [own.id]
    assert r.integrity is I.OK


def test_prediction_from_other_inspection(world):
    db, storage, insp, other = world
    own = add_image(db, storage, insp)
    f_img = add_image(db, storage, other)
    _, (fp,) = add_run(db, f_img, preds=1)
    r = one(world, claim(images=[own.id], preds=[fp.id]))
    assert r.association is A.UNRESOLVED
    assert r.problems == (f"prediction_not_in_inspection:{fp.id}",)
    assert r.prediction_ids == ()


def test_prediction_on_different_image(world):
    db, storage, insp, _ = world
    a = add_image(db, storage, insp, 0)
    b = add_image(db, storage, insp, 1)
    _, (p,) = add_run(db, b, preds=1)
    r = one(world, claim(images=[a.id], preds=[p.id]))
    assert r.association is A.UNRESOLVED
    assert r.problems == (f"prediction_image_mismatch:{p.id}",)
    assert r.detection_match is D.NOT_APPLICABLE


def test_prediction_without_image(world):
    db, storage, insp, _ = world
    a = add_image(db, storage, insp)
    _, (p,) = add_run(db, a, preds=1)
    r = one(world, claim(preds=[p.id]))
    assert r.association is A.UNRESOLVED and "prediction_without_image" in r.problems
    assert r.detection_match is D.NOT_APPLICABLE


def test_evidence_ref_prediction_on_other_image(world):
    db, storage, insp, _ = world
    a = add_image(db, storage, insp, 0)
    b = add_image(db, storage, insp, 1)
    _, (p,) = add_run(db, b, preds=1)
    ref = EvidenceReference(image_id=a.id, prediction_id=p.id)
    r = one(world, claim(images=[a.id, b.id], preds=[p.id], evidence=[ref]))
    assert r.association is A.UNRESOLVED
    assert f"prediction_image_mismatch:{p.id}" in r.problems
    assert r.crops == ()
    assert len(r.problems) == len(set(r.problems))


def test_hash_once_for_many_claims(world, monkeypatch):
    db, storage, insp, _ = world
    img = add_image(db, storage, insp)
    calls = []
    real = ea._sha256_of
    monkeypatch.setattr(ea, "_sha256_of", lambda p: calls.append(p) or real(p))
    claims = [claim(f"c{i}", images=[img.id]) for i in range(5)]
    res = associate_claims(db, storage, insp.id, claims)
    assert len(calls) == 1
    assert all(r.integrity is I.OK for r in res)


def test_file_missing_no_path_leak(world):
    db, storage, insp, _ = world
    img = add_image(db, storage, insp, write=False)
    r = one(world, claim(images=[img.id]))
    assert r.integrity is I.MISSING and r.problems == (f"image_file_missing:{img.id}",)
    assert all(str(storage.root) not in p for p in r.problems)


def test_unsafe_stored_path(world):
    db, storage, insp, _ = world
    img = add_image(db, storage, insp, storage_path="../../etc/passwd")
    r = one(world, claim(images=[img.id]))
    assert r.integrity is I.MISSING and r.problems == (f"image_file_missing:{img.id}",)
    assert r.association is A.ASSOCIATED and r.image_paths == ()


def test_checksum_absent_and_mismatch(world):
    db, storage, insp, _ = world
    a = add_image(db, storage, insp, 0, sha256=None)
    b = add_image(db, storage, insp, 1, sha256="0" * 64)
    ra = one(world, claim(images=[a.id]))
    assert ra.integrity is I.UNCHECKED and ra.problems == (f"image_checksum_absent:{a.id}",)
    rb = one(world, claim(images=[b.id]))
    assert rb.integrity is I.MISMATCHED and rb.problems == (f"image_checksum_mismatch:{b.id}",)


def test_unreadable_image(world):
    db, storage, insp, _ = world
    img = add_image(db, storage, insp, data=b"not an image at all")
    r = one(world, claim(images=[img.id]))
    assert r.integrity is I.CORRUPT and r.problems == (f"image_unreadable:{img.id}",)


def test_dimension_mismatch(world):
    db, storage, insp, _ = world
    img = add_image(db, storage, insp, width_px=10, height_px=10)
    r = one(world, claim(images=[img.id]))
    assert r.integrity is I.MISMATCHED and r.problems == (f"image_dimensions_mismatch:{img.id}",)


def test_worst_integrity_wins(world):
    db, storage, insp, _ = world
    ok = add_image(db, storage, insp, 0)
    mism = add_image(db, storage, insp, 1, sha256="0" * 64)
    miss = add_image(db, storage, insp, 2, write=False)
    assert one(world, claim(images=[ok.id, mism.id])).integrity is I.MISMATCHED
    assert one(world, claim(images=[ok.id, mism.id, miss.id])).integrity is I.MISSING


def test_check_integrity_false(world):
    db, storage, insp, _ = world
    img = add_image(db, storage, insp, write=False)
    r = one(world, claim(images=[img.id]), check_integrity=False)
    assert r.association is A.ASSOCIATED and r.integrity is I.UNCHECKED and r.problems == ()


def test_detection_no_detection_unavailable_and_inconclusive(world):
    db, storage, insp, _ = world
    zero = add_image(db, storage, insp, 0)
    add_run(db, zero, preds=0)
    never = add_image(db, storage, insp, 1)
    failed = add_image(db, storage, insp, 2)
    add_run(db, failed, status=RunStatus.FAILED)
    running = add_image(db, storage, insp, 3)
    add_run(db, running, status=RunStatus.RUNNING)
    has = add_image(db, storage, insp, 4)
    add_run(db, has, preds=1)
    assert one(world, claim(images=[zero.id])).detection_match is D.NO_DETECTION
    assert one(world, claim(images=[never.id])).detection_match is D.UNAVAILABLE
    assert one(world, claim(images=[failed.id])).detection_match is D.UNAVAILABLE
    assert one(world, claim(images=[running.id])).detection_match is D.UNAVAILABLE
    assert one(world, claim(images=[zero.id, never.id])).detection_match is D.UNAVAILABLE
    r = one(world, claim(images=[has.id]))
    assert r.detection_match is D.INCONCLUSIVE and r.detection_details == (f"uncited_detections:{has.id}",)


def test_latest_completed_run_wins_over_failed_and_older(world):
    db, storage, insp, _ = world
    img = add_image(db, storage, insp)
    old, _ = add_run(db, img, preds=2)
    old.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    new, _ = add_run(db, img, preds=0)
    new.created_at = datetime(2021, 1, 1, tzinfo=timezone.utc)
    bad, _ = add_run(db, img, status=RunStatus.FAILED)
    bad.created_at = datetime(2022, 1, 1, tzinfo=timezone.utc)
    db.flush()
    assert one(world, claim(images=[img.id])).detection_match is D.NO_DETECTION


@pytest.mark.parametrize(
    "ctype",
    [t for t in ClaimType if t not in (
        ClaimType.VISIBLE_DAMAGE_PRESENCE, ClaimType.DAMAGE_TYPE, ClaimType.DAMAGE_LOCATION)],
)
def test_non_visual_types_not_applicable(world, ctype):
    db, storage, insp, _ = world
    img = add_image(db, storage, insp)
    add_run(db, img, preds=0)
    _, (p,) = add_run(db, img, preds=1)
    r = one(world, claim(ctype=ctype, images=[img.id], preds=[p.id]))
    assert r.association is A.ASSOCIATED and r.detection_match is D.NOT_APPLICABLE


def test_mixed_batch_order_and_readonly(world):
    db, storage, insp, other = world
    good = add_image(db, storage, insp, 0)
    foreign = add_image(db, storage, other, 0)
    gone = add_image(db, storage, insp, 1, write=False)
    _, (p,) = add_run(db, good, preds=1)
    claims = [
        claim("c1"),
        claim("c2", images=[uuid.uuid4()]),
        claim("c3", images=[foreign.id]),
        claim("c4", images=[good.id], preds=[p.id]),
        claim("c5", images=[gone.id]),
        claim("c6", preds=[p.id]),
        claim("c7", ClaimType.DAMAGE_TYPE, images=[good.id]),
        claim("c8", ClaimType.REPAIR_COST, images=[good.id]),
    ]
    db.commit()
    res = associate_claims(db, storage, insp.id, claims)
    assert [r.claim_id for r in res] == [c.claim_id for c in claims]
    assert not db.new and not db.dirty and not db.deleted
    for r in res:
        EvidenceRecord.model_validate(r.to_record().model_dump(mode="json"))
    assert associate_claims(db, storage, insp.id, []) == []


def test_unknown_inspection_raises(world):
    db, storage, _, _ = world
    with pytest.raises(ValueError):
        associate_claims(db, storage, uuid.uuid4(), [])


def test_to_record_truncates_problems():
    ev = ea.ClaimEvidence("c", A.UNRESOLVED, I.UNCHECKED, D.NOT_APPLICABLE, (), (), (),
                          tuple(f"p{i}" for i in range(80)))
    rec = ev.to_record()
    assert len(rec.problems) == 50 and rec.problems[-1] == "problems_truncated"


BOX = {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}
FAR = {"x_min": 0.6, "y_min": 0.6, "x_max": 0.9, "y_max": 0.9}


def test_damage_type_compared_with_the_stored_prediction(world):
    db, storage, insp, _ = world
    img = add_image(db, storage, insp)
    _, (p,) = add_run(db, img, preds=1)

    def typed(cid, damage_type):
        return ExtractedClaim(claim_id=cid, text=f"A {damage_type or 'damaged area'} is visible",
                              claim_type=ClaimType.DAMAGE_TYPE,
                              damage_type=damage_type, image_ids=[img.id], prediction_ids=[p.id])

    res = associate_claims(db, storage, insp.id, [typed("a", "Dent"), typed("b", "scratch"), typed("c", None)])
    assert [r.detection_match for r in res] == [D.MATCHES_CLAIM, D.CONFLICTS_WITH_CLAIM, D.INCONCLUSIVE]
    assert res[1].to_record().detection_details[0] == f"damage_type_mismatch:{p.id}"


def test_location_compared_with_the_stored_box(world):
    db, storage, insp, _ = world
    img = add_image(db, storage, insp)
    _, (p,) = add_run(db, img, preds=1)
    p.bounding_box = BOX
    db.flush()

    def located(cid, crop):
        return ExtractedClaim(
            claim_id=cid, text="t", claim_type=ClaimType.DAMAGE_LOCATION, image_ids=[img.id],
            prediction_ids=[p.id], evidence=[{"image_id": str(img.id), "prediction_id": str(p.id), "crop": crop}],
        )

    res = associate_claims(db, storage, insp.id, [located("a", BOX), located("b", FAR)])
    assert [r.detection_match for r in res] == [D.MATCHES_CLAIM, D.CONFLICTS_WITH_CLAIM]


def test_cited_prediction_from_a_superseded_run_is_inconclusive(world):
    db, storage, insp, _ = world
    img = add_image(db, storage, insp)
    old, (p,) = add_run(db, img, preds=1)
    old.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    new, _ = add_run(db, img, preds=1)
    new.created_at = datetime(2021, 1, 1, tzinfo=timezone.utc)
    db.flush()
    r = one(world, claim(ctype=ClaimType.VISIBLE_DAMAGE_PRESENCE, images=[img.id], preds=[p.id]))
    assert r.detection_match is D.INCONCLUSIVE and r.detection_details == (f"prediction_not_latest:{p.id}",)
