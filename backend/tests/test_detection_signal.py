"""Semantic claim/detection comparison (evidence_association.detection_signal); no database."""

import uuid

import pytest

from app.schemas.claims import ClaimType, ExtractedClaim
from app.schemas.damage import BoundingBox
from app.services.claim_policy import DetectionMatch as D
from app.services.evidence_association import (
    PredictionFacts,
    box_iou,
    detection_signal,
    normalize_damage_type,
    text_mentions_damage_type,
)

IMG, IMG2 = uuid.uuid4(), uuid.uuid4()
RUN, RUN2, OLD_RUN = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
P1, P2 = uuid.uuid4(), uuid.uuid4()
BOX = {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}
NEAR = {"x_min": 0.15, "y_min": 0.15, "x_max": 0.5, "y_max": 0.5}
FAR = {"x_min": 0.6, "y_min": 0.6, "x_max": 0.9, "y_max": 0.9}
EDGE = {"x_min": 0.4, "y_min": 0.4, "x_max": 0.8, "y_max": 0.8}


def claim(ctype=ClaimType.DAMAGE_TYPE, damage_type="dent", preds=(P1,), crops=None, images=(IMG,), text=None):
    crops = crops or {}
    evidence = [{"image_id": str(IMG), "prediction_id": str(p), "crop": crops[p]} for p in preds if p in crops]
    return ExtractedClaim(
        claim_id="c1", text=text or f"There is a {damage_type or 'damaged area'} on the door.",
        claim_type=ctype, damage_type=damage_type,
        image_ids=list(images), prediction_ids=list(preds), evidence=evidence,
    )


def facts(damage_type="dent", box=BOX, run=RUN, image=IMG):
    return PredictionFacts(image_id=image, run_id=run, damage_type=damage_type, bounding_box=box)


def signal(c, preds=None, latest_run=None, latest_count=None, images=(IMG,)):
    preds = {P1: facts()} if preds is None else preds
    latest_run = {IMG: RUN} if latest_run is None else latest_run
    latest_count = {IMG: len(preds) or 0} if latest_count is None else latest_count
    return detection_signal(c, list(images), preds, latest_run, latest_count)


def test_matching_damage_type_matches():
    match, details = signal(claim())
    assert match is D.MATCHES_CLAIM and f"damage_type_match:{P1}" in details


@pytest.mark.parametrize("claimed,detected", [("Dent", "dent"), ("paint scratch", "Paint-Scratch")])
def test_damage_type_comparison_is_normalised(claimed, detected):
    assert signal(claim(damage_type=claimed), {P1: facts(detected)})[0] is D.MATCHES_CLAIM


def test_different_damage_type_at_the_cited_place_conflicts():
    match, details = signal(claim(damage_type="scratch"))
    assert match is D.CONFLICTS_WITH_CLAIM and f"damage_type_mismatch:{P1}" in details
    assert signal(claim(damage_type="scratch", crops={P1: NEAR}))[0] is D.CONFLICTS_WITH_CLAIM


def test_different_damage_type_somewhere_else_is_not_a_conflict():
    assert signal(claim(damage_type="scratch", crops={P1: FAR}))[0] is D.INCONCLUSIVE


def test_claim_without_structured_damage_type_is_never_a_match():
    match, details = signal(claim(damage_type=None))
    assert match is D.INCONCLUSIVE and "claim_damage_type_absent" in details


def test_mixed_cited_detections_are_inconclusive():
    preds = {P1: facts("dent"), P2: facts("scratch")}
    c = claim(preds=(P1, P2))
    assert signal(c, preds, latest_count={IMG: 2})[0] is D.INCONCLUSIVE


def test_every_cited_detection_must_agree_to_match_or_conflict():
    preds = {P1: facts("dent"), P2: facts("dent", box=NEAR)}
    assert signal(claim(preds=(P1, P2)), preds)[0] is D.MATCHES_CLAIM
    preds = {P1: facts("crack"), P2: facts("scratch")}
    assert signal(claim(preds=(P1, P2)), preds)[0] is D.CONFLICTS_WITH_CLAIM


def test_same_class_at_a_different_place_is_inconclusive():
    assert signal(claim(crops={P1: FAR}))[0] is D.INCONCLUSIVE


def test_location_overlap_matches():
    match, details = signal(claim(ClaimType.DAMAGE_LOCATION, crops={P1: NEAR}))
    assert match is D.MATCHES_CLAIM and f"location_overlap:{P1}" in details


def test_location_without_any_overlap_conflicts():
    match, details = signal(claim(ClaimType.DAMAGE_LOCATION, crops={P1: FAR}))
    assert match is D.CONFLICTS_WITH_CLAIM and f"location_no_overlap:{P1}" in details


def test_location_with_small_overlap_is_unclear():
    match, details = signal(claim(ClaimType.DAMAGE_LOCATION, crops={P1: EDGE}))
    assert match is D.INCONCLUSIVE and f"location_unclear:{P1}" in details


def test_location_claim_without_crop_or_detection_box_is_inconclusive():
    assert signal(claim(ClaimType.DAMAGE_LOCATION))[0] is D.INCONCLUSIVE
    match, details = signal(claim(ClaimType.DAMAGE_LOCATION, crops={P1: BOX}), {P1: facts(box=None)})
    assert match is D.INCONCLUSIVE and f"detection_box_absent:{P1}" in details
    bad_box = {"x_min": 0.9, "y_min": 0.1, "x_max": 0.1, "y_max": 0.5}
    assert signal(claim(ClaimType.DAMAGE_LOCATION, crops={P1: BOX}), {P1: facts(box=bad_box)})[0] is D.INCONCLUSIVE


def test_presence_with_a_cited_detection_matches_but_never_conflicts():
    assert signal(claim(ClaimType.VISIBLE_DAMAGE_PRESENCE, damage_type=None))[0] is D.MATCHES_CLAIM
    far = claim(ClaimType.VISIBLE_DAMAGE_PRESENCE, damage_type=None, crops={P1: FAR})
    assert signal(far)[0] is D.INCONCLUSIVE


def test_image_without_completed_run_is_unavailable_not_no_detection():
    c = claim(preds=())
    match, details = signal(c, preds={}, latest_run={}, latest_count={})
    assert match is D.UNAVAILABLE and details == (f"detection_not_run:{IMG}",)


def test_one_unanalysed_image_makes_the_signal_unavailable():
    c = claim(preds=(), images=(IMG, IMG2))
    match, _ = signal(c, preds={}, latest_run={IMG: RUN}, latest_count={IMG: 0}, images=(IMG, IMG2))
    assert match is D.UNAVAILABLE


def test_zero_detections_on_every_cited_image_is_no_detection():
    c = claim(preds=(), images=(IMG, IMG2))
    match, details = signal(c, preds={}, latest_run={IMG: RUN, IMG2: RUN2},
                            latest_count={IMG: 0, IMG2: 0}, images=(IMG, IMG2))
    assert match is D.NO_DETECTION and set(details) == {f"no_detections:{IMG}", f"no_detections:{IMG2}"}


def test_uncited_detections_are_inconclusive():
    match, details = signal(claim(preds=()), preds={}, latest_count={IMG: 3})
    assert match is D.INCONCLUSIVE and details == (f"uncited_detections:{IMG}",)


def test_prediction_from_an_older_run_is_not_a_match():
    match, details = signal(claim(), {P1: facts(run=OLD_RUN)})
    assert match is D.INCONCLUSIVE and details == (f"prediction_not_latest:{P1}",)


@pytest.mark.parametrize("ctype", [t for t in ClaimType if t not in (
    ClaimType.VISIBLE_DAMAGE_PRESENCE, ClaimType.DAMAGE_TYPE, ClaimType.DAMAGE_LOCATION)])
def test_non_visual_claims_are_not_applicable(ctype):
    assert signal(claim(ctype)) == (D.NOT_APPLICABLE, ())


def test_iou_and_normalisation():
    a = BoundingBox(**BOX)
    assert box_iou(a, a) == pytest.approx(1.0)
    assert box_iou(a, BoundingBox(**FAR)) == 0.0
    touching = BoundingBox(x_min=0.5, y_min=0.1, x_max=0.7, y_max=0.5)
    assert box_iou(a, touching) == 0.0
    assert 0 < box_iou(a, BoundingBox(**EDGE)) < 0.3
    assert normalize_damage_type("  Glass  Crack ") == "glass_crack"


def test_text_mentioning_the_damage_type_can_match():
    assert signal(claim(text="A dent on the rear door."))[0] is D.MATCHES_CLAIM


def test_regression_text_not_mentioning_the_damage_type_is_inconclusive():
    match, details = signal(claim(text="Scratch on the door"))
    assert match is D.INCONCLUSIVE and "claim_text_damage_type_mismatch" in details


def test_mismatched_text_is_not_turned_into_a_conflict_either():
    match, details = signal(claim(damage_type="scratch", text="A dent on the door"))
    assert match is D.INCONCLUSIVE and "claim_text_damage_type_mismatch" in details


def test_hyphenated_damage_type_matches_spaced_text():
    c = claim(damage_type="Paint-Scratch", text="a paint scratch on the door")
    assert signal(c, {P1: facts("paint_scratch")})[0] is D.MATCHES_CLAIM


def test_text_rule_is_strict_about_plurals_and_synonyms():
    assert signal(claim(damage_type="scratch", text="scratches on the door"), {P1: facts("scratch")})[0] is D.INCONCLUSIVE
    assert not text_mentions_damage_type("shattered glass on the lamp", "glass shatter")
    assert text_mentions_damage_type("Glass shatter visible", "glass_shatter")
    assert not text_mentions_damage_type("redentify", "dent")


@pytest.mark.parametrize("text", [
    "Image 1 shows a dent.", "A dent, near the lamp.", "(dent) visible", "Dent!", "a DENT on the door",
])
def test_punctuation_around_the_damage_type_still_counts_as_a_mention(text):
    from app.services.evidence_association import text_mentions_damage_type

    assert text_mentions_damage_type(text, "dent")
    assert not text_mentions_damage_type(text.lower().replace("dent", "dents"), "dent")
