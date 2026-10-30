"""Damage type / location consistency of detection_signal (pure; no database, no models).

The only machine-readable location is the claim's image-relative crop versus the detector's box.
Location words ("rear bumper") are never checked, and a crop may be copied from the prompt's detector
boxes, so crop overlap is not independent evidence that the stated location is right.
"""

import itertools
import uuid

from app.schemas.claims import ClaimType, ExtractedClaim
from app.schemas.damage import BoundingBox
from app.services.claim_policy import DetectionMatch as D
from app.services.evidence_association import (
    LOCATION_MATCH_IOU,
    PredictionFacts,
    box_iou,
    detection_signal,
    normalize_damage_type,
    text_mentions_damage_type,
)

IMG, IMG2 = uuid.UUID(int=1), uuid.UUID(int=2)
RUN, RUN2, OLD_RUN = uuid.UUID(int=21), uuid.UUID(int=22), uuid.UUID(int=23)
P1, P2 = uuid.UUID(int=11), uuid.UUID(int=12)
BOX = {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}
NEAR = {"x_min": 0.15, "y_min": 0.15, "x_max": 0.5, "y_max": 0.5}
EDGE = {"x_min": 0.4, "y_min": 0.4, "x_max": 0.8, "y_max": 0.8}
FAR = {"x_min": 0.6, "y_min": 0.6, "x_max": 0.9, "y_max": 0.9}
LATEST = {IMG: RUN, IMG2: RUN2}

T, L, P = ClaimType.DAMAGE_TYPE, ClaimType.DAMAGE_LOCATION, ClaimType.VISIBLE_DAMAGE_PRESENCE


def facts(damage_type="dent", box=BOX, image=IMG, run=None):
    return PredictionFacts(image, run or LATEST[image], damage_type, box)


def make(ctype=T, damage_type="dent", preds=(), refs=(), images=(IMG,), text=None):
    """refs: (image_id, prediction_id | None, crop | None)."""
    return ExtractedClaim(
        claim_id="c1",
        text=text or (f"There is a {damage_type} on the rear bumper." if damage_type else "Damage on the rear bumper."),
        claim_type=ctype, damage_type=damage_type, image_ids=list(images), prediction_ids=list(preds),
        evidence=[{"image_id": str(i), "prediction_id": str(p) if p else None, "crop": c} for i, p, c in refs],
    )


def run(c, preds, images=None, latest_run=None):
    images = list(images or c.image_ids)
    latest_run = {i: LATEST[i] for i in images} if latest_run is None else latest_run
    count = {i: max(1, sum(f.image_id == i for f in preds.values())) for i in images}
    return detection_signal(c, images, preds, latest_run, count)


# Reference: detection_signal exactly as it was at HEAD 49c397d (claim-policy-6), before the
# location-consistency demotions. Kept frozen so the grid below can prove nothing was loosened.


def _ref_location(claim, pid, f):
    crops = [r.crop for r in claim.evidence if r.prediction_id == pid and r.crop is not None]
    if not crops:
        return None
    try:
        box = BoundingBox.model_validate(f.bounding_box) if f.bounding_box is not None else None
    except ValueError:
        box = None
    if box is None:
        return "unknown"
    best = max(box_iou(c, box) for c in crops)
    return "match" if best >= LOCATION_MATCH_IOU else ("conflict" if best == 0 else "unknown")


def _ref_compare(claim, pid, f):
    loc = _ref_location(claim, pid, f)
    if claim.claim_type is L:
        return loc if loc is not None else "unknown"
    ok = loc in (None, "match")
    same = claim.damage_type is not None and (
        normalize_damage_type(claim.damage_type) == normalize_damage_type(f.damage_type))
    if claim.claim_type is P:
        if claim.damage_type is not None and not same:
            return "unknown"
        return "match" if ok else "unknown"
    if claim.damage_type is None or not text_mentions_damage_type(claim.text, claim.damage_type):
        return "unknown"
    if same:
        return "match" if ok else "unknown"
    return "conflict" if ok else "unknown"


def reference_signal(claim, images, preds, latest_run, latest_count):
    if claim.claim_type not in (T, L, P):
        return D.NOT_APPLICABLE
    if not images:
        return D.INCONCLUSIVE
    if any(i not in latest_run for i in images):
        return D.UNAVAILABLE
    if not preds:
        return D.NO_DETECTION if not any(latest_count.get(i, 0) for i in images) else D.INCONCLUSIVE
    out = ["unknown" if latest_run.get(f.image_id) != f.run_id else _ref_compare(claim, p, f)
           for p, f in preds.items()]
    if all(o == "match" for o in out):
        return D.MATCHES_CLAIM
    if all(o == "conflict" for o in out):
        return D.CONFLICTS_WITH_CLAIM
    return D.INCONCLUSIVE


# Scenario table ------------------------------------------------------------------------------------

# (name, claim, cited detections, before (HEAD), after, detail expected after)
SCENARIOS = [
    ("type ok, place ok (crop on box)", make(T, preds=(P1,), refs=((IMG, P1, NEAR),)), {P1: facts()},
     D.MATCHES_CLAIM, D.MATCHES_CLAIM, f"location_overlap:{P1}"),
    ("type ok, no location given", make(T, preds=(P1,)), {P1: facts()},
     D.MATCHES_CLAIM, D.MATCHES_CLAIM, f"location_not_given:{P1}"),
    ("type ok, paired crop disjoint", make(T, preds=(P1,), refs=((IMG, P1, FAR),)), {P1: facts()},
     D.INCONCLUSIVE, D.INCONCLUSIVE, f"location_no_overlap:{P1}"),
    ("type ok, paired crop partial", make(T, preds=(P1,), refs=((IMG, P1, EDGE),)), {P1: facts()},
     D.INCONCLUSIVE, D.INCONCLUSIVE, f"location_unclear:{P1}"),
    ("wrong type, place ok", make(T, "scratch", (P1,), ((IMG, P1, NEAR),)), {P1: facts()},
     D.CONFLICTS_WITH_CLAIM, D.CONFLICTS_WITH_CLAIM, f"damage_type_mismatch:{P1}"),
    ("wrong type, place wrong", make(T, "scratch", (P1,), ((IMG, P1, FAR),)), {P1: facts()},
     D.INCONCLUSIVE, D.INCONCLUSIVE, f"location_no_overlap:{P1}"),
    ("type ok, unpaired crop elsewhere in same image", make(T, preds=(P1,), refs=((IMG, None, FAR),)),
     {P1: facts()}, D.MATCHES_CLAIM, D.INCONCLUSIVE, f"location_crop_uncovered:{IMG}"),
    ("type ok, unpaired crop on box", make(T, preds=(P1,), refs=((IMG, None, NEAR),)), {P1: facts()},
     D.MATCHES_CLAIM, D.MATCHES_CLAIM, f"location_not_given:{P1}"),
    ("type ok on img2, claim's crop on img1",
     make(T, preds=(P2,), refs=((IMG, None, BOX),), images=(IMG, IMG2)), {P2: facts(image=IMG2)},
     D.MATCHES_CLAIM, D.INCONCLUSIVE, f"location_crop_uncovered:{IMG}"),
    ("crop paired with P1, P2 same type elsewhere",
     make(T, preds=(P1, P2), refs=((IMG, P1, BOX),)), {P1: facts(), P2: facts(box=FAR)},
     D.MATCHES_CLAIM, D.INCONCLUSIVE, f"location_elsewhere:{P2}"),
    ("two crops for one detection disagree",
     make(T, preds=(P1,), refs=((IMG, P1, BOX), (IMG, P1, FAR))), {P1: facts()},
     D.MATCHES_CLAIM, D.INCONCLUSIVE, f"location_crops_disagree:{P1}"),
    ("crop paired with P1 but drawn on another image",
     make(T, preds=(P1,), refs=((IMG2, P1, BOX),), images=(IMG, IMG2)), {P1: facts()},
     D.MATCHES_CLAIM, D.INCONCLUSIVE, f"location_crop_image_mismatch:{P1}"),
    ("crop over two cited detections of different types",
     make(T, preds=(P1, P2), refs=((IMG, P1, BOX),)), {P1: facts(), P2: facts("scratch", NEAR)},
     D.INCONCLUSIVE, D.INCONCLUSIVE, f"damage_type_mismatch:{P2}"),
    ("cited detections disagree on type", make(T, preds=(P1, P2)), {P1: facts(), P2: facts("scratch")},
     D.INCONCLUSIVE, D.INCONCLUSIVE, f"damage_type_mismatch:{P2}"),
    ("detection without box, crop given", make(T, preds=(P1,), refs=((IMG, P1, BOX),)), {P1: facts(box=None)},
     D.INCONCLUSIVE, D.INCONCLUSIVE, f"detection_box_absent:{P1}"),
    ("detection without box, unpaired crop", make(T, preds=(P1,), refs=((IMG, None, BOX),)),
     {P1: facts(box=None)}, D.MATCHES_CLAIM, D.INCONCLUSIVE, f"detection_box_absent:{P1}"),
    ("detection without box, no crop", make(T, preds=(P1,)), {P1: facts(box=None)},
     D.MATCHES_CLAIM, D.MATCHES_CLAIM, f"location_not_given:{P1}"),
    ("generic claim, no damage_type", make(T, None, (P1,)), {P1: facts()},
     D.INCONCLUSIVE, D.INCONCLUSIVE, "claim_damage_type_absent"),
    ("presence, no type, cited detection", make(P, None, (P1,)), {P1: facts("scratch")},
     D.MATCHES_CLAIM, D.MATCHES_CLAIM, f"detection_present:{P1}"),
    ("presence, no type, unpaired crop elsewhere", make(P, None, (P1,), ((IMG, None, FAR),)),
     {P1: facts("scratch")}, D.MATCHES_CLAIM, D.INCONCLUSIVE, f"location_crop_uncovered:{IMG}"),
    ("location claim, crop on box", make(L, None, (P1,), ((IMG, P1, NEAR),)), {P1: facts()},
     D.MATCHES_CLAIM, D.INCONCLUSIVE, "location_text_unchecked"),
    ("location claim, crop copied exactly", make(L, None, (P1,), ((IMG, P1, BOX),)), {P1: facts()},
     D.MATCHES_CLAIM, D.INCONCLUSIVE, "location_text_unchecked"),
    ("location claim, crop disjoint", make(L, None, (P1,), ((IMG, P1, FAR),)), {P1: facts()},
     D.CONFLICTS_WITH_CLAIM, D.CONFLICTS_WITH_CLAIM, f"location_no_overlap:{P1}"),
    ("location claim, no crop", make(L, None, (P1,)), {P1: facts()},
     D.INCONCLUSIVE, D.INCONCLUSIVE, f"location_not_given:{P1}"),
]


def test_scenario_table_before_and_after():
    for name, c, preds, before, after, detail in SCENARIOS:
        images = list(c.image_ids)
        latest = {i: LATEST[i] for i in images}
        count = {i: 1 for i in images}
        assert reference_signal(c, images, preds, latest, count) is before, name
        match, details = run(c, preds)
        assert match is after, name
        assert detail in details, (name, details)


def test_location_claim_is_never_a_detector_match():
    for crop in (BOX, NEAR):
        match, details = run(make(L, None, (P1,), ((IMG, P1, crop),)), {P1: facts()})
        assert match is D.INCONCLUSIVE
        assert f"location_overlap:{P1}" in details and "location_text_unchecked" in details


def test_demotions_keep_the_original_details_first():
    match, details = run(make(T, preds=(P1,), refs=((IMG, None, FAR),)), {P1: facts()})
    assert match is D.INCONCLUSIVE
    assert details == (f"damage_type_match:{P1}", f"location_not_given:{P1}",
                       f"location_crop_uncovered:{IMG}", f"location_elsewhere:{P1}")


def test_every_cited_detection_must_lie_under_the_claims_crops():
    c = make(T, preds=(P1, P2), refs=((IMG, None, FAR),))
    assert run(c, {P1: facts(), P2: facts(box=FAR)})[0] is D.INCONCLUSIVE  # P1 itself is elsewhere
    c = make(T, preds=(P1, P2), refs=((IMG, P1, BOX), (IMG, P2, FAR)))
    assert run(c, {P1: facts(), P2: facts(box=FAR)})[0] is D.MATCHES_CLAIM


def test_detections_on_an_image_without_crops_are_not_doubted():
    c = make(T, preds=(P1, P2), refs=((IMG, P1, BOX),), images=(IMG, IMG2))
    assert run(c, {P1: facts(), P2: facts(image=IMG2, box=FAR)})[0] is D.MATCHES_CLAIM


def test_stale_prediction_is_still_inconclusive_without_location_doubts():
    match, details = run(make(T, preds=(P1,), refs=((IMG, None, FAR),)), {P1: facts(run=OLD_RUN)})
    assert match is D.INCONCLUSIVE and details == (f"prediction_not_latest:{P1}",)


# No-loosening grid ----------------------------------------------------------------------------------

_CROPS = (None, BOX, NEAR, EDGE, FAR)
_DET_BOXES = (BOX, FAR, None)


def _grid():
    for ctype, dtype, det1, box1, second, crop1, extra, stale in itertools.product(
        (T, L, P), (None, "dent", "scratch"), ("dent", "scratch"), _DET_BOXES,
        (None, ("dent", IMG, FAR), ("scratch", IMG, NEAR), ("dent", IMG2, BOX)),
        _CROPS, (None, (IMG, None, FAR), (IMG, None, BOX), (IMG2, None, BOX), (IMG, "P1", FAR)), (False, True),
    ):
        preds = {P1: facts(det1, box1, run=OLD_RUN if stale else None)}
        images = [IMG]
        if second is not None:
            preds[P2] = facts(second[0], second[2], image=second[1])
            if second[1] not in images:
                images.append(second[1])
        refs = []
        if crop1 is not None:
            refs.append((IMG, P1, crop1))
        if extra is not None:
            image, pid, crop = extra
            if pid == "P1":
                if crop1 is None or crop1 == crop:
                    continue
                pid = P1
            if image not in images:
                images.append(image)
            refs.append((image, pid, crop))
        yield make(ctype, dtype, tuple(preds), tuple(refs), tuple(images)), preds


def test_no_configuration_is_loosened():
    """Only allowed change from HEAD 49c397d: MATCHES_CLAIM -> INCONCLUSIVE (or CONFLICTS)."""
    changed = kept_match = total = 0
    for c, preds in _grid():
        total += 1
        images = list(c.image_ids)
        latest = {i: LATEST[i] for i in images}
        count = {i: 1 for i in images}
        before = reference_signal(c, images, preds, latest, count)
        after, _ = detection_signal(c, images, preds, latest, count)
        if after is not before:
            assert before is D.MATCHES_CLAIM and after in (D.INCONCLUSIVE, D.CONFLICTS_WITH_CLAIM), (c, preds)
            changed += 1
        elif after is D.MATCHES_CLAIM:
            kept_match += 1
            assert c.claim_type is not L
    assert total > 3000 and changed > 100 and kept_match > 100
