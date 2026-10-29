import uuid

import pytest
from pydantic import ValidationError

from app.schemas.claims import (
    ClaimExtractionResult,
    ClaimType,
    EvidenceReference,
    ExtractedClaim,
    Verdict,
)

IMG, IMG2, PRED, PRED2 = (uuid.uuid4() for _ in range(4))
BOX = {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}


def claim(**kw):
    base = {"claim_id": "c1", "text": "Front bumper is dented.", "claim_type": "damage_type"}
    base.update(kw)
    return base


def test_enum_values_stable():
    assert {m.name: m.value for m in ClaimType} == {
        "VISIBLE_DAMAGE_PRESENCE": "visible_damage_presence",
        "DAMAGE_TYPE": "damage_type",
        "DAMAGE_LOCATION": "damage_location",
        "DAMAGE_SEVERITY": "damage_severity",
        "HIDDEN_STRUCTURAL_DAMAGE": "hidden_structural_damage",
        "REPAIR_RECOMMENDATION": "repair_recommendation",
        "REPAIR_COST": "repair_cost",
        "DAMAGE_CAUSE": "damage_cause",
        "OTHER": "other",
    }
    assert {m.value for m in Verdict} == {"supported", "contradicted", "insufficient_evidence"}
    assert len(Verdict) == 3


def test_valid_evidence_and_claim():
    ref = EvidenceReference(image_id=IMG, prediction_id=PRED, crop=BOX, note="  front left  ")
    assert ref.note == "front left"
    c = ExtractedClaim(**claim(image_ids=[IMG], prediction_ids=[PRED], evidence=[ref.model_dump()]))
    assert c.claim_type is ClaimType.DAMAGE_TYPE


def test_minimal_claim_and_empty_result():
    assert ExtractedClaim(**claim()).evidence == ()
    assert ClaimExtractionResult().claims == ()


def test_text_stripped_and_bounds():
    assert ExtractedClaim(**claim(text="  hi  ")).text == "hi"
    for bad in ["", "   ", "x" * 1001]:
        with pytest.raises(ValidationError):
            ExtractedClaim(**claim(text=bad))


@pytest.mark.parametrize("bad", ["", "has space", "a/b", "x" * 65, "é"])
def test_bad_claim_ids(bad):
    with pytest.raises(ValidationError):
        ExtractedClaim(**claim(claim_id=bad))


@pytest.mark.parametrize(
    "box",
    [
        {"x_min": 0.5, "y_min": 0.1, "x_max": 0.1, "y_max": 0.5},
        {"x_min": 0.1, "y_min": 0.5, "x_max": 0.5, "y_max": 0.1},
        {"x_min": 0.2, "y_min": 0.1, "x_max": 0.2, "y_max": 0.5},
        {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.1},
        {"x_min": -0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5},
        {"x_min": 0.1, "y_min": 0.1, "x_max": 1.1, "y_max": 0.5},
    ],
)
def test_invalid_boxes(box):
    with pytest.raises(ValidationError):
        EvidenceReference(image_id=IMG, crop=box)


@pytest.mark.parametrize("note", ["", "   ", "n" * 501])
def test_bad_notes(note):
    with pytest.raises(ValidationError):
        EvidenceReference(image_id=IMG, note=note)


def test_malformed_uuids():
    with pytest.raises(ValidationError):
        EvidenceReference(image_id="not-a-uuid")
    with pytest.raises(ValidationError):
        ExtractedClaim(**claim(image_ids=["nope"]))
    with pytest.raises(ValidationError):
        ExtractedClaim(**claim(prediction_ids=[123]))
    with pytest.raises(ValidationError):
        ClaimExtractionResult(report_id="x")


def test_missing_required_fields():
    with pytest.raises(ValidationError):
        EvidenceReference()
    for key in ("claim_id", "text", "claim_type"):
        data = claim()
        del data[key]
        with pytest.raises(ValidationError):
            ExtractedClaim(**data)


def test_wrong_types_and_bad_enum():
    with pytest.raises(ValidationError):
        ExtractedClaim(**claim(claim_type="bogus"))
    with pytest.raises(ValidationError):
        ExtractedClaim(**claim(text=5))
    with pytest.raises(ValidationError):
        ClaimExtractionResult(claims="x")


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        EvidenceReference(image_id=IMG, extra=1)
    with pytest.raises(ValidationError):
        ExtractedClaim(**claim(extra=1))
    with pytest.raises(ValidationError):
        ClaimExtractionResult(claims=[], extra=1)


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError):
        ExtractedClaim(**claim(image_ids=[IMG, IMG]))
    with pytest.raises(ValidationError):
        ExtractedClaim(**claim(prediction_ids=[PRED, PRED]))
    with pytest.raises(ValidationError):
        ClaimExtractionResult(claims=[claim(), claim(text="other")])


def test_cross_reference_rules():
    with pytest.raises(ValidationError):
        ExtractedClaim(**claim(image_ids=[IMG], evidence=[{"image_id": str(IMG2)}]))
    with pytest.raises(ValidationError):
        ExtractedClaim(**claim(evidence=[{"image_id": str(IMG)}]))
    with pytest.raises(ValidationError):
        ExtractedClaim(
            **claim(
                image_ids=[IMG],
                prediction_ids=[PRED],
                evidence=[{"image_id": str(IMG), "prediction_id": str(PRED2)}],
            )
        )


def test_limits():
    many = [uuid.uuid4() for _ in range(51)]
    with pytest.raises(ValidationError):
        ExtractedClaim(**claim(image_ids=many))
    with pytest.raises(ValidationError):
        ClaimExtractionResult(claims=[claim(claim_id=f"c{i}") for i in range(101)])


def test_frozen():
    c = ExtractedClaim(**claim())
    with pytest.raises(ValidationError):
        c.text = "changed"


def test_json_round_trip():
    result = ClaimExtractionResult(
        report_id=uuid.uuid4(),
        claims=[
            claim(
                image_ids=[IMG],
                prediction_ids=[PRED],
                evidence=[{"image_id": str(IMG), "prediction_id": str(PRED), "crop": BOX}],
            ),
            claim(claim_id="c2", claim_type="repair_cost"),
        ],
    )
    dumped = result.model_dump(mode="json")
    assert dumped["claims"][1]["claim_type"] == "repair_cost"
    assert ClaimExtractionResult.model_validate(dumped) == result


def test_crop_rejects_unknown_keys():
    import uuid

    from pydantic import ValidationError

    from app.schemas.claims import EvidenceReference

    box = {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}
    EvidenceReference(image_id=uuid.uuid4(), crop=box)
    with pytest.raises(ValidationError):
        EvidenceReference(image_id=uuid.uuid4(), crop={**box, "extra": 1})


# --- immutability and duplicate evidence (hardening) ------------------------------------


def _full_claim():
    return ExtractedClaim(**claim(
        image_ids=[IMG], prediction_ids=[PRED],
        evidence=[{"image_id": str(IMG), "prediction_id": str(PRED), "crop": BOX}],
    ))


def test_validated_collections_cannot_be_mutated_to_bypass_validation():
    c = _full_claim()
    # The audit's reproduction: appending a duplicate id used to succeed silently.
    with pytest.raises(AttributeError):
        c.image_ids.append(IMG)  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        c.prediction_ids[0] = PRED2  # type: ignore[index]
    with pytest.raises(AttributeError):
        c.evidence.append(EvidenceReference(image_id=IMG))  # type: ignore[attr-defined]
    with pytest.raises(ValidationError):
        c.evidence[0].image_id = IMG2  # nested reference is frozen
    with pytest.raises(ValidationError):
        c.evidence[0].crop.x_max = 0.9  # nested crop is frozen
    result = ClaimExtractionResult(claims=[c])
    with pytest.raises(AttributeError):
        result.claims.append(c)  # type: ignore[attr-defined]
    assert c.image_ids == (IMG,) and c.evidence[0].crop.x_max == 0.5


def test_input_lists_are_copied_not_aliased():
    ids = [IMG]
    c = ExtractedClaim(**claim(image_ids=ids))
    ids.append(IMG)  # mutating the caller's list cannot reach the validated object
    assert c.image_ids == (IMG,)


def test_validated_objects_are_hashable():
    assert hash(_full_claim()) == hash(_full_claim())


def test_json_format_unchanged_arrays():
    dumped = _full_claim().model_dump(mode="json")
    assert dumped["image_ids"] == [str(IMG)] and isinstance(dumped["evidence"], list)
    schema = ExtractedClaim.model_json_schema()
    assert schema["properties"]["image_ids"]["type"] == "array"
    assert schema["properties"]["image_ids"]["maxItems"] == 50


@pytest.mark.parametrize("first,second", [
    ({"image_id": str(IMG)}, {"image_id": str(IMG)}),
    ({"image_id": str(IMG), "note": "a"}, {"image_id": str(IMG), "note": "b"}),  # note is not identity
    ({"image_id": str(IMG), "crop": BOX}, {"image_id": str(IMG), "crop": dict(BOX)}),
    ({"image_id": str(IMG), "prediction_id": str(PRED)}, {"image_id": str(IMG), "prediction_id": str(PRED)}),
])
def test_duplicate_evidence_references_rejected(first, second):
    with pytest.raises(ValidationError, match="duplicate references"):
        ExtractedClaim(**claim(image_ids=[IMG], prediction_ids=[PRED], evidence=[first, second]))


def test_distinct_evidence_references_allowed():
    other_box = {**BOX, "x_max": 0.6}
    refs = [
        {"image_id": str(IMG)},
        {"image_id": str(IMG), "prediction_id": str(PRED)},
        {"image_id": str(IMG), "crop": BOX},
        {"image_id": str(IMG), "crop": other_box},
        {"image_id": str(IMG2)},
    ]
    c = ExtractedClaim(**claim(image_ids=[IMG, IMG2], prediction_ids=[PRED], evidence=refs))
    assert len(c.evidence) == 5


def test_structured_damage_type():
    assert ExtractedClaim(**claim()).damage_type is None
    assert ExtractedClaim(**claim(damage_type="  dent ")).damage_type == "dent"
    for bad in ["", "   ", "x" * 101, 5]:
        with pytest.raises(ValidationError):
            ExtractedClaim(**claim(damage_type=bad))
