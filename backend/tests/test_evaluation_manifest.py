import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.manifest import (
    Manifest,
    gold_labels,
    load_manifest,
    manifest_digest,
    parse_manifest,
    split_by_inspection,
)
from app.evaluation.records import GoldLabel

FIXTURE = Path(__file__).parent / "fixtures" / "evaluation" / "synthetic_manifest_v0.json"


def _annotation(status: str = "adjudicated", annotator: str | None = "ann-01") -> dict:
    return {"annotator": annotator, "status": status, "annotated_at": "2026-10-01T12:00:00Z"}


def _base() -> dict:
    return {
        "schema_version": "1",
        "dataset_version": "real-v1",
        "dataset_kind": "real",
        "description": "Test manifest.",
        "created": "2026-10-01",
        "images": [
            {
                "fixture_id": "img-a1",
                "inspection_key": "insp-a",
                "source": "images/a/front.jpg",
                "sha256": "a" * 64,
                "visible_damage": [
                    {
                        "damage_type": "Dent",
                        "location": "front bumper",
                        "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5},
                    }
                ],
                "image_quality": "ok",
            },
            {
                "fixture_id": "img-b1",
                "inspection_key": "insp-b",
                "source": "images/b/rear.jpg",
                "sha256": "b" * 64,
                "visible_damage": [],
                "image_quality": "blurry",
            },
        ],
        "claims": [
            {
                "claim_key": "a-c1",
                "inspection_key": "insp-a",
                "text": "Dent on the front bumper.",
                "claim_type": "visible_damage_presence",
                "damage_type": "dent",
                "cited_fixture_ids": ["img-a1"],
                "label": "supported",
                "annotation": _annotation(),
            },
            {
                "claim_key": "b-c1",
                "inspection_key": "insp-b",
                "text": "Scratch on the rear door.",
                "claim_type": "visible_damage_presence",
                "damage_type": "scratch",
                "cited_fixture_ids": ["img-b1"],
                "label": "unverifiable",
                "annotation": _annotation("draft"),
            },
            {
                "claim_key": "b-c2",
                "inspection_key": "insp-b",
                "text": "Repair costs 500 USD.",
                "claim_type": "repair_cost",
                "label": None,
                "annotation": _annotation("disputed"),
            },
        ],
    }


def _parse(data: dict) -> Manifest:
    return parse_manifest(json.dumps(data))


def test_valid_manifest_loads_and_normalizes():
    m = _parse(_base())
    assert not m.is_synthetic
    assert "real dataset real-v1" == m.result_label
    assert m.images[0].visible_damage[0].damage_type == "dent"
    assert len(m.claims) == 3


def _mutated(fn) -> dict:
    data = copy.deepcopy(_base())
    fn(data)
    return data


BAD_CASES = {
    "extra_top_level_field": lambda d: d.update(extra=1),
    "extra_claim_field": lambda d: d["claims"][0].update(confidence=0.9),
    "wrong_schema_version": lambda d: d.update(schema_version="2"),
    "schema_version_as_int": lambda d: d.update(schema_version=1),
    "missing_dataset_version": lambda d: d.pop("dataset_version"),
    "missing_dataset_kind": lambda d: d.pop("dataset_kind"),
    "unknown_dataset_kind": lambda d: d.update(dataset_kind="mixed"),
    "duplicate_fixture_id": lambda d: d["images"][1].update(fixture_id="img-a1"),
    "duplicate_claim_key": lambda d: d["claims"][1].update(claim_key="a-c1"),
    "unknown_inspection": lambda d: d["claims"][0].update(inspection_key="insp-zzz"),
    "bad_slug": lambda d: d["images"][0].update(fixture_id="Img A1"),
    "absolute_source": lambda d: d["images"][0].update(source="/home/user/front.jpg"),
    "windows_source": lambda d: d["images"][0].update(source="C:/data/front.jpg"),
    "parent_traversal": lambda d: d["images"][0].update(source="images/../../etc/passwd"),
    "url_source": lambda d: d["images"][0].update(source="https://example.com/a.jpg"),
    "backslash_source": lambda d: d["images"][0].update(source="images\\a.jpg"),
    "bad_synthetic_source": lambda d: d["images"][0].update(source="synthetic:Bad Name"),
    "synthetic_source_in_real": lambda d: d["images"][0].update(source="synthetic:solid-color"),
    "file_source_missing_sha": lambda d: d["images"][0].update(sha256=None),
    "bad_sha": lambda d: d["images"][0].update(sha256="XYZ"),
    "bad_quality": lambda d: d["images"][0].update(image_quality="great"),
    "bad_bbox_order": lambda d: d["images"][0]["visible_damage"][0].update(
        bbox={"x_min": 0.5, "y_min": 0.1, "x_max": 0.4, "y_max": 0.5}
    ),
    "bbox_extra_key": lambda d: d["images"][0]["visible_damage"][0]["bbox"].update(label="dent"),
    "empty_location": lambda d: d["images"][0]["visible_damage"][0].update(location="  "),
    "bad_claim_type": lambda d: d["claims"][0].update(claim_type="vibes"),
    "bad_label": lambda d: d["claims"][0].update(label="maybe"),
    "label_on_disputed": lambda d: d["claims"][2].update(label="supported"),
    "label_on_unannotated": lambda d: d["claims"][2].update(
        label="supported", annotation={"annotator": None, "status": "unannotated"}
    ),
    "adjudicated_without_label": lambda d: d["claims"][0].update(label=None),
    "annotator_missing": lambda d: d["claims"][0]["annotation"].update(annotator=None),
    "annotator_email": lambda d: d["claims"][0]["annotation"].update(annotator="jane@example.com"),
    "bad_status": lambda d: d["claims"][0]["annotation"].update(status="final"),
    "duplicate_citation": lambda d: d["claims"][0].update(cited_fixture_ids=["img-a1", "img-a1"]),
    "fabricated_ref_untagged": lambda d: d["claims"][0].update(cited_fixture_ids=["img-zzz"]),
    "cross_inspection_untagged": lambda d: d["claims"][0].update(cited_fixture_ids=["img-b1"]),
    "fabricated_ref_wrong_tag": lambda d: d["claims"][0].update(
        cited_fixture_ids=["img-zzz"], adversarial_tag="cross_inspection_reference"
    ),
    "unknown_tag": lambda d: d["claims"][0].update(adversarial_tag="sneaky"),
    "empty_text": lambda d: d["claims"][0].update(text=" "),
    "bad_created": lambda d: d.update(created="yesterday"),
}


@pytest.mark.parametrize("case", sorted(BAD_CASES))
def test_validation_rejects(case):
    with pytest.raises(ValidationError):
        _parse(_mutated(BAD_CASES[case]))


def test_synthetic_manifest_rejects_file_source():
    data = _base()
    data["dataset_kind"] = "synthetic"
    with pytest.raises(ValidationError, match="synthetic: source"):
        _parse(data)


def test_tagged_adversarial_references_are_allowed():
    data = _base()
    data["claims"][0].update(cited_fixture_ids=["img-zzz"], adversarial_tag="fabricated_reference")
    data["claims"][1].update(cited_fixture_ids=["img-a1"], adversarial_tag="cross_inspection_reference")
    m = _parse(data)
    assert m.claims[0].adversarial_tag == "fabricated_reference"


def test_duplicate_json_keys_rejected():
    text = json.dumps(_base())
    text = text.replace('"dataset_version": "real-v1"', '"dataset_version": "real-v1", "dataset_version": "x"', 1)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_manifest(text)


def test_nan_rejected():
    text = json.dumps(_base()).replace('"x_min": 0.1', '"x_min": NaN', 1)
    with pytest.raises(ValueError):
        parse_manifest(text)


def test_string_number_not_coerced():
    data = _base()
    data["images"][0]["visible_damage"][0]["bbox"]["x_min"] = "0.1"
    with pytest.raises(ValidationError):
        _parse(data)


def test_manifest_is_frozen():
    m = _parse(_base())
    with pytest.raises(ValidationError):
        m.dataset_version = "other"


def test_gold_labels_default_only_adjudicated():
    gold = gold_labels(_parse(_base()))
    assert gold == {"a-c1": GoldLabel.SUPPORTED, "b-c1": None, "b-c2": None}


def test_gold_labels_can_include_drafts():
    gold = gold_labels(_parse(_base()), require_status=("adjudicated", "draft"))
    assert gold == {"a-c1": GoldLabel.SUPPORTED, "b-c1": GoldLabel.UNVERIFIABLE, "b-c2": None}


def test_gold_labels_rejects_unknown_or_empty_status():
    m = _parse(_base())
    with pytest.raises(ValueError):
        gold_labels(m, require_status=("final",))
    with pytest.raises(ValueError):
        gold_labels(m, require_status=())


def test_digest_stable_under_key_reordering():
    data = _base()
    reordered = dict(reversed(list(data.items())))
    reordered["claims"] = [dict(reversed(list(c.items()))) for c in data["claims"]]
    assert manifest_digest(_parse(data)) == manifest_digest(_parse(reordered))
    assert len(manifest_digest(_parse(data))) == 64


def test_digest_changes_on_content_change():
    data = _base()
    changed = copy.deepcopy(data)
    changed["claims"][0]["label"] = "contradicted"
    assert manifest_digest(_parse(data)) != manifest_digest(_parse(changed))


def test_split_by_inspection_keeps_inspections_whole():
    m = _parse(_base())
    split = split_by_inspection(m, test_fraction=0.5, salt="s")
    assert set(split.dev_inspections).isdisjoint(split.test_inspections)
    assert set(split.dev_claims) | set(split.test_claims) == {c.claim_key for c in m.claims}
    owner = {c.claim_key: c.inspection_key for c in m.claims}
    assert all(owner[k] in split.dev_inspections for k in split.dev_claims)
    assert all(owner[k] in split.test_inspections for k in split.test_claims)
    assert split == split_by_inspection(m, test_fraction=0.5, salt="s")
    assert split_by_inspection(m, test_fraction=0.0).test_claims == ()
    assert split_by_inspection(m, test_fraction=1.0).dev_claims == ()
    with pytest.raises(ValueError):
        split_by_inspection(m, test_fraction=1.5)


def test_synthetic_fixture_loads_and_is_flagged():
    m = load_manifest(FIXTURE)
    assert m.is_synthetic
    assert m.dataset_kind == "synthetic"
    assert "SYNTHETIC" in m.result_label
    assert "not human ground truth" in m.description.lower()
    assert all(img.source.startswith("synthetic:") for img in m.images)
    assert len({img.inspection_key for img in m.images}) == 4
    assert 12 <= len(m.claims) <= 16
    labelled = [c for c in m.claims if c.label is not None]
    assert all(c.annotation.annotator == "synthetic-fixture" for c in labelled)
    assert all(c.annotation.status == "adjudicated" for c in labelled)


def test_synthetic_fixture_coverage():
    m = load_manifest(FIXTURE)
    tags = {c.adversarial_tag for c in m.claims}
    for tag in (
        "wrong_location",
        "invented_damage",
        "wrong_damage_type",
        "area_not_visible",
        "low_quality_evidence",
        "ambiguous",
        "fabricated_reference",
        "cross_inspection_reference",
    ):
        assert tag in tags
    types = {c.claim_type.value for c in m.claims}
    assert {"repair_cost", "damage_cause", "hidden_structural_damage", "visible_damage_presence"} <= types
    assert {img.image_quality for img in m.images} >= {"ok", "blurry", "low_resolution"}
    gold = gold_labels(m)
    assert set(gold) == {c.claim_key for c in m.claims}
    assert {g for g in gold.values() if g is not None} == set(GoldLabel)
    assert None in gold.values()
