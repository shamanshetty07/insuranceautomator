"""DB-free tests for the development-only stubs."""

import ast
import inspect
import uuid
from pathlib import Path

import pytest

from app.schemas.claims import ClaimType, StructuredReport
from app.services import dev_stubs
from app.services.dev_stubs import StubClaimVerifier, StubStructuredReportGenerator, StubVerifierError
from app.services.interfaces import (
    ClaimVerificationService,
    ReportGenerationService,
    StructuredReportGenerationService,
)
from app.services.types import Detection, EvidenceBundle, ImageRef

A, B = uuid.uuid4(), uuid.uuid4()
P1, P2 = uuid.uuid4(), uuid.uuid4()
BOX = {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.6}


def ev(with_det=True, images=(A, B)):
    dets = [
        Detection(A, "dent", 0.9, BOX),
        Detection(A, "scratch", 0.8, {"x_min": 0.9, "y_min": 0, "x_max": 0.1, "y_max": 1}),
        Detection(B, "crack", 0.7),
    ] if with_det else []
    return EvidenceBundle([ImageRef(i, Path(f"/x/{i}.jpg")) for i in images], dets)


def use_structured(s: StructuredReportGenerationService) -> StructuredReport:
    return s.generate_structured(ev())


def use_report(s: ReportGenerationService) -> str:
    return s.generate(ev())


def use_verifier(s: ClaimVerificationService):
    return s.extract_claims("x.")


def test_deterministic():
    g = StubStructuredReportGenerator({A: [P1]})
    assert g.generate_structured(ev()) == g.generate_structured(ev())
    assert g.generate(ev()) == g.generate_structured(ev()).report_text


def test_empty_images():
    r = StubStructuredReportGenerator().generate_structured(ev(images=()))
    assert r.report_text == "No images were provided."
    assert r.claims.claims == []
    StructuredReport.model_validate(r.model_dump())


def test_schema_roundtrip_and_ids_from_evidence():
    r = StubStructuredReportGenerator({A: [P1, P2]}).generate_structured(ev())
    StructuredReport.model_validate(r.model_dump())
    assert [c.claim_id for c in r.claims.claims] == [f"c{i}" for i in range(1, len(r.claims.claims) + 1)]
    for c in r.claims.claims:
        assert set(c.image_ids) <= {A, B}
        assert set(c.prediction_ids) <= {P1, P2}


def test_crops_and_predictions():
    r = StubStructuredReportGenerator({A: [P1, P2]}, include_unverifiable_claims=False)
    cl = [c for c in r.generate_structured(ev()).claims.claims if c.claim_type == ClaimType.DAMAGE_TYPE]
    assert len(cl) == 3
    assert cl[0].evidence[0].crop.x_max == 0.5 and "dent" in cl[0].text
    assert cl[0].prediction_ids == [P1] and cl[1].prediction_ids == [P2]
    assert cl[1].evidence[0].crop is None  # invalid box dropped
    assert cl[2].prediction_ids == [] and cl[2].image_ids == [B]


def test_unverifiable_flag():
    on = StubStructuredReportGenerator().generate_structured(ev()).claims.claims
    off = StubStructuredReportGenerator(include_unverifiable_claims=False).generate_structured(ev()).claims.claims
    types_on = {c.claim_type for c in on}
    assert {ClaimType.HIDDEN_STRUCTURAL_DAMAGE, ClaimType.REPAIR_COST, ClaimType.DAMAGE_SEVERITY} <= types_on
    assert not types_on - {c.claim_type for c in off} - {
        ClaimType.HIDDEN_STRUCTURAL_DAMAGE, ClaimType.REPAIR_COST, ClaimType.DAMAGE_SEVERITY}
    assert len(off) == len(on) - 3


def test_stub_markers():
    for cls in (StubStructuredReportGenerator, StubClaimVerifier):
        assert cls.IS_STUB is True
        assert cls.model.name.startswith("stub-")
        assert "not AI" in cls.__doc__
    assert "NOT AI" in dev_stubs.__doc__


def test_protocol_conformance_and_imports():
    g, v = StubStructuredReportGenerator(), StubClaimVerifier()
    assert use_structured(g).claims.claims and isinstance(use_report(g), str)
    assert use_verifier(v)
    tree = ast.parse(inspect.getsource(dev_stubs))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    assert not mods & {"torch", "ultralytics", "sqlalchemy", "transformers"}


def test_verifier_labels_and_calls():
    from app.services.types import ReportClaim as RC
    v = StubClaimVerifier(labels={"a": "contradicted", "b": "bogus"}, explanations={"a": "why"})
    out = v.verify([RC("a"), RC("b"), RC("c")], ev())
    assert [x.label for x in out] == ["contradicted", "bogus", "supported"]
    assert [x.claim_text for x in out] == ["a", "b", "c"]
    assert out[0].explanation == "why" and out[0].verifier == v.model
    assert out[0].evidence_reference == {"image_ids": [str(A), str(B)]}
    assert v.calls == [(("a", "b", "c"), (str(A), str(B)))]
    assert StubClaimVerifier("unverifiable").verify([RC("z")], ev())[0].label == "unverifiable"


def test_verifier_failure_modes():
    from app.services.types import ReportClaim as RC
    with pytest.raises(StubVerifierError):
        StubClaimVerifier(raise_on=["a"]).verify([RC("a")], ev())
    assert StubClaimVerifier(empty_for=["a"]).verify([RC("a")], ev()) == []
    assert len(StubClaimVerifier(extra_for=["a"]).verify([RC("a"), RC("b")], ev())) == 3


def test_extract_and_omissions():
    v = StubClaimVerifier()
    got = [c.text for c in v.extract_claims("One thing. Two things!\nThree?  Four")]
    assert got == ["One thing.", "Two things!", "Three?", "Four"]
    assert v.extract_claims("") == []
    assert v.find_omissions("r", ev()) == []
