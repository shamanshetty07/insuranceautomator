"""Migration 0004 (claim-level verification) on a database holding data."""

import uuid

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.models import VerificationLabel, VerificationResult
from app.schemas.claims import ClaimType, Verdict
from app.services.verdict_mapping import legacy_verdict
from tests.conftest import requires_db
from tests.test_migrations import _cfg

pytestmark = requires_db

INSP = "00000000-0000-0000-0000-0000000000a1"
REP = "00000000-0000-0000-0000-0000000000a2"
VRUN = "00000000-0000-0000-0000-0000000000a3"


def _reset(engine, target: str) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    command.upgrade(_cfg(), target)


@pytest.fixture
def at_0003(db_engine):
    _reset(db_engine, "0003")
    yield db_engine
    _reset(db_engine, "head")


def _seed_parents(conn) -> None:
    conn.execute(text("INSERT INTO inspections (id) VALUES (:i)"), {"i": INSP})
    conn.execute(
        text(
            "INSERT INTO generated_reports (id, inspection_id, content, model_name, model_version, "
            "evidence_strategy) VALUES (:r, :i, 'c', 'm', '1', 'full_image')"
        ),
        {"r": REP, "i": INSP},
    )
    conn.execute(
        text(
            "INSERT INTO verification_runs (id, generated_report_id, verifier_model, verifier_version) "
            "VALUES (:v, :r, 'qwen', '3')"
        ),
        {"v": VRUN, "r": REP},
    )


def _seed_legacy(conn) -> dict[str, str]:
    _seed_parents(conn)
    ids = {}
    for label in VerificationLabel:
        rid = str(uuid.uuid4())
        ids[label.value] = rid
        conn.execute(
            text(
                "INSERT INTO verification_results (id, verification_run_id, claim_text, "
                "verification_label, explanation, evidence_reference) "
                "VALUES (:id, :v, :c, :l, :e, CAST(:ev AS jsonb))"
            ),
            {"id": rid, "v": VRUN, "c": f"claim {label.value}", "l": label.value,
             "e": f"expl {label.value}", "ev": '{"image_ids": ["x"]}'},
        )
    return ids


ORIG = "id::text, verification_run_id::text, claim_text, verification_label, explanation, evidence_reference, created_at"


def _orig_rows(engine):
    with engine.connect() as conn:
        return [tuple(r) for r in conn.execute(text(f"SELECT {ORIG} FROM verification_results ORDER BY id"))]


def _new_result(conn, claim_id=None, **cols) -> None:
    vals = {"id": str(uuid.uuid4()), "v": VRUN, "claim_id": claim_id, "lbl": "supported", **cols}
    vals.setdefault("ct", None)
    vals.setdefault("vd", None)
    conn.execute(
        text(
            "INSERT INTO verification_results (id, verification_run_id, claim_text, verification_label, "
            "explanation, claim_id, claim_type, verdict) "
            "VALUES (:id, :v, 'c', :lbl, 'e', :claim_id, :ct, :vd)"
        ),
        vals,
    )


def test_backfill_matches_python_mapping_and_preserves_data(at_0003):
    with at_0003.begin() as conn:
        ids = _seed_legacy(conn)
    before = _orig_rows(at_0003)
    command.upgrade(_cfg(), "0004")
    assert _orig_rows(at_0003) == before
    with at_0003.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT verification_label, verdict, claim_id, claim_type, needs_human_review, "
                "policy_reason_codes FROM verification_results"
            )
        ).all()
    assert len(rows) == len(ids) == 7
    for label, verdict, claim_id, claim_type, review, codes in rows:
        expected = legacy_verdict(label)
        assert verdict == (expected.value if expected else None), label
        assert (claim_id, claim_type, review, codes) == (None, None, False, [])


def test_new_constraints(at_0003):
    command.upgrade(_cfg(), "0004")
    with at_0003.begin() as conn:
        _seed_parents(conn)
    bad = [
        dict(vd="bogus"),
        dict(ct="bogus"),
        dict(lbl="omitted_damage", vd="supported"),
        dict(claim_id="c1"),  # first is fine; see below for duplicate
    ]
    for kw in bad[:3]:
        with pytest.raises(IntegrityError):
            with at_0003.begin() as conn:
                _new_result(conn, **kw)
    with at_0003.begin() as conn:
        _new_result(conn, claim_id="c1", ct=ClaimType.DAMAGE_TYPE.value, vd=Verdict.SUPPORTED.value)
        _new_result(conn, lbl=None)  # NULL label accepted
        _new_result(conn)  # two NULL claim_ids allowed
        _new_result(conn, lbl="omitted_damage")  # omitted with NULL verdict is fine
    with pytest.raises(IntegrityError):
        with at_0003.begin() as conn:
            _new_result(conn, claim_id="c1")
    with at_0003.begin() as conn:
        assert conn.execute(text("SELECT count(*) FROM verification_results")).scalar() == 4


def test_cascade_delete_removes_results(at_0003):
    command.upgrade(_cfg(), "0004")
    with at_0003.begin() as conn:
        _seed_parents(conn)
        _new_result(conn, claim_id="c1")
        _new_result(conn, claim_id="c2")
        conn.execute(text("DELETE FROM verification_runs WHERE id = :v"), {"v": VRUN})
        assert conn.execute(text("SELECT count(*) FROM verification_results")).scalar() == 0
    with at_0003.begin() as conn:
        conn.execute(text("DELETE FROM generated_reports"))
        conn.execute(text("DELETE FROM inspections"))
        _seed_parents(conn)
        _new_result(conn, claim_id="c1")
        conn.execute(text("DELETE FROM inspections WHERE id = :i"), {"i": INSP})
        assert conn.execute(text("SELECT count(*) FROM verification_results")).scalar() == 0


def test_downgrade_keeps_legacy_rows_and_fills_null_labels(at_0003):
    with at_0003.begin() as conn:
        _seed_legacy(conn)
    before = _orig_rows(at_0003)
    command.upgrade(_cfg(), "0004")
    with at_0003.begin() as conn:
        _new_result(conn, claim_id="n1", ct="damage_type", vd="contradicted")
        _new_result(conn, lbl=None)
    command.downgrade(_cfg(), "0003")
    after = _orig_rows(at_0003)
    assert len(after) == 9
    assert all(row in after for row in before)
    new_labels = sorted(r[3] for r in after if r not in before)
    assert new_labels == ["supported", "unverifiable"]
    with at_0003.connect() as conn:
        cols = {r[0] for r in conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name='verification_results'"))}
        nullable = conn.execute(text(
            "SELECT is_nullable FROM information_schema.columns WHERE table_name='verification_results' "
            "AND column_name='verification_label'")).scalar()
    assert not cols & {"claim_id", "claim_type", "verdict", "needs_human_review", "policy_reason_codes"}
    assert nullable == "NO"


def test_round_trip(at_0003):
    with at_0003.begin() as conn:
        _seed_legacy(conn)
    command.upgrade(_cfg(), "0004")
    command.downgrade(_cfg(), "0003")
    command.upgrade(_cfg(), "0004")
    with at_0003.connect() as conn:
        verdicts = dict(conn.execute(text("SELECT verification_label, verdict FROM verification_results")).all())
    for label, verdict in verdicts.items():
        expected = legacy_verdict(label)
        assert verdict == (expected.value if expected else None)


def test_model_matches_migrated_schema(db_engine):
    # db_engine is at head here (other fixtures restore it on teardown).
    table = VerificationResult.__table__
    with db_engine.connect() as conn:
        info = {
            r[0]: (r[1], r[2])
            for r in conn.execute(text(
                "SELECT column_name, is_nullable, column_default FROM information_schema.columns "
                "WHERE table_name='verification_results'"))
        }
    for name in ("verification_label", "claim_id", "claim_type", "verdict",
                 "needs_human_review", "policy_reason_codes"):
        assert (info[name][0] == "YES") == table.c[name].nullable, name
        assert (info[name][1] is not None) == (table.c[name].server_default is not None), name
    assert info["needs_human_review"][1] == "false"
    assert "'[]'::jsonb" in info["policy_reason_codes"][1]
    names = {i.name for i in table.indexes}
    assert {"uq_verification_results_run_claim", "ix_verification_results_review"} <= names
    with db_engine.connect() as conn:
        db_idx = {r[0] for r in conn.execute(text(
            "SELECT indexname FROM pg_indexes WHERE tablename='verification_results'"))}
    assert names <= db_idx
