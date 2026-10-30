"""Migration 0006 (report_claims) on a database holding data. Needs TEST_DATABASE_URL."""

import uuid

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from tests.conftest import requires_db
from tests.test_migrations import _cfg
from tests.test_verification_migration import REP, VRUN, _reset, _seed_parents

pytestmark = requires_db


@pytest.fixture
def at_0005(db_engine):
    _reset(db_engine, "0005")
    yield db_engine
    _reset(db_engine, "head")


def _legacy_result(conn) -> str:
    rid = str(uuid.uuid4())
    conn.execute(
        text(
            "INSERT INTO verification_results (id, verification_run_id, claim_text, verification_label, "
            "explanation, claim_id) VALUES (:id, :v, 'c', 'supported', 'e', 'c1')"
        ),
        {"id": rid, "v": VRUN},
    )
    return rid


def _claim(conn, claim_id="c1", position=0, **cols) -> str:
    cid = str(uuid.uuid4())
    vals = {"id": cid, "r": REP, "p": position, "c": claim_id, "t": "A dent.", "ct": "damage_type",
            "img": "[]", **cols}
    conn.execute(
        text(
            "INSERT INTO report_claims (id, generated_report_id, position, claim_id, text, claim_type, image_ids) "
            "VALUES (:id, :r, :p, :c, :t, :ct, CAST(:img AS jsonb))"
        ),
        vals,
    )
    return cid


def test_upgrade_keeps_existing_rows_and_backfills_nothing(at_0005):
    with at_0005.begin() as conn:
        _seed_parents(conn)
        rid = _legacy_result(conn)
    command.upgrade(_cfg(), "0006")
    with at_0005.connect() as conn:
        assert conn.execute(text("SELECT report_claim_id FROM verification_results WHERE id = :i"),
                            {"i": rid}).scalar() is None
        assert conn.execute(text("SELECT count(*) FROM report_claims")).scalar() == 0


@pytest.mark.parametrize("cols", [
    {"ct": "bogus"},
    {"c": "has space"},
    {"t": ""},
    {"p": -1},
    {"img": '{"not": "an array"}'},
])
def test_report_claim_constraints(at_0005, cols):
    command.upgrade(_cfg(), "0006")
    with at_0005.begin() as conn:
        _seed_parents(conn)
    with pytest.raises(IntegrityError):
        with at_0005.begin() as conn:
            _claim(conn, **cols)


def test_claim_id_and_position_unique_per_report(at_0005):
    command.upgrade(_cfg(), "0006")
    with at_0005.begin() as conn:
        _seed_parents(conn)
        _claim(conn, "c1", 0)
    for claim_id, position in (("c1", 1), ("c2", 0)):
        with pytest.raises(IntegrityError):
            with at_0005.begin() as conn:
                _claim(conn, claim_id, position)


def test_one_result_per_claim_per_run_and_cascades(at_0005):
    command.upgrade(_cfg(), "0006")
    with at_0005.begin() as conn:
        _seed_parents(conn)
        cid = _claim(conn)
        rid = _legacy_result(conn)
        conn.execute(text("UPDATE verification_results SET report_claim_id = :c WHERE id = :r"), {"c": cid, "r": rid})
    with pytest.raises(IntegrityError):
        with at_0005.begin() as conn:
            other = _legacy_result(conn)
            conn.execute(text("UPDATE verification_results SET claim_id = 'c2', report_claim_id = :c WHERE id = :r"),
                         {"c": cid, "r": other})
    with at_0005.begin() as conn:
        conn.execute(text("DELETE FROM generated_reports WHERE id = :r"), {"r": REP})
        assert conn.execute(text("SELECT count(*) FROM report_claims")).scalar() == 0


def test_downgrade_drops_claims_and_keeps_result_snapshots(at_0005):
    command.upgrade(_cfg(), "0006")
    with at_0005.begin() as conn:
        _seed_parents(conn)
        cid = _claim(conn)
        rid = _legacy_result(conn)
        conn.execute(text("UPDATE verification_results SET report_claim_id = :c WHERE id = :r"), {"c": cid, "r": rid})
    command.downgrade(_cfg(), "0005")
    with at_0005.connect() as conn:
        assert conn.execute(text("SELECT claim_id FROM verification_results WHERE id = :i"),
                            {"i": rid}).scalar() == "c1"
        tables = {r[0] for r in conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))}
    assert not tables & {"report_claims", "report_claim_images"}
    command.upgrade(_cfg(), "0006")
