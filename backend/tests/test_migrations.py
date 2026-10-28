"""Migration 0002 behaviour on a database holding data. Rebuilds the test schema at the end."""

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from tests.conftest import BACKEND_DIR, TEST_DATABASE_URL, requires_db

pytestmark = requires_db


def _cfg() -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL.replace("%", "%%"))
    return cfg


@pytest.fixture
def at_0001(db_engine):
    def reset(target: str):
        with db_engine.begin() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
        command.upgrade(_cfg(), target)

    reset("0001")
    yield db_engine
    reset("head")  # leave the shared test schema as other tests expect


def test_upgrade_refuses_to_guess_metadata_for_existing_rows(at_0001):
    with at_0001.begin() as conn:
        conn.execute(text("INSERT INTO inspections (id) VALUES ('00000000-0000-0000-0000-000000000001')"))
        conn.execute(
            text(
                "INSERT INTO generated_reports (id, inspection_id, content, model_name, model_version) "
                "VALUES ('00000000-0000-0000-0000-000000000002', '00000000-0000-0000-0000-000000000001', "
                "'c', 'm', '1')"
            )
        )
    with pytest.raises(RuntimeError, match="generated_reports"):
        command.upgrade(_cfg(), "0002")
    with at_0001.connect() as conn:  # transactional DDL: nothing was half-applied
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0001"


def test_upgrade_keeps_existing_inspections(at_0001):
    with at_0001.begin() as conn:
        conn.execute(text("INSERT INTO inspections (id, title) VALUES (gen_random_uuid(), 'keep me')"))
    command.upgrade(_cfg(), "0002")
    with at_0001.connect() as conn:
        assert conn.execute(text("SELECT title FROM inspections")).scalar() == "keep me"


def test_downgrade_restores_columns_from_runs(db_engine, at_0001):
    command.upgrade(_cfg(), "0002")
    ids = {k: "00000000-0000-0000-0000-00000000000%d" % i for i, k in enumerate(
        ["insp", "img", "run", "pred", "rep", "vrun", "res"], start=1)}
    with db_engine.begin() as conn:
        for sql in [
            "INSERT INTO inspections (id) VALUES (:insp)",
            "INSERT INTO inspection_images (id, inspection_id, storage_path, original_filename, media_type, "
            "file_size_bytes, image_order) VALUES (:img, :insp, 'p', 'f', 'image/jpeg', 1, 0)",
            "INSERT INTO detection_runs (id, inspection_image_id, model_name, model_version) "
            "VALUES (:run, :img, 'yolo', '9')",
            "INSERT INTO damage_predictions (id, detection_run_id, damage_type, confidence) "
            "VALUES (:pred, :run, 'dent', 0.5)",
            "INSERT INTO generated_reports (id, inspection_id, content, model_name, model_version, "
            "evidence_strategy) VALUES (:rep, :insp, 'c', 'm', '1', 'full_image')",
            "INSERT INTO verification_runs (id, generated_report_id, verifier_model, verifier_version) "
            "VALUES (:vrun, :rep, 'qwen', '3')",
            "INSERT INTO verification_results (id, verification_run_id, claim_text, verification_label, "
            "explanation) VALUES (:res, :vrun, 'c', 'supported', 'e')",
        ]:
            conn.execute(text(sql), ids)
    command.downgrade(_cfg(), "0001")
    with db_engine.connect() as conn:
        pred = conn.execute(
            text("SELECT inspection_image_id::text, model_name, model_version FROM damage_predictions")
        ).one()
        res = conn.execute(
            text("SELECT generated_report_id::text, verifier_model FROM verification_results")
        ).one()
    assert tuple(pred) == (ids["img"], "yolo", "9")
    assert tuple(res) == (ids["rep"], "qwen")
