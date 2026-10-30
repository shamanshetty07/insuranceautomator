"""Test setup. DB tests need TEST_DATABASE_URL pointing at a dedicated, disposable database."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
if TEST_DATABASE_URL and TEST_DATABASE_URL == os.environ.get("DATABASE_URL"):
    raise RuntimeError("TEST_DATABASE_URL must differ from DATABASE_URL; tests wipe their database.")

# Must run before app imports: without a test DB, use an unroutable URL so tests never touch a real one.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL or "postgresql+psycopg://none:none@127.0.0.1:1/none"
os.environ["APP_ENV"] = "test"

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from app.api.deps import get_storage  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.db.session import get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.services.image_storage import ImageStorage  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[1]

requires_db = pytest.mark.skipif(
    not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set; database tests skipped"
)


@pytest.fixture(scope="session")
def db_engine():
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL.replace("%", "%%"))
    command.upgrade(cfg, "head")
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Iterator[Session]:
    with sessionmaker(bind=db_engine, expire_on_commit=False)() as session:
        yield session
    with db_engine.begin() as conn:
        conn.execute(text("TRUNCATE inspections CASCADE"))


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Small limits and a temporary storage root; never the real storage directory."""
    return Settings(
        database_url=TEST_DATABASE_URL or "postgresql+psycopg://none:none@127.0.0.1:1/none",
        storage_root=tmp_path / "storage",
        max_upload_bytes=200_000,
        max_images_per_inspection=3,
        min_image_side_px=16,
        max_image_pixels=1_000_000,
    )


@pytest.fixture
def storage(test_settings: Settings) -> ImageStorage:
    return ImageStorage(test_settings.storage_root)


@pytest.fixture
def client(db_session: Session, test_settings: Settings, storage: ImageStorage) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_storage] = lambda: storage
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def plain_client() -> Iterator[TestClient]:
    """Client with no database override (for endpoints that never touch the DB)."""
    with TestClient(app) as c:
        yield c
