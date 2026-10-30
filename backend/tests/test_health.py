from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.session import get_db
from app.main import app
from tests.conftest import requires_db


def test_root_health(plain_client: TestClient):
    r = plain_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_api_health(plain_client: TestClient):
    r = plain_client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@requires_db
def test_db_health_ok(client: TestClient):
    r = client.get("/api/v1/health/db")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "database": "reachable"}


def test_db_health_reports_failure_honestly():
    engine = create_engine("postgresql+psycopg://x:x@127.0.0.1:1/x", connect_args={"connect_timeout": 2})

    def broken_db():
        with sessionmaker(bind=engine)() as session:
            yield session

    app.dependency_overrides[get_db] = broken_db
    try:
        with TestClient(app) as c:
            r = c.get("/api/v1/health/db")
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
    assert r.status_code == 503
    assert r.json() == {"status": "unavailable", "database": "unreachable"}
