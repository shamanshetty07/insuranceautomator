"""QA gap tests for the image API and upload service (needs TEST_DATABASE_URL)."""

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from starlette.middleware.cors import CORSMiddleware

from app.core.errors import AppError
from app.main import app
from app.models import Inspection, InspectionImage
from app.services import image_service
from app.services.image_storage import ImageStorage
from tests.conftest import requires_db
from tests.image_utils import make_image_bytes

pytestmark = requires_db


def _inspection(client: TestClient) -> str:
    return client.post("/api/v1/inspections", json={"title": "qa"}).json()["id"]


def _upload(client, iid, data=None, name="photo.jpg", extra_files=None):
    files = [("file", (name, data if data is not None else make_image_bytes(), "image/jpeg"))]
    files += extra_files or []
    return client.post(f"/api/v1/inspections/{iid}/images", files=files)


def _stored_files(storage: ImageStorage) -> list:
    return [p for p in storage.root.rglob("*") if p.is_file()] if storage.root.exists() else []


def test_hostile_filename_is_display_only_and_never_shapes_the_path(client, storage, tmp_path):
    iid = _inspection(client)
    body = _upload(client, iid, name="..\\../../etc/evil name.jpg").json()
    assert body["original_filename"] == "evil name.jpg"
    files = _stored_files(storage)
    assert len(files) == 1 and files[0].parent.parent == storage.root
    assert files[0].relative_to(storage.root).as_posix() == f"{iid}/{body['id']}.jpg"
    assert not (tmp_path / "etc").exists()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("C:\\Users\\me\\car\x00.jpg", "car.jpg"), ("a/b/\x1b[31mred\x7f.png", "[31mred.png"),
     ("   ", "unnamed"), (None, "unnamed"), ("x" * 300 + ".jpg", "x" * 255)],
)
def test_clean_filename_strips_paths_and_control_characters(raw, expected):
    assert image_service.clean_filename(raw) == expected


def test_upload_size_limit_is_inclusive(client, test_settings):
    iid = _inspection(client)
    image = make_image_bytes()
    limit = test_settings.max_upload_bytes
    exact = image + b"\x00" * (limit - len(image))
    assert len(exact) == limit
    assert _upload(client, iid, exact).status_code == 201
    over = make_image_bytes(color=(1, 2, 3))
    over += b"\x00" * (limit + 1 - len(over))
    r = _upload(client, iid, over)
    assert (r.status_code, r.json()["error"]["code"]) == (413, "file_too_large")


def test_request_body_far_larger_than_the_limit_is_rejected(client, test_settings):
    iid = _inspection(client)
    junk = b"\x00" * (3 * test_settings.max_upload_bytes)
    r = _upload(client, iid, extra_files=[("junk", ("junk.bin", junk, "application/octet-stream"))])
    assert (r.status_code, r.json()["error"]["code"]) == (413, "file_too_large")


def test_chunked_body_without_content_length_is_refused(client):
    iid = _inspection(client)

    def body():
        yield b"x" * 10

    r = client.post(f"/api/v1/inspections/{iid}/images", content=body())
    assert (r.status_code, r.json()["error"]["code"]) == (411, "length_required")


def test_content_endpoint_headers(client):
    iid = _inspection(client)
    content = client.get(_upload(client, iid).json()["content_url"])
    assert content.headers["cache-control"] == "private, max-age=3600"
    assert content.headers["x-content-type-options"] == "nosniff"
    assert "attachment" not in content.headers.get("content-disposition", "")


def test_unsafe_database_path_is_never_served(client, db_session, storage):
    iid = _inspection(client)
    image = _upload(client, iid).json()
    row = db_session.get(InspectionImage, uuid.UUID(image["id"]))
    row.storage_path = "../../../../etc/passwd"
    db_session.commit()
    r = client.get(image["content_url"])
    assert (r.status_code, r.json()["error"]["code"]) == (404, "image_file_missing")
    assert "passwd" not in r.text and str(storage.root) not in r.text


def test_missing_file_on_disk_is_a_clean_404(client, storage):
    iid = _inspection(client)
    image = _upload(client, iid).json()
    for path in _stored_files(storage):
        path.unlink()
    r = client.get(image["content_url"])
    assert (r.status_code, r.json()["error"]["code"]) == (404, "image_file_missing")
    assert str(storage.root) not in r.text


def test_unexpected_storage_failure_returns_generic_500_without_details(
    client, db_session, storage, monkeypatch
):
    iid = _inspection(client)

    def disk_full(self, relative_path, data):
        raise OSError(28, "No space left on device", f"/very/secret/{relative_path}")

    monkeypatch.setattr(ImageStorage, "write", disk_full)
    raw_client = TestClient(app, raise_server_exceptions=False)
    r = _upload(raw_client, iid)
    assert (r.status_code, r.json()) == (
        500, {"error": {"code": "internal_error", "message": "An unexpected error occurred."}}
    )
    assert "secret" not in r.text and "No space" not in r.text
    db_session.rollback()
    monkeypatch.undo()
    assert client.get(f"/api/v1/inspections/{iid}/images").json()["total"] == 0
    assert _upload(client, iid).status_code == 201


def _allowed_origins() -> list[str]:
    cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    return list(cors.kwargs["allow_origins"])


def _preflight(client, origin: str, method: str):
    return client.options(
        "/api/v1/inspections",
        headers={"Origin": origin, "Access-Control-Request-Method": method},
    )


def test_cors_allows_only_configured_origins_and_methods(client):
    origins = _allowed_origins()
    assert "*" not in origins
    ok = _preflight(client, origins[0], "DELETE")
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == origins[0]
    evil = _preflight(client, "https://evil.example", "POST")
    assert "access-control-allow-origin" not in evil.headers
    assert _preflight(client, origins[0], "PUT").status_code == 400
    simple = client.get("/api/v1/inspections", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in simple.headers
    assert "access-control-allow-credentials" not in ok.headers


def test_repeated_delete_is_404_and_keeps_other_images(client, storage):
    iid = _inspection(client)
    first = _upload(client, iid).json()
    second = _upload(client, iid, make_image_bytes(color=(9, 9, 200))).json()
    url = f"/api/v1/inspections/{iid}/images/{first['id']}"
    assert client.delete(url).status_code == 204
    r = client.delete(url)
    assert (r.status_code, r.json()["error"]["code"]) == (404, "not_found")
    assert client.get(second["content_url"]).status_code == 200
    assert len(_stored_files(storage)) == 1


def test_concurrent_identical_uploads_store_exactly_one(db_engine, db_session, storage, test_settings):
    inspection = Inspection(title="race")
    db_session.add(inspection)
    db_session.commit()
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    data = make_image_bytes(color=(10, 120, 240))

    def attempt(i: int):
        with factory() as session:
            try:
                return image_service.upload_image(
                    session, storage, test_settings, inspection.id, data, f"{i}.jpg"
                ).id
            except AppError as exc:
                return exc.code

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, range(4)))

    assert sum(isinstance(r, uuid.UUID) for r in results) == 1
    assert results.count("duplicate_image") == 3
    assert len(_stored_files(storage)) == 1
