import uuid

from fastapi.testclient import TestClient

from tests.conftest import requires_db

pytestmark = requires_db


def test_list_empty(client: TestClient):
    r = client.get("/api/v1/inspections")
    assert r.status_code == 200
    assert r.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_create_with_all_fields(client: TestClient):
    r = client.post("/api/v1/inspections", json={"title": "  Front bumper  ", "notes": "Rear-ended"})
    assert r.status_code == 201
    body = r.json()
    assert uuid.UUID(body["id"])
    assert body["title"] == "Front bumper"
    assert body["notes"] == "Rear-ended"
    assert body["status"] == "created"
    assert body["created_at"] and body["updated_at"]


def test_create_with_no_fields(client: TestClient):
    r = client.post("/api/v1/inspections", json={})
    assert r.status_code == 201
    assert r.json()["title"] is None
    assert r.json()["notes"] is None


def test_blank_strings_become_null(client: TestClient):
    r = client.post("/api/v1/inspections", json={"title": "   ", "notes": ""})
    assert r.status_code == 201
    assert r.json()["title"] is None
    assert r.json()["notes"] is None


def test_get_inspection(client: TestClient):
    created = client.post("/api/v1/inspections", json={"title": "A"}).json()
    r = client.get(f"/api/v1/inspections/{created['id']}")
    assert r.status_code == 200
    assert r.json() == created


def test_list_pagination_newest_first(client: TestClient):
    for i in range(3):
        client.post("/api/v1/inspections", json={"title": f"t{i}"})
    page = client.get("/api/v1/inspections", params={"limit": 2, "offset": 0}).json()
    assert page["total"] == 3
    assert [i["title"] for i in page["items"]] == ["t2", "t1"]
    rest = client.get("/api/v1/inspections", params={"limit": 2, "offset": 2}).json()
    assert [i["title"] for i in rest["items"]] == ["t0"]


def test_get_unknown_returns_404(client: TestClient):
    r = client.get(f"/api/v1/inspections/{uuid.uuid4()}")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_get_malformed_id_returns_422(client: TestClient):
    r = client.get("/api/v1/inspections/not-a-uuid")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_create_rejects_overlong_title(client: TestClient):
    r = client.post("/api/v1/inspections", json={"title": "x" * 201})
    assert r.status_code == 422
    assert r.json()["error"]["details"][0]["field"] == "body.title"


def test_create_rejects_wrong_type_and_unknown_fields(client: TestClient):
    assert client.post("/api/v1/inspections", json={"title": 5}).status_code == 422
    assert client.post("/api/v1/inspections", json={"status": "completed"}).status_code == 422


def test_list_rejects_bad_pagination(client: TestClient):
    assert client.get("/api/v1/inspections", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/inspections", params={"limit": 101}).status_code == 422
    assert client.get("/api/v1/inspections", params={"offset": -1}).status_code == 422
