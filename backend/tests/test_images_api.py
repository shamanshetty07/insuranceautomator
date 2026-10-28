import hashlib
import io
import uuid

from fastapi.testclient import TestClient
from PIL import Image

from app.models import ReportPrediction
from app.services.image_storage import ImageStorage
from tests.conftest import requires_db
from tests.factories import make_image as make_image_row
from tests.factories import make_prediction, make_report, make_run
from tests.image_utils import make_image_bytes

pytestmark = requires_db


def _inspection(client: TestClient) -> str:
    return client.post("/api/v1/inspections", json={"title": "t"}).json()["id"]


def _upload(client, inspection_id, data=None, name="photo.jpg", mime="image/jpeg", color=(200, 30, 30)):
    data = data if data is not None else make_image_bytes(color=color)
    return client.post(
        f"/api/v1/inspections/{inspection_id}/images", files={"file": (name, data, mime)}
    )


def test_upload_jpeg_and_png(client: TestClient):
    iid = _inspection(client)
    jpeg = _upload(client, iid)
    png = _upload(client, iid, make_image_bytes("PNG", color=(1, 2, 3)), "p.png", "image/png")
    assert jpeg.status_code == png.status_code == 201
    body = jpeg.json()
    assert body["media_type"] == "image/jpeg" and body["image_order"] == 0
    assert (body["width_px"], body["height_px"]) == (64, 32)
    assert png.json()["media_type"] == "image/png" and png.json()["image_order"] == 1
    assert body["content_url"] == f"/api/v1/inspections/{iid}/images/{body['id']}/content"


def test_response_never_leaks_storage_details(client: TestClient, storage: ImageStorage):
    iid = _inspection(client)
    r = _upload(client, iid, name="../../etc/passwd.jpg")
    text = r.text
    assert "storage_path" not in r.json()
    assert str(storage.root) not in text
    assert r.json()["original_filename"] == "passwd.jpg"  # display copy only
    stored = [p for p in storage.root.rglob("*") if p.is_file()]
    assert len(stored) == 1 and "passwd" not in stored[0].name  # server-generated name


def test_upload_to_missing_inspection(client: TestClient):
    r = _upload(client, uuid.uuid4())
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_malformed_inspection_id(client: TestClient):
    assert _upload(client, "nope").status_code == 422


def test_missing_file_field(client: TestClient):
    iid = _inspection(client)
    r = client.post(f"/api/v1/inspections/{iid}/images")
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_rejects_empty_non_image_oversized_and_unsupported(client: TestClient, storage: ImageStorage):
    iid = _inspection(client)
    assert _upload(client, iid, b"").json()["error"]["code"] == "empty_file"
    spoof = _upload(client, iid, b"not an image at all", "evil.jpg", "image/jpeg")
    assert (spoof.status_code, spoof.json()["error"]["code"]) == (415, "unsupported_media_type")
    big = _upload(client, iid, b"\xff\xd8" + b"0" * 300_000)
    assert (big.status_code, big.json()["error"]["code"]) == (413, "file_too_large")
    gif = _upload(client, iid, make_image_bytes("GIF"), "a.gif", "image/gif")
    assert gif.status_code == 415
    assert not any(p.is_file() for p in storage.root.rglob("*")) if storage.root.exists() else True
    assert client.get(f"/api/v1/inspections/{iid}/images").json()["total"] == 0


def test_real_content_wins_over_declared_mime_type(client: TestClient):
    iid = _inspection(client)
    r = _upload(client, iid, make_image_bytes("PNG"), "x.jpg", "image/jpeg")  # PNG labelled JPEG
    assert r.status_code == 201 and r.json()["media_type"] == "image/png"


def test_list_get_and_content(client: TestClient):
    iid = _inspection(client)
    created = _upload(client, iid).json()
    listing = client.get(f"/api/v1/inspections/{iid}/images").json()
    assert listing["total"] == 1 and listing["items"][0]["id"] == created["id"]
    assert client.get(f"/api/v1/inspections/{iid}/images/{created['id']}").json() == created

    content = client.get(created["content_url"])
    assert content.status_code == 200
    assert content.headers["content-type"] == "image/jpeg"
    assert content.headers["x-content-type-options"] == "nosniff"
    assert hashlib.sha256(content.content).hexdigest() == created["sha256"]
    assert len(content.content) == created["file_size_bytes"]
    assert Image.open(io.BytesIO(content.content)).size == (created["width_px"], created["height_px"])


def test_list_empty_and_missing_inspection(client: TestClient):
    iid = _inspection(client)
    assert client.get(f"/api/v1/inspections/{iid}/images").json() == {"items": [], "total": 0}
    assert client.get(f"/api/v1/inspections/{uuid.uuid4()}/images").status_code == 404


def test_image_through_wrong_inspection_is_not_found(client: TestClient):
    a, b = _inspection(client), _inspection(client)
    image = _upload(client, a).json()
    for suffix in ("", "/content"):
        r = client.get(f"/api/v1/inspections/{b}/images/{image['id']}{suffix}")
        assert r.status_code == 404
    assert client.delete(f"/api/v1/inspections/{b}/images/{image['id']}").status_code == 404
    assert client.get(image["content_url"]).status_code == 200  # untouched


def test_exif_rotated_upload_reports_normalized_dimensions(client: TestClient):
    iid = _inspection(client)
    body = _upload(client, iid, make_image_bytes("JPEG", (64, 32), orientation=6)).json()
    assert (body["width_px"], body["height_px"], body["exif_orientation"]) == (32, 64, 6)
    stored = Image.open(io.BytesIO(client.get(body["content_url"]).content))
    assert stored.size == (32, 64)


def test_image_count_limit_enforced(client: TestClient):
    iid = _inspection(client)
    for i in range(3):
        assert _upload(client, iid, color=(i * 40, 0, 0)).status_code == 201
    r = _upload(client, iid, color=(250, 1, 1))
    assert (r.status_code, r.json()["error"]["code"]) == (409, "image_limit_reached")


def test_ordering_is_sequential_and_survives_deletion(client: TestClient):
    iid = _inspection(client)
    ids = [_upload(client, iid, color=(i * 50, 5, 5)).json() for i in range(3)]
    assert [i["image_order"] for i in ids] == [0, 1, 2]
    assert client.delete(f"/api/v1/inspections/{iid}/images/{ids[1]['id']}").status_code == 204
    after = _upload(client, iid, color=(9, 9, 200)).json()
    assert after["image_order"] == 3  # never reuses an order, so no collision
    listing = client.get(f"/api/v1/inspections/{iid}/images").json()["items"]
    assert [i["image_order"] for i in listing] == [0, 2, 3]


def test_duplicate_submission_is_rejected_safely(client: TestClient, storage: ImageStorage):
    iid = _inspection(client)
    data = make_image_bytes()
    assert _upload(client, iid, data).status_code == 201
    again = _upload(client, iid, data)
    assert (again.status_code, again.json()["error"]["code"]) == (409, "duplicate_image")
    assert client.get(f"/api/v1/inspections/{iid}/images").json()["total"] == 1
    assert len([p for p in storage.root.rglob("*") if p.is_file()]) == 1  # no stray file
    other = _inspection(client)
    assert _upload(client, other, data).status_code == 201  # same photo in another inspection is fine


def test_delete_removes_row_and_file(client: TestClient, storage: ImageStorage):
    iid = _inspection(client)
    image = _upload(client, iid).json()
    assert client.delete(f"/api/v1/inspections/{iid}/images/{image['id']}").status_code == 204
    assert client.get(f"/api/v1/inspections/{iid}/images/{image['id']}").status_code == 404
    assert not any(p.is_file() for p in storage.root.rglob("*"))


def test_delete_refused_when_report_provenance_depends_on_it(client: TestClient, db_session, storage):
    iid = _inspection(client)
    image = _upload(client, iid).json()
    from app.models import InspectionImage, Inspection

    row = db_session.get(InspectionImage, uuid.UUID(image["id"]))
    run = make_run(row)
    pred = make_prediction(run)
    db_session.add_all([run, pred])
    db_session.flush()
    report = make_report(db_session.get(Inspection, uuid.UUID(iid)))
    db_session.add(report)
    db_session.flush()
    db_session.add(ReportPrediction(generated_report_id=report.id, damage_prediction_id=pred.id))
    db_session.commit()

    r = client.delete(f"/api/v1/inspections/{iid}/images/{image['id']}")
    assert (r.status_code, r.json()["error"]["code"]) == (409, "image_in_use")
    assert client.get(image["content_url"]).status_code == 200  # row and file both intact
