import time

from fastapi.testclient import TestClient

from tuxun.server import create_app


def test_web_flow(library, photos):
    client = TestClient(create_app(library))
    assert "图寻" in client.get("/").text

    assert client.post("/api/folders", json={"path": str(photos)}).status_code == 200
    assert client.post("/api/folders", json={"path": "/definitely/missing"}).status_code == 400

    assert client.post("/api/index").json()["started"] is True
    for _ in range(100):
        status = client.get("/api/status").json()
        if not status["indexing"]:
            break
        time.sleep(0.05)
    assert status["count"] == 3 and status["error"] is None

    results = client.get("/api/search", params={"q": "红"}).json()["results"]
    top = results[0]
    assert top["name"] == "red.jpg"

    thumb = client.get(f"/thumb/{top['id']}")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    assert client.get(f"/image/{top['id']}").status_code == 200
    assert client.get("/image/99999").status_code == 404

    similar = client.get(f"/api/similar/{top['id']}").json()["results"]
    assert top["id"] not in [r["id"] for r in similar]

    assert client.get("/api/search", params={"q": "红", "date_from": "bad"}).status_code == 400
    assert client.get("/api/search", params={"q": "红", "folder": str(photos.resolve())}).json()["results"]
