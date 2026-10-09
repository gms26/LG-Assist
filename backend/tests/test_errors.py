"""Structured error envelope tests."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_404_uses_error_envelope():
    r = client.get("/api/v1/does-not-exist")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "not_found"
    assert err["message"]
    assert err["request_id"]


def test_request_id_header_matches_envelope():
    r = client.get("/api/v1/does-not-exist")
    err = r.json()["error"]
    assert r.headers["x-request-id"] == err["request_id"]


def test_client_supplied_request_id_is_preserved():
    r = client.get("/api/v1/does-not-exist", headers={"X-Request-ID": "abc123"})
    assert r.json()["error"]["request_id"] == "abc123"
    assert r.headers["x-request-id"] == "abc123"
