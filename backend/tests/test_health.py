"""Health endpoint tests. Run without Postgres/Ollama: the endpoint must
still return 200 and report each dependency separately (degraded, not 500)."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_health_status_code_and_shape():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert set(body["checks"]) == {"database", "ollama"}
    for check in body["checks"].values():
        assert check["status"] in ("ok", "degraded", "unreachable")


def test_health_reports_provider_info():
    body = client.get("/api/v1/health").json()
    provider = body["provider"]
    assert provider["active"] == "ollama"
    assert provider["configured"] == "ollama"
    assert provider["chat_model"]
    assert provider["embed_model"] == "nomic-embed-text"
    assert provider["cloud_configured"] is False


def test_health_degraded_without_dependencies():
    # In CI/sandbox there is no Postgres or Ollama; health must say so
    # honestly instead of crashing.
    body = client.get("/api/v1/health").json()
    assert body["status"] == "degraded"
    assert body["checks"]["database"]["status"] == "unreachable"
    assert body["checks"]["ollama"]["status"] == "unreachable"
